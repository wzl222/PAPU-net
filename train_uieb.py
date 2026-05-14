import argparse
import os
import random
import subprocess
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from net.model import PromptIR
from utils.loss_utils import (
    ColorBalanceLoss,
    ColorDistributionLoss,
    EdgeIntensityLoss,
    GradientStructureLoss,
    HighFrequencyLoss,
    LocalContrastLoss,
)
from utils.schedulers import LinearWarmupCosineAnnealingLR


class UIEBTrainDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        patch_size=128,
        names=None,
        augment=True,
        random_crop=True,
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.random_crop = random_crop
        self.to_tensor = ToTensor()
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        if names is None:
            self.names = sorted(
                p.name for p in self.input_dir.iterdir() if p.suffix.lower() in extensions
            )
        else:
            self.names = list(names)
        if not self.names:
            raise RuntimeError(f"No image files found in {self.input_dir}")
        missing = [name for name in self.names if not (self.target_dir / name).exists()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} target files are missing, first one: {missing[0]}"
            )

    def __len__(self):
        return len(self.names)

    def _paired_crop(self, degraded, clean):
        target_crop = max(8, self.patch_size - self.patch_size % 8)
        h, w = degraded.shape[:2]
        pad_h = max(0, target_crop - h)
        pad_w = max(0, target_crop - w)
        if pad_h or pad_w:
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad
            degraded = np.pad(
                degraded,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="reflect",
            )
            clean = np.pad(
                clean,
                ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                mode="reflect",
            )
            h, w = degraded.shape[:2]

        crop = target_crop
        if self.random_crop:
            top = random.randint(0, h - crop)
            left = random.randint(0, w - crop)
        else:
            top = (h - crop) // 2
            left = (w - crop) // 2
        degraded = degraded[top : top + crop, left : left + crop]
        clean = clean[top : top + crop, left : left + crop]
        return degraded, clean

    def _paired_augment(self, degraded, clean):
        if random.random() < 0.5:
            degraded = np.flip(degraded, axis=1)
            clean = np.flip(clean, axis=1)
        if random.random() < 0.5:
            degraded = np.flip(degraded, axis=0)
            clean = np.flip(clean, axis=0)
        k = random.randint(0, 3)
        if k:
            degraded = np.rot90(degraded, k)
            clean = np.rot90(clean, k)
        return degraded.copy(), clean.copy()

    def __getitem__(self, idx):
        name = self.names[idx]
        degraded = np.array(Image.open(self.input_dir / name).convert("RGB"))
        clean = np.array(Image.open(self.target_dir / name).convert("RGB"))
        degraded, clean = self._paired_crop(degraded, clean)
        if self.augment:
            degraded, clean = self._paired_augment(degraded, clean)
        return [Path(name).stem], self.to_tensor(degraded), self.to_tensor(clean)


class WaterPromptIRModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        backbone_lr=None,
        module_lr=None,
        max_epochs=40,
        warmup_epochs=3,
        lambda_color=0.03,
        lambda_structure=0.05,
        lambda_color_dist=0.0,
        lambda_edge=0.0,
        lambda_local_contrast=0.0,
        lambda_high_freq=0.0,
        lambda_consistency=0.0,
        jitter_brightness=0.15,
        jitter_contrast=0.15,
        jitter_saturation=0.15,
        color_correction=False,
        frequency_refinement=False,
        local_contrast_refinement=False,
        prior_coupling=True,
        disabled_priors=None,
        prompt_conditioning=True,
        modulation=True,
        stage_specific_priors=True,
        refinement_conditioning=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(
            decoder=True,
            water_aware=True,
            prior_coupling=prior_coupling,
            disabled_priors=disabled_priors,
            prompt_conditioning=prompt_conditioning,
            modulation=modulation,
            stage_specific_priors=stage_specific_priors,
            refinement_conditioning=refinement_conditioning,
            color_correction=color_correction,
            frequency_refinement=frequency_refinement,
            local_contrast_refinement=local_contrast_refinement,
        )
        self.rec_loss = nn.L1Loss()
        self.color_loss = ColorBalanceLoss()
        self.color_dist_loss = ColorDistributionLoss()
        self.structure_loss = GradientStructureLoss()
        self.edge_loss = EdgeIntensityLoss()
        self.local_contrast_loss = LocalContrastLoss()
        self.high_freq_loss = HighFrequencyLoss()

        if freeze_backbone:
            for name, param in self.net.named_parameters():
                param.requires_grad = name.startswith("water_")

    def forward(self, x):
        return self.net(x)

    def _color_jitter(self, image):
        batch_size = image.shape[0]
        view_shape = (batch_size, 1, 1, 1)
        jittered = image

        if self.hparams.jitter_brightness > 0:
            scale = 1.0 + torch.empty(
                view_shape, device=image.device, dtype=image.dtype
            ).uniform_(-self.hparams.jitter_brightness, self.hparams.jitter_brightness)
            jittered = jittered * scale

        if self.hparams.jitter_contrast > 0:
            mean = jittered.mean(dim=(-2, -1), keepdim=True)
            scale = 1.0 + torch.empty(
                view_shape, device=image.device, dtype=image.dtype
            ).uniform_(-self.hparams.jitter_contrast, self.hparams.jitter_contrast)
            jittered = (jittered - mean) * scale + mean

        if self.hparams.jitter_saturation > 0:
            gray = (
                0.299 * jittered[:, 0:1]
                + 0.587 * jittered[:, 1:2]
                + 0.114 * jittered[:, 2:3]
            )
            scale = 1.0 + torch.empty(
                view_shape, device=image.device, dtype=image.dtype
            ).uniform_(-self.hparams.jitter_saturation, self.hparams.jitter_saturation)
            jittered = (jittered - gray) * scale + gray

        return jittered.clamp(0.0, 1.0)

    def training_step(self, batch, batch_idx):
        _, degraded, clean = batch
        restored = self.net(degraded)
        rec_loss = self.rec_loss(restored, clean)
        color_loss = self.color_loss(restored)
        color_dist_loss = self.color_dist_loss(restored, clean)
        structure_loss = self.structure_loss(restored, clean)
        edge_loss = self.edge_loss(restored, clean)
        local_contrast_loss = self.local_contrast_loss(restored, clean)
        high_freq_loss = self.high_freq_loss(restored, clean)
        consistency_loss = restored.new_tensor(0.0)
        if self.hparams.lambda_consistency > 0:
            restored_aug = self.net(self._color_jitter(degraded))
            consistency_loss = self.rec_loss(restored_aug, restored.detach())
        loss = (
            rec_loss
            + self.hparams.lambda_color * color_loss
            + self.hparams.lambda_color_dist * color_dist_loss
            + self.hparams.lambda_structure * structure_loss
            + self.hparams.lambda_edge * edge_loss
            + self.hparams.lambda_local_contrast * local_contrast_loss
            + self.hparams.lambda_high_freq * high_freq_loss
            + self.hparams.lambda_consistency * consistency_loss
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rec_loss", rec_loss)
        self.log("train_color_loss", color_loss)
        self.log("train_color_dist_loss", color_dist_loss)
        self.log("train_structure_loss", structure_loss)
        self.log("train_edge_loss", edge_loss)
        self.log("train_local_contrast_loss", local_contrast_loss)
        self.log("train_high_freq_loss", high_freq_loss)
        self.log("train_consistency_loss", consistency_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, degraded, clean = batch
        restored = self.net(degraded)
        rec_loss = self.rec_loss(restored, clean)
        psnr = -10.0 * torch.log10(torch.mean((restored - clean) ** 2) + 1e-8)
        self.log("val_loss", rec_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": rec_loss, "val_psnr": psnr}

    def _build_param_groups(self):
        module_prefixes = (
            "net.water_encoder",
            "net.water_mod_",
            "net.frequency_refine",
            "net.local_contrast_refine",
            "net.color_head",
        )
        backbone_lr = self.hparams.backbone_lr or self.hparams.lr
        module_lr = self.hparams.module_lr or self.hparams.lr
        backbone_params = []
        module_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(module_prefixes):
                module_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if module_params:
            param_groups.append({"params": module_params, "lr": module_lr})
        return param_groups

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self._build_param_groups(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]


def load_pretrain(model, ckpt_path):
    if ckpt_path is None or ckpt_path == "" or str(ckpt_path).lower() == "none":
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(
        "Loaded pretrain checkpoint with "
        f"{len(missing)} missing, {len(unexpected)} unexpected, "
        f"and {len(skipped)} shape-mismatched keys skipped."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uieb_root", type=str, default="/mnt/disk1new/wzl/wt/data/UIEB")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--pretrain_ckpt", type=str, default="ckpt/model.ckpt")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="train_ckpt/uieb_water_promptir")
    parser.add_argument("--log_dir", type=str, default="logs/uieb_water_promptir")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--val_patch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--module_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lambda_color", type=float, default=0.03)
    parser.add_argument("--lambda_structure", type=float, default=0.05)
    parser.add_argument("--lambda_color_dist", type=float, default=0.02)
    parser.add_argument("--lambda_edge", type=float, default=0.01)
    parser.add_argument("--lambda_local_contrast", type=float, default=0.0)
    parser.add_argument("--lambda_high_freq", type=float, default=0.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.0)
    parser.add_argument("--jitter_brightness", type=float, default=0.15)
    parser.add_argument("--jitter_contrast", type=float, default=0.15)
    parser.add_argument("--jitter_saturation", type=float, default=0.15)
    parser.add_argument("--color_correction", action="store_true")
    parser.add_argument("--frequency_refinement", action="store_true")
    parser.add_argument("--local_contrast_refinement", action="store_true")
    parser.add_argument(
        "--disabled_priors",
        nargs="*",
        default=None,
        choices=["color", "dcp", "luminance", "structure"],
        help="Disable selected physical prior tokens for ablation.",
    )
    parser.add_argument(
        "--disable_prior_coupling",
        action="store_true",
        help="Disable self-attention coupling among physical prior tokens for ablation.",
    )
    parser.add_argument(
        "--disable_prompt_conditioning",
        action="store_true",
        help="Disable physics-guided prompt conditioning and keep prompts feature-driven only.",
    )
    parser.add_argument(
        "--disable_modulation",
        action="store_true",
        help="Disable physics-guided feature modulation for ablation.",
    )
    parser.add_argument(
        "--shared_prior_token",
        action="store_true",
        help="Use a shared global prior token for all stages instead of stage-specific prior tokens.",
    )
    parser.add_argument(
        "--disable_refinement_conditioning",
        action="store_true",
        help="Disable physics-guided conditioning in the output refinement branches for ablation.",
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir or os.path.join(args.uieb_root, "train", "input")
    target_dir = args.target_dir or os.path.join(args.uieb_root, "train", "target")
    subprocess.check_output(["mkdir", "-p", args.ckpt_dir])

    pl.seed_everything(0, workers=True)
    all_names = sorted(
        p.name
        for p in Path(input_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not all_names:
        raise RuntimeError(f"No training images found in {input_dir}")
    val_count = max(1, int(len(all_names) * args.val_split))
    train_names = all_names[:-val_count]
    val_names = all_names[-val_count:]
    train_set = UIEBTrainDataset(
        input_dir,
        target_dir,
        patch_size=args.patch_size,
        names=train_names,
        augment=True,
        random_crop=True,
    )
    val_set = UIEBTrainDataset(
        input_dir,
        target_dir,
        patch_size=args.val_patch_size,
        names=val_names,
        augment=False,
        random_crop=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = WaterPromptIRModel(
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        module_lr=args.module_lr,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lambda_color=args.lambda_color,
        lambda_structure=args.lambda_structure,
        lambda_color_dist=args.lambda_color_dist,
        lambda_edge=args.lambda_edge,
        lambda_local_contrast=args.lambda_local_contrast,
        lambda_high_freq=args.lambda_high_freq,
        lambda_consistency=args.lambda_consistency,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
        color_correction=args.color_correction,
        frequency_refinement=args.frequency_refinement,
        local_contrast_refinement=args.local_contrast_refinement,
        prior_coupling=not args.disable_prior_coupling,
        disabled_priors=args.disabled_priors,
        prompt_conditioning=not args.disable_prompt_conditioning,
        modulation=not args.disable_modulation,
        stage_specific_priors=not args.shared_prior_token,
        refinement_conditioning=not args.disable_refinement_conditioning,
        freeze_backbone=args.freeze_backbone,
    )
    if args.resume_ckpt is None:
        load_pretrain(model, args.pretrain_ckpt)

    logger = CSVLogger(save_dir=args.log_dir, name="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="uieb-water-promptir-best-{epoch:03d}-{val_psnr:.3f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="uieb-water-promptir-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.cuda],
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback, epoch_checkpoint_callback],
        log_every_n_steps=20,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
