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
    def __init__(self, input_dir, target_dir, patch_size=128):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.to_tensor = ToTensor()
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.names = sorted(
            p.name for p in self.input_dir.iterdir() if p.suffix.lower() in extensions
        )
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
        h, w = degraded.shape[:2]
        crop = min(self.patch_size, h, w)
        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
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
        degraded, clean = self._paired_augment(degraded, clean)
        return [Path(name).stem], self.to_tensor(degraded), self.to_tensor(clean)


class WaterPromptIRModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        max_epochs=40,
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
        freeze_backbone=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(
            decoder=True,
            water_aware=True,
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

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=3, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]


def load_pretrain(model, ckpt_path):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        "Loaded pretrain checkpoint with "
        f"{len(missing)} missing and {len(unexpected)} unexpected keys."
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
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_color", type=float, default=0.03)
    parser.add_argument("--lambda_structure", type=float, default=0.05)
    parser.add_argument("--lambda_color_dist", type=float, default=0.0)
    parser.add_argument("--lambda_edge", type=float, default=0.0)
    parser.add_argument("--lambda_local_contrast", type=float, default=0.0)
    parser.add_argument("--lambda_high_freq", type=float, default=0.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.0)
    parser.add_argument("--jitter_brightness", type=float, default=0.15)
    parser.add_argument("--jitter_contrast", type=float, default=0.15)
    parser.add_argument("--jitter_saturation", type=float, default=0.15)
    parser.add_argument("--color_correction", action="store_true")
    parser.add_argument("--frequency_refinement", action="store_true")
    parser.add_argument("--local_contrast_refinement", action="store_true")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir or os.path.join(args.uieb_root, "train", "input")
    target_dir = args.target_dir or os.path.join(args.uieb_root, "train", "target")
    subprocess.check_output(["mkdir", "-p", args.ckpt_dir])

    pl.seed_everything(0, workers=True)
    train_set = UIEBTrainDataset(input_dir, target_dir, patch_size=args.patch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = WaterPromptIRModel(
        lr=args.lr,
        max_epochs=args.epochs,
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
        freeze_backbone=args.freeze_backbone,
    )
    if args.resume_ckpt is None:
        load_pretrain(model, args.pretrain_ckpt)

    logger = CSVLogger(save_dir=args.log_dir, name="")
    checkpoint_callback = ModelCheckpoint(
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
        callbacks=[checkpoint_callback],
        log_every_n_steps=20,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
