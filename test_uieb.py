import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

import lightning.pytorch as pl

from net.model import PromptIR
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter


class PromptIRModel(pl.LightningModule):
    def __init__(
        self,
        water_aware=False,
        frequency_refinement=False,
        local_contrast_refinement=False,
    ):
        super().__init__()
        self.net = PromptIR(
            decoder=True,
            water_aware=water_aware,
            frequency_refinement=frequency_refinement,
            local_contrast_refinement=local_contrast_refinement,
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


class UIEBTestDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
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

    def __getitem__(self, idx):
        name = self.names[idx]
        degraded = np.array(Image.open(self.input_dir / name).convert("RGB"))
        clean = np.array(Image.open(self.target_dir / name).convert("RGB"))

        return [Path(name).stem], self.to_tensor(degraded), self.to_tensor(clean)


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    h_pad = (img_multiple_of - height % img_multiple_of) % img_multiple_of
    w_pad = (img_multiple_of - width % img_multiple_of) % img_multiple_of
    if h_pad or w_pad:
        input_ = F.pad(input_, (0, w_pad, 0, h_pad), "reflect")
    return input_, height, width


def tile_eval(model, input_, tile=256, tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    output = torch.zeros(b, c, h, w).type_as(input_)
    weight = torch.zeros_like(output)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            output[..., h_idx : h_idx + tile, w_idx : w_idx + tile].add_(out_patch)
            weight[..., h_idx : h_idx + tile, w_idx : w_idx + tile].add_(
                torch.ones_like(out_patch)
            )

    return torch.clamp(output.div_(weight), 0, 1)


def compute_metrics(restored, clean):
    restored = np.clip(restored.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)

    psnr = 0.0
    ssim = 0.0
    for i in range(restored.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], restored[i], data_range=1)
        ssim += structural_similarity(
            clean[i], restored[i], data_range=1, channel_axis=-1
        )
    count = restored.shape[0]
    return psnr / count, ssim / count, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--uieb_root", type=str, default="/mnt/disk1new/wzl/wt/data/UIEB/test"
    )
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output/uieb/")
    parser.add_argument("--ckpt_name", type=str, default="model.ckpt")
    parser.add_argument("--water_aware", action="store_true")
    parser.add_argument("--frequency_refinement", action="store_true")
    parser.add_argument("--local_contrast_refinement", action="store_true")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    opt = parser.parse_args()

    input_dir = opt.input_dir or os.path.join(opt.uieb_root, "input")
    target_dir = opt.target_dir or os.path.join(opt.uieb_root, "target")
    ckpt_path = opt.ckpt_name if os.path.exists(opt.ckpt_name) else os.path.join("ckpt", opt.ckpt_name)

    subprocess.check_output(["mkdir", "-p", opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)

    print(f"CKPT name : {ckpt_path}")
    print(f"UIEB input: {input_dir}")
    print(f"UIEB target: {target_dir}")

    net = PromptIRModel.load_from_checkpoint(
        ckpt_path,
        water_aware=opt.water_aware,
        frequency_refinement=opt.frequency_refinement,
        local_contrast_refinement=opt.local_contrast_refinement,
        strict=False,
    ).to(device)
    net.eval()

    test_set = UIEBTestDataset(input_dir, target_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([name], degraded, clean) in tqdm(test_loader):
            degraded = degraded.to(device)
            clean = clean.to(device)

            degraded, height, width = pad_input(degraded)
            if opt.tile:
                restored = tile_eval(
                    net,
                    degraded,
                    tile=opt.tile_size,
                    tile_overlap=opt.tile_overlap,
                )
            else:
                restored = net(degraded)
            restored = restored[:, :, :height, :width]
            temp_psnr, temp_ssim, count = compute_metrics(restored, clean)
            psnr.update(temp_psnr, count)
            ssim.update(temp_ssim, count)

            save_image_tensor(restored, os.path.join(opt.output_path, name[0] + ".png"))

    print(f"UIEB: images: {len(test_set)}, PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")
    print(f"Saved results to: {opt.output_path}")


if __name__ == "__main__":
    main()
