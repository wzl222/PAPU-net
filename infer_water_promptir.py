import argparse
import os
import subprocess
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from net.model import PromptIR
from utils.image_io import save_image_tensor
from utils.uiqm_postprocess_utils import enhance_rgb_float, preset_params


class PromptIRModel(pl.LightningModule):
    def __init__(self, water_aware=False, color_correction=False):
        super().__init__()
        self.net = PromptIR(
            decoder=True,
            water_aware=water_aware,
            color_correction=color_correction,
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


class ImageFolderDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.to_tensor = ToTensor()
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(
            p for p in self.input_dir.iterdir() if p.suffix.lower() in extensions
        )
        if not self.paths:
            raise RuntimeError(f"No image files found in {self.input_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = np.array(Image.open(path).convert("RGB"))
        return path.stem, self.to_tensor(image)


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
    tile = tile - tile % 8
    assert tile % 8 == 0, "tile size should be multiple of 8"
    if tile <= tile_overlap:
        tile_overlap = max(0, tile // 4)

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


def apply_uiqm_postprocess_batch(restored, clahe, saturation, contrast, sharpness):
    restored_np = restored.detach().cpu().permute(0, 2, 3, 1).numpy()
    processed = [
        enhance_rgb_float(image, clahe, saturation, contrast, sharpness)
        for image in restored_np
    ]
    processed = np.stack(processed, axis=0)
    return torch.from_numpy(processed).permute(0, 3, 1, 2).to(restored.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--water_aware", action="store_true")
    parser.add_argument("--color_correction", action="store_true")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--uiqm_postprocess", action="store_true")
    parser.add_argument("--uiqm_preset", choices=["mild", "strong"], default=None)
    parser.add_argument("--uiqm_clahe", type=float, default=0.0)
    parser.add_argument("--uiqm_saturation", type=float, default=1.0)
    parser.add_argument("--uiqm_contrast", type=float, default=1.0)
    parser.add_argument("--uiqm_sharpness", type=float, default=0.0)
    args = parser.parse_args()

    subprocess.check_output(["mkdir", "-p", args.output_path])
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    net = PromptIRModel.load_from_checkpoint(
        args.ckpt_path,
        water_aware=args.water_aware,
        color_correction=args.color_correction,
        strict=False,
    ).to(device)
    net.eval()

    dataset = ImageFolderDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Images: {len(dataset)}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_path}")
    if args.uiqm_postprocess:
        params = preset_params(args.uiqm_preset) if args.uiqm_preset else {
            "clahe": args.uiqm_clahe,
            "saturation": args.uiqm_saturation,
            "contrast": args.uiqm_contrast,
            "sharpness": args.uiqm_sharpness,
        }
        print(
            "UIQM postprocess: "
            f"clahe={params['clahe']} saturation={params['saturation']} "
            f"contrast={params['contrast']} sharpness={params['sharpness']}"
        )
    with torch.no_grad():
        for names, degraded in tqdm(loader):
            degraded = degraded.to(device)
            degraded, height, width = pad_input(degraded)
            if args.tile:
                restored = tile_eval(
                    net,
                    degraded,
                    tile=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
            else:
                restored = net(degraded)
            restored = restored[:, :, :height, :width]
            if args.uiqm_postprocess:
                restored = apply_uiqm_postprocess_batch(
                    restored,
                    params["clahe"],
                    params["saturation"],
                    params["contrast"],
                    params["sharpness"],
                )
                restored = torch.clamp(restored, 0, 1)
            save_image_tensor(restored, os.path.join(args.output_path, names[0] + ".png"))


if __name__ == "__main__":
    main()
