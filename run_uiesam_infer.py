import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


UIESAM_ROOT = Path("/mnt/disk1new/wzl/UIE_SAM")
UIESAM_CODE = UIESAM_ROOT / "UIE_SAM"
if str(UIESAM_CODE) not in sys.path:
    sys.path.insert(0, str(UIESAM_CODE))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402
from tool import High_pass, compensate, fusion  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(UIESAM_ROOT / "sam_vit_h_4b8939.pth"),
    )
    parser.add_argument("--model_type", default="vit_h")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gbc", type=float, default=0.92)
    parser.add_argument("--rgc", type=float, default=0.92)
    parser.add_argument("--high_pass_sigma", type=float, default=5.0)
    return parser.parse_args()


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(path: Path):
    image = Image.open(path).convert("RGB")
    image_u8 = np.array(image)
    image_f32 = image_u8.astype(np.float32) / 255.0
    return image_f32, image_u8


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
        device = "cuda"
    else:
        device = args.device

    image_paths = sorted(path for path in input_dir.iterdir() if is_image_file(path))
    print(f"UIE-SAM input: {input_dir}")
    print(f"UIE-SAM output: {output_dir}")
    print(f"UIE-SAM images: {len(image_paths)}")
    print(f"UIE-SAM checkpoint: {args.checkpoint}")
    print(f"UIE-SAM device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)

    with torch.no_grad():
        for idx, path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] {path.name}", flush=True)
            out_name = path.stem + ".jpg"
            out_path = output_dir / out_name
            if out_path.exists():
                continue
            raw0, i0 = load_rgb_image(path)
            masks = mask_generator.generate(i0)
            if not masks:
                mask = np.zeros(i0.shape[:2], dtype=np.uint8)
            else:
                mask = np.array(masks[0]["segmentation"], dtype=np.uint8)
            forward = np.where(mask == 0)
            back = np.where(mask == 1)
            forward_image = compensate(raw0.copy(), forward, args.gbc, args.rgc)
            back_image = compensate(raw0.copy(), back, args.gbc, args.rgc)
            output_image = fusion(forward_image, back_image, mask)
            output_image = High_pass(raw0, output_image, args.high_pass_sigma)
            output_image = np.float32(np.minimum(np.maximum(output_image, 0), 1))
            plt.imsave(out_path, output_image)


if __name__ == "__main__":
    main()
