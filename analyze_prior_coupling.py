import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from net.model import PromptIR


DATASET_PATHS = {
    "UIEB_test": "/mnt/disk1new/wzl/wt/data/UIEB/test/input",
    "LSUI": "/mnt/disk1new/wzl/Semi-UIR/data/test/LSUI/input",
    "RUIE": "/mnt/disk1new/wzl/Semi-UIR/data/test/RUIE/input",
    "FUVD": "/mnt/disk1new/wzl/Semi-UIR/data/test/FUVD/input",
}

PRIOR_NAMES = ["Color", "DCP", "Luminance", "Structure"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default=(
            "train_ckpt/uieb_water_physicscoupled_v2_e10/"
            "uieb-water-promptir-best-epoch=003-val_psnr=24.257.ckpt"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="UIEB_test",
        choices=sorted(DATASET_PATHS.keys()),
    )
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        default="analysis/prior_coupling/uieb_attention_heatmap",
    )
    return parser.parse_args()


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    h_pad = (img_multiple_of - height % img_multiple_of) % img_multiple_of
    w_pad = (img_multiple_of - width % img_multiple_of) % img_multiple_of
    if h_pad or w_pad:
        input_ = F.pad(input_, (0, w_pad, 0, h_pad), "reflect")
    return input_


def load_model(ckpt_path, device):
    net = PromptIR(
        decoder=True,
        water_aware=True,
        frequency_refinement=True,
        local_contrast_refinement=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[4:] if key.startswith("net.") else key] = value
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    net.to(device).eval()
    return net


def sample_images(dataset_name, sample_count, seed):
    root = Path(DATASET_PATHS[dataset_name])
    paths = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if sample_count <= 0 or sample_count >= len(paths):
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, sample_count)


def extract_attentions(net, image_paths, device):
    to_tensor = ToTensor()
    attention_mats = []

    with torch.no_grad():
        for idx, path in enumerate(image_paths, start=1):
            image = np.array(Image.open(path).convert("RGB"))
            tensor = to_tensor(image).unsqueeze(0).to(device)
            tensor = pad_input(tensor)
            features = net.water_encoder.encode(tensor)
            attention = features["attention"]
            if attention.dim() == 4:
                attention = attention.mean(dim=1)
            attention_mats.append(attention.squeeze(0).cpu().numpy())
            if idx % 50 == 0 or idx == len(image_paths):
                print(f"Extracted attention {idx}/{len(image_paths)}")

    return np.stack(attention_mats, axis=0)


def save_matrix_csv(path, matrix, names):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source/target"] + names)
        for row_name, row in zip(names, matrix):
            writer.writerow([row_name] + [f"{value:.6f}" for value in row])


def save_summary(path, mean_matrix, std_matrix):
    upper_pairs = []
    for i in range(len(PRIOR_NAMES)):
        for j in range(len(PRIOR_NAMES)):
            if i == j:
                continue
            upper_pairs.append(
                (PRIOR_NAMES[i], PRIOR_NAMES[j], float(mean_matrix[i, j]))
            )
    upper_pairs.sort(key=lambda item: item[2], reverse=True)

    lines = [
        "Prior coupling summary",
        "",
        "Average attention matrix:",
    ]
    for i, name in enumerate(PRIOR_NAMES):
        row = ", ".join(
            f"{target}={mean_matrix[i, j]:.4f}" for j, target in enumerate(PRIOR_NAMES)
        )
        lines.append(f"- {name}: {row}")
    lines.extend(
        [
            "",
            "Average standard-deviation matrix:",
        ]
    )
    for i, name in enumerate(PRIOR_NAMES):
        row = ", ".join(
            f"{target}={std_matrix[i, j]:.4f}" for j, target in enumerate(PRIOR_NAMES)
        )
        lines.append(f"- {name}: {row}")
    lines.extend(
        [
            "",
            "Strongest directed couplings:",
        ]
    )
    for src, dst, value in upper_pairs[:6]:
        lines.append(f"- {src} -> {dst}: {value:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(matrix, save_path, title):
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=220)
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(PRIOR_NAMES)))
    ax.set_yticks(range(len(PRIOR_NAMES)))
    ax.set_xticklabels(PRIOR_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(PRIOR_NAMES)
    ax.set_xlabel("Target prior token")
    ax.set_ylabel("Source prior token")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < matrix.max() * 0.65 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    image_paths = sample_images(args.dataset, args.sample_count, args.seed)
    print(f"Dataset: {args.dataset}")
    print(f"Images: {len(image_paths)}")

    net = load_model(args.ckpt_path, device)
    attention_mats = extract_attentions(net, image_paths, device)

    mean_matrix = attention_mats.mean(axis=0)
    std_matrix = attention_mats.std(axis=0)

    np.save(output_dir / "prior_attention_matrices.npy", attention_mats)
    np.save(output_dir / "prior_attention_mean.npy", mean_matrix)
    np.save(output_dir / "prior_attention_std.npy", std_matrix)
    (output_dir / "sample_paths.txt").write_text(
        "\n".join(str(path) for path in image_paths), encoding="utf-8"
    )

    save_matrix_csv(output_dir / "prior_attention_mean.csv", mean_matrix, PRIOR_NAMES)
    save_matrix_csv(output_dir / "prior_attention_std.csv", std_matrix, PRIOR_NAMES)
    save_summary(output_dir / "prior_attention_summary.txt", mean_matrix, std_matrix)

    plot_heatmap(
        mean_matrix,
        output_dir / "prior_attention_mean_heatmap.png",
        "Average Prior-Coupling Attention",
    )
    plot_heatmap(
        std_matrix,
        output_dir / "prior_attention_std_heatmap.png",
        "Std of Prior-Coupling Attention",
    )

    print("Average attention matrix:")
    print(mean_matrix)
    print("Std attention matrix:")
    print(std_matrix)
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
