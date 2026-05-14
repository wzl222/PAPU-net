import argparse
import subprocess
from pathlib import Path


BASE_PRETRAIN = (
    "train_ckpt/uieb_water_physicscoupled_tuned_e20/"
    "uieb-water-promptir-best-epoch=014-val_psnr=24.112.ckpt"
)


ABLATIONS = {
    "w_o_dcp": ["--disabled_priors", "dcp"],
    "w_o_structure": ["--disabled_priors", "structure"],
    "w_o_color": ["--disabled_priors", "color"],
    "w_o_luminance": ["--disabled_priors", "luminance"],
    "independent_priors": ["--disable_prior_coupling"],
    "only_modulation": ["--disable_prompt_conditioning", "--disable_refinement_conditioning"],
    "only_prompt": ["--disable_modulation", "--disable_refinement_conditioning"],
    "shared_token": ["--shared_prior_token"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["w_o_dcp", "w_o_structure", "only_modulation", "only_prompt", "shared_token"],
        choices=sorted(ABLATIONS.keys()),
    )
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--val_patch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--module_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lambda_color_dist", type=float, default=0.02)
    parser.add_argument("--lambda_edge", type=float, default=0.01)
    parser.add_argument("--lambda_local_contrast", type=float, default=0.03)
    parser.add_argument("--lambda_high_freq", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_cmd(args, variant_name):
    ckpt_dir = f"train_ckpt/uieb_{variant_name}"
    log_dir = f"logs/uieb_{variant_name}"
    cmd = [
        "conda",
        "run",
        "-p",
        "/mnt/disk1new/wzl/env/bioir",
        "python",
        "train_uieb.py",
        "--pretrain_ckpt",
        BASE_PRETRAIN,
        "--ckpt_dir",
        ckpt_dir,
        "--log_dir",
        log_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--patch_size",
        str(args.patch_size),
        "--val_patch_size",
        str(args.val_patch_size),
        "--val_split",
        str(args.val_split),
        "--backbone_lr",
        str(args.backbone_lr),
        "--module_lr",
        str(args.module_lr),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--lambda_color_dist",
        str(args.lambda_color_dist),
        "--lambda_edge",
        str(args.lambda_edge),
        "--lambda_local_contrast",
        str(args.lambda_local_contrast),
        "--lambda_high_freq",
        str(args.lambda_high_freq),
        "--cuda",
        str(args.cuda),
        "--frequency_refinement",
        "--local_contrast_refinement",
    ]
    cmd.extend(ABLATIONS[variant_name])
    return cmd


def main():
    args = parse_args()
    for variant in args.variants:
        cmd = build_cmd(args, variant)
        print(f"\n[{variant}]")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        Path(f"logs/uieb_{variant}").mkdir(parents=True, exist_ok=True)
        Path(f"train_ckpt/uieb_{variant}").mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
