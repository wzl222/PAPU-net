import argparse
import shutil
from pathlib import Path

import numpy as np

from evaluate_new import getUCIQE, getUIQM, getUIQM_torch
from utils.uiqm_postprocess_utils import (
    enhance_rgb_uint8,
    is_image_file,
    preset_params,
    read_rgb,
    save_rgb,
)


def list_image_paths(input_dir, max_images=0):
    input_dir = Path(input_dir)
    paths = sorted(p for p in input_dir.iterdir() if is_image_file(p.name))
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def process_dir(input_dir, output_dir, clahe, saturation, contrast, sharpness, max_images=0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_image_paths(input_dir, max_images=max_images)
    for path in paths:
        image = read_rgb(path)
        enhanced = enhance_rgb_uint8(image, clahe, saturation, contrast, sharpness)
        save_rgb(output_dir / path.name, enhanced)


def score_dir(input_dir, device="cpu", max_images=0):
    input_dir = Path(input_dir)
    uiqm_scores = []
    uciqe_scores = []
    paths = sorted(p for p in input_dir.iterdir() if is_image_file(p.name))
    if max_images > 0:
        paths = paths[:max_images]
    for path in paths:
        image = read_rgb(path).astype(np.float32) / 255.0
        if device != "cpu":
            uiqm_scores.append(getUIQM_torch(image, device)[0])
        else:
            uiqm_scores.append(getUIQM(image)[0])
        uciqe_scores.append(getUCIQE(image))
    return float(np.mean(uiqm_scores)), float(np.mean(uciqe_scores))


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clahe", type=float, default=1.5)
    parser.add_argument("--saturation", type=float, default=1.05)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=0.2)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--preset", choices=["mild", "strong"], default=None)
    parser.add_argument("--clahe_grid", default="0,1.0,1.5,2.0")
    parser.add_argument("--saturation_grid", default="1.0,1.05,1.1,1.15")
    parser.add_argument("--contrast_grid", default="1.0,1.03,1.06")
    parser.add_argument("--sharpness_grid", default="0,0.15,0.3,0.45")
    parser.add_argument(
        "--search_images",
        type=int,
        default=0,
        help="Only process this many images per trial during search. Final best output still uses all images.",
    )
    parser.add_argument(
        "--search_log",
        default="",
        help="Optional txt log file for search trials and final best params.",
    )
    args = parser.parse_args()

    log_path = Path(args.search_log) if args.search_log else None

    def log(message):
        print(message)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    if not args.search:
        if args.preset:
            params = preset_params(args.preset)
            args.clahe = params["clahe"]
            args.saturation = params["saturation"]
            args.contrast = params["contrast"]
            args.sharpness = params["sharpness"]
        process_dir(
            args.input_dir,
            args.output_dir,
            args.clahe,
            args.saturation,
            args.contrast,
            args.sharpness,
        )
        uiqm, uciqe = score_dir(args.output_dir, args.device, args.max_images)
        log(f"UIQM={uiqm:.3f} UCIQE={uciqe:.3f}")
        return

    best = None
    output_dir = Path(args.output_dir)
    trial_dir = output_dir.with_name(output_dir.name + "_trial")
    clahe_values = parse_float_list(args.clahe_grid)
    saturation_values = parse_float_list(args.saturation_grid)
    contrast_values = parse_float_list(args.contrast_grid)
    sharpness_values = parse_float_list(args.sharpness_grid)
    total_trials = (
        len(clahe_values)
        * len(saturation_values)
        * len(contrast_values)
        * len(sharpness_values)
    )
    trial_index = 0
    for clahe in clahe_values:
        for saturation in saturation_values:
            for contrast in contrast_values:
                for sharpness in sharpness_values:
                    trial_index += 1
                    log(
                        f"Trial {trial_index}/{total_trials} "
                        f"clahe={clahe} saturation={saturation} contrast={contrast} sharpness={sharpness}"
                    )
                    process_dir(
                        args.input_dir,
                        trial_dir,
                        clahe,
                        saturation,
                        contrast,
                        sharpness,
                        max_images=args.search_images,
                    )
                    uiqm, uciqe = score_dir(trial_dir, args.device, args.max_images)
                    result = (uiqm, uciqe, clahe, saturation, contrast, sharpness)
                    log(
                        "UIQM={:.3f} UCIQE={:.3f} clahe={} saturation={} contrast={} sharpness={}".format(
                            uiqm, uciqe, clahe, saturation, contrast, sharpness
                        )
                    )
                    if best is None or result[:2] > best[:2]:
                        best = result
                        log(
                            "BestSoFar UIQM={:.3f} UCIQE={:.3f} clahe={} saturation={} contrast={} sharpness={}".format(
                                uiqm, uciqe, clahe, saturation, contrast, sharpness
                            )
                        )

    uiqm, uciqe, clahe, saturation, contrast, sharpness = best
    process_dir(args.input_dir, output_dir, clahe, saturation, contrast, sharpness)
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    log(
        "Best UIQM={:.3f} UCIQE={:.3f} clahe={} saturation={} contrast={} sharpness={}".format(
            uiqm, uciqe, clahe, saturation, contrast, sharpness
        )
    )


if __name__ == "__main__":
    main()
