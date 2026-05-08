import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS = {
    "fuvd": {
        "input_dir": "output/fuvd_water_promptir_e60",
        "output_dir": "output/fuvd_water_promptir_e60_uiqm_post_searchbest",
        "metrics_file": "metrics_fuvd_water_promptir_e60_uiqm_post_searchbest_fixed.txt",
        "search_log": "logs/uiqm_search_fuvd.log",
        "clahe_grid": "0,0.3,0.6",
        "saturation_grid": "1.0,1.05,1.1",
        "contrast_grid": "1.0,1.03",
        "sharpness_grid": "0.45,0.6,0.75",
    },
    "ruie": {
        "input_dir": "output/ruie_water_promptir_e60",
        "output_dir": "output/ruie_water_promptir_e60_uiqm_post_searchbest",
        "metrics_file": "metrics_ruie_water_promptir_e60_uiqm_post_searchbest_fixed.txt",
        "search_log": "logs/uiqm_search_ruie.log",
        "clahe_grid": "0,0.6,1.2",
        "saturation_grid": "1.0,1.05,1.1,1.15",
        "contrast_grid": "1.0,1.03,1.06",
        "sharpness_grid": "0.3,0.45,0.6",
    },
}


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    command_text = " ".join(command)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{timestamp}] $ {command_text}\n")
        handle.flush()
        print(f"[{timestamp}] $ {command_text}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fuvd", "ruie"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--search_images",
        type=int,
        default=120,
        help="Number of images used for each search trial.",
    )
    parser.add_argument(
        "--score_images",
        type=int,
        default=120,
        help="Number of processed images used for scoring each search trial.",
    )
    parser.add_argument("--clahe_grid", default="")
    parser.add_argument("--saturation_grid", default="")
    parser.add_argument("--contrast_grid", default="")
    parser.add_argument("--sharpness_grid", default="")
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable used to launch uiqm_postprocess.py and evaluate_new.py.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python_exe = args.python_exe
    summary_lines = []

    for dataset in args.datasets:
        config = DATASETS[dataset]
        input_dir = root / config["input_dir"]
        output_dir = root / config["output_dir"]
        metrics_file = root / config["metrics_file"]
        log_path = root / config["search_log"]

        if not input_dir.exists():
            summary_lines.append(f"{dataset}: skipped, missing input dir {input_dir}")
            continue

        print(f"Starting dataset: {dataset}", flush=True)
        clahe_grid = args.clahe_grid or config["clahe_grid"]
        saturation_grid = args.saturation_grid or config["saturation_grid"]
        contrast_grid = args.contrast_grid or config["contrast_grid"]
        sharpness_grid = args.sharpness_grid or config["sharpness_grid"]

        search_cmd = [
            python_exe,
            "uiqm_postprocess.py",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--search",
            "--device",
            args.device,
            "--search_images",
            str(args.search_images),
            "--max_images",
            str(args.score_images),
            "--clahe_grid",
            clahe_grid,
            "--saturation_grid",
            saturation_grid,
            "--contrast_grid",
            contrast_grid,
            "--sharpness_grid",
            sharpness_grid,
            "--search_log",
            str(log_path),
        ]
        eval_cmd = [
            python_exe,
            "evaluate_new.py",
            "--input_dir",
            str(output_dir),
            "--save_txt",
            str(metrics_file),
            "--device",
            args.device,
        ]

        search_code = run_command(search_cmd, log_path)
        if search_code != 0:
            summary_lines.append(f"{dataset}: search failed, see {log_path}")
            print(summary_lines[-1], flush=True)
            continue

        eval_code = run_command(eval_cmd, log_path)
        if eval_code != 0:
            summary_lines.append(f"{dataset}: evaluation failed, see {log_path}")
            print(summary_lines[-1], flush=True)
            continue

        summary_lines.append(
            f"{dataset}: done, output={output_dir.name}, metrics={metrics_file.name}, log={log_path.name}"
        )
        print(summary_lines[-1], flush=True)

    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
