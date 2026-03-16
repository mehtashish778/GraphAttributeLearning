from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chair-only Visual Genome data pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="visual_genome",
        help="Dataset selector. Only visual_genome is supported in this version.",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (default behavior).")
    return parser.parse_args()


def run_step(script_path: Path, config_path: Path) -> None:
    cmd: List[str] = [sys.executable, str(script_path), "--config", str(config_path)]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    if args.dataset_name != "visual_genome":
        raise RuntimeError("Only --dataset-name visual_genome is supported currently.")

    selected_flags = [
        args.download_only,
        args.extract_only,
        args.process_only,
        args.split_only,
        args.all,
    ]
    if sum(1 for flag in selected_flags if flag) > 1:
        raise RuntimeError("Use only one stage selector at a time.")

    scripts_dir = REPO_ROOT / "scripts" / "data"
    steps = {
        "download": scripts_dir / "download_visual_genome.py",
        "extract": scripts_dir / "extract_visual_genome.py",
        "process": scripts_dir / "process_visual_genome.py",
        "split": scripts_dir / "build_splits.py",
    }

    if args.download_only:
        pipeline = ["download"]
    elif args.extract_only:
        pipeline = ["extract"]
    elif args.process_only:
        pipeline = ["process"]
    elif args.split_only:
        pipeline = ["split"]
    else:
        pipeline = ["download", "extract", "process", "split"]

    for step in pipeline:
        run_step(script_path=steps[step], config_path=args.config)
    print("[ok] Data pipeline completed.")


if __name__ == "__main__":
    main()
