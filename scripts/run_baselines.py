from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [run_baselines] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline A/B/C experiments.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def run_train(
    mode: str,
    run_name: str,
    overrides: Dict[str, Any],
    dataset_config: Path,
    model_config: Path,
    train_config: Path,
    eval_config: Path,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_baseline.py"),
        "--run-name",
        run_name,
        "--mode",
        mode,
        "--dataset-config",
        str(dataset_config),
        "--model-config",
        str(model_config),
        "--train-config",
        str(train_config),
        "--eval-config",
        str(eval_config),
        "--overrides-json",
        json.dumps(overrides),
    ]
    env = dict(**os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd.insert(1, "-u")
    log(f"Launching run '{run_name}' in mode='{mode}'")
    log(f"Python executable: {sys.executable}")
    log(f"Overrides: {json.dumps(overrides)}")
    log(f"Command: {' '.join(cmd)}")
    started = time.time()
    completed = subprocess.run(cmd, check=False, cwd=REPO_ROOT, env=env)
    elapsed = time.time() - started
    log(f"Run '{run_name}' exited with code={completed.returncode} after {elapsed:.1f}s")
    if completed.returncode != 0:
        raise RuntimeError(f"Baseline run failed: {run_name}")


def main() -> None:
    args = parse_args()
    log("Starting baseline orchestrator")
    log(f"Repo root: {REPO_ROOT}")
    log(f"Mode: {args.mode}")
    log(f"Experiment config: {args.experiment_config}")
    experiment_cfg = load_yaml(args.experiment_config)
    baseline_runs = experiment_cfg.get("baseline_runs", [])
    if not baseline_runs:
        raise RuntimeError("No baseline_runs found in experiment config.")
    log(f"Discovered {len(baseline_runs)} baseline runs")

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison: Dict[str, Any] = {"mode": args.mode, "runs": []}

    for row in baseline_runs:
        run_name = f"{row['name']}_{args.mode}"
        overrides = row.get("overrides", {})
        log(f"Preparing run: {run_name}")
        run_train(
            mode=args.mode,
            run_name=run_name,
            overrides=overrides,
            dataset_config=args.dataset_config,
            model_config=args.model_config,
            train_config=args.train_config,
            eval_config=args.eval_config,
        )
        metrics_path = output_dir / run_name / "metrics.json"
        log(f"Reading metrics from: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        comparison["runs"].append(
            {
                "run_name": run_name,
                "test_map": metrics.get("test_metrics", {}).get("map"),
                "test_macro_f1": metrics.get("test_metrics", {}).get("macro_f1"),
                "test_micro_f1": metrics.get("test_metrics", {}).get("micro_f1"),
            }
        )

    comparison_path = output_dir / f"baseline_comparison_{args.mode}.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=True)
    log(f"Wrote comparison report: {comparison_path}")


if __name__ == "__main__":
    main()
