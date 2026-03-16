from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show saved metrics for a trained baseline run.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.output_dir / args.run_name / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)
    report = {
        "run_name": payload.get("run_name"),
        "mode": payload.get("mode"),
        "best_val_map": payload.get("best_val_map"),
        "test_map": payload.get("test_metrics", {}).get("map"),
        "test_macro_f1": payload.get("test_metrics", {}).get("macro_f1"),
        "test_micro_f1": payload.get("test_metrics", {}).get("micro_f1"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
