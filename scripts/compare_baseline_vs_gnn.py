from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_metrics(run_name: str, output_dir: Path) -> Dict[str, Any]:
    metrics_path = output_dir / run_name / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs GNN runs on test metrics.")
    parser.add_argument("--baseline-run", type=str, required=True, help="Baseline run_name (e.g. smoke_baseline).")
    parser.add_argument("--gnn-run", type=str, required=True, help="GNN run_name (e.g. smoke_gnn).")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline_run, args.output_dir)
    gnn = load_metrics(args.gnn_run, args.output_dir)

    def extract(row: Dict[str, Any]) -> Dict[str, float]:
        test = row.get("test_metrics", {})
        return {
            "map": float(test.get("map", 0.0)),
            "macro_f1": float(test.get("macro_f1", 0.0)),
            "micro_f1": float(test.get("micro_f1", 0.0)),
        }

    b = extract(baseline)
    g = extract(gnn)
    delta = {k: g[k] - b[k] for k in b.keys()}

    report = {
        "baseline_run": args.baseline_run,
        "gnn_run": args.gnn_run,
        "baseline_test": b,
        "gnn_test": g,
        "delta": delta,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

