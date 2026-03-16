from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from infer.pipeline import load_checkpoint, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference for baseline or GNN checkpoints.")
    parser.add_argument("--model-type", type=str, choices=["auto", "baseline", "gnn"], default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/smoke_baseline/best.pt"),
        help="Path to checkpoint (.pt).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--benchmark", action="store_true", help="Print simple latency timing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    loaded = load_checkpoint(args.checkpoint, device=device)

    if args.benchmark:
        import time

        start = time.time()
        result = predict_image(
            image_path=args.image,
            loaded=loaded,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        elapsed = time.time() - start
    else:
        result = predict_image(
            image_path=args.image,
            loaded=loaded,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        elapsed = None

    payload: Dict[str, Any] = {
        "image": str(args.image),
        "top_k": args.top_k,
        "threshold": args.threshold,
        "labels": result.labels,
        "scores": result.scores,
        "positives": result.positives,
    }
    if elapsed is not None:
        payload["latency_sec"] = elapsed

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

