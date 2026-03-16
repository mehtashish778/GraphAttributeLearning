from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import ensure_dir, load_yaml, write_json  # noqa: E402


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return to_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic train/val/test split manifests for chair samples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config yaml.",
    )
    return parser.parse_args()


def _load_samples_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "image_id": int(row["image_id"]),
                    "object_id": int(row["object_id"]),
                    "object_name": row.get("object_name", "chair"),
                    "image_url": row.get("image_url", ""),
                    "image_path": row.get("image_path", ""),
                    "attributes_raw": json.loads(row.get("attributes_raw", "[]")),
                    "attributes_norm": json.loads(row.get("attributes_norm", "[]")),
                }
            )
    return rows


def load_samples(processed_root: Path, processing_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    parquet_path = processed_root / processing_cfg.get("samples_file_parquet", "samples.parquet")
    csv_path = processed_root / processing_cfg.get("samples_file_csv", "samples.csv")
    if parquet_path.exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(parquet_path)
            rows = df.to_dict(orient="records")
            for row in rows:
                row = to_json_safe(row)
                if isinstance(row.get("attributes_raw"), str):
                    row["attributes_raw"] = json.loads(row["attributes_raw"])
                if isinstance(row.get("attributes_norm"), str):
                    row["attributes_norm"] = json.loads(row["attributes_norm"])
                row["image_id"] = int(row["image_id"])
                row["object_id"] = int(row["object_id"])
                row["attributes_raw"] = to_json_safe(row.get("attributes_raw", []))
                row["attributes_norm"] = to_json_safe(row.get("attributes_norm", []))
            return rows
        except Exception:
            pass
    if csv_path.exists():
        return _load_samples_csv(csv_path)
    raise RuntimeError("No processed samples file found (parquet/csv). Run processing first.")


def allocate_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratios = [train_ratio, val_ratio, test_ratio]
    names = ["train", "val", "test"]
    base = [int(total * r) for r in ratios]
    remaining = total - sum(base)
    frac = sorted(
        [(i, (total * ratios[i]) - base[i]) for i in range(3)],
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remaining):
        base[frac[i % len(frac)][0]] += 1
    counts = dict(zip(names, base))
    return counts["train"], counts["val"], counts["test"]


def signature(labels: Iterable[str]) -> str:
    uniq = sorted(set(labels))
    return "|".join(uniq) if uniq else "__no_label__"


def digest_ids(values: Iterable[int]) -> str:
    joined = ",".join(str(v) for v in sorted(values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg["dataset"]
    split_cfg = cfg.get("splits", {})
    processing_cfg = cfg.get("processing", {})

    processed_root = REPO_ROOT / dataset_cfg["processed_dir"]
    ensure_dir(processed_root)
    samples = load_samples(processed_root=processed_root, processing_cfg=processing_cfg)
    if not samples:
        raise RuntimeError("No samples available for split generation.")

    seed = int(split_cfg.get("split_seed", 42))
    random.seed(seed)

    by_image: Dict[int, Dict[str, Any]] = {}
    for row in samples:
        image_id = int(row["image_id"])
        if image_id not in by_image:
            by_image[image_id] = {"labels": set(), "rows": []}
        by_image[image_id]["rows"].append(row)
        by_image[image_id]["labels"].update(row.get("attributes_norm", []))

    buckets: Dict[str, List[int]] = defaultdict(list)
    for image_id, payload in by_image.items():
        buckets[signature(payload["labels"])].append(image_id)

    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise RuntimeError("Split ratios must sum to 1.0")

    train_ids: List[int] = []
    val_ids: List[int] = []
    test_ids: List[int] = []

    for _, image_ids in sorted(buckets.items(), key=lambda item: item[0]):
        random.shuffle(image_ids)
        n_train, n_val, n_test = allocate_counts(
            total=len(image_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_ids.extend(image_ids[:n_train])
        val_ids.extend(image_ids[n_train : n_train + n_val])
        test_ids.extend(image_ids[n_train + n_val : n_train + n_val + n_test])

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Leakage detected across train/val/test image IDs.")

    split_rows = {"train": [], "val": [], "test": []}
    for row in samples:
        row = to_json_safe(row)
        image_id = int(row["image_id"])
        if image_id in train_set:
            split_rows["train"].append(row)
        elif image_id in val_set:
            split_rows["val"].append(row)
        elif image_id in test_set:
            split_rows["test"].append(row)

    split_dir = processed_root / processing_cfg.get("split_dir", "splits")
    ensure_dir(split_dir)
    for split_name in ("train", "val", "test"):
        write_json(split_dir / f"{split_name}.json", {"samples": split_rows[split_name]})

    summary = {
        "seed": seed,
        "total_rows": len(samples),
        "total_unique_images": len(by_image),
        "train_rows": len(split_rows["train"]),
        "val_rows": len(split_rows["val"]),
        "test_rows": len(split_rows["test"]),
        "train_unique_images": len(train_set),
        "val_unique_images": len(val_set),
        "test_unique_images": len(test_set),
        "train_image_digest": digest_ids(train_set),
        "val_image_digest": digest_ids(val_set),
        "test_image_digest": digest_ids(test_set),
        "stratify_mode": "multilabel_signature_bucket",
        "fallback_used": str(split_cfg.get("stratify_fallback", "iterative_bucket")),
        "leakage_check_passed": True,
    }
    write_json(split_dir / "split_report.json", summary)
    print("[ok] Split manifests created with leakage check passed.")


if __name__ == "__main__":
    main()
