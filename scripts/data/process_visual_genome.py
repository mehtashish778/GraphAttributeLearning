from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import ensure_dir, load_yaml, write_json  # noqa: E402
from data.normalization import (  # noqa: E402
    build_label_vocab,
    filter_by_min_support,
    normalize_labels,
    normalize_text,
)
from data.schemas import SampleRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Visual Genome attributes to chair-only normalized samples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config yaml.",
    )
    return parser.parse_args()


def flatten_attribute_groups(groups_cfg: Dict[str, Iterable[str]]) -> List[str]:
    labels: List[str] = []
    for values in groups_cfg.values():
        labels.extend(values)
    return labels


def find_json_file(raw_root: Path, target_name: str) -> Path:
    direct = raw_root / target_name
    if direct.exists():
        return direct
    matches = list(raw_root.rglob(target_name))
    if not matches:
        raise RuntimeError(f"Required json file not found: {target_name}")
    return matches[0]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_image_metadata_map(image_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in image_data:
        image_id = int(row.get("image_id"))
        image_url = str(row.get("url") or "")
        file_name = Path(image_url).name if image_url else ""
        image_path = str(Path("data/raw/visual_genome/images") / file_name) if file_name else ""
        out[image_id] = {"image_url": image_url, "image_path": image_path}
    return out


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def object_is_chair(names: List[str], lowercase: bool, lemmatize: bool) -> bool:
    normalized = [
        normalize_text(name, lowercase=lowercase, lemmatize=lemmatize) for name in names
    ]
    return "chair" in set(normalized)


def parse_chair_records(
    attributes_data: List[Dict[str, Any]],
    image_meta: Dict[int, Dict[str, str]],
    lowercase: bool,
    lemmatize: bool,
    synonym_map: Dict[str, str],
    keep_unmapped_attributes: bool,
    allowed_labels: List[str],
) -> Tuple[List[SampleRecord], Dict[str, int]]:
    allowed_set = set(
        normalize_text(label, lowercase=lowercase, lemmatize=lemmatize) for label in allowed_labels
    )
    row_stats = {
        "total_attribute_objects": 0,
        "non_chair_objects_dropped": 0,
        "chair_objects_kept_pre_filter": 0,
    }
    output: List[SampleRecord] = []

    for image_item in attributes_data:
        image_id = int(image_item.get("image_id"))
        image_attributes = image_item.get("attributes", [])
        metadata = image_meta.get(image_id, {})
        for obj in image_attributes:
            row_stats["total_attribute_objects"] += 1
            names = as_list(obj.get("names"))
            if not object_is_chair(names, lowercase=lowercase, lemmatize=lemmatize):
                row_stats["non_chair_objects_dropped"] += 1
                continue

            object_id = int(obj.get("object_id") or 0)
            raw_attributes = as_list(obj.get("attributes"))
            normalized = normalize_labels(
                labels=raw_attributes,
                lowercase=lowercase,
                lemmatize=lemmatize,
                synonym_map=synonym_map,
                keep_unmapped=keep_unmapped_attributes,
                allowed_labels=allowed_set,
            )
            record = SampleRecord(
                image_id=image_id,
                object_id=object_id,
                object_name="chair",
                image_url=metadata.get("image_url", ""),
                image_path=metadata.get("image_path", ""),
                attributes_raw=raw_attributes,
                attributes_norm=normalized,
            )
            output.append(record)
            row_stats["chair_objects_kept_pre_filter"] += 1

    return output, row_stats


def deduplicate_records(records: List[SampleRecord]) -> List[SampleRecord]:
    seen = set()
    out: List[SampleRecord] = []
    for record in records:
        key = (record.image_id, record.object_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def save_samples(
    rows: List[Dict[str, Any]],
    output_format: str,
    csv_fallback: bool,
    parquet_path: Path,
    csv_path: Path,
) -> str:
    if output_format == "parquet":
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows)
            df.to_parquet(parquet_path, index=False)
            return str(parquet_path)
        except Exception as exc:  # pragma: no cover
            if not csv_fallback:
                raise RuntimeError(f"Failed to write parquet and csv fallback disabled: {exc}") from exc

    ensure_dir(csv_path.parent)
    fieldnames = [
        "image_id",
        "object_id",
        "object_name",
        "image_url",
        "image_path",
        "attributes_raw",
        "attributes_norm",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["attributes_raw"] = json.dumps(row_copy["attributes_raw"], ensure_ascii=True)
            row_copy["attributes_norm"] = json.dumps(row_copy["attributes_norm"], ensure_ascii=True)
            writer.writerow(row_copy)
    return str(csv_path)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg["dataset"]
    processing_cfg = cfg.get("processing", {})
    filtering_cfg = cfg.get("filtering", {})
    attributes_cfg = cfg.get("attributes", {})

    strict_chair_only = bool(processing_cfg.get("strict_chair_only", True))
    primary_object = str(cfg["project"].get("primary_object_class", "chair")).strip().lower()
    if strict_chair_only and primary_object != "chair":
        raise RuntimeError("This pipeline currently supports strict chair-only processing.")

    raw_root = REPO_ROOT / dataset_cfg["root_dir"]
    processed_root = REPO_ROOT / dataset_cfg["processed_dir"]
    ensure_dir(processed_root)

    attributes_path = find_json_file(raw_root, "attributes.json")
    image_data_path = find_json_file(raw_root, "image_data.json")
    attributes_data = load_json(attributes_path)
    image_data = load_json(image_data_path)

    lowercase = bool(attributes_cfg.get("lowercase", True))
    lemmatize = bool(attributes_cfg.get("lemmatize", True))
    synonym_map = {
        normalize_text(str(k), lowercase=lowercase, lemmatize=lemmatize): normalize_text(
            str(v), lowercase=lowercase, lemmatize=lemmatize
        )
        for k, v in attributes_cfg.get("synonym_map", {}).items()
    }
    allowed_labels = flatten_attribute_groups(attributes_cfg.get("groups", {}))
    keep_unmapped = bool(processing_cfg.get("keep_unmapped_attributes", False))

    image_meta = build_image_metadata_map(image_data)
    records, row_stats = parse_chair_records(
        attributes_data=attributes_data,
        image_meta=image_meta,
        lowercase=lowercase,
        lemmatize=lemmatize,
        synonym_map=synonym_map,
        keep_unmapped_attributes=keep_unmapped,
        allowed_labels=allowed_labels,
    )

    if bool(filtering_cfg.get("deduplicate_images", True)):
        records = deduplicate_records(records)

    if bool(filtering_cfg.get("drop_empty_labels", True)):
        records = [r for r in records if len(r.attributes_norm) > 0]

    min_support = int(attributes_cfg.get("min_support", 1))
    filtered_label_lists, label_freq = filter_by_min_support(
        [r.attributes_norm for r in records],
        min_support=min_support,
    )
    for idx, labels in enumerate(filtered_label_lists):
        records[idx].attributes_norm = labels

    if bool(filtering_cfg.get("drop_empty_labels", True)):
        records = [r for r in records if len(r.attributes_norm) > 0]

    max_samples_per_attribute = filtering_cfg.get("max_samples_per_attribute")
    if max_samples_per_attribute is not None:
        per_attr_limit = int(max_samples_per_attribute)
        attr_counts: Dict[str, int] = {}
        limited: List[SampleRecord] = []
        for rec in records:
            if not rec.attributes_norm:
                continue
            can_take = True
            for label in rec.attributes_norm:
                if attr_counts.get(label, 0) >= per_attr_limit:
                    can_take = False
                    break
            if can_take:
                limited.append(rec)
                for label in rec.attributes_norm:
                    attr_counts[label] = attr_counts.get(label, 0) + 1
        records = limited

    label_freq = {}
    for rec in records:
        for label in set(rec.attributes_norm):
            label_freq[label] = label_freq.get(label, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_vocab = build_label_vocab(label_freq)

    rows = [record.to_dict() for record in records]
    samples_parquet = processed_root / processing_cfg.get("samples_file_parquet", "samples.parquet")
    samples_csv = processed_root / processing_cfg.get("samples_file_csv", "samples.csv")
    output_format = str(processing_cfg.get("output_format", "parquet")).lower()
    csv_fallback = bool(processing_cfg.get("csv_fallback", True))
    samples_written = save_samples(
        rows=rows,
        output_format=output_format,
        csv_fallback=csv_fallback,
        parquet_path=samples_parquet,
        csv_path=samples_csv,
    )

    write_json(
        processed_root / processing_cfg.get("label_vocab_file", "label_vocab.json"),
        label_vocab,
    )
    write_json(
        processed_root / processing_cfg.get("label_frequency_file", "label_frequencies.json"),
        label_freq,
    )
    report = {
        "raw_attributes_file": str(attributes_path),
        "raw_image_data_file": str(image_data_path),
        "samples_output": samples_written,
        "strict_chair_only": strict_chair_only,
        "primary_object_class": primary_object,
        "total_samples_final": len(records),
        "labels_kept": list(label_vocab.keys()),
        "num_labels_kept": len(label_vocab),
        "min_support": min_support,
        **row_stats,
    }
    write_json(
        processed_root / processing_cfg.get("processing_report_file", "processing_report.json"),
        report,
    )
    print(f"[ok] Processed {len(records)} chair samples.")


if __name__ == "__main__":
    main()
