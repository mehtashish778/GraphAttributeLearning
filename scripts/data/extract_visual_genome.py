from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import ensure_dir, extract_archive, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Visual Genome raw archives.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force extraction even if marker exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg["dataset"]
    extraction_cfg = cfg.get("extraction", {})

    if not extraction_cfg.get("enabled", True):
        print("Extraction disabled in config. Nothing to do.")
        return

    raw_root = REPO_ROOT / dataset_cfg["root_dir"]
    ensure_dir(raw_root)

    marker_file = raw_root / extraction_cfg.get("marker_file", ".extract_complete")
    force_extract = args.force_extract or bool(extraction_cfg.get("force_extract", False))
    safe_extract = bool(extraction_cfg.get("safe_extract", True))

    if marker_file.exists() and not force_extract:
        print(f"[skip] Extraction marker found: {marker_file}")
        return

    archives = extraction_cfg.get("archives", [])
    if not archives:
        raise RuntimeError("No extraction archives configured.")

    for archive_name in archives:
        archive_path = raw_root / archive_name
        if not archive_path.exists():
            raise RuntimeError(f"Archive missing: {archive_path}")
        print(f"[extract] {archive_name}")
        extract_archive(archive_path=archive_path, output_dir=raw_root, safe_extract=safe_extract)

    marker_file.write_text("ok\n", encoding="utf-8")
    print(f"[ok] Extraction completed. Marker written at {marker_file}")


if __name__ == "__main__":
    main()
