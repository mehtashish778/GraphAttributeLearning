from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import download_file, ensure_dir, load_yaml, verify_checksum  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Visual Genome raw artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg["dataset"]
    download_cfg = cfg.get("download", {})

    if not download_cfg.get("enabled", True):
        print("Download disabled in config. Nothing to do.")
        return

    raw_root = REPO_ROOT / dataset_cfg["root_dir"]
    ensure_dir(raw_root)

    timeout_seconds = int(download_cfg.get("timeout_seconds", 120))
    retries = int(download_cfg.get("retries", 3))
    chunk_size = int(download_cfg.get("chunk_size_bytes", 1024 * 1024))
    force_download = args.force_download or bool(download_cfg.get("force_download", False))

    artifacts = download_cfg.get("artifacts", [])
    if not artifacts:
        raise RuntimeError("No download artifacts configured.")

    for artifact in artifacts:
        filename = artifact["filename"]
        url = artifact["url"]
        expected_sha256 = artifact.get("sha256")
        target = raw_root / filename

        if target.exists() and not force_download:
            if verify_checksum(target, expected_sha256):
                print(f"[skip] {filename} already exists and checksum passed.")
                continue
            print(f"[warn] {filename} exists but checksum mismatch. Re-downloading.")

        print(f"[download] {filename} <- {url}")
        download_file(
            url=url,
            destination=target,
            timeout_seconds=timeout_seconds,
            retries=retries,
            chunk_size=chunk_size,
        )

        if not verify_checksum(target, expected_sha256):
            raise RuntimeError(f"Checksum validation failed for {target}")
        print(f"[ok] {filename}")

    print("All configured Visual Genome artifacts are downloaded.")


if __name__ == "__main__":
    main()
