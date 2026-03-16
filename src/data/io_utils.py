from __future__ import annotations

import hashlib
import json
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _download_once(
    url: str,
    destination: Path,
    timeout_seconds: int,
    chunk_size: int,
) -> None:
    temp_file = destination.with_suffix(destination.suffix + ".part")
    resume_pos = temp_file.stat().st_size if temp_file.exists() else 0

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    request = Request(url=url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        supports_resume = status == 206 and resume_pos > 0
        mode = "ab" if supports_resume else "wb"
        with temp_file.open(mode) as out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    temp_file.replace(destination)


def download_file(
    url: str,
    destination: Path,
    timeout_seconds: int = 120,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
) -> None:
    ensure_dir(destination.parent)
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            _download_once(
                url=url,
                destination=destination,
                timeout_seconds=timeout_seconds,
                chunk_size=chunk_size,
            )
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"Download failed after {attempts} attempts for {url}: {exc}"
                ) from exc
            sleep_s = min(10, attempt * 2)
            time.sleep(sleep_s)


def verify_checksum(path: Path, expected_sha256: Optional[str]) -> bool:
    if not expected_sha256:
        return True
    actual = sha256_file(path)
    return actual.lower() == expected_sha256.lower()


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    try:
        base = base_dir.resolve(strict=False)
        target = target_path.resolve(strict=False)
        target.relative_to(base)
        return True
    except ValueError:
        return False


def extract_archive(archive_path: Path, output_dir: Path, safe_extract: bool = True) -> None:
    ensure_dir(output_dir)
    lower_name = archive_path.name.lower()

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            if safe_extract:
                for member in members:
                    target = output_dir / member.filename
                    if not _is_within_directory(output_dir, target):
                        raise RuntimeError(
                            f"Unsafe zip member path detected: {member.filename}"
                        )
            zf.extractall(output_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            if safe_extract:
                for member in members:
                    target = output_dir / member.name
                    if not _is_within_directory(output_dir, target):
                        raise RuntimeError(
                            f"Unsafe tar member path detected: {member.name}"
                        )
            tf.extractall(output_dir)
        return

    raise RuntimeError(f"Unsupported archive format: {lower_name}")
