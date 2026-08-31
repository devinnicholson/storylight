#!/usr/bin/env python3
"""Download a private TensorRT checkpoint directly to a Jetson and verify it."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest contains an invalid file path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or value != candidate.as_posix():
        raise ValueError(f"manifest contains an unsafe file path: {value!r}")
    return Path(*candidate.parts)


def _fetch(token: str, bucket: str, object_name: str, destination: Path) -> tuple[str, int]:
    encoded = urllib.parse.quote(object_name, safe="")
    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{encoded}?alt=media"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    digest = hashlib.sha256()
    size = 0
    partial = destination.with_name(f".{destination.name}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _download_one(
    token: str,
    bucket: str,
    prefix: str,
    root: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    relative = _safe_relative_path(entry.get("path"))
    expected_size = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"invalid size for {relative.as_posix()}")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"invalid checksum for {relative.as_posix()}")
    actual_sha, actual_size = _fetch(
        token,
        bucket,
        f"{prefix}/onnx/{relative.as_posix()}",
        root / "onnx" / relative,
    )
    if actual_size != expected_size or actual_sha != expected_sha:
        raise ValueError(f"download verification failed for {relative.as_posix()}")
    return {"bytes": actual_size, "path": relative.as_posix(), "sha256": actual_sha}


def download_bundle(
    token: str,
    bucket: str,
    prefix: str,
    destination: Path,
    workers: int,
) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        _fetch(
            token,
            bucket,
            f"{prefix}/export.manifest.json",
            temporary / "export.manifest.json",
        )
        manifest = json.loads((temporary / "export.manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("result") != "complete":
            raise ValueError("GCS prefix has no complete export manifest")
        expected_prefix = f"gs://{bucket}/{prefix}"
        if manifest.get("gcs_prefix") != expected_prefix:
            raise ValueError("manifest GCS prefix does not match the requested source")
        files = manifest.get("files")
        if not isinstance(files, list) or not files or manifest.get("file_count") != len(files):
            raise ValueError("manifest has an invalid file list")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_download_one, token, bucket, prefix, temporary, entry)
                for entry in files
                if isinstance(entry, dict)
            ]
            completed = [future.result() for future in concurrent.futures.as_completed(futures)]
        if len(completed) != len(files):
            raise ValueError("manifest contains an invalid file entry")
        total_bytes = sum(item["bytes"] for item in completed)
        if total_bytes != manifest.get("total_bytes"):
            raise ValueError("downloaded byte count does not match the manifest")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "destination": str(destination),
        "export_run_id": manifest.get("export_run_id"),
        "file_count": len(completed),
        "result": "downloaded_and_verified",
        "total_bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--access-token-stdin", action="store_true", required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise ValueError("an OAuth access token is required on stdin")
    result = download_bundle(token, args.bucket, args.prefix.strip("/"), args.destination, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
