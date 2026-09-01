#!/usr/bin/env python3
"""Fetch and verify one immutable Vertex JAX release from its private GCS bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast

PROJECT_ID = "your-gcp-project"
Download = Callable[[str, Path], None]


class BlobLike(Protocol):
    name: str
    generation: int

    def reload(self) -> None: ...

    def download_to_filename(self, filename: str, *, if_generation_match: int) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("release file path must be a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("release file path is unsafe")
    return relative


def _verify_portable_package(root: Path) -> None:
    required = {
        "adapter.manifest.json",
        "package.manifest.json",
        "runtime.lock.json",
        "training/run.json",
        "training/completion.json",
    }
    package_path = root / "package.manifest.json"
    if not required.issubset(
        {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    ):
        raise ValueError("portable training package is incomplete")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    rows = package.get("files") if isinstance(package, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("portable package manifest has no files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("portable package entry is malformed")
        relative = _safe_relative(row.get("path"))
        if relative.as_posix() in declared:
            raise ValueError("portable package path is duplicated")
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != row.get("bytes")
            or _sha256(candidate) != row.get("sha256")
        ):
            raise ValueError(f"portable artifact failed verification: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != package_path and path != root / "completion.json"
    }
    if actual != declared:
        raise ValueError("portable package contains undeclared or missing files")


def fetch_release(
    *,
    run_id: str,
    expected_completion_sha256: str,
    destination: Path,
    download: Download,
    remote_objects: Iterable[str],
) -> dict[str, object]:
    """Fetch completion first, then only its declared immutable payload."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists")
    if len(expected_completion_sha256) != 64:
        raise ValueError("expected completion SHA-256 is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bookforge-gcs-release-", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        completion_path = temporary / "completion.json"
        download("completion.json", completion_path)
        if _sha256(completion_path) != expected_completion_sha256:
            raise ValueError("GCS completion checksum changed")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            not isinstance(completion, dict)
            or completion.get("run_id") != run_id
            or completion.get("status") != "succeeded"
            or completion.get("backend") != "vertex-tpu-v6e"
        ):
            raise ValueError("GCS release is not the requested successful Vertex run")
        rows = completion.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("GCS release completion has no files")
        payload = temporary / "payload"
        declared: set[str] = set()
        for row in cast(list[object], rows):
            if not isinstance(row, dict):
                raise ValueError("GCS release entry is malformed")
            relative = _safe_relative(row.get("path"))
            name = relative.as_posix()
            if name == "completion.json" or name in declared:
                raise ValueError("GCS release path is reserved or duplicated")
            declared.add(name)
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            download(name, target)
            if (
                not target.is_file()
                or target.is_symlink()
                or target.stat().st_size != row.get("bytes")
                or _sha256(target) != row.get("sha256")
            ):
                raise ValueError(f"GCS release artifact failed verification: {relative}")
        objects = set(remote_objects)
        if objects != declared | {"completion.json"}:
            raise ValueError("GCS release prefix contains undeclared or missing objects")
        _verify_portable_package(payload)
        shutil.copy2(completion_path, payload / "completion.json")
        shutil.copytree(payload, destination)
    return completion


def fetch_from_gcs(
    *,
    bucket_name: str,
    run_id: str,
    expected_completion_sha256: str,
    destination: Path,
) -> dict[str, object]:
    """Use generation-matched reads from the configured private release bucket."""

    from google.cloud import storage

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    bucket.reload()
    iam = bucket.iam_configuration
    if (
        iam.public_access_prevention != "enforced"
        or iam.uniform_bucket_level_access_enabled is not True
    ):
        raise RuntimeError("release bucket must be private with uniform access")
    prefix = f"releases/{run_id}/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    names = [blob.name.removeprefix(prefix) for blob in blobs]
    by_name = {name: blob for name, blob in zip(names, blobs, strict=True)}
    if len(by_name) != len(blobs) or any(not name for name in names):
        raise ValueError("release prefix contains a duplicate or invalid object")
    initial_generations = {name: int(blob.generation) for name, blob in by_name.items()}

    def download(relative: str, target: Path) -> None:
        blob: BlobLike | None = by_name.get(relative)
        if blob is None:
            raise ValueError(f"release object is missing: {relative}")
        blob.reload()
        blob.download_to_filename(str(target), if_generation_match=blob.generation)

    result = fetch_release(
        run_id=run_id,
        expected_completion_sha256=expected_completion_sha256,
        destination=destination,
        download=download,
        remote_objects=names,
    )
    try:
        final_blobs = list(client.list_blobs(bucket, prefix=prefix))
        final_generations = {
            blob.name.removeprefix(prefix): int(blob.generation) for blob in final_blobs
        }
        if final_generations != initial_generations:
            raise RuntimeError("release prefix changed during verified retrieval")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = fetch_from_gcs(
        bucket_name=args.bucket,
        run_id=args.run_id,
        expected_completion_sha256=args.completion_sha256,
        destination=args.destination,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
