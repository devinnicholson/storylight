#!/usr/bin/env python3
"""Download or verify one checksum-bound private fidelity TensorRT export."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "export.manifest.json"
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[a-f0-9]{20}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")

Fetch = Callable[[str, str, str, Path], tuple[str, int]]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("export manifest contains an invalid file path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
        or not candidate.parts
        or candidate.parts[0] != "llm"
    ):
        raise ValueError(f"export manifest contains an unsafe file path: {value!r}")
    return Path(*candidate.parts)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"export manifest repeats JSON key: {key}")
        result[key] = value
    return result


def _load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _sha(expected_sha256, "expected manifest hash")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{MANIFEST_NAME} is missing or unsafe")
    if sha256_file(path) != expected_sha256:
        raise ValueError("export manifest checksum mismatch")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"export manifest is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("export manifest must be a JSON object")
    return value


def _validate_document(
    document: Mapping[str, Any],
    *,
    expected_private_prefix: str | None = None,
    allow_modal_private: bool = True,
) -> list[tuple[Path, int, str]]:
    required = {
        "schema_version": "1.0",
        "status": "succeeded",
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "quantization": "int4_awq",
        "calibration_dataset": "wikitext",
        "calibration_samples": 128,
        "components": ["thinker"],
        "skip_visual": True,
        "skip_audio": True,
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": "v0.10.0",
        "tensorrt_edge_llm_revision": EDGELLM_REVISION,
        "engine_built_in_cloud": False,
    }
    for field, expected in required.items():
        if document.get(field) != expected:
            raise ValueError(f"export manifest {field!r} does not match the pinned value")

    candidate_id = document.get("candidate_id")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("export manifest has an invalid fidelity candidate_id")
    training_run_id = document.get("training_run_id")
    if not isinstance(training_run_id, str) or not _RUN_ID.fullmatch(training_run_id):
        raise ValueError("export manifest has an invalid training_run_id")
    for field in (
        "source_release_manifest_sha256",
        "source_files_content_sha256",
        "config_sha256",
        "dataset_manifest_sha256",
    ):
        _sha(document.get(field), field)

    private_prefix = document.get("private_output_prefix")
    if not isinstance(private_prefix, str):
        raise ValueError("export manifest has no private output prefix")
    parsed_prefix = urllib.parse.urlsplit(private_prefix)
    path_parts = PurePosixPath(parsed_prefix.path.lstrip("/")).parts
    canonical_path = bool(path_parts) and all(part not in {"", ".", ".."} for part in path_parts)
    canonical_prefix = f"{parsed_prefix.scheme}://{parsed_prefix.netloc}/{'/'.join(path_parts)}"
    gcs_prefix = (
        parsed_prefix.scheme == "gs"
        and bool(_BUCKET.fullmatch(parsed_prefix.netloc))
        and canonical_path
    )
    modal_prefix = (
        allow_modal_private
        and parsed_prefix.scheme == "modal-private"
        and parsed_prefix.netloc == "storylight-tensorrt-edge-llm-fidelity"
        and canonical_path
    )
    if (
        parsed_prefix.query
        or parsed_prefix.fragment
        or private_prefix != canonical_prefix
        or not (gcs_prefix or modal_prefix)
    ):
        raise ValueError("export manifest has an invalid private output prefix")
    if private_prefix.endswith("/"):
        raise ValueError("export manifest private output prefix is not canonical")
    if expected_private_prefix is not None and private_prefix != expected_private_prefix:
        raise ValueError("export manifest private output prefix differs from the requested source")

    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("export manifest has no files")
    if type(document.get("file_count")) is not int or document["file_count"] != len(files):
        raise ValueError("export manifest file_count does not match files")
    parsed: list[tuple[Path, int, str]] = []
    declared: set[Path] = set()
    total = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("export manifest contains an invalid file entry")
        relative = _safe_relative_path(entry.get("path"))
        if relative in declared:
            raise ValueError(f"export manifest repeats {relative.as_posix()}")
        declared.add(relative)
        size = entry.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError(f"invalid byte count for {relative.as_posix()}")
        digest = _sha(entry.get("sha256"), f"files[{relative.as_posix()}].sha256")
        parsed.append((relative, size, digest))
        total += size
    if [item[0].as_posix() for item in parsed] != sorted(item[0].as_posix() for item in parsed):
        raise ValueError("export manifest files must be sorted by path")
    if type(document.get("total_bytes")) is not int or document["total_bytes"] != total:
        raise ValueError("export manifest total_bytes does not match files")
    paths = {item[0] for item in parsed}
    if not {
        Path("llm/config.json"),
        Path("llm/model.onnx"),
    }.issubset(paths):
        raise ValueError("export lacks the Gemma graph or configuration")
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("export lacks externalized INT4 weights")
    return parsed


def verify_export_bundle(
    bundle_root: Path | str,
    expected_manifest_sha256: str,
    *,
    expected_private_prefix: str | None = None,
    allow_modal_private: bool = True,
) -> dict[str, Any]:
    """Verify manifest lineage, every byte, and exact bundle membership."""

    unresolved = Path(bundle_root).expanduser()
    if unresolved.is_symlink():
        raise ValueError("export bundle root may not be a symbolic link")
    root = unresolved.resolve()
    if not root.is_dir():
        raise ValueError(f"export bundle does not exist: {root}")
    document = _load_manifest(root / MANIFEST_NAME, expected_manifest_sha256)
    files = _validate_document(
        document,
        expected_private_prefix=expected_private_prefix,
        allow_modal_private=allow_modal_private,
    )
    declared: set[Path] = set()
    total = 0
    for relative, expected_bytes, expected_sha in files:
        bundle_relative = Path("onnx") / relative
        declared.add(bundle_relative)
        path = root / bundle_relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"export file is missing or unsafe: {bundle_relative.as_posix()}")
        size = path.stat().st_size
        if size != expected_bytes:
            raise ValueError(f"export file size mismatch: {bundle_relative.as_posix()}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"export file checksum mismatch: {bundle_relative.as_posix()}")
        total += size

    actual: set[Path] = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            if (directory_path / name).is_symlink():
                raise ValueError("export bundle contains a symbolic-link directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("export bundle contains a symbolic-link file")
            relative = path.relative_to(root)
            if relative != Path(MANIFEST_NAME):
                actual.add(relative)
    if actual != declared:
        raise ValueError("export bundle contains undeclared or missing files")
    if total != document["total_bytes"]:
        raise ValueError("verified export byte count differs from the manifest")
    return document


def _fetch(token: str, bucket: str, object_name: str, destination: Path) -> tuple[str, int]:
    encoded = urllib.parse.quote(object_name, safe="")
    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{encoded}?alt=media"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    digest = hashlib.sha256()
    size = 0
    partial = destination.with_name(f".{destination.name}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _atomic_destination(destination: Path) -> Path:
    destination = destination.expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    return destination


def download_gcs_bundle(
    token: str,
    bucket: str,
    prefix: str,
    destination: Path | str,
    expected_manifest_sha256: str,
    *,
    workers: int = 3,
    fetch: Fetch = _fetch,
) -> dict[str, Any]:
    """Download a private GCS export into a new, atomically published directory."""

    if not token:
        raise ValueError("an OAuth access token is required")
    if not _BUCKET.fullmatch(bucket):
        raise ValueError("invalid private GCS bucket name")
    prefix = prefix.strip("/")
    if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError("invalid private GCS object prefix")
    if workers not in range(1, 5):
        raise ValueError("workers must be between one and four")
    target = _atomic_destination(Path(destination))
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        manifest_path = temporary / MANIFEST_NAME
        actual_sha, _ = fetch(token, bucket, f"{prefix}/{MANIFEST_NAME}", manifest_path)
        if actual_sha != _sha(expected_manifest_sha256, "expected manifest hash"):
            raise ValueError("downloaded export manifest checksum mismatch")
        document = _load_manifest(manifest_path, expected_manifest_sha256)
        expected_prefix = f"gs://{bucket}/{prefix}"
        files = _validate_document(
            document,
            expected_private_prefix=expected_prefix,
            allow_modal_private=False,
        )

        def download_one(item: tuple[Path, int, str]) -> None:
            relative, expected_bytes, expected_sha = item
            actual_digest, actual_bytes = fetch(
                token,
                bucket,
                f"{prefix}/onnx/{relative.as_posix()}",
                temporary / "onnx" / relative,
            )
            if actual_bytes != expected_bytes or actual_digest != expected_sha:
                raise ValueError(f"download verification failed for {relative.as_posix()}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_one, item) for item in files]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        verify_export_bundle(
            temporary,
            expected_manifest_sha256,
            expected_private_prefix=expected_prefix,
            allow_modal_private=False,
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "candidate_id": document["candidate_id"],
        "destination": str(target),
        "file_count": document["file_count"],
        "manifest_sha256": expected_manifest_sha256,
        "result": "downloaded_and_verified",
        "total_bytes": document["total_bytes"],
    }


def copy_local_bundle(
    source: Path | str,
    destination: Path | str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify then atomically copy only declared files from a local export."""

    source_path = Path(source).expanduser()
    document = verify_export_bundle(source_path, expected_manifest_sha256)
    source_path = source_path.resolve()
    target = _atomic_destination(Path(destination))
    if target == source_path or source_path in target.parents:
        raise ValueError("local destination may not be nested in its source bundle")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        for entry in document["files"]:
            relative = _safe_relative_path(entry["path"])
            source_file = source_path / "onnx" / relative
            target_file = temporary / "onnx" / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target_file, follow_symlinks=False)
        shutil.copyfile(
            source_path / MANIFEST_NAME,
            temporary / MANIFEST_NAME,
            follow_symlinks=False,
        )
        verify_export_bundle(temporary, expected_manifest_sha256)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "candidate_id": document["candidate_id"],
        "destination": str(target),
        "file_count": document["file_count"],
        "manifest_sha256": expected_manifest_sha256,
        "result": "copied_and_verified",
        "total_bytes": document["total_bytes"],
    }


def verification_summary(
    document: Mapping[str, Any], expected_manifest_sha256: str
) -> dict[str, Any]:
    """Return only validated lineage needed by the isolated Jetson builder."""

    return {
        "candidate_id": document["candidate_id"],
        "config_sha256": document["config_sha256"],
        "dataset_manifest_sha256": document["dataset_manifest_sha256"],
        "file_count": document["file_count"],
        "manifest_sha256": expected_manifest_sha256,
        "result": "verified",
        "source_files_content_sha256": document["source_files_content_sha256"],
        "source_release_manifest_sha256": document["source_release_manifest_sha256"],
        "total_bytes": document["total_bytes"],
        "training_run_id": document["training_run_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-bundle", type=Path)
    source.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--access-token-stdin", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.local_bundle is not None:
        if args.prefix or args.access_token_stdin:
            parser.error("local verification does not accept GCS credentials or a prefix")
        if args.verify_only:
            if args.destination is not None:
                parser.error("--verify-only does not accept --destination")
            document = verify_export_bundle(
                args.local_bundle,
                args.expected_manifest_sha256,
            )
            result = verification_summary(document, args.expected_manifest_sha256)
        else:
            if args.destination is None:
                parser.error("local copy requires --destination")
            result = copy_local_bundle(
                args.local_bundle,
                args.destination,
                args.expected_manifest_sha256,
            )
    else:
        if args.verify_only:
            parser.error("GCS source cannot be verified without downloading it")
        if args.destination is None or not args.prefix or not args.access_token_stdin:
            parser.error("GCS download requires --prefix, --destination, and --access-token-stdin")
        token = sys.stdin.readline().strip()
        result = download_gcs_bundle(
            token,
            args.bucket,
            args.prefix,
            args.destination,
            args.expected_manifest_sha256,
            workers=args.workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
