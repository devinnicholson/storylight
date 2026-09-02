"""Clone one verified roundtrip base into an immutable full-training population."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from infra.gcp.jax.full_input_clone import (
    approval_token,
    build_full_training_manifest,
    canonical_bytes,
    manifest_sha256,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
APP_NAME = "bookforge-jax-full-input-clone"
TIMEOUT_SECONDS = 1_200
MAX_CONTAINERS = 1
_INPUT_ROOT = Path("/inputs")
_RELEASE_ROOT = Path("/releases/roundtrip")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        REPOSITORY_ROOT / "infra/gcp/jax",
        "/opt/bookforge/infra/gcp/jax",
        copy=True,
    )
    .env({"PYTHONPATH": "/opt/bookforge"})
)
input_volume = modal.Volume.from_name(
    "bookforge-jax-fidelity-inputs", create_if_missing=False
)
release_volume = modal.Volume.from_name(
    "bookforge-jax-fidelity-release", create_if_missing=False
)
app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read required JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def _validate_request(request: dict[str, object]) -> tuple[str, str, str, str, str]:
    source_run_id = request.get("source_run_id")
    source_input_manifest_sha = request.get("source_input_manifest_sha256")
    roundtrip_completion_sha = request.get("roundtrip_completion_sha256")
    target_run_id = request.get("target_run_id")
    target_manifest_sha = request.get("target_manifest_sha256")
    if (
        not isinstance(source_run_id, str)
        or not isinstance(target_run_id, str)
        or _RUN_ID.fullmatch(source_run_id) is None
        or _RUN_ID.fullmatch(target_run_id) is None
        or source_run_id == target_run_id
    ):
        raise ValueError("source and target run IDs must be distinct immutable slugs")
    for name, value in (
        ("source_input_manifest_sha256", source_input_manifest_sha),
        ("roundtrip_completion_sha256", roundtrip_completion_sha),
        ("target_manifest_sha256", target_manifest_sha),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    expected = approval_token(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_input_manifest_sha,
        roundtrip_completion_sha256=roundtrip_completion_sha,
        target_run_id=target_run_id,
        target_manifest_sha256=target_manifest_sha,
    )
    if request.get("approval_token") != expected:
        raise ValueError("full-training input clone approval token is not exact")
    return (
        source_run_id,
        source_input_manifest_sha,
        roundtrip_completion_sha,
        target_run_id,
        target_manifest_sha,
    )


def _safe_relative(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != raw:
        raise RuntimeError("input clone declaration contains an unsafe path")
    return Path(*pure.parts)


def _copy_verified(source: Path, destination: Path, row: dict[str, Any]) -> None:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise RuntimeError(f"input clone source is not a regular file: {source}") from error
    source_before = os.fstat(source_descriptor)
    if not stat.S_ISREG(source_before.st_mode) or source_before.st_size != row["bytes"]:
        os.close(source_descriptor)
        raise RuntimeError(f"input clone source size or type changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = os.open(destination, flags, 0o400)
        with os.fdopen(source_descriptor, "rb", closefd=False) as reader, os.fdopen(
            descriptor, "wb", closefd=False
        ) as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        source_after = os.fstat(source_descriptor)
        os.close(source_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
    stable = (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
        source_before.st_mtime_ns,
    ) == (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_size,
        source_after.st_mtime_ns,
    )
    if not stable or digest.hexdigest() != row["sha256"]:
        raise RuntimeError(f"input clone changed while copying: {source}")


def _write_manifest_last(path: Path, document: dict[str, Any]) -> None:
    payload = canonical_bytes(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def clone_population(request: dict[str, object]) -> dict[str, object]:
    """Copy and verify the full population. Exposed separately for filesystem tests."""

    (
        source_run_id,
        source_input_manifest_sha,
        roundtrip_completion_sha,
        target_run_id,
        target_manifest_sha,
    ) = _validate_request(request)
    source_input = _INPUT_ROOT / source_run_id
    source_release = _RELEASE_ROOT / source_run_id
    target = _INPUT_ROOT / target_run_id
    partial = _INPUT_ROOT / f".full-input-clone-{target_run_id}.partial"
    if target.exists() or target.is_symlink() or partial.exists() or partial.is_symlink():
        raise RuntimeError("target run ID already has input state and cannot be retried")

    source_manifest_path = source_input / "inputs.manifest.json"
    completion_path = source_release / "completion.json"
    if _sha256(source_manifest_path) != source_input_manifest_sha:
        raise RuntimeError("source input manifest checksum changed")
    if _sha256(completion_path) != roundtrip_completion_sha:
        raise RuntimeError("roundtrip completion checksum changed")
    source_manifest = _json_object(source_manifest_path)
    completion = _json_object(completion_path)
    base_receipt_path = source_release / "evidence/base-orbax.receipt.json"
    base_manifest_path = source_release / "base-orbax.manifest.json"
    base_receipt = _json_object(base_receipt_path)
    base_manifest = _json_object(base_manifest_path)
    manifest, source_input_paths = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_input_manifest_sha,
        source_input_manifest=source_manifest,
        target_run_id=target_run_id,
        roundtrip_completion_sha256=roundtrip_completion_sha,
        roundtrip_completion=completion,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
    )
    if manifest_sha256(manifest) != target_manifest_sha:
        raise RuntimeError("derived full-training input manifest checksum changed")

    rows = {row["path"]: row for row in manifest["files"]}
    partial.mkdir(mode=0o700)
    for relative in source_input_paths:
        row = rows[relative]
        _copy_verified(
            source_input / _safe_relative(relative),
            partial / _safe_relative(relative),
            row,
        )
    relative_leaf = str(base_receipt["relative_path"])
    for row in base_manifest["files"]:
        target_relative = f"checkpoint/{relative_leaf}/{row['path']}"
        base_source = (
            source_release
            / "base-orbax"
            / _safe_relative(relative_leaf)
            / _safe_relative(row["path"])
        )
        _copy_verified(
            base_source,
            partial / _safe_relative(target_relative),
            rows[target_relative],
        )
    for source, relative in (
        (base_receipt_path, "checkpoint.receipt.json"),
        (base_manifest_path, "checkpoint.manifest.json"),
    ):
        _copy_verified(source, partial / relative, rows[relative])

    actual = {
        path.relative_to(partial).as_posix()
        for path in partial.rglob("*")
        if path.is_file()
    }
    expected = set(rows)
    if actual != expected:
        raise RuntimeError("cloned input population has undeclared or missing files")
    os.replace(partial, target)
    _write_manifest_last(target / "inputs.manifest.json", manifest)
    return {
        "schema_version": "1.0",
        "status": "staged",
        "producer": "bookforge-modal-jax-full-input-clone",
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_input_manifest_sha,
        "roundtrip_completion_sha256": roundtrip_completion_sha,
        "target_run_id": target_run_id,
        "input_manifest_sha256": target_manifest_sha,
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows.values()),
        "manifest_uploaded_last": True,
        "overwrite_enabled": False,
    }


@app.function(
    image=image,
    cpu=4,
    memory=8_192,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=MAX_CONTAINERS,
    volumes={str(_INPUT_ROOT): input_volume, "/releases": release_volume},
)
def clone_finite(request: dict[str, object]) -> dict[str, object]:
    input_volume.reload()
    release_volume.reload()
    result = clone_population(request)
    input_volume.commit()
    return result


@app.local_entrypoint()
def run_cli(
    source_run_id: str,
    source_input_manifest_sha256: str,
    roundtrip_completion_sha256: str,
    target_run_id: str,
    target_manifest_sha256: str,
    approval_token_value: str,
) -> None:
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_input_manifest_sha256,
        "roundtrip_completion_sha256": roundtrip_completion_sha256,
        "target_run_id": target_run_id,
        "target_manifest_sha256": target_manifest_sha256,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    print(json.dumps(clone_finite.remote(request), indent=2, sort_keys=True))
