"""Clone cached roundtrip model bytes into a sealed v3 recovery input prefix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from infra.gcp.jax.full_input_v3 import (
    OVERLAY_PREFIX,
    PREPARED_PATH,
    build_full_training_manifest,
    canonical_bytes,
    clone_approval_token,
    manifest_sha256,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
APP_NAME = "bookforge-jax-full-input-v3-recovery-clone"
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
input_volume = modal.Volume.from_name("bookforge-jax-fidelity-inputs", create_if_missing=False)
release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required v3 clone input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required v3 clone input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read required JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def _safe_relative(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != raw:
        raise RuntimeError("v3 clone declaration contains an unsafe path")
    return Path(*pure.parts)


def _copy_verified(source: Path, destination: Path, row: dict[str, Any]) -> None:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise RuntimeError(f"v3 clone source is not a regular file: {source}") from error
    source_before = os.fstat(source_descriptor)
    if not stat.S_ISREG(source_before.st_mode) or source_before.st_size != row["bytes"]:
        os.close(source_descriptor)
        raise RuntimeError(f"v3 clone source size or type changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        destination_descriptor = os.open(destination, flags, 0o400)
        with (
            os.fdopen(source_descriptor, "rb", closefd=False) as reader,
            os.fdopen(destination_descriptor, "wb", closefd=False) as writer,
        ):
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.fchmod(destination_descriptor, 0o400)
    finally:
        source_after = os.fstat(source_descriptor)
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
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
        raise RuntimeError(f"v3 clone source changed while copying: {source}")


def _write_manifest_last(path: Path, document: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _validate_request(
    request: dict[str, object],
) -> tuple[str, str, str, str, str, str]:
    source_run_id = request.get("source_run_id")
    source_manifest_sha = request.get("source_input_manifest_sha256")
    completion_sha = request.get("roundtrip_completion_sha256")
    overlay_sha = request.get("overlay_manifest_sha256")
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
        ("source_input_manifest_sha256", source_manifest_sha),
        ("roundtrip_completion_sha256", completion_sha),
        ("overlay_manifest_sha256", overlay_sha),
        ("target_manifest_sha256", target_manifest_sha),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    expected = clone_approval_token(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_manifest_sha,
        roundtrip_completion_sha256=completion_sha,
        overlay_manifest_sha256=overlay_sha,
        target_run_id=target_run_id,
        target_manifest_sha256=target_manifest_sha,
    )
    if request.get("approval_token") != expected:
        raise ValueError("v3 recovery clone approval token is not exact")
    return (
        source_run_id,
        source_manifest_sha,
        completion_sha,
        overlay_sha,
        target_run_id,
        target_manifest_sha,
    )


def clone_population(request: dict[str, object]) -> dict[str, object]:
    """Clone a complete v3 recovery population without host model upload."""

    (
        source_run_id,
        source_manifest_sha,
        completion_sha,
        overlay_sha,
        target_run_id,
        target_manifest_sha,
    ) = _validate_request(request)
    source_input = _INPUT_ROOT / source_run_id
    source_release = _RELEASE_ROOT / source_run_id
    overlay = _INPUT_ROOT / OVERLAY_PREFIX / target_run_id
    target = _INPUT_ROOT / target_run_id
    partial = _INPUT_ROOT / f".full-input-v3-recovery-{target_run_id}.partial"
    if target.exists() or target.is_symlink() or partial.exists() or partial.is_symlink():
        raise RuntimeError("target run ID already has input state and cannot be retried")

    source_manifest_path = source_input / "inputs.manifest.json"
    completion_path = source_release / "completion.json"
    overlay_manifest_path = overlay / "staging.manifest.json"
    if _sha256(source_manifest_path) != source_manifest_sha:
        raise RuntimeError("source input manifest checksum changed")
    if _sha256(completion_path) != completion_sha:
        raise RuntimeError("roundtrip completion checksum changed")
    if _sha256(overlay_manifest_path) != overlay_sha:
        raise RuntimeError("v3 overlay manifest checksum changed")

    source_manifest = _json_object(source_manifest_path)
    completion = _json_object(completion_path)
    overlay_manifest = _json_object(overlay_manifest_path)
    base_receipt_path = source_release / "evidence/base-orbax.receipt.json"
    base_manifest_path = source_release / "base-orbax.manifest.json"
    base_receipt = _json_object(base_receipt_path)
    base_manifest = _json_object(base_manifest_path)
    manifest, tokenizer_paths, overlay_paths = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_manifest_sha,
        source_input_manifest=source_manifest,
        roundtrip_completion_sha256=completion_sha,
        roundtrip_completion=completion,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
        overlay_manifest_sha256=overlay_sha,
        overlay_manifest=overlay_manifest,
        target_run_id=target_run_id,
    )
    if manifest_sha256(manifest) != target_manifest_sha:
        raise RuntimeError("derived v3 target manifest checksum changed")

    rows = {row["path"]: row for row in manifest["files"]}
    partial.mkdir(mode=0o700)
    for relative in overlay_paths:
        _copy_verified(
            overlay / _safe_relative(relative),
            partial / _safe_relative(relative),
            rows[relative],
        )
    _copy_verified(
        overlay / "recovery/overfit-canary.train.jsonl",
        partial / PREPARED_PATH,
        rows[PREPARED_PATH],
    )
    for relative in tokenizer_paths:
        _copy_verified(
            source_input / _safe_relative(relative),
            partial / _safe_relative(relative),
            rows[relative],
        )
    relative_leaf = str(base_receipt["relative_path"])
    for row in base_manifest["files"]:
        target_relative = f"checkpoint/{relative_leaf}/{row['path']}"
        source = (
            source_release
            / "base-orbax"
            / _safe_relative(relative_leaf)
            / _safe_relative(row["path"])
        )
        _copy_verified(source, partial / target_relative, rows[target_relative])
    for source, relative in (
        (base_receipt_path, "checkpoint.receipt.json"),
        (base_manifest_path, "checkpoint.manifest.json"),
    ):
        _copy_verified(source, partial / relative, rows[relative])

    actual = {path.relative_to(partial).as_posix() for path in partial.rglob("*") if path.is_file()}
    if actual != set(rows):
        raise RuntimeError("v3 target population has undeclared or missing files")
    os.replace(partial, target)
    _write_manifest_last(target / "inputs.manifest.json", manifest)
    return {
        "schema_version": "1.0",
        "producer": "bookforge-modal-jax-full-input-v3-recovery-cloner",
        "status": "complete",
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_manifest_sha,
        "roundtrip_completion_sha256": completion_sha,
        "overlay_manifest_sha256": overlay_sha,
        "target_run_id": target_run_id,
        "target_manifest_sha256": target_manifest_sha,
        "prepared_source": "recovery/overfit-canary.train.jsonl",
        "public_probe_held_out": True,
        "hidden_records_included": False,
        "cached_model_upload_bytes": 0,
    }


@app.function(
    image=image,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=MAX_CONTAINERS,
    volumes={str(_INPUT_ROOT): input_volume, str(_RELEASE_ROOT): release_volume},
)
def clone_finite(request: dict[str, object]) -> dict[str, object]:
    input_volume.reload()
    release_volume.reload()
    result = clone_population(request)
    input_volume.commit()
    return result


@app.local_entrypoint()
def main(
    source_run_id: str,
    source_input_manifest_sha256: str,
    roundtrip_completion_sha256: str,
    overlay_manifest_sha256: str,
    target_run_id: str,
    target_manifest_sha256: str,
    approval_token: str,
) -> None:
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_input_manifest_sha256,
        "roundtrip_completion_sha256": roundtrip_completion_sha256,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "target_run_id": target_run_id,
        "target_manifest_sha256": target_manifest_sha256,
        "approval_token": approval_token,
    }
    print(json.dumps(clone_finite.remote(request), indent=2, sort_keys=True))
