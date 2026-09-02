"""Restore immutable Hugging Face generation metadata after MaxText export."""

from __future__ import annotations

import errno
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import EOS_TOKEN_IDS
from .integrity import artifact_manifest, canonical_json_bytes, sha256_file

SCHEMA_VERSION = "bookforge-hf-generation-normalization-v1"
GENERATION_CONFIG = "generation_config.json"


class GenerationNormalizationError(ValueError):
    """Generation metadata cannot be restored without changing model bytes."""


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationNormalizationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise GenerationNormalizationError(f"{label} must contain one JSON object")
    return value


def _manifest_files(manifest: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise GenerationNormalizationError(f"{label} has no files")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise GenerationNormalizationError(f"{label} contains a malformed file")
        path = row["path"]
        if path in result:
            raise GenerationNormalizationError(f"{label} contains a duplicate file")
        result[path] = row
    return result


def _generation_row(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    row = _manifest_files(manifest, "original checkpoint manifest").get(GENERATION_CONFIG)
    if row is None:
        raise GenerationNormalizationError("original checkpoint has no generation config")
    return row


def _validate_generation_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise GenerationNormalizationError("generation config is not a safe regular file")
    document = _json_object(path, "generation config")
    if document.get("eos_token_id") != list(EOS_TOKEN_IDS):
        raise GenerationNormalizationError("generation config changed the pinned EOS token set")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EPERM,
            errno.EXDEV,
        }
        if error.errno not in unsupported:
            raise
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())


def _copy_checkpoint(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise GenerationNormalizationError("raw export is not a safe directory")
    if destination.exists() or destination.is_symlink():
        raise GenerationNormalizationError("normalized checkpoint destination already exists")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise GenerationNormalizationError("raw export contains a symbolic link")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=False)
        elif path.is_file():
            _copy_file(path, target)
        else:
            raise GenerationNormalizationError("raw export contains a special file")


def _expected_output_manifest(
    source_manifest: Mapping[str, Any],
    generation_row: Mapping[str, Any],
) -> dict[str, Any]:
    files = _manifest_files(source_manifest, "raw export manifest")
    existing = files.get(GENERATION_CONFIG)
    if existing is not None and dict(existing) != dict(generation_row):
        raise GenerationNormalizationError("MaxText exported different generation metadata")
    files[GENERATION_CONFIG] = dict(generation_row)
    ordered = [files[name] for name in sorted(files)]
    from .integrity import canonical_sha256

    return {
        "schema_version": "1.0",
        "files": ordered,
        "content_sha256": canonical_sha256(ordered),
    }


def normalize_generation_config(
    *,
    source_checkpoint: Path | str,
    original_checkpoint: Path | str,
    destination: Path | str,
    receipt_path: Path | str,
    prior_conversion_completion_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a byte-identical export plus the original pinned generation config."""

    source = Path(source_checkpoint).resolve()
    original = Path(original_checkpoint).resolve()
    output = Path(destination).resolve()
    receipt = Path(receipt_path).resolve()
    if receipt.exists() or receipt.is_symlink():
        raise GenerationNormalizationError("normalization receipt already exists")
    generation = original / GENERATION_CONFIG
    _validate_generation_file(generation)
    source_manifest = artifact_manifest(source)
    original_manifest = artifact_manifest(original)
    generation_row = _generation_row(original_manifest)
    if (
        generation_row.get("sha256") != sha256_file(generation)
        or generation_row.get("bytes") != generation.stat().st_size
    ):
        raise GenerationNormalizationError("original generation config is not manifest-bound")
    expected_output = _expected_output_manifest(source_manifest, generation_row)

    _copy_checkpoint(source, output)
    output_generation = output / GENERATION_CONFIG
    if not output_generation.exists():
        _copy_file(generation, output_generation)
    _validate_generation_file(output_generation)
    output_manifest = artifact_manifest(output)
    if output_manifest != expected_output:
        raise GenerationNormalizationError("normalization changed files beyond generation metadata")

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "operation": "restore-original-generation-config",
        "prior_conversion_completion_sha256": prior_conversion_completion_sha256,
        "source_checkpoint_manifest": source_manifest,
        "original_checkpoint_content_sha256": original_manifest["content_sha256"],
        "original_generation_config": dict(generation_row),
        "output_checkpoint_manifest": output_manifest,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    return document


def validate_generation_normalization(
    receipt: Mapping[str, Any],
    *,
    source_checkpoint_manifest: Mapping[str, Any],
    original_checkpoint_manifest: Mapping[str, Any],
    normalized_checkpoint: Path | str,
    source_generation_config: Path | str,
    prior_conversion_completion_sha256: str | None,
) -> None:
    """Verify a normalization receipt using only packaged custody evidence."""

    generation = Path(source_generation_config).resolve()
    _validate_generation_file(generation)
    generation_row = _generation_row(original_checkpoint_manifest)
    if (
        generation_row.get("sha256") != sha256_file(generation)
        or generation_row.get("bytes") != generation.stat().st_size
    ):
        raise GenerationNormalizationError("packaged generation config changed")
    expected_output = _expected_output_manifest(source_checkpoint_manifest, generation_row)
    actual_output = artifact_manifest(normalized_checkpoint)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "succeeded",
        "operation": "restore-original-generation-config",
        "prior_conversion_completion_sha256": prior_conversion_completion_sha256,
        "source_checkpoint_manifest": dict(source_checkpoint_manifest),
        "original_checkpoint_content_sha256": original_checkpoint_manifest.get(
            "content_sha256"
        ),
        "original_generation_config": dict(generation_row),
        "output_checkpoint_manifest": expected_output,
    }
    if dict(receipt) != expected or actual_output != expected_output:
        raise GenerationNormalizationError("generation normalization receipt is inconsistent")
