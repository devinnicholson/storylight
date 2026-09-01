"""Content-addressed dataset and artifact validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class DatasetIntegrityError(ValueError):
    """A declared dataset artifact is missing, mutable, or malformed."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{rendered}\n".encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetIntegrityError(f"{path} must contain a JSON object")
    return value


def _safe_child(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise DatasetIntegrityError("split path must be non-empty text")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DatasetIntegrityError("split path must remain beneath its manifest directory")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise DatasetIntegrityError("dataset split may not be a symbolic link")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetIntegrityError("split path escaped its manifest directory") from error
    return resolved


def _jsonl_records(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n"):
                    raise DatasetIntegrityError(f"{path}:{line_number} lacks a final newline")
                if not line.strip():
                    raise DatasetIntegrityError(f"{path}:{line_number} is blank")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetIntegrityError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from error
                if not isinstance(value, dict):
                    raise DatasetIntegrityError(f"{path}:{line_number} must be an object")
                count += 1
    except OSError as error:
        raise DatasetIntegrityError(f"could not read split {path}: {error}") from error
    return count


@dataclass(frozen=True)
class ValidatedSplit:
    name: str
    path: Path
    sha256: str
    records: int


@dataclass(frozen=True)
class ValidatedDataset:
    manifest_path: Path
    manifest_sha256: str
    splits: tuple[ValidatedSplit, ...]


def validate_dataset_manifest(
    manifest_path: Path | str,
    *,
    expected_manifest_sha256: str | None = None,
    required_split_records: Mapping[str, int] | None = None,
    allow_hash_only_splits: frozenset[str] = frozenset({"hidden"}),
) -> ValidatedDataset:
    """Verify the manifest and every accessible JSONL byte-for-byte.

    Private hidden splits may intentionally declare only a hash and record count.
    They are never accepted as inputs to training.
    """

    path = Path(manifest_path).resolve()
    actual_manifest_sha = sha256_file(path)
    if expected_manifest_sha256 is not None and actual_manifest_sha != expected_manifest_sha256:
        raise DatasetIntegrityError("dataset manifest SHA-256 does not match the approved value")
    document = _load_json(path)
    splits = document.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise DatasetIntegrityError("dataset manifest must contain a non-empty splits object")
    if required_split_records is not None and not set(required_split_records).issubset(splits):
        missing = sorted(set(required_split_records) - set(splits))
        raise DatasetIntegrityError(f"dataset manifest is missing required splits: {missing}")

    validated: list[ValidatedSplit] = []
    for name in sorted(splits):
        declaration = splits[name]
        if not isinstance(name, str) or not isinstance(declaration, dict):
            raise DatasetIntegrityError("each split declaration must be an object")
        declared_sha = declaration.get("sha256")
        declared_records = declaration.get("records")
        if not isinstance(declared_sha, str) or not _HEX_SHA256.fullmatch(declared_sha):
            raise DatasetIntegrityError(f"split {name} has an invalid SHA-256")
        if type(declared_records) is not int or declared_records < 1:
            raise DatasetIntegrityError(f"split {name} has an invalid record count")
        if (
            required_split_records is not None
            and name in required_split_records
            and declared_records != required_split_records[name]
        ):
            raise DatasetIntegrityError(f"split {name} record count differs from the contract")
        relative = declaration.get("path")
        if relative is None and name in allow_hash_only_splits:
            continue
        split_path = _safe_child(path.parent, relative)
        if not split_path.is_file():
            raise DatasetIntegrityError(f"split {name} is not an accessible regular file")
        if sha256_file(split_path) != declared_sha:
            raise DatasetIntegrityError(f"split {name} SHA-256 mismatch")
        actual_records = _jsonl_records(split_path)
        if actual_records != declared_records:
            raise DatasetIntegrityError(f"split {name} record count mismatch")
        validated.append(ValidatedSplit(name, split_path, declared_sha, declared_records))
    return ValidatedDataset(path, actual_manifest_sha, tuple(validated))


def artifact_manifest(root: Path | str) -> dict[str, Any]:
    """Describe every regular file below a checkpoint directory."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        raise DatasetIntegrityError(f"artifact root is not a directory: {directory}")
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DatasetIntegrityError(f"artifact may not contain symlinks: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not files:
        raise DatasetIntegrityError("artifact directory contains no files")
    return {
        "schema_version": "1.0",
        "files": files,
        "content_sha256": canonical_sha256(files),
    }


def verify_artifact_manifest(root: Path | str, manifest: Mapping[str, Any]) -> None:
    actual = artifact_manifest(root)
    if actual != dict(manifest):
        raise DatasetIntegrityError("checkpoint artifact manifest does not match on-disk bytes")


def artifact_binding(root: Path | str) -> dict[str, Any]:
    """Return a compact binding for all bytes in an artifact directory."""

    manifest = artifact_manifest(root)
    files = manifest["files"]
    return {
        "content_sha256": manifest["content_sha256"],
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
    }


def verify_conversion_manifest(
    manifest_path: Path | str,
    *,
    expected_manifest_sha256: str,
    artifact_roots: Mapping[str, Path | str],
) -> Mapping[str, Any]:
    """Verify every named conversion input against a single approved manifest."""

    path = Path(manifest_path).resolve()
    if sha256_file(path) != expected_manifest_sha256:
        raise DatasetIntegrityError("input artifact manifest SHA-256 mismatch")
    document = _load_json(path)
    if document.get("schema_version") != "1.0":
        raise DatasetIntegrityError("conversion manifest schema_version must be 1.0")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_roots):
        raise DatasetIntegrityError("conversion manifest artifact names do not match the command")
    for name, root in artifact_roots.items():
        declaration = artifacts[name]
        if not isinstance(declaration, dict):
            raise DatasetIntegrityError(f"conversion artifact {name} must be an object")
        verify_artifact_manifest(root, declaration)
    return document
