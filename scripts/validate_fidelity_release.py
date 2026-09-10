#!/usr/bin/env python3
"""Validate a checksum-bound merged-HF Storylight fidelity release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MAXTEXT_REVISION = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
BASE_EXPORT_ID = "gemma4-e2b-it-int4-awq-v010"
SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")


class ReleaseValidationError(ValueError):
    """The merged checkpoint release is incomplete or has changed."""


@dataclass(frozen=True, slots=True)
class ValidatedRelease:
    manifest_path: Path
    manifest_sha256: str
    release_root: Path
    candidate_id: str
    config_sha256: str
    dataset_manifest_sha256: str
    files_content_sha256: str
    training_run_id: str
    document: Mapping[str, Any]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode()).hexdigest()


def expected_candidate_id(document: Mapping[str, Any]) -> str:
    """Derive a unique candidate from lineage and merged checkpoint bytes."""

    base_model = document.get("base_model")
    if not isinstance(base_model, dict):
        raise ReleaseValidationError("release has no base_model lineage")
    lineage = {
        "base_model": base_model,
        "config_sha256": document.get("config_sha256"),
        "dataset_manifest_sha256": document.get("dataset_manifest_sha256"),
        "training_run_id": document.get("training_run_id"),
        "files_content_sha256": document.get("files_content_sha256"),
    }
    return f"fidelity-{canonical_sha256(lineage)[:20]}"


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"could not load release manifest: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseValidationError("release manifest must contain one JSON object")
    return value


def _valid_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReleaseValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReleaseValidationError("release file path must be non-empty text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReleaseValidationError("release file path must stay beneath the release root")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise ReleaseValidationError("release files may not be symbolic links")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ReleaseValidationError("release file escaped its declared root") from error
    return resolved


def validate_release(
    manifest_path: Path | str,
    release_root: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_config_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_candidate: str | None = None,
) -> ValidatedRelease:
    """Validate lineage, every byte, and the content-addressed candidate ID."""

    manifest = Path(manifest_path).resolve()
    root = Path(release_root).resolve()
    actual_manifest_sha = sha256_file(manifest)
    if actual_manifest_sha != _valid_sha(expected_manifest_sha256, "expected manifest hash"):
        raise ReleaseValidationError("release manifest SHA-256 differs from the approved value")
    document = _json_object(manifest)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseValidationError("release schema_version must be 1.0")
    if document.get("status") != "succeeded" or document.get("release_type") != "merged-hf":
        raise ReleaseValidationError("release must be a successful merged-hf checkpoint")
    base_model = document.get("base_model")
    if base_model != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ReleaseValidationError("release does not descend from the exact Gemma base revision")
    if document.get("maxtext_revision") != MAXTEXT_REVISION:
        raise ReleaseValidationError("release MaxText revision differs from the pinned converter")
    config_sha = _valid_sha(document.get("config_sha256"), "config_sha256")
    dataset_sha = _valid_sha(document.get("dataset_manifest_sha256"), "dataset_manifest_sha256")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise ReleaseValidationError("release config hash differs from the approved value")
    if (
        expected_dataset_manifest_sha256 is not None
        and dataset_sha != expected_dataset_manifest_sha256
    ):
        raise ReleaseValidationError("release dataset hash differs from the approved value")
    training_run_id = document.get("training_run_id")
    if not isinstance(training_run_id, str) or not _RUN_ID.fullmatch(training_run_id):
        raise ReleaseValidationError("training_run_id must be an immutable lowercase slug")

    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseValidationError("release must declare a non-empty file manifest")
    ordered_files = sorted(
        files,
        key=lambda item: item.get("path", "") if isinstance(item, dict) else "",
    )
    if files != ordered_files:
        raise ReleaseValidationError("release file manifest must be sorted by path")
    declared_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseValidationError("each release file declaration must be an object")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in declared_paths:
            raise ReleaseValidationError("release file paths must be unique strings")
        declared_paths.add(relative)
        path = _safe_file(root, relative)
        if not path.is_file():
            raise ReleaseValidationError(f"release file is missing: {relative}")
        declared_bytes = item.get("bytes")
        if type(declared_bytes) is not int or declared_bytes != path.stat().st_size:
            raise ReleaseValidationError(f"release file size mismatch: {relative}")
        if sha256_file(path) != _valid_sha(item.get("sha256"), f"files[{relative}].sha256"):
            raise ReleaseValidationError(f"release file SHA-256 mismatch: {relative}")

    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseValidationError("release root may not contain symbolic links")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != declared_paths:
        raise ReleaseValidationError("release root has missing or unmanifested files")
    required = {"config.json", "tokenizer_config.json"}
    if not required.issubset(declared_paths):
        raise ReleaseValidationError("release lacks required Gemma configuration/tokenizer files")
    if not any(path.endswith(".safetensors") for path in declared_paths):
        raise ReleaseValidationError("release lacks merged SafeTensors weights")
    if not any(path in declared_paths for path in ("tokenizer.json", "tokenizer.model")):
        raise ReleaseValidationError("release lacks tokenizer model data")

    files_content_sha = _valid_sha(document.get("files_content_sha256"), "files_content_sha256")
    if files_content_sha != canonical_sha256(files):
        raise ReleaseValidationError("release file-manifest content hash changed")
    candidate_id = document.get("candidate_id")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ReleaseValidationError("candidate_id must be a content-addressed fidelity ID")
    if candidate_id == BASE_EXPORT_ID or candidate_id != expected_candidate_id(document):
        raise ReleaseValidationError("candidate_id does not match the merged release lineage")
    if expected_candidate is not None and candidate_id != expected_candidate:
        raise ReleaseValidationError("candidate_id differs from the approved candidate")
    return ValidatedRelease(
        manifest_path=manifest,
        manifest_sha256=actual_manifest_sha,
        release_root=root,
        candidate_id=candidate_id,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        files_content_sha256=files_content_sha,
        training_run_id=training_run_id,
        document=document,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--candidate-id")
    args = parser.parse_args()
    validated = validate_release(
        args.manifest,
        args.release_root,
        expected_manifest_sha256=args.manifest_sha256,
        expected_config_sha256=args.config_sha256,
        expected_dataset_manifest_sha256=args.dataset_manifest_sha256,
        expected_candidate=args.candidate_id,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "candidate_id": validated.candidate_id,
                "manifest_sha256": validated.manifest_sha256,
                "config_sha256": validated.config_sha256,
                "dataset_manifest_sha256": validated.dataset_manifest_sha256,
                "files_content_sha256": validated.files_content_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
