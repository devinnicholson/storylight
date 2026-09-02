"""Create and verify a checksum-only merged-HF development candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .configuration import load_config
from .integrity import artifact_manifest, canonical_json_bytes, sha256_file
from .release import candidate_id_for_checkpoint, candidate_id_from_lineage

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")
_FIELDS = {
    "schema_version",
    "status",
    "manifest_type",
    "candidate_id",
    "base_model",
    "config_sha256",
    "dataset_manifest_sha256",
    "training_run_id",
    "checkpoint_manifest",
    "checkpoint_manifest_sha256",
    "checkpoint_content_sha256",
    "eligibility",
}


class MergedCandidateError(ValueError):
    """A provisional merged candidate is incomplete or has changed."""


def _write_once(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def build_merged_candidate_manifest(
    *,
    config_path: Path | str,
    dataset_manifest_sha256: str,
    training_run_id: str,
    merged_hf_checkpoint: Path | str,
) -> dict[str, Any]:
    """Bind candidate identity before development evaluation, without releasing it."""

    if _SHA256.fullmatch(dataset_manifest_sha256) is None:
        raise MergedCandidateError("dataset manifest hash must be a lowercase SHA-256")
    if _RUN_ID.fullmatch(training_run_id) is None:
        raise MergedCandidateError("training run ID must be an immutable lowercase slug")
    config = load_config(config_path)
    checkpoint = Path(merged_hf_checkpoint)
    checkpoint_manifest = artifact_manifest(checkpoint)
    checkpoint_manifest_sha = hashlib.sha256(canonical_json_bytes(checkpoint_manifest)).hexdigest()
    candidate_id = candidate_id_for_checkpoint(
        config_path=config_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_run_id=training_run_id,
        merged_hf_checkpoint=checkpoint,
    )
    return {
        "schema_version": "1.0",
        "status": "candidate-only",
        "manifest_type": "merged-hf-development-candidate",
        "candidate_id": candidate_id,
        "base_model": {
            "id": config.production["model_id"],
            "revision": config.production["model_revision"],
        },
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
        "eligibility": {
            "development_evaluated": False,
            "hidden_evaluated": False,
            "release_authorized": False,
        },
    }


def validate_merged_candidate_manifest(
    manifest_path: Path | str,
    merged_hf_checkpoint: Path | str,
    *,
    config_path: Path | str,
    expected_manifest_sha256: str,
    expected_config_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_candidate_id: str | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    document = validate_merged_candidate_declaration(
        manifest,
        config_path=config_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_candidate_id=expected_candidate_id,
    )
    checkpoint_manifest = artifact_manifest(merged_hf_checkpoint)
    checkpoint_manifest_sha = hashlib.sha256(canonical_json_bytes(checkpoint_manifest)).hexdigest()
    if (
        document.get("checkpoint_manifest") != checkpoint_manifest
        or document.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha
        or document.get("checkpoint_content_sha256") != checkpoint_manifest["content_sha256"]
    ):
        raise MergedCandidateError("merged candidate checkpoint bytes changed")
    return document


def _validate_checkpoint_declaration(document: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint = document.get("checkpoint_manifest")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version",
        "files",
        "content_sha256",
    }:
        raise MergedCandidateError("merged candidate checkpoint declaration changed")
    files = checkpoint.get("files")
    if checkpoint.get("schema_version") != "1.0" or not isinstance(files, list) or not files:
        raise MergedCandidateError("merged candidate checkpoint declaration is malformed")
    observed: set[str] = set()
    for row in files:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or _SHA256.fullmatch(row["sha256"]) is None
        ):
            raise MergedCandidateError("merged candidate checkpoint file table is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in observed:
            raise MergedCandidateError("merged candidate checkpoint file path is unsafe")
        observed.add(row["path"])
    if (
        checkpoint.get("content_sha256") != hashlib.sha256(canonical_json_bytes(files)).hexdigest()
        or document.get("checkpoint_content_sha256") != checkpoint.get("content_sha256")
        or document.get("checkpoint_manifest_sha256")
        != hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest()
    ):
        raise MergedCandidateError("merged candidate checkpoint hashes changed")
    return files


def validate_merged_candidate_declaration(
    manifest_path: Path | str,
    *,
    config_path: Path | str,
    expected_manifest_sha256: str,
    expected_config_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_candidate_id: str | None = None,
) -> dict[str, Any]:
    """Validate a checksum-only candidate declaration without fetching model bytes."""

    manifest = Path(manifest_path)
    if (
        _SHA256.fullmatch(expected_manifest_sha256) is None
        or sha256_file(manifest) != expected_manifest_sha256
    ):
        raise MergedCandidateError("merged candidate manifest checksum changed")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MergedCandidateError("merged candidate manifest is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise MergedCandidateError("merged candidate manifest fields changed")
    candidate_id = document.get("candidate_id")
    config_sha = document.get("config_sha256")
    dataset_sha = document.get("dataset_manifest_sha256")
    training_run_id = document.get("training_run_id")
    config = load_config(config_path)
    if (
        document.get("schema_version") != "1.0"
        or document.get("status") != "candidate-only"
        or document.get("manifest_type") != "merged-hf-development-candidate"
        or not isinstance(candidate_id, str)
        or _CANDIDATE_ID.fullmatch(candidate_id) is None
        or not isinstance(config_sha, str)
        or _SHA256.fullmatch(config_sha) is None
        or not isinstance(dataset_sha, str)
        or _SHA256.fullmatch(dataset_sha) is None
        or not isinstance(training_run_id, str)
        or _RUN_ID.fullmatch(training_run_id) is None
        or config.sha256 != config_sha
        or document.get("base_model")
        != {
            "id": config.production["model_id"],
            "revision": config.production["model_revision"],
        }
        or document.get("eligibility")
        != {
            "development_evaluated": False,
            "hidden_evaluated": False,
            "release_authorized": False,
        }
    ):
        raise MergedCandidateError("merged candidate identity or eligibility changed")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise MergedCandidateError("merged candidate config hash changed")
    if (
        expected_dataset_manifest_sha256 is not None
        and dataset_sha != expected_dataset_manifest_sha256
    ):
        raise MergedCandidateError("merged candidate dataset hash changed")
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise MergedCandidateError("merged candidate ID changed")
    files = _validate_checkpoint_declaration(document)
    expected_id = candidate_id_from_lineage(
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset_sha,
        training_run_id=training_run_id,
        base_model=document["base_model"],
        files=files,
    )
    if expected_id != candidate_id:
        raise MergedCandidateError("merged candidate ID is not derived from its exact bytes")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--merged-hf-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise MergedCandidateError("merged candidate manifest is write-once")
    document = build_merged_candidate_manifest(
        config_path=args.config,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        training_run_id=args.training_run_id,
        merged_hf_checkpoint=args.merged_hf_checkpoint,
    )
    _write_once(args.output, document)
    print(
        json.dumps(
            {
                "candidate_id": document["candidate_id"],
                "manifest_sha256": sha256_file(args.output),
                "development_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
