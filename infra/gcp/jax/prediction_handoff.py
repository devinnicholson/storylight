"""Build and verify an immutable Modal prediction reference manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

INPUT_VOLUME = "bookforge-jax-fidelity-inputs"
RELEASE_VOLUME = "bookforge-jax-fidelity-release"
PREDICTION_ROOT = "prediction"
PRODUCER = "bookforge-modal-jax-prediction-reference-stager"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")
_PUBLIC_PATHS = (
    "config.json",
    "dataset/manifest.json",
    "dataset/development.jsonl",
)


def canonical_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_sha256(document: object) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_rows(document: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} has no files")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
            or not isinstance(row.get("sha256"), str)
            or _SHA256.fullmatch(row["sha256"]) is None
        ):
            raise ValueError(f"{label} contains a malformed file declaration")
        pure = PurePosixPath(row["path"])
        relative = pure.as_posix()
        if pure.is_absolute() or ".." in pure.parts or relative != row["path"]:
            raise ValueError(f"{label} contains an unsafe path")
        if relative in result:
            raise ValueError(f"{label} contains a duplicate path")
        if any("hidden" in part.casefold() for part in pure.parts):
            raise ValueError(f"{label} contains hidden data")
        result[relative] = dict(row)
    return result


def _require_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_run_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable lowercase slug")
    return value


def build_prediction_reference_manifest(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    source_input_manifest: dict[str, Any],
    merge_run_id: str,
    merge_completion_sha256: str,
    merge_completion: dict[str, Any],
    candidate_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    prediction_run_id: str,
) -> dict[str, Any]:
    """Derive the exact reference-only population from trusted source receipts."""

    source_run_id = _require_run_id("source_run_id", source_run_id)
    merge_run_id = _require_run_id("merge_run_id", merge_run_id)
    prediction_run_id = _require_run_id("prediction_run_id", prediction_run_id)
    if len({source_run_id, merge_run_id, prediction_run_id}) != 3:
        raise ValueError("source, merge, and prediction run IDs must be distinct")
    source_manifest_sha = _require_sha(
        "source_input_manifest_sha256", source_input_manifest_sha256
    )
    completion_sha = _require_sha("merge_completion_sha256", merge_completion_sha256)
    if manifest_sha256(source_input_manifest) != source_manifest_sha:
        raise ValueError("training source input manifest checksum changed")
    if manifest_sha256(merge_completion) != completion_sha:
        raise ValueError("merge completion checksum changed")
    if (
        source_input_manifest.get("schema_version") != "1.0"
        or source_input_manifest.get("producer") != "bookforge-gcp-jax-input-stager"
        or source_input_manifest.get("status") != "complete"
        or source_input_manifest.get("run_id") != source_run_id
    ):
        raise ValueError("training source input identity changed")
    source_rows = _safe_rows(source_input_manifest, "training source input manifest")
    if not set(_PUBLIC_PATHS).issubset(source_rows):
        raise ValueError("training source lacks the public prediction population")

    if (
        merge_completion.get("schema_version") != "1.0"
        or merge_completion.get("status") != "succeeded"
        or merge_completion.get("backend") != "modal-l4"
        or merge_completion.get("release_type")
        != "provisional-merged-hf-development-candidate"
        or merge_completion.get("merge_run_id") != merge_run_id
        or merge_completion.get("training_input_run_id") != source_run_id
        or merge_completion.get("training_input_manifest_sha256")
        != source_manifest_sha
        or merge_completion.get("development_evaluated") is not False
        or merge_completion.get("release_authorized") is not False
    ):
        raise ValueError("merge completion identity, backend, or eligibility changed")
    release_rows = _safe_rows(merge_completion, "merge completion")
    for binding, source_path in (
        ("config_sha256", "config.json"),
        ("dataset_manifest_sha256", "dataset/manifest.json"),
    ):
        if merge_completion.get(binding) != source_rows[source_path]["sha256"]:
            raise ValueError(f"merge completion lost its source {binding} binding")

    candidate_id = candidate_manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("merged candidate ID is invalid")
    checkpoint_sha = manifest_sha256(checkpoint_manifest)
    checkpoint_content_sha = checkpoint_manifest.get("content_sha256")
    checkpoint_rows = _safe_rows(checkpoint_manifest, "merged checkpoint manifest")
    candidate_sha = manifest_sha256(candidate_manifest)
    expected_eligibility = {
        "development_evaluated": False,
        "hidden_evaluated": False,
        "release_authorized": False,
    }
    if (
        checkpoint_manifest.get("schema_version") != "1.0"
        or not isinstance(checkpoint_content_sha, str)
        or _SHA256.fullmatch(checkpoint_content_sha) is None
        or candidate_manifest.get("schema_version") != "1.0"
        or candidate_manifest.get("status") != "candidate-only"
        or candidate_manifest.get("manifest_type")
        != "merged-hf-development-candidate"
        or candidate_manifest.get("eligibility") != expected_eligibility
        or candidate_manifest.get("config_sha256")
        != source_rows["config.json"]["sha256"]
        or candidate_manifest.get("dataset_manifest_sha256")
        != source_rows["dataset/manifest.json"]["sha256"]
        or candidate_manifest.get("checkpoint_manifest") != checkpoint_manifest
        or candidate_manifest.get("checkpoint_manifest_sha256") != checkpoint_sha
        or candidate_manifest.get("checkpoint_content_sha256") != checkpoint_content_sha
    ):
        raise ValueError("merged candidate or checkpoint manifest binding changed")
    expected_release_bindings = {
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": candidate_sha,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "checkpoint_content_sha256": checkpoint_content_sha,
        "merged_hf_manifest_sha256": checkpoint_sha,
    }
    if any(
        merge_completion.get(name) != value
        for name, value in expected_release_bindings.items()
    ):
        raise ValueError("merge completion lost its candidate/checkpoint binding")
    candidate_row = release_rows.get("candidate.manifest.json")
    checkpoint_manifest_row = release_rows.get("merged-hf.manifest.json")
    if (
        candidate_row is None
        or candidate_row["sha256"] != candidate_sha
        or checkpoint_manifest_row is None
        or checkpoint_manifest_row["sha256"] != checkpoint_sha
    ):
        raise ValueError("merge file table does not bind the candidate manifests")

    references = [
        {
            **source_rows[path],
            "path": path,
            "source_volume": INPUT_VOLUME,
            "source_path": f"{source_run_id}/{path}",
        }
        for path in _PUBLIC_PATHS
    ]
    references.append(
        {
            **candidate_row,
            "path": "candidate/candidate.manifest.json",
            "source_volume": RELEASE_VOLUME,
            "source_path": f"merged/{merge_run_id}/candidate.manifest.json",
        }
    )
    for path, row in sorted(checkpoint_rows.items()):
        release_path = f"merged-hf/{path}"
        expected_release_row = {**row, "path": release_path}
        if release_rows.get(release_path) != expected_release_row:
            raise ValueError("merge completion does not bind every checkpoint byte")
        references.append(
            {
                **row,
                "path": f"candidate/merged-hf/{path}",
                "source_volume": RELEASE_VOLUME,
                "source_path": f"merged/{merge_run_id}/{release_path}",
            }
        )
    references.sort(key=lambda row: str(row["path"]))
    bindings = {
        "candidate_id": candidate_id,
        "config_sha256": source_rows["config.json"]["sha256"],
        "dataset_manifest_sha256": source_rows["dataset/manifest.json"]["sha256"],
        "development_records_sha256": source_rows["dataset/development.jsonl"]["sha256"],
        "candidate_manifest_sha256": candidate_sha,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "checkpoint_content_sha256": checkpoint_content_sha,
    }
    return {
        "schema_version": "1.0",
        "producer": PRODUCER,
        "run_id": prediction_run_id,
        "status": "complete",
        "prefix": f"{PREDICTION_ROOT}/{prediction_run_id}",
        "privacy": {
            "split": "development",
            "public_records_only": True,
            "hidden_records_included": False,
        },
        "bindings": bindings,
        "derivation": {
            "source_run_id": source_run_id,
            "source_input_manifest_sha256": source_manifest_sha,
            "merge_run_id": merge_run_id,
            "merge_completion_sha256": completion_sha,
            "merge_backend": "modal-l4",
            "candidate_manifest_sha256": candidate_sha,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "checkpoint_content_sha256": checkpoint_content_sha,
        },
        "references": references,
    }


def approval_token(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    merge_run_id: str,
    merge_completion_sha256: str,
    prediction_run_id: str,
    candidate_id: str,
    candidate_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_content_sha256: str,
    target_manifest_sha256: str,
) -> str:
    return (
        "APPROVE_MODAL_JAX_PREDICTION_HANDOFF:"
        f"{source_run_id}:{source_input_manifest_sha256}:{merge_run_id}:"
        f"{merge_completion_sha256}:{prediction_run_id}:{candidate_id}:"
        f"{candidate_manifest_sha256}:{checkpoint_manifest_sha256}:"
        f"{checkpoint_content_sha256}:{target_manifest_sha256}"
    )


def prediction_approval_token(
    *,
    run_id: str,
    bindings: dict[str, Any],
    input_manifest_sha256: str,
    batch_size: int = 4,
) -> str:
    """Render the unchanged prediction-worker token for a handoff manifest."""

    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("prediction run ID must be an immutable lowercase slug")
    if type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise ValueError("prediction batch size must be an integer in 1..4")
    candidate_id = bindings.get("candidate_id")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("prediction candidate ID is invalid")
    names = (
        "config_sha256",
        "dataset_manifest_sha256",
        "development_records_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
    )
    values = [_require_sha(name, bindings.get(name)) for name in names]
    input_sha = _require_sha("input_manifest_sha256", input_manifest_sha256)
    return (
        f"APPROVE_MODAL_JAX_PREDICTION:{run_id}:{candidate_id}:{values[0]}:"
        f"{values[1]}:{values[2]}:{values[3]}:{values[4]}:{values[5]}:"
        f"{input_sha}:{batch_size}"
    )


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required handoff input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required handoff input is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"required handoff input must be a JSON object: {path}")
    return value


def _reject_symlinks_or_hidden(root: Path, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{label} is not a regular directory")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"{label} contains a symbolic link")
        if any("hidden" in part.casefold() for part in relative.parts):
            raise RuntimeError(f"{label} contains hidden data")


def _verify_file(path: Path, row: dict[str, Any], label: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a regular file") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != row["bytes"]:
            raise RuntimeError(f"{label} size or type changed")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise RuntimeError(f"{label} checksum changed")
    finally:
        os.close(descriptor)


def _validate_public_dataset(manifest_path: Path, records_path: Path) -> None:
    document = _json_object(manifest_path)
    splits = document.get("splits")
    development = splits.get("development") if isinstance(splits, dict) else None
    hidden = splits.get("hidden") if isinstance(splits, dict) else None
    if (
        not isinstance(development, dict)
        or development.get("public") is not True
        or development.get("path") != "development.jsonl"
        or development.get("records") != 512
        or development.get("sha256") != sha256_file(records_path)
        or not isinstance(hidden, dict)
        or hidden.get("public") is not False
        or hidden.get("path") is not None
    ):
        raise RuntimeError("prediction handoff dataset is not public development-only")
    records = 0
    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("development record is not a JSON object")
            records += 1
    if records != 512:
        raise RuntimeError("prediction handoff development population is incomplete")


def verify_reference_sources(
    *,
    source_input_root: Path,
    source_input_manifest_sha256: str,
    merge_release_root: Path,
    merge_completion_sha256: str,
    prediction_run_id: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify referenced bytes and return the rebuilt manifest and logical paths."""

    _reject_symlinks_or_hidden(source_input_root, "training source input")
    _reject_symlinks_or_hidden(merge_release_root, "merged release")
    source_manifest_path = source_input_root / "inputs.manifest.json"
    completion_path = merge_release_root / "completion.json"
    if sha256_file(source_manifest_path) != source_input_manifest_sha256:
        raise RuntimeError("training source input manifest checksum changed")
    if sha256_file(completion_path) != merge_completion_sha256:
        raise RuntimeError("merge completion checksum changed")
    source_manifest = _json_object(source_manifest_path)
    completion = _json_object(completion_path)
    source_run_id = source_manifest.get("run_id")
    merge_run_id = completion.get("merge_run_id")
    if not isinstance(source_run_id, str) or not isinstance(merge_run_id, str):
        raise RuntimeError("handoff source run identity is missing")
    source_rows = _safe_rows(source_manifest, "training source input manifest")
    release_rows = _safe_rows(completion, "merge completion")
    candidate_path = merge_release_root / "candidate.manifest.json"
    checkpoint_manifest_path = merge_release_root / "merged-hf.manifest.json"
    candidate = _json_object(candidate_path)
    checkpoint_manifest = _json_object(checkpoint_manifest_path)
    manifest = build_prediction_reference_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_input_manifest_sha256,
        source_input_manifest=source_manifest,
        merge_run_id=merge_run_id,
        merge_completion_sha256=merge_completion_sha256,
        merge_completion=completion,
        candidate_manifest=candidate,
        checkpoint_manifest=checkpoint_manifest,
        prediction_run_id=prediction_run_id,
    )
    for relative in _PUBLIC_PATHS:
        _verify_file(source_input_root / relative, source_rows[relative], relative)
    _validate_public_dataset(
        source_input_root / "dataset/manifest.json",
        source_input_root / "dataset/development.jsonl",
    )
    _verify_file(candidate_path, release_rows["candidate.manifest.json"], "candidate manifest")
    _verify_file(
        checkpoint_manifest_path,
        release_rows["merged-hf.manifest.json"],
        "checkpoint manifest",
    )
    checkpoint_rows = _safe_rows(checkpoint_manifest, "merged checkpoint manifest")
    checkpoint_root = merge_release_root / "merged-hf"
    for relative, row in checkpoint_rows.items():
        _verify_file(checkpoint_root / relative, row, f"merged checkpoint {relative}")
    actual_checkpoint = {
        path.relative_to(checkpoint_root).as_posix()
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    }
    if actual_checkpoint != set(checkpoint_rows):
        raise RuntimeError("merged checkpoint contains undeclared or missing files")
    logical_paths = {
        "config": source_input_root / "config.json",
        "dataset_manifest": source_input_root / "dataset/manifest.json",
        "development_records": source_input_root / "dataset/development.jsonl",
        "candidate_manifest": candidate_path,
        "checkpoint": checkpoint_root,
    }
    return manifest, logical_paths
