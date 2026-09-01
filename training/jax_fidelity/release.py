"""Produce one immutable merged-HF release for TensorRT candidate export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import load_config
from .integrity import (
    artifact_manifest,
    sha256_file,
    validate_dataset_manifest,
    verify_artifact_manifest,
)
from .roundtrip_smoke import validate_roundtrip_evidence

_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")


class ReleaseError(RuntimeError):
    """Release inputs or output state do not satisfy the immutable contract."""


def _json_object(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"could not read {source}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{source} must contain one JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _verified_json(path: Path | str, expected_sha256: str) -> dict[str, Any]:
    source = Path(path)
    if sha256_file(source) != expected_sha256:
        raise ReleaseError(f"approved SHA-256 changed for {source}")
    document = _json_object(source)
    if not document:
        raise ReleaseError(f"terminal evidence is empty: {source}")
    return document


def verified_artifact_binding(
    root: Path | str, manifest_path: Path | str, manifest_sha256: str
) -> dict[str, Any]:
    declaration = _verified_json(manifest_path, manifest_sha256)
    try:
        verify_artifact_manifest(root, declaration)
    except ValueError as error:
        raise ReleaseError(str(error)) from error
    return {
        "manifest_sha256": manifest_sha256,
        "content_sha256": declaration["content_sha256"],
        "files": len(declaration["files"]),
        "bytes": sum(item["bytes"] for item in declaration["files"]),
    }


def _copy_checkpoint(source: Path, destination: Path) -> list[dict[str, Any]]:
    if destination.exists():
        raise ReleaseError(f"release checkpoint already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ReleaseError(f"merged checkpoint may not contain symlinks: {path}")
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return artifact_manifest(destination)["files"]


def candidate_id_from_lineage(
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    training_run_id: str,
    base_model: Mapping[str, Any],
    files: list[dict[str, Any]],
) -> str:
    lineage = {
        "base_model": dict(base_model),
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "files_content_sha256": _canonical_sha256(files),
    }
    return f"fidelity-{_canonical_sha256(lineage)[:20]}"


def candidate_id_for_checkpoint(
    *,
    config_path: Path | str,
    dataset_manifest_sha256: str,
    training_run_id: str,
    merged_hf_checkpoint: Path | str,
) -> str:
    """Derive the release identity before development inference and packaging."""

    config = load_config(config_path)
    files = artifact_manifest(merged_hf_checkpoint)["files"]
    return candidate_id_from_lineage(
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_run_id=training_run_id,
        base_model={
            "id": config.production["model_id"],
            "revision": config.production["model_revision"],
        },
        files=files,
    )


def verify_training_lineage(
    *,
    run: Mapping[str, Any],
    run_sha256: str,
    completion: Mapping[str, Any],
    config_sha256: str,
    dataset_manifest_sha256: str,
    release_inputs: Mapping[str, Any],
    adapter_root: Path,
    adapter_manifest: Mapping[str, Any],
) -> None:
    expected_inputs = {
        "base_checkpoint": {
            name: release_inputs["base_checkpoint"][name]
            for name in ("content_sha256", "files", "bytes")
        },
        "prepared_train": release_inputs["prepared_train"],
        "tokenizer_checkpoint": {
            name: release_inputs["tokenizer_checkpoint"][name]
            for name in ("content_sha256", "files", "bytes")
        },
    }
    metadata = run.get("metadata")
    if (
        run.get("schema_version") != "1.0"
        or run.get("status") != "planned"
        or run.get("stage") != "lora-train"
        or run.get("config_sha256") != config_sha256
        or run.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or not isinstance(metadata, dict)
        or metadata.get("inputs") != expected_inputs
    ):
        raise ReleaseError("training run does not match the approved release inputs")
    if (
        completion.get("run_manifest_sha256") != run_sha256
        or completion.get("evidence", {}).get("inputs") != expected_inputs
    ):
        raise ReleaseError("training completion is not bound to its run and input bytes")

    completion_rows = completion.get("artifacts")
    if not isinstance(completion_rows, list):
        raise ReleaseError("training completion has no artifact rows")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in completion_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ReleaseError("training completion artifact row is malformed")
        if row["path"] in by_path:
            raise ReleaseError("training completion contains duplicate artifact paths")
        by_path[row["path"]] = row
    matched_paths: set[str] = set()
    for file_row in adapter_manifest["files"]:
        artifact = adapter_root / file_row["path"]
        if artifact.is_symlink():
            raise ReleaseError("adapter completion artifact may not be a symbolic link")
        resolved = artifact.resolve()
        try:
            resolved.relative_to(adapter_root.resolve())
        except ValueError as error:
            raise ReleaseError("adapter completion artifact escaped its root") from error
        absolute_path = str(resolved)
        portable_path = Path(file_row["path"]).as_posix()
        row = by_path.get(absolute_path, by_path.get(portable_path))
        expected_path = absolute_path if absolute_path in by_path else portable_path
        if row != {
            "path": expected_path,
            "sha256": file_row["sha256"],
            "bytes": file_row["bytes"],
        }:
            raise ReleaseError("adapter bytes are not artifacts of the approved training run")
        matched_paths.add(expected_path)
    if matched_paths != set(by_path):
        raise ReleaseError("training completion contains artifacts outside the adapter manifest")


def verify_development_evaluation(evaluation: Mapping[str, Any], *, candidate_id: str) -> None:
    if (
        evaluation.get("schema_version") != "1.0"
        or evaluation.get("candidate_id") != candidate_id
        or evaluation.get("stage") != "development"
        or evaluation.get("eligibility_decision") != {"passed": True, "hidden_evaluated": False}
        or not isinstance(evaluation.get("summary"), dict)
        or not evaluation["summary"]
    ):
        raise ReleaseError("evaluation is not a passing development-only eligibility decision")


def produce_release(
    *,
    config_path: Path | str,
    dataset_manifest_path: Path | str,
    dataset_manifest_sha256: str,
    prepared_train_path: Path | str,
    prepared_train_sha256: str,
    base_checkpoint: Path | str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    tokenizer_checkpoint: Path | str,
    tokenizer_manifest_path: Path | str,
    tokenizer_manifest_sha256: str,
    adapter_checkpoint: Path | str,
    adapter_manifest_path: Path | str,
    adapter_manifest_sha256: str,
    merged_hf_checkpoint: Path | str,
    training_run_id: str,
    training_run_path: Path | str,
    training_run_sha256: str,
    training_completion_path: Path | str,
    training_completion_sha256: str,
    roundtrip_evidence_path: Path | str,
    roundtrip_evidence_sha256: str,
    evaluation_evidence_path: Path | str,
    evaluation_evidence_sha256: str,
    release_directory: Path | str,
) -> dict[str, Any]:
    """Validate all lineage and bytes, copy the checkpoint, then write completion last."""

    if not _RUN_ID.fullmatch(training_run_id):
        raise ReleaseError("training_run_id must be an immutable lowercase slug")
    config = load_config(config_path)
    dataset = validate_dataset_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        required_split_records=config.dataset["required_split_records"],
    )
    prepared = Path(prepared_train_path).resolve()
    if sha256_file(prepared) != prepared_train_sha256:
        raise ReleaseError("prepared training data SHA-256 changed")

    adapter_manifest = _verified_json(adapter_manifest_path, adapter_manifest_sha256)
    try:
        verify_artifact_manifest(adapter_checkpoint, adapter_manifest)
    except ValueError as error:
        raise ReleaseError(str(error)) from error
    inputs = {
        "adapter_checkpoint": {
            "manifest_sha256": adapter_manifest_sha256,
            "content_sha256": adapter_manifest["content_sha256"],
            "files": len(adapter_manifest["files"]),
            "bytes": sum(item["bytes"] for item in adapter_manifest["files"]),
        },
        "base_checkpoint": verified_artifact_binding(
            base_checkpoint, base_manifest_path, base_manifest_sha256
        ),
        "prepared_train": {
            "sha256": prepared_train_sha256,
            "bytes": prepared.stat().st_size,
        },
        "tokenizer_checkpoint": verified_artifact_binding(
            tokenizer_checkpoint, tokenizer_manifest_path, tokenizer_manifest_sha256
        ),
    }
    run = _verified_json(training_run_path, training_run_sha256)
    completion = _verified_json(training_completion_path, training_completion_sha256)
    completion_evidence = completion.get("evidence")
    if (
        completion.get("status") != "succeeded"
        or completion.get("run_id") != training_run_id
        or not completion.get("artifacts")
        or not isinstance(completion_evidence, dict)
        or not completion_evidence
    ):
        raise ReleaseError("training completion is not successful, nonempty terminal evidence")
    if run.get("run_id") != training_run_id:
        raise ReleaseError("training run ID differs from the release")
    verify_training_lineage(
        run=run,
        run_sha256=training_run_sha256,
        completion=completion,
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset.manifest_sha256,
        release_inputs=inputs,
        adapter_root=Path(adapter_checkpoint).resolve(),
        adapter_manifest=adapter_manifest,
    )
    roundtrip = _verified_json(roundtrip_evidence_path, roundtrip_evidence_sha256)
    validate_roundtrip_evidence(config, roundtrip, exported_checkpoint=merged_hf_checkpoint)
    source_files = artifact_manifest(merged_hf_checkpoint)["files"]
    base_model = {
        "id": config.production["model_id"],
        "revision": config.production["model_revision"],
    }
    candidate_id = candidate_id_from_lineage(
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset.manifest_sha256,
        training_run_id=training_run_id,
        base_model=base_model,
        files=source_files,
    )
    evaluation = _verified_json(evaluation_evidence_path, evaluation_evidence_sha256)
    verify_development_evaluation(evaluation, candidate_id=candidate_id)

    release_root = Path(release_directory).resolve()
    if release_root.exists():
        raise ReleaseError(f"release directory already exists: {release_root}")
    release_root.mkdir(parents=True, exist_ok=False)
    merged_root = release_root / "merged-hf"
    files = _copy_checkpoint(Path(merged_hf_checkpoint).resolve(), merged_root)
    if files != source_files:
        raise ReleaseError("merged checkpoint bytes changed while creating the release")
    files_content_sha256 = _canonical_sha256(files)
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "release_type": "merged-hf",
        "base_model": base_model,
        "maxtext_revision": config.versions["maxtext_revision"],
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "training_run_id": training_run_id,
        "files_content_sha256": files_content_sha256,
        "files": files,
        "inputs": inputs,
        "terminal_evidence": {
            "training_run": training_run_sha256,
            "training_completion": training_completion_sha256,
            "roundtrip": roundtrip_evidence_sha256,
            "evaluation": evaluation_evidence_sha256,
        },
    }
    document["candidate_id"] = candidate_id
    manifest_path = release_root / "release.manifest.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "candidate_id": document["candidate_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "release_root": str(merged_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--prepared-train-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-train-sha256", required=True)
    for name in ("base", "tokenizer", "adapter"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-manifest-sha256", required=True)
    parser.add_argument("--merged-hf-checkpoint", type=Path, required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-run-sha256", required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--training-completion-sha256", required=True)
    parser.add_argument("--roundtrip-evidence", type=Path, required=True)
    parser.add_argument("--roundtrip-evidence-sha256", required=True)
    parser.add_argument("--evaluation-evidence", type=Path, required=True)
    parser.add_argument("--evaluation-evidence-sha256", required=True)
    parser.add_argument("--release-directory", type=Path, required=True)
    args = parser.parse_args()
    result = produce_release(
        config_path=args.config,
        dataset_manifest_path=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        prepared_train_path=args.prepared_train_jsonl,
        prepared_train_sha256=args.prepared_train_sha256,
        base_checkpoint=args.base_checkpoint,
        base_manifest_path=args.base_manifest,
        base_manifest_sha256=args.base_manifest_sha256,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        tokenizer_manifest_path=args.tokenizer_manifest,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_manifest_path=args.adapter_manifest,
        adapter_manifest_sha256=args.adapter_manifest_sha256,
        merged_hf_checkpoint=args.merged_hf_checkpoint,
        training_run_id=args.training_run_id,
        training_run_path=args.training_run,
        training_run_sha256=args.training_run_sha256,
        training_completion_path=args.training_completion,
        training_completion_sha256=args.training_completion_sha256,
        roundtrip_evidence_path=args.roundtrip_evidence,
        roundtrip_evidence_sha256=args.roundtrip_evidence_sha256,
        evaluation_evidence_path=args.evaluation_evidence,
        evaluation_evidence_sha256=args.evaluation_evidence_sha256,
        release_directory=args.release_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
