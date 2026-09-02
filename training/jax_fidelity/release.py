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
from .development_eligibility import (
    DevelopmentEligibilityError,
    validate_development_eligibility,
)
from .integrity import (
    artifact_manifest,
    sha256_file,
    validate_dataset_manifest,
    verify_artifact_manifest,
    verify_conversion_manifest,
)
from .roundtrip_smoke import validate_roundtrip_contract

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


def verify_development_evaluation(
    evaluation: Mapping[str, Any],
    *,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    training_run_id: str,
) -> None:
    try:
        validate_development_eligibility(
            evaluation,
            candidate_id=candidate_id,
            config_sha256=config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            training_run_id=training_run_id,
        )
    except DevelopmentEligibilityError as error:
        raise ReleaseError(str(error)) from error


def _verify_merge_file_table(root: Path, completion: Mapping[str, Any]) -> None:
    rows = completion.get("files")
    if not isinstance(rows, list) or not rows:
        raise ReleaseError("merge completion has no file table")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ReleaseError("merge file declaration is malformed")
        relative = Path(row["path"])
        name = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in declared:
            raise ReleaseError("merge file path is unsafe or duplicated")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise ReleaseError(f"merge file failed checksum verification: {name}")
        declared.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "completion.json"
    }
    if actual != declared:
        raise ReleaseError("merge release has undeclared or missing files")


def _required_merge_file(root: Path, path: Path | str, relative: str) -> Path:
    expected = (root / relative).resolve()
    candidate = Path(path).resolve()
    if candidate != expected or not candidate.is_file() or candidate.is_symlink():
        raise ReleaseError(f"merge evidence is not the declared {relative}")
    return candidate


def verify_merge_provenance(
    *,
    config_path: Path | str,
    dataset_manifest_sha256: str,
    training_run_id: str,
    merged_hf_checkpoint: Path | str,
    original_hf_checkpoint: Path | str,
    original_hf_manifest_path: Path | str,
    original_hf_manifest_sha256: str,
    base_checkpoint_root: Path | str,
    base_checkpoint: Path | str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    base_receipt_path: Path | str,
    base_receipt_sha256: str,
    adapter_checkpoint: Path | str,
    adapter_manifest_path: Path | str,
    adapter_manifest_sha256: str,
    adapter_receipt_path: Path | str,
    adapter_receipt_sha256: str,
    merge_release_root: Path | str,
    merge_completion_path: Path | str,
    merge_completion_sha256: str,
    candidate_manifest_path: Path | str,
    candidate_manifest_sha256: str,
    source_bindings_path: Path | str,
    source_bindings_sha256: str,
    conversion_input_path: Path | str,
    conversion_input_sha256: str,
    conversion_run_path: Path | str,
    conversion_run_sha256: str,
    conversion_completion_path: Path | str,
    conversion_completion_sha256: str,
    training_run_sha256: str,
    training_completion_sha256: str,
) -> dict[str, str]:
    """Reproduce the exact full-adapter MaxText-to-HF custody chain."""

    from .manifests import stable_run_id
    from .merged_candidate import validate_merged_candidate_manifest
    from .orbax_receipt import terminal_checkpoint_step, verify_orbax_leaf_receipt

    config = load_config(config_path)
    merge_root = Path(merge_release_root).resolve()
    completion_path = _required_merge_file(
        merge_root, merge_completion_path, "completion.json"
    )
    completion = _verified_json(completion_path, merge_completion_sha256)
    merged = Path(merged_hf_checkpoint).resolve()
    if merged != (merge_root / "merged-hf").resolve() or not merged.is_dir():
        raise ReleaseError("merged checkpoint is not the trusted merge release payload")
    if (
        completion.get("schema_version") != "1.0"
        or completion.get("status") != "succeeded"
        or completion.get("backend") != "modal-l4"
        or completion.get("release_type")
        != "provisional-merged-hf-development-candidate"
        or completion.get("config_sha256") != config.sha256
        or completion.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or completion.get("training_run_id") != training_run_id
        or completion.get("development_evaluated") is not False
        or completion.get("release_authorized") is not False
    ):
        raise ReleaseError("merge completion identity or eligibility changed")
    _verify_merge_file_table(merge_root, completion)

    candidate_path = _required_merge_file(
        merge_root, candidate_manifest_path, "candidate.manifest.json"
    )
    bindings_path = _required_merge_file(
        merge_root, source_bindings_path, "evidence/source-bindings.json"
    )
    input_path = _required_merge_file(
        merge_root, conversion_input_path, "evidence/maxtext-to-hf.inputs.json"
    )
    run_path = _required_merge_file(
        merge_root, conversion_run_path, "evidence/maxtext-to-hf.run.json"
    )
    conversion_path = _required_merge_file(
        merge_root,
        conversion_completion_path,
        "evidence/maxtext-to-hf.completion.json",
    )
    adapter_receipt_path = _required_merge_file(
        merge_root,
        adapter_receipt_path,
        "evidence/adapter-orbax.receipt.json",
    )
    merged_manifest_path = _required_merge_file(
        merge_root,
        merge_root / "merged-hf.manifest.json",
        "merged-hf.manifest.json",
    )
    expected_files = {
        candidate_path: candidate_manifest_sha256,
        bindings_path: source_bindings_sha256,
        input_path: conversion_input_sha256,
        run_path: conversion_run_sha256,
        conversion_path: conversion_completion_sha256,
    }
    if any(sha256_file(path) != digest for path, digest in expected_files.items()):
        raise ReleaseError("approved merge or conversion receipt checksum changed")
    if (
        completion.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or completion.get("merged_hf_manifest_sha256")
        != sha256_file(merged_manifest_path)
        or completion.get("conversion_input_manifest_sha256") != conversion_input_sha256
        or completion.get("conversion_run_sha256") != conversion_run_sha256
        or completion.get("conversion_completion_sha256") != conversion_completion_sha256
    ):
        raise ReleaseError("merge completion does not bind the conversion receipts")

    candidate = validate_merged_candidate_manifest(
        candidate_path,
        merged,
        config_path=config_path,
        expected_manifest_sha256=candidate_manifest_sha256,
        expected_config_sha256=config.sha256,
        expected_dataset_manifest_sha256=dataset_manifest_sha256,
        expected_candidate_id=str(completion.get("candidate_id")),
    )
    if candidate.get("training_run_id") != training_run_id:
        raise ReleaseError("merged candidate training lineage changed")
    merged_manifest = _json_object(merged_manifest_path)
    try:
        verify_artifact_manifest(merged, merged_manifest)
    except ValueError as error:
        raise ReleaseError(str(error)) from error
    if (
        completion.get("checkpoint_manifest_sha256")
        != _canonical_sha256(merged_manifest)
        or completion.get("checkpoint_content_sha256")
        != merged_manifest.get("content_sha256")
    ):
        raise ReleaseError("merge completion does not bind the merged HF manifest")

    base_receipt = _verified_json(base_receipt_path, base_receipt_sha256)
    adapter_receipt = _verified_json(adapter_receipt_path, adapter_receipt_sha256)
    try:
        selected_base = verify_orbax_leaf_receipt(
            base_checkpoint_root,
            base_receipt,
            expected_step=0,
            role="base-maxtext",
        )
        selected_adapter = verify_orbax_leaf_receipt(
            adapter_checkpoint,
            adapter_receipt,
            expected_step=terminal_checkpoint_step(config.training["steps"]),
            role="full-lora",
        )
    except ValueError as error:
        raise ReleaseError(str(error)) from error
    if selected_base != Path(base_checkpoint).resolve():
        raise ReleaseError("release base checkpoint is not the receipt-selected Orbax leaf")
    verified_artifact_binding(selected_base, base_manifest_path, base_manifest_sha256)
    verified_artifact_binding(
        original_hf_checkpoint,
        original_hf_manifest_path,
        original_hf_manifest_sha256,
    )
    # The whole adapter manifest proves the selected leaf belongs to the approved
    # training package; the receipt proves which terminal step the conversion used.
    verified_artifact_binding(
        adapter_checkpoint,
        adapter_manifest_path,
        adapter_manifest_sha256,
    )
    try:
        verify_conversion_manifest(
            input_path,
            expected_manifest_sha256=conversion_input_sha256,
            artifact_roots={
                "adapter_checkpoint": selected_adapter,
                "base_checkpoint": selected_base,
                "hf_checkpoint": original_hf_checkpoint,
            },
        )
    except ValueError as error:
        raise ReleaseError(str(error)) from error

    source = _verified_json(bindings_path, source_bindings_sha256)
    expected_source = {
        "hf_snapshot_manifest_sha256": original_hf_manifest_sha256,
        "base_orbax_receipt_sha256": base_receipt_sha256,
        "base_orbax_manifest_sha256": base_manifest_sha256,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "adapter_orbax_receipt_sha256": adapter_receipt_sha256,
        "training_run_sha256": training_run_sha256,
        "training_completion_sha256": training_completion_sha256,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "conversion_input_manifest_sha256": conversion_input_sha256,
        "conversion_run_sha256": conversion_run_sha256,
        "conversion_completion_sha256": conversion_completion_sha256,
    }
    if any(source.get(name) != digest for name, digest in expected_source.items()):
        raise ReleaseError("merge source bindings differ from the approved conversion lineage")
    inherited_hashes = (
        "input_manifest_sha256",
        "roundtrip_completion_sha256",
        "training_release_completion_sha256",
    )
    for name in inherited_hashes:
        digest = source.get(name)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or completion.get(name) != digest
        ):
            raise ReleaseError(f"merge completion changed inherited source hash: {name}")
    conversion_run_id = source.get("conversion_run_id")
    expected_conversion_id = stable_run_id(
        stage="maxtext-to-hf",
        config_sha256=config.sha256,
        dataset_manifest_sha256=conversion_input_sha256,
    )
    run = _verified_json(run_path, conversion_run_sha256)
    conversion = _verified_json(conversion_path, conversion_completion_sha256)
    metadata = run.get("metadata")
    evidence = conversion.get("evidence")
    observed_merged_manifest = artifact_manifest(merged)
    if (
        conversion_run_id != expected_conversion_id
        or run.get("schema_version") != "1.0"
        or run.get("run_id") != conversion_run_id
        or run.get("stage") != "maxtext-to-hf"
        or run.get("status") != "planned"
        or run.get("config_sha256") != config.sha256
        or run.get("dataset_manifest_sha256") != conversion_input_sha256
        or not isinstance(metadata, dict)
        or metadata.get("conversion_input_manifest_sha256") != conversion_input_sha256
        or metadata.get("direction") != "maxtext-to-hf"
        or conversion.get("schema_version") != "1.0"
        or conversion.get("run_id") != conversion_run_id
        or conversion.get("status") != "succeeded"
        or conversion.get("run_manifest_sha256") != conversion_run_sha256
        or not conversion.get("artifacts")
        or not isinstance(evidence, dict)
        or evidence.get("direction") != "maxtext-to-hf"
        or evidence.get("input_manifest_sha256") != conversion_input_sha256
        or evidence.get("output_manifest") != observed_merged_manifest
        or merged_manifest != observed_merged_manifest
    ):
        raise ReleaseError("MaxText-to-HF conversion receipt or output binding changed")
    return {
        "merge_completion": merge_completion_sha256,
        "candidate_manifest": candidate_manifest_sha256,
        "merge_source_bindings": source_bindings_sha256,
        "conversion_input": conversion_input_sha256,
        "conversion_run": conversion_run_sha256,
        "conversion_completion": conversion_completion_sha256,
        "original_hf_manifest": original_hf_manifest_sha256,
        "base_orbax_receipt": base_receipt_sha256,
        "base_orbax_manifest": base_manifest_sha256,
        "full_adapter_manifest": adapter_manifest_sha256,
        "adapter_orbax_receipt": adapter_receipt_sha256,
        "merged_hf_manifest": sha256_file(merged_manifest_path),
        "staged_input_manifest": str(source["input_manifest_sha256"]),
        "roundtrip_completion": str(source["roundtrip_completion_sha256"]),
        "training_release_completion": str(
            source["training_release_completion_sha256"]
        ),
    }


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
    original_hf_checkpoint: Path | str,
    original_hf_manifest_path: Path | str,
    original_hf_manifest_sha256: str,
    base_checkpoint_root: Path | str,
    base_receipt_path: Path | str,
    base_receipt_sha256: str,
    adapter_receipt_path: Path | str,
    adapter_receipt_sha256: str,
    merge_release_root: Path | str,
    merge_completion_path: Path | str,
    merge_completion_sha256: str,
    candidate_manifest_path: Path | str,
    candidate_manifest_sha256: str,
    source_bindings_path: Path | str,
    source_bindings_sha256: str,
    conversion_input_path: Path | str,
    conversion_input_sha256: str,
    conversion_run_path: Path | str,
    conversion_run_sha256: str,
    conversion_completion_path: Path | str,
    conversion_completion_sha256: str,
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
    # The roundtrip checkpoint is a fixed smoke canary. The trained checkpoint is
    # expected to contain different weights, so only the checksum-bound canary
    # contract and lineage apply to this release.
    validate_roundtrip_contract(config, roundtrip)
    merge_evidence = verify_merge_provenance(
        config_path=config_path,
        dataset_manifest_sha256=dataset.manifest_sha256,
        training_run_id=training_run_id,
        merged_hf_checkpoint=merged_hf_checkpoint,
        original_hf_checkpoint=original_hf_checkpoint,
        original_hf_manifest_path=original_hf_manifest_path,
        original_hf_manifest_sha256=original_hf_manifest_sha256,
        base_checkpoint_root=base_checkpoint_root,
        base_checkpoint=base_checkpoint,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        base_receipt_path=base_receipt_path,
        base_receipt_sha256=base_receipt_sha256,
        adapter_checkpoint=adapter_checkpoint,
        adapter_manifest_path=adapter_manifest_path,
        adapter_manifest_sha256=adapter_manifest_sha256,
        adapter_receipt_path=adapter_receipt_path,
        adapter_receipt_sha256=adapter_receipt_sha256,
        merge_release_root=merge_release_root,
        merge_completion_path=merge_completion_path,
        merge_completion_sha256=merge_completion_sha256,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_manifest_sha256,
        source_bindings_path=source_bindings_path,
        source_bindings_sha256=source_bindings_sha256,
        conversion_input_path=conversion_input_path,
        conversion_input_sha256=conversion_input_sha256,
        conversion_run_path=conversion_run_path,
        conversion_run_sha256=conversion_run_sha256,
        conversion_completion_path=conversion_completion_path,
        conversion_completion_sha256=conversion_completion_sha256,
        training_run_sha256=training_run_sha256,
        training_completion_sha256=training_completion_sha256,
    )
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
    verify_development_evaluation(
        evaluation,
        candidate_id=candidate_id,
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset.manifest_sha256,
        training_run_id=training_run_id,
    )

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
            **merge_evidence,
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
    parser.add_argument("--original-hf-checkpoint", type=Path, required=True)
    parser.add_argument("--original-hf-manifest", type=Path, required=True)
    parser.add_argument("--original-hf-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--base-receipt-sha256", required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--adapter-receipt-sha256", required=True)
    parser.add_argument("--merge-release-root", type=Path, required=True)
    parser.add_argument("--merge-completion", type=Path, required=True)
    parser.add_argument("--merge-completion-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--source-bindings", type=Path, required=True)
    parser.add_argument("--source-bindings-sha256", required=True)
    parser.add_argument("--conversion-input", type=Path, required=True)
    parser.add_argument("--conversion-input-sha256", required=True)
    parser.add_argument("--conversion-run", type=Path, required=True)
    parser.add_argument("--conversion-run-sha256", required=True)
    parser.add_argument("--conversion-completion", type=Path, required=True)
    parser.add_argument("--conversion-completion-sha256", required=True)
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
        original_hf_checkpoint=args.original_hf_checkpoint,
        original_hf_manifest_path=args.original_hf_manifest,
        original_hf_manifest_sha256=args.original_hf_manifest_sha256,
        base_checkpoint_root=args.base_checkpoint_root,
        base_receipt_path=args.base_receipt,
        base_receipt_sha256=args.base_receipt_sha256,
        adapter_receipt_path=args.adapter_receipt,
        adapter_receipt_sha256=args.adapter_receipt_sha256,
        merge_release_root=args.merge_release_root,
        merge_completion_path=args.merge_completion,
        merge_completion_sha256=args.merge_completion_sha256,
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        source_bindings_path=args.source_bindings,
        source_bindings_sha256=args.source_bindings_sha256,
        conversion_input_path=args.conversion_input,
        conversion_input_sha256=args.conversion_input_sha256,
        conversion_run_path=args.conversion_run,
        conversion_run_sha256=args.conversion_run_sha256,
        conversion_completion_path=args.conversion_completion,
        conversion_completion_sha256=args.conversion_completion_sha256,
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
