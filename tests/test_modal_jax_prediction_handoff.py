# ruff: noqa: E402
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "BOOKFORGE_NVIDIA_PYTORCH_IMAGE",
    "nvcr.io/nvidia/pytorch:26.01-py3@sha256:" + "1" * 64,
)

from deploy import modal_jax_prediction as prediction
from deploy import modal_jax_prediction_handoff as worker
from infra.gcp.jax import stage_modal_prediction_inputs as materialized_stager
from infra.gcp.jax.prediction_handoff import (
    approval_token,
    canonical_bytes,
    manifest_sha256,
    prediction_approval_token,
    sha256_file,
)
from scripts.plan_modal_jax_prediction_handoff import build_plan
from training.jax_fidelity.integrity import artifact_manifest
from training.jax_fidelity.merged_candidate import build_merged_candidate_manifest

CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
DATASET_MANIFEST = ROOT / "datasets/story-fidelity-v2/manifest.json"
DEVELOPMENT = ROOT / "datasets/story-fidelity-v2/development.jsonl"
LEGACY_CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"
LEGACY_DATASET_MANIFEST = ROOT / "datasets/story-fidelity-v1/manifest.json"
LEGACY_DEVELOPMENT = ROOT / "datasets/story-fidelity-v1/development.jsonl"


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))


def _row(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _population(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source_run_id = "jax-full-20260902-l4x2-v2"
    merge_run_id = "jax-merge-20260902-candidate"
    prediction_run_id = "jax-prediction-20260902-final"
    inputs = tmp_path / "inputs"
    releases = tmp_path / "releases/merged"
    source = inputs / source_run_id
    source_files = {
        "config.json": CONFIG,
        "dataset/manifest.json": DATASET_MANIFEST,
        "dataset/development.jsonl": DEVELOPMENT,
    }
    source_rows = []
    for relative, original in source_files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        source_rows.append(_row(target, relative))
    source_manifest = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "purpose": "hf-maxtext-roundtrip-smoke",
        "run_id": source_run_id,
        "status": "complete",
        "files": sorted(source_rows, key=lambda row: str(row["path"])),
    }
    source_manifest_path = source / "inputs.manifest.json"
    _write_json(source_manifest_path, source_manifest)

    release = releases / merge_run_id
    checkpoint = release / "merged-hf"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_bytes(b"{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"merged weights")
    checkpoint_manifest = artifact_manifest(checkpoint)
    checkpoint_manifest_path = release / "merged-hf.manifest.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    candidate = build_merged_candidate_manifest(
        config_path=CONFIG,
        dataset_manifest_sha256=sha256_file(DATASET_MANIFEST),
        training_run_id="lora-train-aaaaaaaaaaaaaaaaaaaa",
        merged_hf_checkpoint=checkpoint,
    )
    candidate_path = release / "candidate.manifest.json"
    _write_json(candidate_path, candidate)
    release_rows = [
        _row(path, path.relative_to(release).as_posix())
        for path in sorted(item for item in release.rglob("*") if item.is_file())
    ]
    completion = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l4",
        "release_type": "provisional-merged-hf-development-candidate",
        "merge_run_id": merge_run_id,
        "roundtrip_run_id": "jax-roundtrip-20260902-source",
        "training_input_run_id": source_run_id,
        "training_input_manifest_sha256": sha256_file(source_manifest_path),
        "config_sha256": sha256_file(CONFIG),
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
        "candidate_id": candidate["candidate_id"],
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "merged_hf_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": manifest_sha256(checkpoint_manifest),
        "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
        "development_evaluated": False,
        "release_authorized": False,
        "files": release_rows,
    }
    completion_path = release / "completion.json"
    _write_json(completion_path, completion)
    plan = build_plan(
        source_input_manifest_path=source_manifest_path,
        merge_completion_path=completion_path,
        candidate_manifest_path=candidate_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        dataset_manifest_path=DATASET_MANIFEST,
        development_records_path=DEVELOPMENT,
        prediction_run_id=prediction_run_id,
    )
    request = {
        name: plan[name]
        for name in (
            "source_run_id",
            "source_input_manifest_sha256",
            "merge_run_id",
            "merge_completion_sha256",
            "prediction_run_id",
            "candidate_id",
            "candidate_manifest_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_content_sha256",
            "target_manifest_sha256",
        )
    }
    request["approval_token"] = plan["approval_token"]
    return inputs, releases, request


def test_plan_is_exact_and_references_existing_modal_bytes(tmp_path: Path) -> None:
    inputs, releases, request = _population(tmp_path)
    source = inputs / str(request["source_run_id"])
    release = releases / str(request["merge_run_id"])
    plan = build_plan(
        source_input_manifest_path=source / "inputs.manifest.json",
        merge_completion_path=release / "completion.json",
        candidate_manifest_path=release / "candidate.manifest.json",
        checkpoint_manifest_path=release / "merged-hf.manifest.json",
        dataset_manifest_path=DATASET_MANIFEST,
        development_records_path=DEVELOPMENT,
        prediction_run_id=str(request["prediction_run_id"]),
    )

    assert plan["status"] == "plan-only"
    assert plan["remote_mutation"] is False
    assert plan["checkpoint_bytes_reuploaded"] == 0
    assert plan["approval_token"] == approval_token(
        **{name: request[name] for name in request if name != "approval_token"}
    )
    assert plan["prediction_approval_token"] == prediction._approval_token(
        run_id=str(plan["prediction_run_id"]),
        input_manifest_sha256=str(plan["target_manifest_sha256"]),
        batch_size=4,
        **{
            name: str(plan["target_manifest"]["bindings"][name])
            for name in (
                "candidate_id",
                "config_sha256",
                "dataset_manifest_sha256",
                "development_records_sha256",
                "candidate_manifest_sha256",
                "checkpoint_manifest_sha256",
                "checkpoint_content_sha256",
            )
        },
    )
    assert plan["prediction_approval_token"] == prediction_approval_token(
        run_id=str(plan["prediction_run_id"]),
        bindings=plan["target_manifest"]["bindings"],
        input_manifest_sha256=str(plan["target_manifest_sha256"]),
    )
    references = plan["target_manifest"]["references"]
    checkpoint = next(row for row in references if row["path"].endswith("model.safetensors"))
    assert checkpoint["source_volume"] == "bookforge-jax-fidelity-release"
    assert checkpoint["source_path"].startswith(f"merged/{request['merge_run_id']}/")
    assert not any("hidden" in str(row["path"]).casefold() for row in references)


def test_worker_publishes_only_manifest_and_prediction_resolves_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, releases, request = _population(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(worker, "_RELEASE_ROOT", releases)

    result = worker.stage_reference(request)

    target = inputs / "prediction" / str(request["prediction_run_id"])
    assert result["manifest_uploaded_last"] is True
    assert result["checkpoint_bytes_duplicated"] == 0
    assert {path.name for path in target.iterdir()} == {"inputs.manifest.json"}
    document = json.loads((target / "inputs.manifest.json").read_text())
    paths = prediction._resolve_prediction_inputs(
        target,
        run_id=str(request["prediction_run_id"]),
        expected_manifest_sha256=str(request["target_manifest_sha256"]),
        expected_bindings=document["bindings"],
        input_root=inputs,
        merged_release_root=releases,
    )
    assert paths["checkpoint"] == releases / str(request["merge_run_id"]) / "merged-hf"
    assert paths["development_records"] == (
        inputs / str(request["source_run_id"]) / "dataset/development.jsonl"
    )


def test_prediction_keeps_materialized_stager_compatibility(tmp_path: Path) -> None:
    _population(tmp_path)
    release = tmp_path / "legacy-release"
    checkpoint = release / "merged-hf"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_bytes(b"{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"legacy merged weights")
    candidate = build_merged_candidate_manifest(
        config_path=LEGACY_CONFIG,
        dataset_manifest_sha256=sha256_file(LEGACY_DATASET_MANIFEST),
        training_run_id="lora-train-bbbbbbbbbbbbbbbbbbbb",
        merged_hf_checkpoint=checkpoint,
    )
    _write_json(release / "candidate.manifest.json", candidate)
    run_id = "jax-prediction-20260902-legacy"
    manifest, sources = materialized_stager.build_prediction_input_manifest(
        run_id=run_id,
        config_path=LEGACY_CONFIG,
        dataset_manifest_path=LEGACY_DATASET_MANIFEST,
        development_records_path=LEGACY_DEVELOPMENT,
        candidate_directory=release,
    )
    root = tmp_path / "materialized"
    for relative, source in sources.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    _write_json(root / "inputs.manifest.json", manifest)

    paths = prediction._resolve_prediction_inputs(
        root,
        run_id=run_id,
        expected_manifest_sha256=manifest_sha256(manifest),
        expected_bindings=manifest["bindings"],
    )

    assert paths["checkpoint"] == root / "candidate/merged-hf"
    assert paths["development_records"] == root / "dataset/development.jsonl"


def test_handoff_rejects_inexact_approval_legacy_backend_and_hidden_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, releases, request = _population(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(worker, "_RELEASE_ROOT", releases)
    with pytest.raises(ValueError, match="approval token"):
        worker.stage_reference({**request, "approval_token": "approve"})

    release = releases / str(request["merge_run_id"])
    completion_path = release / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["backend"] = "modal-l40s"
    _write_json(completion_path, completion)
    legacy_request = dict(request)
    legacy_request["merge_completion_sha256"] = sha256_file(completion_path)
    legacy_request["approval_token"] = approval_token(
        **{name: legacy_request[name] for name in request if name != "approval_token"}
    )
    with pytest.raises(ValueError, match="backend"):
        worker.stage_reference(legacy_request)

    completion["backend"] = "modal-l4"
    _write_json(completion_path, completion)
    hidden = inputs / str(request["source_run_id"]) / "dataset/hidden.jsonl"
    hidden.write_text("{}\n")
    request["merge_completion_sha256"] = sha256_file(completion_path)
    request["approval_token"] = approval_token(
        **{name: request[name] for name in request if name != "approval_token"}
    )
    with pytest.raises(RuntimeError, match="hidden data"):
        worker.stage_reference(request)
    hidden.unlink()
    symlink = release / "unsafe-link"
    symlink.symlink_to(release / "candidate.manifest.json")
    with pytest.raises(RuntimeError, match="symbolic link"):
        worker.stage_reference(request)
