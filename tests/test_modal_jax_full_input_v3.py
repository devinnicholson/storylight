# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import modal_jax_fidelity as training_worker
from deploy import modal_jax_full_input_v3 as worker
from infra.gcp.jax.full_input_v3 import (
    OVERLAY_PATHS,
    OVERLAY_PREFIX,
    build_full_training_manifest,
    canonical_bytes,
    clone_approval_token,
    manifest_sha256,
)
from infra.gcp.jax.stage_modal_v3_inputs import build_stage_plan, stage_overlay
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import canonical_json_bytes, sha256_file
from training.jax_fidelity.recovery_inputs import build_recovery_inputs
from training.jax_fidelity.recovery_staging import (
    RecoveryStagingError,
    verify_recovery_training_input,
)

V2_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
V3_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"
DATASET = ROOT / "datasets/story-fidelity-v2"


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))


def _row(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path]:
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    rejected_config = lineage / "config-v2.json"
    rejected_config.write_bytes(V2_CONFIG.read_bytes())
    training = lineage / "training-completion.json"
    rejection = lineage / "development-rejection.json"
    events = lineage / "training.tfevents"
    training_run_id = "lora-train-2cd8c181b1ba1183c17d"
    training.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "run_id": training_run_id,
            }
        )
    )
    rejection.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "bookforge-jax-development-rejection-v1",
                "status": "rejected",
                "training_run_id": training_run_id,
                "config_sha256": sha256_file(rejected_config),
                "dataset_manifest_sha256": sha256_file(DATASET / "manifest.json"),
                "checkpoint_bytes_downloaded": False,
            }
        )
    )
    events.write_bytes(b"sealed zero-gradient evidence")
    config = json.loads(V3_CONFIG.read_text())
    config["recovery"]["previous_attempt"].update(
        {
            "config_sha256": sha256_file(rejected_config),
            "training_completion_sha256": sha256_file(training),
            "development_rejection_sha256": sha256_file(rejection),
            "training_events_sha256": sha256_file(events),
        }
    )
    config_path = tmp_path / "config-v3.json"
    config_path.write_bytes(canonical_json_bytes(config))
    recovery = tmp_path / "sealed-recovery"
    build_recovery_inputs(
        config_path=config_path,
        dataset_manifest_path=DATASET / "manifest.json",
        rejected_config_path=rejected_config,
        rejected_training_completion_path=training,
        rejected_development_rejection_path=rejection,
        rejected_training_events_path=events,
        output_directory=recovery,
    )
    return config_path, recovery


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_run_id = "jax-roundtrip-20260902-source"
    target_run_id = "jax-recovery-20260902-l4x2-v3"
    inputs = tmp_path / "inputs"
    releases = tmp_path / "releases/roundtrip"
    source_input = inputs / source_run_id
    source_release = releases / source_run_id
    source_payloads = {
        "config.json": b"cached-source-config\n",
        "dataset/manifest.json": b"cached-source-dataset\n",
        "tokenizer.manifest.json": b"cached-tokenizer-manifest\n",
        "tokenizer/tokenizer.json": b"cached-tokenizer-bytes\n",
    }
    for relative, payload in source_payloads.items():
        path = source_input / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    source_rows = [_row(source_input / path, path) for path in source_payloads]
    source_manifest = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": source_run_id,
        "status": "complete",
        "files": sorted(source_rows, key=lambda row: row["path"]),
    }
    source_manifest_path = source_input / "inputs.manifest.json"
    _write_json(source_manifest_path, source_manifest)
    source_by_path = {row["path"]: row for row in source_rows}

    base_leaf = source_release / "base-orbax/0/items"
    base_leaf.mkdir(parents=True)
    (base_leaf / "weights.bin").write_bytes(b"cached-base-weights")
    base_rows = [_row(base_leaf / "weights.bin", "weights.bin")]
    base_manifest = {
        "schema_version": "1.0",
        "files": base_rows,
        "content_sha256": hashlib.sha256(canonical_bytes(base_rows)).hexdigest(),
    }
    base_receipt = {
        "schema_version": "1.0",
        "format": "maxtext-orbax-items",
        "role": "base-maxtext",
        "expected_step": 0,
        "relative_path": "0/items",
        "artifact_manifest": base_manifest,
    }
    base_manifest_path = source_release / "base-orbax.manifest.json"
    receipt_path = source_release / "evidence/base-orbax.receipt.json"
    _write_json(base_manifest_path, base_manifest)
    _write_json(receipt_path, base_receipt)
    release_rows = [
        _row(path, path.relative_to(source_release).as_posix())
        for path in sorted(item for item in source_release.rglob("*") if item.is_file())
    ]
    completion = {
        "schema_version": "1.0",
        "run_id": source_run_id,
        "status": "succeeded",
        "backend": "modal-l4x2",
        "input_manifest_sha256": sha256_file(source_manifest_path),
        "config_sha256": source_by_path["config.json"]["sha256"],
        "dataset_manifest_sha256": source_by_path["dataset/manifest.json"]["sha256"],
        "tokenizer_manifest_sha256": source_by_path["tokenizer.manifest.json"]["sha256"],
        "base_orbax_receipt_sha256": sha256_file(receipt_path),
        "base_orbax_manifest_sha256": sha256_file(base_manifest_path),
        "files": release_rows,
    }
    completion_path = source_release / "completion.json"
    _write_json(completion_path, completion)

    config_path, recovery = _recovery_fixture(tmp_path)
    sources = {
        "config.json": config_path,
        "dataset/manifest.json": DATASET / "manifest.json",
        "dataset/train.jsonl": DATASET / "train.jsonl",
        "dataset/development.jsonl": DATASET / "development.jsonl",
        "recovery/inputs.manifest.json": recovery / "inputs.manifest.json",
        "recovery/overfit-canary.source.jsonl": recovery / "overfit-canary.source.jsonl",
        "recovery/overfit-canary.train.jsonl": recovery / "overfit-canary.train.jsonl",
        "recovery/public-probe.source.jsonl": recovery / "public-probe.source.jsonl",
        "recovery/public-probe.teacher.jsonl": recovery / "public-probe.teacher.jsonl",
    }
    overlay, normalized, stage_plan = build_stage_plan(
        target_run_id=target_run_id,
        sources=sources,
        source_tokenizer_manifest_sha256=str(source_by_path["tokenizer.manifest.json"]["sha256"]),
    )
    overlay_root = inputs / OVERLAY_PREFIX / target_run_id
    for relative, source in normalized.items():
        destination = overlay_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    overlay_path = overlay_root / "staging.manifest.json"
    _write_json(overlay_path, overlay)
    target_manifest, _, _ = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=sha256_file(source_manifest_path),
        source_input_manifest=source_manifest,
        roundtrip_completion_sha256=sha256_file(completion_path),
        roundtrip_completion=completion,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
        overlay_manifest_sha256=sha256_file(overlay_path),
        overlay_manifest=overlay,
        target_run_id=target_run_id,
    )
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": sha256_file(source_manifest_path),
        "roundtrip_completion_sha256": sha256_file(completion_path),
        "overlay_manifest_sha256": sha256_file(overlay_path),
        "target_run_id": target_run_id,
        "target_manifest_sha256": manifest_sha256(target_manifest),
    }
    request["approval_token"] = clone_approval_token(**request)
    return {
        "inputs": inputs,
        "releases": releases,
        "target_run_id": target_run_id,
        "config": config_path,
        "request": request,
        "stage_plan": stage_plan,
    }


def test_v3_clone_reuses_cached_model_and_maps_only_canary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", fixture["inputs"])
    monkeypatch.setattr(worker, "_RELEASE_ROOT", fixture["releases"])
    result = worker.clone_population(dict(fixture["request"]))
    target = fixture["inputs"] / fixture["target_run_id"]
    manifest = json.loads((target / "inputs.manifest.json").read_text())

    assert result["cached_model_upload_bytes"] == 0
    assert result["public_probe_held_out"] is True
    assert (target / "prepared/train.jsonl").read_bytes() == (
        target / "recovery/overfit-canary.train.jsonl"
    ).read_bytes()
    assert (target / "prepared/train.jsonl").read_bytes() != (
        target / "recovery/public-probe.teacher.jsonl"
    ).read_bytes()
    assert (target / "tokenizer/tokenizer.json").read_bytes() == b"cached-tokenizer-bytes\n"
    assert (target / "checkpoint/0/items/weights.bin").read_bytes() == b"cached-base-weights"
    assert not any("hidden" in row["path"].split("/") for row in manifest["files"])

    experiment = load_config(fixture["config"])
    binding = verify_recovery_training_input(
        target,
        experiment=experiment,
        input_manifest=manifest,
        config_sha256=experiment.sha256,
        prepared_sha256=sha256_file(target / "prepared/train.jsonl"),
        tokenizer_manifest_sha256=manifest["prepared_training"]["tokenizer_manifest_sha256"],
    )
    assert binding == manifest["prepared_training"]
    assert (
        training_worker._verify_v3_recovery_evidence(
            target,
            experiment=experiment,
            input_manifest=manifest,
            config_sha256=experiment.sha256,
            prepared_sha256=sha256_file(target / "prepared/train.jsonl"),
            tokenizer_manifest_sha256=manifest["prepared_training"]["tokenizer_manifest_sha256"],
        )
        == manifest["prepared_training"]
    )


def test_v3_stage_plan_uploads_only_public_overlay_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan = fixture["stage_plan"]

    assert plan["remote_mutation"] is False
    assert plan["cached_model_upload_bytes"] == 0
    assert set(plan["host_upload_paths"]) == set(OVERLAY_PATHS)
    assert "prepared/train.jsonl" not in plan["host_upload_paths"]
    assert not any(
        path.startswith(("tokenizer/", "checkpoint/")) or "hidden" in path.split("/")
        for path in plan["host_upload_paths"]
    )


def test_v3_stager_publishes_manifest_last_and_verifies_readback(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    plan = fixture["stage_plan"]
    manifest = plan["overlay_manifest"]
    source_root = tmp_path / "inputs" / OVERLAY_PREFIX / fixture["target_run_id"]
    sources = {relative: source_root / relative for relative in plan["host_upload_paths"]}
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[2] == "get":
            Path(command[-1]).write_bytes(canonical_bytes(manifest))
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    result = stage_overlay(
        target_run_id=str(fixture["target_run_id"]),
        manifest=manifest,
        sources=sources,
        approval_token_value=plan["approval_token"],
        runner=runner,
    )

    puts = [command for command in commands if command[2] == "put"]
    assert result["manifest_uploaded_last"] is True
    assert puts[-1][-1].endswith("/staging.manifest.json")
    assert commands[-1][2] == "get"


def test_v3_runtime_rejects_hidden_declarations_and_probe_as_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", fixture["inputs"])
    monkeypatch.setattr(worker, "_RELEASE_ROOT", fixture["releases"])
    worker.clone_population(dict(fixture["request"]))
    target = fixture["inputs"] / fixture["target_run_id"]
    manifest = json.loads((target / "inputs.manifest.json").read_text())
    experiment = load_config(fixture["config"])

    hidden = json.loads(json.dumps(manifest))
    hidden["files"].append({"path": "dataset/hidden/data.jsonl", "bytes": 0, "sha256": "0" * 64})
    with pytest.raises(RecoveryStagingError, match="hidden bytes"):
        verify_recovery_training_input(
            target,
            experiment=experiment,
            input_manifest=hidden,
            config_sha256=experiment.sha256,
            prepared_sha256=sha256_file(target / "prepared/train.jsonl"),
            tokenizer_manifest_sha256=manifest["prepared_training"]["tokenizer_manifest_sha256"],
        )

    probe = target / "recovery/public-probe.teacher.jsonl"
    with pytest.raises(RecoveryStagingError, match="not exactly the overfit canary"):
        verify_recovery_training_input(
            target,
            experiment=experiment,
            input_manifest=manifest,
            config_sha256=experiment.sha256,
            prepared_sha256=sha256_file(probe),
            tokenizer_manifest_sha256=manifest["prepared_training"]["tokenizer_manifest_sha256"],
        )
