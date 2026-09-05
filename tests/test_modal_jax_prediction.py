# ruff: noqa: E402
from __future__ import annotations

import hashlib
import importlib.util
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

from infra.gcp.jax import stage_modal_prediction_inputs as stager
from scripts import fetch_modal_jax_predictions as fetcher
from training.jax_fidelity.integrity import canonical_json_bytes, sha256_file
from training.jax_fidelity.merged_candidate import build_merged_candidate_manifest
from training.jax_fidelity.release import candidate_id_for_checkpoint, candidate_id_from_lineage


def _load_worker():
    path = ROOT / "deploy/modal_jax_prediction.py"
    spec = importlib.util.spec_from_file_location("modal_jax_prediction", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _release(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "release"
    checkpoint = root / "merged-hf"
    checkpoint.mkdir(parents=True)
    for relative, content in {
        "config.json": b"{}\n",
        "model.safetensors": b"merged-weights",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
    }.items():
        (checkpoint / relative).write_bytes(content)
    document = build_merged_candidate_manifest(
        config_path=ROOT / "experiments/jax-fidelity-lab/config.json",
        dataset_manifest_sha256=sha256_file(stager.PUBLIC_MANIFEST),
        training_run_id="lora-train-fidelity-001",
        merged_hf_checkpoint=checkpoint,
    )
    (root / "candidate.manifest.json").write_bytes(canonical_json_bytes(document))
    return root, document


def _request(manifest: dict[str, object], **updates: object) -> dict[str, object]:
    bindings = dict(manifest["bindings"])
    request: dict[str, object] = {
        "run_id": manifest["run_id"],
        **bindings,
        "input_manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "batch_size": 4,
    }
    request["approval_token"] = worker._approval_token(
        run_id=str(request["run_id"]),
        candidate_id=str(request["candidate_id"]),
        config_sha256=str(request["config_sha256"]),
        dataset_manifest_sha256=str(request["dataset_manifest_sha256"]),
        development_records_sha256=str(request["development_records_sha256"]),
        candidate_manifest_sha256=str(request["candidate_manifest_sha256"]),
        checkpoint_manifest_sha256=str(request["checkpoint_manifest_sha256"]),
        checkpoint_content_sha256=str(request["checkpoint_content_sha256"]),
        input_manifest_sha256=str(request["input_manifest_sha256"]),
        batch_size=4,
    )
    request.update(updates)
    return request


def test_prediction_stager_accepts_only_committed_public_development_and_merged_release(
    tmp_path: Path,
) -> None:
    release, release_document = _release(tmp_path)
    manifest, sources = stager.build_prediction_input_manifest(
        run_id="prediction-fidelity-20260901",
        config_path=ROOT / "experiments/jax-fidelity-lab/config.json",
        dataset_manifest_path=stager.PUBLIC_MANIFEST,
        development_records_path=stager.PUBLIC_DEVELOPMENT,
        candidate_directory=release,
    )

    assert manifest["privacy"] == {
        "split": "development",
        "public_records_only": True,
        "hidden_records_included": False,
    }
    assert manifest["bindings"]["candidate_id"] == release_document["candidate_id"]
    assert manifest["bindings"]["candidate_manifest_sha256"] == sha256_file(
        release / "candidate.manifest.json"
    )
    assert len(str(manifest["bindings"]["checkpoint_manifest_sha256"])) == 64
    assert "dataset/development.jsonl" in sources
    assert "dataset/manifest.json" in sources
    assert not any("hidden" in Path(relative).name for relative in sources)
    assert not any("train.jsonl" in relative for relative in sources)
    expected_candidate = candidate_id_for_checkpoint(
        config_path=ROOT / "experiments/jax-fidelity-lab/config.json",
        dataset_manifest_sha256=sha256_file(stager.PUBLIC_MANIFEST),
        training_run_id="lora-train-fidelity-001",
        merged_hf_checkpoint=release / "merged-hf",
    )
    assert release_document["candidate_id"] == expected_candidate
    final_release_candidate = candidate_id_from_lineage(
        config_sha256=str(release_document["config_sha256"]),
        dataset_manifest_sha256=str(release_document["dataset_manifest_sha256"]),
        training_run_id=str(release_document["training_run_id"]),
        base_model=release_document["base_model"],
        files=release_document["checkpoint_manifest"]["files"],
    )
    assert final_release_candidate == expected_candidate
    assert release_document["eligibility"] == {
        "development_evaluated": False,
        "hidden_evaluated": False,
        "release_authorized": False,
    }
    assert "terminal_evidence" not in release_document


def test_prediction_stager_rejects_noncanonical_or_exposed_hidden_dataset(tmp_path: Path) -> None:
    copied = tmp_path / "development.jsonl"
    shutil.copyfile(stager.PUBLIC_DEVELOPMENT, copied)
    with pytest.raises(ValueError, match="repository public development"):
        stager._require_committed_public_sources(stager.PUBLIC_MANIFEST, copied)

    manifest = json.loads(stager.PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    manifest["splits"]["hidden"]["path"] = "hidden.jsonl"
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="external and hash-only"):
        stager.validate_public_dataset(altered, stager.PUBLIC_DEVELOPMENT)


def test_prediction_stage_is_write_once_and_uploads_manifest_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release, _ = _release(tmp_path)
    run_id = "prediction-fidelity-20260901"
    manifest, sources = stager.build_prediction_input_manifest(
        run_id=run_id,
        config_path=ROOT / "experiments/jax-fidelity-lab/config.json",
        dataset_manifest_path=stager.PUBLIC_MANIFEST,
        development_records_path=stager.PUBLIC_DEVELOPMENT,
        candidate_directory=release,
    )
    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    monkeypatch.setenv(
        stager.APPROVAL_ENVIRONMENT,
        stager.approval_token(run_id, manifest_sha),
    )
    commands: list[list[str]] = []
    uploaded_manifest: bytes | None = None

    def runner(command: list[str], **_: object):
        nonlocal uploaded_manifest
        commands.append(command)
        if command[2:4] == ["ls", stager.VOLUME_NAME]:
            return subprocess_result(command, stdout="[]")
        if command[2:4] == ["put", stager.VOLUME_NAME]:
            if command[-1].endswith("/inputs.manifest.json"):
                uploaded_manifest = Path(command[-2]).read_bytes()
            return subprocess_result(command)
        assert command[2:4] == ["get", stager.VOLUME_NAME]
        assert uploaded_manifest is not None
        Path(command[-1]).write_bytes(uploaded_manifest)
        return subprocess_result(command)

    result = stager.stage_prediction_inputs(
        run_id=run_id,
        manifest=manifest,
        sources=sources,
        runner=runner,
    )

    put_commands = [command for command in commands if command[2] == "put"]
    assert put_commands[-1][-1].endswith("/inputs.manifest.json")
    assert result["manifest_uploaded_last"] is True
    assert result["hidden_records_uploaded"] is False


def subprocess_result(command: list[str], stdout: str = ""):
    return __import__("subprocess").CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_prediction_request_rejects_batch_approval_and_candidate_hash_drift(tmp_path: Path) -> None:
    release, _ = _release(tmp_path)
    manifest, _ = stager.build_prediction_input_manifest(
        run_id="prediction-fidelity-20260901",
        config_path=ROOT / "experiments/jax-fidelity-lab/config.json",
        dataset_manifest_path=stager.PUBLIC_MANIFEST,
        development_records_path=stager.PUBLIC_DEVELOPMENT,
        candidate_directory=release,
    )
    request = _request(manifest)
    validated = worker._validate_request(request)
    assert validated[0] == manifest["run_id"]
    assert validated[-1] == 4
    with pytest.raises(ValueError, match="batch_size"):
        worker._validate_request({**request, "batch_size": 8})
    with pytest.raises(ValueError, match="approval"):
        worker._validate_request({**request, "approval_token": "approve"})
    with pytest.raises(ValueError, match="SHA-256"):
        worker._validate_request({**request, "candidate_manifest_sha256": "latest"})


def test_prediction_fetch_requires_completion_hash_and_verifies_public_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "prediction-fidelity-20260901"
    predictions = b"".join(
        canonical_json_bytes({"record_id": f"development-{index:04d}", "raw": "SETTING: x"})
        for index in range(512)
    )
    bindings = {
        "candidate_id": "fidelity-" + "a" * 20,
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "development_records_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64,
        "checkpoint_manifest_sha256": "e" * 64,
        "checkpoint_content_sha256": "f" * 64,
    }
    intent = canonical_json_bytes(
        {
            "schema_version": "1.0",
            "status": "prediction-intent-recorded",
            "retry_allowed": False,
            "run_id": run_id,
            "candidate_id": bindings["candidate_id"],
            "input_manifest_sha256": "9" * 64,
            "bindings": bindings,
            "batch_size": 4,
        }
    )
    files = [
        {
            "path": "intent.json",
            "bytes": len(intent),
            "sha256": hashlib.sha256(intent).hexdigest(),
        },
        {
            "path": "predictions.jsonl",
            "bytes": len(predictions),
            "sha256": hashlib.sha256(predictions).hexdigest(),
        },
    ]
    completion = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l4-cuda",
        "run_id": run_id,
        **bindings,
        "input_manifest_sha256": "9" * 64,
        "batch_size": 4,
        "split": "development",
        "predictions": 512,
        "hidden_evaluated": False,
        "passages_retained": False,
        "files": files,
    }
    completion_bytes = canonical_json_bytes(completion)
    completion_sha = hashlib.sha256(completion_bytes).hexdigest()

    def fake_get(remote: str, destination: Path) -> None:
        if remote.endswith("/completion.json"):
            destination.write_bytes(completion_bytes)
            return
        payload = destination / run_id
        payload.mkdir(parents=True)
        (payload / "intent.json").write_bytes(intent)
        (payload / "predictions.jsonl").write_bytes(predictions)
        (payload / "completion.json").write_bytes(completion_bytes)

    monkeypatch.setattr(fetcher, "_modal_get", fake_get)
    destination = tmp_path / "predictions"
    receipt = fetcher.fetch_predictions(
        run_id=run_id,
        expected_completion_sha256=completion_sha,
        destination=destination,
    )
    assert receipt["status"] == "fetched-and-verified"
    assert receipt["hidden_evaluated"] is False
    assert (destination / "predictions.jsonl").read_bytes() == predictions

    legacy_completion = dict(completion)
    legacy_completion["backend"] = "modal-l40s-cuda"
    legacy_path = tmp_path / "legacy-completion.json"
    legacy_path.write_bytes(canonical_json_bytes(legacy_completion))
    with pytest.raises(ValueError, match="identity"):
        fetcher._completion(
            legacy_path,
            run_id=run_id,
            expected_sha256=sha256_file(legacy_path),
        )

    with pytest.raises(ValueError, match="completion checksum"):
        fetcher.fetch_predictions(
            run_id=run_id,
            expected_completion_sha256="0" * 64,
            destination=tmp_path / "rejected",
        )
