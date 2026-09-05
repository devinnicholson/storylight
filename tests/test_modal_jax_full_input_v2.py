# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import modal_jax_full_input_v2 as worker
from infra.gcp.jax.full_input_v2 import (
    OVERLAY_PATHS,
    OVERLAY_PREFIX,
    build_full_training_manifest,
    canonical_bytes,
    clone_approval_token,
    manifest_sha256,
)
from infra.gcp.jax.stage_modal_v2_inputs import build_stage_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": "", "bytes": len(payload), "sha256": _sha(path)}


def _write_json(path: Path, document: object, *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(canonical_bytes(document))
    else:
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _row(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)}


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_run_id = "jax-roundtrip-20260902-source"
    target_run_id = "jax-full-20260902-l4x2-v2"
    inputs = tmp_path / "inputs"
    releases = tmp_path / "releases/roundtrip"
    source_input = inputs / source_run_id
    source_release = releases / source_run_id

    source_payloads = {
        "config.json": b"old-config\n",
        "dataset/manifest.json": b"old-dataset\n",
        "dataset/train.jsonl": b"old-train\n",
        "dataset/development.jsonl": b"old-development\n",
        "prepared/train.jsonl": b"old-prepared\n",
        "tokenizer.manifest.json": b"cached-tokenizer-manifest\n",
        "tokenizer/tokenizer.json": b"cached-tokenizer-bytes\n",
    }
    source_rows = []
    for relative, payload in source_payloads.items():
        path = source_input / relative
        _write(path, payload)
        source_rows.append(_row(path, relative))
    source_manifest = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": source_run_id,
        "status": "complete",
        "files": sorted(source_rows, key=lambda row: row["path"]),
    }
    source_manifest_path = source_input / "inputs.manifest.json"
    _write_json(source_manifest_path, source_manifest)

    base_leaf = source_release / "base-orbax/0/items"
    _write(base_leaf / "weights.bin", b"cached-base-weights")
    _write(base_leaf / "nested/state.bin", b"cached-base-state")
    base_rows = [
        _row(path, path.relative_to(base_leaf).as_posix())
        for path in sorted(item for item in base_leaf.rglob("*") if item.is_file())
    ]
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
    base_receipt_path = source_release / "evidence/base-orbax.receipt.json"
    _write_json(base_manifest_path, base_manifest)
    _write_json(base_receipt_path, base_receipt)
    release_rows = [
        _row(path, path.relative_to(source_release).as_posix())
        for path in sorted(item for item in source_release.rglob("*") if item.is_file())
    ]
    source_by_path = {row["path"]: row for row in source_rows}
    completion = {
        "schema_version": "1.0",
        "run_id": source_run_id,
        "status": "succeeded",
        "backend": "modal-l4x2",
        "input_manifest_sha256": _sha(source_manifest_path),
        "config_sha256": source_by_path["config.json"]["sha256"],
        "dataset_manifest_sha256": source_by_path["dataset/manifest.json"]["sha256"],
        "tokenizer_manifest_sha256": source_by_path["tokenizer.manifest.json"]["sha256"],
        "base_orbax_receipt_sha256": _sha(base_receipt_path),
        "base_orbax_manifest_sha256": _sha(base_manifest_path),
        "files": release_rows,
    }
    completion_path = source_release / "completion.json"
    _write_json(completion_path, completion, canonical=False)

    local = tmp_path / "local-v2"
    prompt_sha = "a" * 64
    config = {
        "schema_version": "1.0",
        "experiment_id": "bookforge-gemma4-e2b-lora-r16-v2",
        "production_contract": {"prompt_contract_sha256": prompt_sha},
        "dataset": {"manifest_path": "datasets/story-fidelity-v2/manifest.json"},
        "training": {"preparation_policy": "balanced-counterfactual-pairs-v1"},
    }
    _write_json(local / "config.json", config)
    _write(local / "dataset/train.jsonl", b'{"split":"train"}\n')
    _write(local / "dataset/development.jsonl", b'{"split":"development"}\n')
    dataset = {
        "dataset_id": "story-fidelity-v2",
        "splits": {
            "train": {
                "path": "train.jsonl",
                "public": True,
                "sha256": _sha(local / "dataset/train.jsonl"),
            },
            "development": {
                "path": "development.jsonl",
                "public": True,
                "sha256": _sha(local / "dataset/development.jsonl"),
            },
            "hidden": {"path": None, "public": False, "sha256": "b" * 64},
        },
    }
    _write_json(local / "dataset/manifest.json", dataset)
    _write(local / "prepared/train.jsonl", b'{"messages":[]}\n')
    preparation = {
        "schema_version": "bookforge-jax-training-preparation-v2",
        "policy": "balanced-counterfactual-pairs-v1",
        "source_train_sha256": _sha(local / "dataset/train.jsonl"),
        "prepared_sha256": _sha(local / "prepared/train.jsonl"),
        "prompt_contract_sha256": prompt_sha,
        "prepared_records": 320,
        "assistant_turns_per_record": 1,
        "pair_adjacency_preserved": True,
    }
    preparation_path = local / "prepared/preparation.manifest.json"
    _write_json(preparation_path, preparation)
    validation = {
        "schema_version": "bookforge-jax-prepared-validation-v1",
        "status": "passed",
        "config_sha256": _sha(local / "config.json"),
        "prepared_sha256": _sha(local / "prepared/train.jsonl"),
        "preparation_manifest_sha256": _sha(preparation_path),
        "tokenizer_manifest_sha256": source_by_path["tokenizer.manifest.json"]["sha256"],
        "prompt_contract_sha256": prompt_sha,
        "records": 320,
        "assistant_turns_per_record": 1,
    }
    _write_json(local / "prepared/prepared-validation.json", validation)
    sources = {relative: local / relative for relative in OVERLAY_PATHS}
    overlay_manifest, normalized, stage_plan = build_stage_plan(
        target_run_id=target_run_id,
        sources=sources,
        source_tokenizer_manifest_sha256=str(
            source_by_path["tokenizer.manifest.json"]["sha256"]
        ),
    )
    overlay_root = inputs / OVERLAY_PREFIX / target_run_id
    for relative, source in normalized.items():
        destination = overlay_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    overlay_manifest_path = overlay_root / "staging.manifest.json"
    _write_json(overlay_manifest_path, overlay_manifest)
    target_manifest, _, _ = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=_sha(source_manifest_path),
        source_input_manifest=source_manifest,
        roundtrip_completion_sha256=_sha(completion_path),
        roundtrip_completion=completion,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
        overlay_manifest_sha256=_sha(overlay_manifest_path),
        overlay_manifest=overlay_manifest,
        target_run_id=target_run_id,
    )
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": _sha(source_manifest_path),
        "roundtrip_completion_sha256": _sha(completion_path),
        "overlay_manifest_sha256": _sha(overlay_manifest_path),
        "target_run_id": target_run_id,
        "target_manifest_sha256": manifest_sha256(target_manifest),
    }
    request["approval_token"] = clone_approval_token(**request)
    return {
        "inputs": inputs,
        "releases": releases,
        "source_run_id": source_run_id,
        "target_run_id": target_run_id,
        "source_manifest_path": source_manifest_path,
        "source_release": source_release,
        "overlay_manifest_path": overlay_manifest_path,
        "request": request,
        "stage_plan": stage_plan,
        "stage_sources": normalized,
        "overlay_manifest": overlay_manifest,
    }


def test_clone_uses_cached_model_and_v2_files_with_manifest_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", fixture["inputs"])
    monkeypatch.setattr(worker, "_RELEASE_ROOT", fixture["releases"])
    observed: dict[str, set[str]] = {}
    original_write = worker._write_manifest_last

    def observe(path: Path, document: dict[str, object]) -> None:
        observed["before_manifest"] = {
            item.relative_to(path.parent).as_posix()
            for item in path.parent.rglob("*")
            if item.is_file()
        }
        original_write(path, document)

    monkeypatch.setattr(worker, "_write_manifest_last", observe)
    result = worker.clone_population(dict(fixture["request"]))

    target = fixture["inputs"] / fixture["target_run_id"]
    manifest_path = target / "inputs.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert result["cached_model_upload_bytes"] == 0
    assert manifest["producer"] == "bookforge-gcp-jax-input-stager"
    assert set(manifest) == {
        "schema_version",
        "producer",
        "run_id",
        "status",
        "files",
        "base_orbax",
        "prepared_training",
    }
    assert set(manifest["prepared_training"]) == {
        "policy",
        "preparation_manifest_sha256",
        "prepared_sha256",
        "prepared_validation_sha256",
        "prompt_contract_sha256",
        "records",
        "source_train_sha256",
        "tokenizer_manifest_sha256",
    }
    assert manifest["prepared_training"]["records"] == 320
    assert "derivation" not in manifest
    assert result["source_run_id"] == fixture["source_run_id"]
    assert result["source_input_manifest_sha256"] == fixture["request"][
        "source_input_manifest_sha256"
    ]
    assert result["roundtrip_completion_sha256"] == fixture["request"][
        "roundtrip_completion_sha256"
    ]
    assert result["overlay_manifest_sha256"] == fixture["request"][
        "overlay_manifest_sha256"
    ]
    assert (target / "config.json").read_bytes() != b"old-config\n"
    assert (target / "tokenizer/tokenizer.json").read_bytes() == b"cached-tokenizer-bytes\n"
    assert (target / "checkpoint/0/items/weights.bin").read_bytes() == b"cached-base-weights"
    assert "inputs.manifest.json" not in observed["before_manifest"]
    assert observed["before_manifest"] == {row["path"] for row in manifest["files"]}


def test_clone_rejects_inexact_approval_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", fixture["inputs"])
    monkeypatch.setattr(worker, "_RELEASE_ROOT", fixture["releases"])
    request = dict(fixture["request"])
    request["approval_token"] = "approve"

    with pytest.raises(ValueError, match="approval token"):
        worker.clone_population(request)
    assert not (fixture["inputs"] / fixture["target_run_id"]).exists()


def test_overlay_rejects_hidden_file_and_tokenizer_lineage_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    hidden_sources = dict(fixture["stage_sources"])
    hidden = tmp_path / "hidden.jsonl"
    hidden.write_text("{}\n")
    hidden_sources["dataset/hidden.jsonl"] = hidden
    with pytest.raises(ValueError, match="approved relative paths"):
        build_stage_plan(
            target_run_id=str(fixture["target_run_id"]),
            sources=hidden_sources,
            source_tokenizer_manifest_sha256="c" * 64,
        )

    source_manifest = json.loads(fixture["source_manifest_path"].read_text())
    completion = json.loads((fixture["source_release"] / "completion.json").read_text())
    receipt = json.loads(
        (fixture["source_release"] / "evidence/base-orbax.receipt.json").read_text()
    )
    base_manifest = json.loads(
        (fixture["source_release"] / "base-orbax.manifest.json").read_text()
    )
    overlay = dict(fixture["overlay_manifest"])
    overlay["bindings"] = dict(overlay["bindings"])
    overlay["bindings"]["tokenizer_manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="cached tokenizer"):
        build_full_training_manifest(
            source_run_id=str(fixture["source_run_id"]),
            source_input_manifest_sha256=_sha(fixture["source_manifest_path"]),
            source_input_manifest=source_manifest,
            roundtrip_completion_sha256=_sha(
                fixture["source_release"] / "completion.json"
            ),
            roundtrip_completion=completion,
            base_receipt=receipt,
            base_manifest=base_manifest,
            overlay_manifest_sha256=manifest_sha256(overlay),
            overlay_manifest=overlay,
            target_run_id=str(fixture["target_run_id"]),
        )
