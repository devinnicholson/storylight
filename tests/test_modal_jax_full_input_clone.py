# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import modal_jax_full_input_clone as worker
from infra.gcp.jax.full_input_clone import (
    approval_token,
    build_full_training_manifest,
    canonical_bytes,
    manifest_sha256,
)
from scripts.plan_modal_jax_full_input_clone import build_plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, document: object, *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(canonical_bytes(document))
    else:
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _population(tmp_path: Path) -> tuple[Path, Path, str, str, dict[str, object]]:
    source_run_id = "jax-roundtrip-20260902-source"
    target_run_id = "jax-full-20260902-target"
    inputs = tmp_path / "inputs"
    releases = tmp_path / "releases/roundtrip"
    source_input = inputs / source_run_id
    source_release = releases / source_run_id
    files = {
        "config.json": b"{\"config\":true}\n",
        "dataset/manifest.json": b"{\"splits\":{}}\n",
        "dataset/train.jsonl": b"{\"split\":\"train\"}\n",
        "dataset/development.jsonl": b"{\"split\":\"development\"}\n",
        "prepared/train.jsonl": b"{\"prepared\":true}\n",
        "tokenizer.manifest.json": b"{\"tokenizer\":true}\n",
        "tokenizer/tokenizer.model": b"tokenizer",
    }
    rows = []
    for relative, payload in files.items():
        path = source_input / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(_row(path, relative))
    source_manifest = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": source_run_id,
        "status": "complete",
        "files": sorted(rows, key=lambda row: row["path"]),
    }
    source_manifest_path = source_input / "inputs.manifest.json"
    _write_json(source_manifest_path, source_manifest)

    base_leaf = source_release / "base-orbax/0/items"
    (base_leaf / "nested").mkdir(parents=True)
    (base_leaf / "weights.bin").write_bytes(b"weights")
    (base_leaf / "nested/state.bin").write_bytes(b"state")
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
    completion = {
        "schema_version": "1.0",
        "run_id": source_run_id,
        "status": "succeeded",
        "backend": "modal-l4x2",
        "input_manifest_sha256": _sha256(source_manifest_path),
        "config_sha256": next(row["sha256"] for row in rows if row["path"] == "config.json"),
        "dataset_manifest_sha256": next(
            row["sha256"] for row in rows if row["path"] == "dataset/manifest.json"
        ),
        "tokenizer_manifest_sha256": next(
            row["sha256"] for row in rows if row["path"] == "tokenizer.manifest.json"
        ),
        "base_orbax_receipt_sha256": _sha256(base_receipt_path),
        "base_orbax_manifest_sha256": _sha256(base_manifest_path),
        "files": release_rows,
    }
    completion_path = source_release / "completion.json"
    _write_json(completion_path, completion, canonical=False)
    target_manifest, _ = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=_sha256(source_manifest_path),
        source_input_manifest=source_manifest,
        target_run_id=target_run_id,
        roundtrip_completion_sha256=_sha256(completion_path),
        roundtrip_completion=completion,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
    )
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": _sha256(source_manifest_path),
        "roundtrip_completion_sha256": _sha256(completion_path),
        "target_run_id": target_run_id,
        "target_manifest_sha256": manifest_sha256(target_manifest),
    }
    request["approval_token"] = approval_token(
        source_run_id=source_run_id,
        source_input_manifest_sha256=str(request["source_input_manifest_sha256"]),
        roundtrip_completion_sha256=str(request["roundtrip_completion_sha256"]),
        target_run_id=target_run_id,
        target_manifest_sha256=str(request["target_manifest_sha256"]),
    )
    return inputs, releases, source_run_id, target_run_id, request


def test_clone_builds_complete_full_training_population_manifest_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, releases, _, target_run_id, request = _population(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(worker, "_RELEASE_ROOT", releases)

    result = worker.clone_population(request)

    target = inputs / target_run_id
    manifest_path = target / "inputs.manifest.json"
    document = json.loads(manifest_path.read_text())
    assert result["status"] == "staged"
    assert result["manifest_uploaded_last"] is True
    assert _sha256(manifest_path) == request["target_manifest_sha256"]
    assert document["base_orbax"]["relative_path"] == "0/items"
    assert document["derivation"]["source_run_id"] == result["source_run_id"]
    assert (target / "checkpoint/0/items/weights.bin").read_bytes() == b"weights"
    assert (target / "dataset/development.jsonl").is_file()
    expected = {row["path"] for row in document["files"]}
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert actual == expected


def test_clone_rejects_inexact_approval_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs, releases, _, target_run_id, request = _population(tmp_path)
    monkeypatch.setattr(worker, "_INPUT_ROOT", inputs)
    monkeypatch.setattr(worker, "_RELEASE_ROOT", releases)
    request["approval_token"] = "approve"

    with pytest.raises(ValueError, match="approval token"):
        worker.clone_population(request)
    assert not (inputs / target_run_id).exists()


def test_plan_derives_exact_approval_without_remote_mutation(tmp_path: Path) -> None:
    inputs, releases, source_run_id, target_run_id, request = _population(tmp_path)

    plan = build_plan(
        source_input_manifest_path=inputs / source_run_id / "inputs.manifest.json",
        roundtrip_release=releases / source_run_id,
        target_run_id=target_run_id,
    )

    assert plan["status"] == "plan-only"
    assert plan["remote_mutation"] is False
    assert plan["target_manifest_sha256"] == request["target_manifest_sha256"]
    assert plan["approval_token"] == request["approval_token"]


def test_manifest_builder_rejects_hidden_or_legacy_roundtrip_data(tmp_path: Path) -> None:
    inputs, releases, source_run_id, target_run_id, _ = _population(tmp_path)
    source_manifest = json.loads(
        (inputs / source_run_id / "inputs.manifest.json").read_text()
    )
    completion = json.loads((releases / source_run_id / "completion.json").read_text())
    base_receipt = json.loads(
        (releases / source_run_id / "evidence/base-orbax.receipt.json").read_text()
    )
    base_manifest = json.loads(
        (releases / source_run_id / "base-orbax.manifest.json").read_text()
    )
    source_manifest["files"].append(
        {"path": "dataset/hidden.jsonl", "bytes": 1, "sha256": "0" * 64}
    )
    completion["input_manifest_sha256"] = manifest_sha256(source_manifest)
    with pytest.raises(ValueError, match="hidden data"):
        build_full_training_manifest(
            source_run_id=source_run_id,
            source_input_manifest_sha256=manifest_sha256(source_manifest),
            source_input_manifest=source_manifest,
            target_run_id=target_run_id,
            roundtrip_completion_sha256="1" * 64,
            roundtrip_completion=completion,
            base_receipt=base_receipt,
            base_manifest=base_manifest,
        )
    source_manifest["files"].pop()
    completion["input_manifest_sha256"] = manifest_sha256(source_manifest)
    completion["backend"] = "modal-l40s"
    with pytest.raises(ValueError, match="round-trip release identity"):
        build_full_training_manifest(
            source_run_id=source_run_id,
            source_input_manifest_sha256=manifest_sha256(source_manifest),
            source_input_manifest=source_manifest,
            target_run_id=target_run_id,
            roundtrip_completion_sha256="1" * 64,
            roundtrip_completion=completion,
            base_receipt=base_receipt,
            base_manifest=base_manifest,
        )
