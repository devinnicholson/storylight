# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
JAX_INFRA = ROOT / "infra/gcp/jax"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JAX_INFRA))

from training.jax_fidelity.integrity import artifact_manifest, sha256_file
from training.jax_fidelity.orbax_receipt import (
    OrbaxReceiptError,
    discover_orbax_items,
    orbax_leaf_receipt,
    verify_orbax_leaf_receipt,
    write_orbax_leaf_receipt,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


roundtrip = _load("bookforge_modal_jax_roundtrip", ROOT / "deploy/modal_jax_roundtrip.py")
staging = _load("bookforge_roundtrip_staging", JAX_INFRA / "stage_roundtrip_inputs.py")
fetcher = _load("bookforge_roundtrip_fetcher", ROOT / "scripts/fetch_modal_jax_roundtrip.py")


def _request(**updates: object) -> dict[str, object]:
    run_id = "bookforge-roundtrip-smoke-20260901"
    hashes = [character * 64 for character in "abcdef"]
    request: dict[str, object] = {
        "run_id": run_id,
        "config_sha256": hashes[0],
        "dataset_manifest_sha256": hashes[1],
        "prepared_train_sha256": hashes[2],
        "input_manifest_sha256": hashes[3],
        "hf_snapshot_manifest_sha256": hashes[4],
        "tokenizer_manifest_sha256": hashes[5],
        "approval_token": roundtrip._approval_token(run_id, *hashes),
    }
    request.update(updates)
    return request


def _jsonl(path: Path, row: dict[str, object]) -> str:
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def test_orbax_leaf_discovery_records_one_exact_step(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    leaf = root / "run/checkpoints/5/items"
    leaf.mkdir(parents=True)
    (leaf / "checkpoint").write_bytes(b"adapter")

    selected = discover_orbax_items(root, expected_step=5)
    receipt = orbax_leaf_receipt(root, selected, expected_step=5, role="smoke-lora")
    receipt_path = tmp_path / "receipt.json"
    write_orbax_leaf_receipt(receipt_path, receipt)

    assert selected == leaf.resolve()
    assert receipt["relative_path"] == "run/checkpoints/5/items"
    assert receipt["artifact_manifest"] == artifact_manifest(leaf)
    assert (
        verify_orbax_leaf_receipt(root, receipt, expected_step=5, role="smoke-lora")
        == leaf.resolve()
    )
    assert json.loads(receipt_path.read_text()) == receipt


def test_orbax_leaf_discovery_rejects_ambiguity_and_tampering(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    first = root / "a/5/items"
    second = root / "b/5/items"
    for leaf in (first, second):
        leaf.mkdir(parents=True)
        (leaf / "checkpoint").write_bytes(leaf.as_posix().encode())
    with pytest.raises(OrbaxReceiptError, match="expected one"):
        discover_orbax_items(root, expected_step=5)

    (root / "b/6").mkdir(parents=True)
    second.rename(root / "b/6/items")
    selected = discover_orbax_items(root, expected_step=5)
    receipt = orbax_leaf_receipt(root, selected, expected_step=5, role="smoke-lora")
    (selected / "checkpoint").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        verify_orbax_leaf_receipt(root, receipt, expected_step=5, role="smoke-lora")


def test_roundtrip_staging_includes_manifest_public_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train = dataset / "train.jsonl"
    development = dataset / "development.jsonl"
    row = {"id": "one"}
    train_sha = _jsonl(train, row)
    development_sha = _jsonl(development, row)
    manifest = dataset / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"path": "train.jsonl", "sha256": train_sha, "records": 1},
                    "development": {
                        "path": "development.jsonl",
                        "sha256": development_sha,
                        "records": 1,
                    },
                    "hidden": {"path": None, "sha256": "9" * 64, "records": 1},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    prepared = tmp_path / "prepared.jsonl"
    config.write_text("{}\n")
    prepared.write_text("{}\n")
    snapshot = tmp_path / "snapshot"
    tokenizer = tmp_path / "tokenizer"
    snapshot.mkdir()
    tokenizer.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"model")
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    snapshot_manifest = tmp_path / "snapshot.manifest.json"
    tokenizer_manifest = tmp_path / "tokenizer.manifest.json"
    snapshot_manifest.write_text(json.dumps(artifact_manifest(snapshot)) + "\n")
    tokenizer_manifest.write_text(json.dumps(artifact_manifest(tokenizer)) + "\n")

    document, sources = staging.build_roundtrip_input_manifest(
        run_id="bookforge-roundtrip-smoke-20260901",
        config=config,
        dataset_manifest=manifest,
        prepared_train=prepared,
        hf_snapshot=snapshot,
        hf_snapshot_manifest=snapshot_manifest,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
    )

    assert document["purpose"] == "hf-maxtext-roundtrip-smoke"
    assert "dataset/train.jsonl" in sources
    assert "dataset/development.jsonl" in sources
    assert all("hidden" not in path for path in sources)
    assert {row["path"] for row in document["files"]} == set(sources)


def test_roundtrip_request_and_modal_function_fail_closed() -> None:
    assert roundtrip._validate_request(_request())[0] == "bookforge-roundtrip-smoke-20260901"
    with pytest.raises(ValueError, match="approval"):
        roundtrip._validate_request(_request(approval_token="approve"))
    with pytest.raises(ValueError, match="SHA-256"):
        roundtrip._validate_request(_request(input_manifest_sha256="latest"))

    source = (ROOT / "deploy/modal_jax_roundtrip.py").read_text()
    plan = json.loads(
        (ROOT / "experiments/jax-fidelity-lab/modal-roundtrip-plan-2026-09.json").read_text()
    )
    assert plan["function_calls"] == 1
    assert plan["gpu"] == "L4"
    assert plan["automatic_retries"] == 0
    assert plan["timeout_seconds"] == 2700
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "@modal.web_endpoint" not in source
    assert source.index('"hf-to-maxtext"') < source.index('stage="lora-smoke"')
    assert source.index('stage="lora-smoke"') < source.index('"maxtext-to-hf"')
    assert source.index('"maxtext-to-hf"') < source.index('"logit-check"')
    assert 'f"adapter_checkpoint={smoke_leaf}"' in source
    assert '"--adapter-checkpoint"' in source
    assert source.index("_release_files(release)") < source.index(
        "_write_once(completion_path, payload)"
    )


def test_roundtrip_fetch_requires_trusted_completion_and_every_declared_byte(
    tmp_path: Path,
) -> None:
    root = tmp_path / "payload"
    root.mkdir()
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"roundtrip")
    rows = [
        {
            "path": "artifact.bin",
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    ]
    completion = root / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "backend": "modal-l40s",
                "run_id": "bookforge-roundtrip-smoke-20260901",
                "files": rows,
            },
            sort_keys=True,
        )
        + "\n"
    )
    document = fetcher._completion(
        completion,
        run_id="bookforge-roundtrip-smoke-20260901",
        expected_sha256=sha256_file(completion),
    )
    fetcher._verify_files(root, document["files"])

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        fetcher._verify_files(root, document["files"])
