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

from training.jax_fidelity.integrity import sha256_file
from training.jax_fidelity.orbax_receipt import (
    OrbaxReceiptError,
    discover_orbax_items,
    orbax_leaf_receipt,
    verify_orbax_leaf_receipt,
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
cloner = _load(
    "bookforge_roundtrip_cloner", ROOT / "scripts/clone_modal_jax_roundtrip_inputs.py"
)


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


def test_roundtrip_request_requires_exact_approval_and_complete_cache_binding() -> None:
    assert roundtrip._validate_request(_request())[0] == "bookforge-roundtrip-smoke-20260901"
    with pytest.raises(ValueError, match="approval"):
        roundtrip._validate_request(_request(approval_token="approve"))
    with pytest.raises(ValueError, match="SHA-256"):
        roundtrip._validate_request(_request(input_manifest_sha256="latest"))

    cache_run_id = "bookforge-roundtrip-cache-20260902"
    receipt_sha = "1" * 64
    completion_sha = "2" * 64
    cache_request = {
        "base_cache_run_id": cache_run_id,
        "base_cache_receipt_sha256": receipt_sha,
        "base_cache_completion_sha256": completion_sha,
        "base_cache_approval_token": roundtrip._base_cache_approval_token(
            cache_run_id, receipt_sha, completion_sha
        ),
    }
    assert roundtrip._validate_base_cache_request(
        cache_request, target_run_id="bookforge-roundtrip-target-20260902"
    ) == (cache_run_id, receipt_sha, completion_sha)
    with pytest.raises(ValueError, match="must be complete"):
        roundtrip._validate_base_cache_request(
            {"base_cache_run_id": cache_run_id},
            target_run_id="bookforge-roundtrip-target-20260902",
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
                "backend": "modal-l4x2",
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


def test_modal_billing_parser_accepts_current_and_legacy_fields() -> None:
    parse = roundtrip._parse_modal_billing_total

    assert parse('[{"cost":"0.25"},{"cost":"1.5"}]') == pytest.approx(1.75)
    assert parse('[{"Cost":"0.25"},{"Cost":"1.5"}]') == pytest.approx(1.75)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[null]",
        '[{"description":"missing"}]',
        '[{"cost":"NaN"}]',
        '[{"cost":"-0.1"}]',
        '[{"cost":"1","Cost":"2"}]',
    ],
)
def test_modal_billing_parser_rejects_unsafe_reports(payload: str) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        roundtrip._parse_modal_billing_total(payload)
