# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra/gcp/jax"))

from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import (
    sha256_file,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


merge = _load("bookforge_modal_jax_merge", ROOT / "deploy/modal_jax_merge.py")
fetcher = _load("bookforge_modal_jax_merge_fetch", ROOT / "scripts/fetch_modal_jax_merge.py")
CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
CONFIG_V3 = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"


def _request(**updates: object) -> dict[str, object]:
    identifiers = {
        "merge_run_id": "bookforge-full-merge-20260901",
        "roundtrip_run_id": "bookforge-roundtrip-smoke-20260901",
        "training_input_run_id": "bookforge-v2-training-input-20260902",
        "training_release_run_id": "bookforge-full-training-20260901",
        "training_run_id": "lora-train-aaaaaaaaaaaaaaaaaaaa",
    }
    bindings = {name: f"{index:x}" * 64 for index, name in enumerate(merge._BINDING_NAMES, 1)}
    request: dict[str, object] = {**identifiers, **bindings}
    request["approval_token"] = merge._approval_token(**identifiers, bindings=bindings)
    request.update(updates)
    return request


def test_merge_request_requires_every_hash_and_exact_approval() -> None:
    validated = merge._validate_request(_request())
    assert validated[0] == "bookforge-full-merge-20260901"
    assert validated[2] == "bookforge-v2-training-input-20260902"
    assert set(validated[-1]) == set(merge._BINDING_NAMES)
    with pytest.raises(ValueError, match="approval"):
        merge._validate_request(_request(approval_token="approve"))
    with pytest.raises(ValueError, match="SHA-256"):
        merge._validate_request(_request(training_run_sha256="latest"))
    with pytest.raises(ValueError, match="training_input_run_id"):
        merge._validate_request(_request(training_input_run_id="latest"))


def test_merge_billing_parser_accepts_current_and_legacy_fields() -> None:
    assert merge._parse_modal_billing_total('[{"cost": "1.25"}, {"Cost": 0.5}]') == pytest.approx(
        1.75
    )
    with pytest.raises(RuntimeError, match="no cost"):
        merge._parse_modal_billing_total('[{"description": "missing"}]')
    with pytest.raises(RuntimeError, match="conflicting"):
        merge._parse_modal_billing_total('[{"cost": 1, "Cost": 2}]')


def test_merge_refuses_v3_diagnostic_canary_and_requires_live_evidence() -> None:
    config = load_config(CONFIG_V3)
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        merge._enforce_merge_policy(config)

    learning = {"status": "passed", "optimizer_steps": 100}
    adapter = {"lora_pair_count": 205, "checkpoint_step": 99}
    completion = {"evidence": {"learning": learning, "terminal_adapter": adapter}}
    merge._require_live_training_evidence(
        completion,
        learning=learning,
        terminal_adapter=adapter,
    )
    with pytest.raises(RuntimeError, match="learning evidence"):
        merge._require_live_training_evidence(
            completion,
            learning={**learning, "optimizer_steps": 99},
            terminal_adapter=adapter,
        )
    with pytest.raises(RuntimeError, match="terminal adapter evidence"):
        merge._require_live_training_evidence(
            completion,
            learning=learning,
            terminal_adapter={**adapter, "lora_pair_count": 204},
        )


def test_merge_release_file_verifier_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.manifest.json"
    artifact.write_bytes(b"candidate")
    rows = [
        {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}
    ]
    fetcher._verify_files(tmp_path, rows)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        fetcher._verify_files(tmp_path, rows)
