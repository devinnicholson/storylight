# ruff: noqa: E402
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.jax_fidelity.checkpoint_evidence as checkpoint_evidence
import training.jax_fidelity.roundtrip_evidence as roundtrip_evidence
from bookforge.fidelity_benchmark import FidelitySummary
from training.jax_fidelity.artifact_contract import create_artifact_contract
from training.jax_fidelity.checkpoint_evidence import (
    CheckpointEvidenceError,
    inspect_hf_roundtrip,
    parse_max_kl_divergence,
)
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.development_eligibility import (
    decide_development_eligibility,
)
from training.jax_fidelity.integrity import artifact_manifest, sha256_file

CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"
DATASET_MANIFEST = ROOT / "datasets/story-fidelity-v1/manifest.json"


def _write_json(path: Path, document: object) -> str:
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    return sha256_file(path)


def _completion(
    path: Path,
    *,
    run_id: str,
    direction: str | None = None,
    smoke: bool = False,
) -> str:
    evidence: dict[str, object] = {}
    if direction is not None:
        evidence["direction"] = direction
    if direction == "logit-check":
        evidence["forward_kl_divergence"] = 0.004
    if smoke:
        evidence["runtime_lock"] = {"sha256": "f" * 64}
    return _write_json(
        path,
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "succeeded",
            "artifacts": [{"sha256": "e" * 64}],
            "evidence": evidence,
        },
    )


def _summary(*, exact: float, semantic: float = 1.0) -> FidelitySummary:
    categories = {
        "attributes": 1.0,
        "negation": 1.0,
        "passive_voice": 1.0,
        "prompt_injection": 1.0,
        "transformation": 1.0,
    }
    return FidelitySummary(
        surface="raw",
        split="development",
        records=512,
        record_ids_sha256="a" * 64,
        category_record_counts={name: 64 for name in categories},
        schema_valid_rate=1.0,
        privacy_pass_rate=1.0,
        semantic_atom_recall=semantic,
        exact_example_pass_rate=exact,
        category_pass_rates=categories,
        counterfactual_pairs=256,
        counterfactual_sensitivity=1.0,
        unsupported_concept_rate=0.0,
        pii_leaks=0,
        privacy_term_leaks=0,
        source_echoes=0,
        injection_leaks=0,
        forbidden_hits=0,
    )


def test_parse_maxtext_kl_uses_the_largest_observed_value() -> None:
    output = "\n".join(
        (
            "Max KL divergence for a single token in the set: 0.002",
            "Max KL divergence for a single token in the set: 1.2e-02",
        )
    )
    assert parse_max_kl_divergence(output) == pytest.approx(0.012)
    with pytest.raises(CheckpointEvidenceError, match="maximum KL"):
        parse_max_kl_divergence("conversion passed")


def test_checkpoint_inspection_binds_architecture_tokenizer_and_ple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    architecture = {
        name: ["global", "local"] if name == "layer_types" else index + 1
        for index, name in enumerate(checkpoint_evidence._ARCHITECTURE_FIELDS)
    }
    architecture["rope_parameters"] = {"rope_theta": 10000.0}
    for root in (tmp_path / "base", tmp_path / "exported"):
        root.mkdir()
        _write_json(root / "config.json", {"model_type": "gemma4_text", **architecture})
        _write_json(root / "generation_config.json", {"eos_token_id": [1, 106, 50]})
        (root / "tokenizer.json").write_bytes(b"tokenizer")
        (root / "tokenizer_config.json").write_bytes(b"config")

    shapes = {
        "model.embed_tokens.weight": (256, 64),
        "model.embed_tokens_per_layer.weight": (2, 256, 64),
    }
    monkeypatch.setattr(checkpoint_evidence, "_safetensor_shapes", lambda _root: shapes)

    result = inspect_hf_roundtrip(
        load_config(CONFIG),
        base_checkpoint=tmp_path / "base",
        exported_checkpoint=tmp_path / "exported",
    )

    assert result["tensor_names"] is True
    assert result["ple_weights"] is True
    assert result["eos_token_ids"] == [1, 106, 50]


def test_roundtrip_evidence_uses_only_terminal_completion_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "model.safetensors").write_bytes(b"merged")
    inspection = {
        "tensor_names": True,
        "tensor_shapes": True,
        "tokenizer": True,
        "special_tokens": True,
        "ple_weights": True,
        "kv_sharing": True,
        "gemma4_metadata": True,
        "eos_token_ids": [1, 106, 50],
        "exported_checkpoint_manifest": artifact_manifest(exported),
        "base_checkpoint_manifest": {"content_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        roundtrip_evidence,
        "inspect_hf_roundtrip",
        lambda *_args, **_kwargs: dict(inspection),
    )
    bindings = {
        "hf": ("hf-to-maxtext-test", "hf-to-maxtext", False),
        "smoke": ("lora-smoke-test", None, True),
        "export": ("maxtext-to-hf-test", "maxtext-to-hf", False),
        "logit": ("logit-check-test", "logit-check", False),
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, (run_id, direction, smoke) in bindings.items():
        paths[name] = tmp_path / f"{name}.json"
        hashes[name] = _completion(
            paths[name], run_id=run_id, direction=direction, smoke=smoke
        )

    evidence = roundtrip_evidence.build_roundtrip_evidence(
        config_path=CONFIG,
        base_checkpoint=tmp_path / "base",
        exported_checkpoint=exported,
        hf_to_maxtext_completion=paths["hf"],
        hf_to_maxtext_completion_sha256=hashes["hf"],
        smoke_completion=paths["smoke"],
        smoke_completion_sha256=hashes["smoke"],
        maxtext_to_hf_completion=paths["export"],
        maxtext_to_hf_completion_sha256=hashes["export"],
        logit_completion=paths["logit"],
        logit_completion_sha256=hashes["logit"],
    )

    assert evidence["checks"]["forward_kl_divergence"] == pytest.approx(0.004)
    assert evidence["lineage"]["smoke_run_id"] == "lora-smoke-test"
    assert evidence["lineage"]["logit_completion_sha256"] == hashes["logit"]


def test_artifact_contract_rejects_duplicate_names_and_binds_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "value").write_bytes(b"first")
    (second / "value").write_bytes(b"second")

    contract = create_artifact_contract([f"base={first}", f"tokenizer={second}"])
    assert contract["artifacts"]["base"] == artifact_manifest(first)
    with pytest.raises(ValueError, match="duplicate"):
        create_artifact_contract([f"base={first}", f"base={second}"])


def test_hf_snapshot_plan_requires_no_token_or_network(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.jax_fidelity.hf_snapshot",
            "--config",
            str(CONFIG),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--snapshot-manifest",
            str(tmp_path / "snapshot.json"),
            "--tokenizer-manifest",
            str(tmp_path / "tokenizer.json"),
            "--completion",
            str(tmp_path / "completion.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["model_revision"] == load_config(CONFIG).production["model_revision"]
    assert plan["approval_token"].startswith("HF-SNAPSHOT:")
    assert not (tmp_path / "snapshot").exists()


def test_candidate_prediction_plan_validates_all_inputs_without_loading_torch(
    tmp_path: Path,
) -> None:
    manifest = json.loads(DATASET_MANIFEST.read_text())
    development = DATASET_MANIFEST.parent / manifest["splits"]["development"]["path"]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"candidate")
    checkpoint_manifest = tmp_path / "checkpoint.manifest.json"
    checkpoint_manifest_sha = _write_json(checkpoint_manifest, artifact_manifest(checkpoint))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.jax_fidelity.predict",
            "--config",
            str(CONFIG),
            "--dataset-manifest",
            str(DATASET_MANIFEST),
            "--dataset-manifest-sha256",
            sha256_file(DATASET_MANIFEST),
            "--records",
            str(development),
            "--records-sha256",
            sha256_file(development),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-manifest",
            str(checkpoint_manifest),
            "--checkpoint-manifest-sha256",
            checkpoint_manifest_sha,
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--completion",
            str(tmp_path / "prediction-completion.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["records"] == 512
    assert plan["maximum_output_tokens"] == 64


def test_development_eligibility_requires_real_improvement() -> None:
    passing = decide_development_eligibility(_summary(exact=0.98), _summary(exact=0.90))
    unchanged = decide_development_eligibility(_summary(exact=0.98), _summary(exact=0.98))
    low_recall = decide_development_eligibility(
        _summary(exact=0.98, semantic=0.90), _summary(exact=0.90)
    )

    assert all(passing.values())
    assert unchanged["development_improvement"] is False
    assert low_recall["semantic_atom_recall"] is False
