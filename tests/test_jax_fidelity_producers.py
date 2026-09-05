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
from training.jax_fidelity.checkpoint_evidence import (
    inspect_hf_roundtrip,
)
from training.jax_fidelity.configuration import load_config
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
        evidence["comparison"] = "adapted-maxtext-vs-merged-hf"
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


def test_checkpoint_inspection_binds_architecture_tokenizer_and_ple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    architecture = {
        name: ["global", "local"] if name == "layer_types" else index + 1
        for index, name in enumerate(checkpoint_evidence._ARCHITECTURE_FIELDS)
    }
    architecture["rope_parameters"] = {"rope_theta": 10000.0}
    architecture["num_kv_shared_layers"] = 0
    for root in (tmp_path / "base", tmp_path / "exported"):
        root.mkdir()
        _write_json(root / "config.json", {"model_type": "gemma4_text", **architecture})
        _write_json(root / "generation_config.json", {"eos_token_id": [1, 106, 50]})
        (root / "tokenizer.json").write_bytes(b"tokenizer")
    tokenizer_config = {"image_token": "<|image|>", "tokenizer_class": "GemmaTokenizer"}
    _write_json(tmp_path / "base/tokenizer_config.json", tokenizer_config)
    _write_json(
        tmp_path / "exported/tokenizer_config.json",
        {
            **tokenizer_config,
            "is_local": True,
            "local_files_only": True,
            "model_specific_special_tokens": {"image_token": "<|image|>"},
        },
    )

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
    assert evidence["checks"]["logit_comparison"] == "adapted-maxtext-vs-merged-hf"
    assert evidence["lineage"]["smoke_run_id"] == "lora-smoke-test"
    assert evidence["lineage"]["logit_completion_sha256"] == hashes["logit"]


def test_hf_snapshot_public_access_mode_is_approval_bound(tmp_path: Path) -> None:
    common = [
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
    ]
    secret = subprocess.run(
        common,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    public = subprocess.run(
        [*common, "--access-mode", "public-anonymous"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    secret_plan = json.loads(secret.stdout)
    public_plan = json.loads(public.stdout)
    assert secret_plan["access_mode"] == "secret-token"
    assert public_plan["access_mode"] == "public-anonymous"
    assert public_plan["approval_token"] != secret_plan["approval_token"]
