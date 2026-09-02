# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.hf_generation_normalization import (
    GenerationNormalizationError,
    normalize_generation_config,
    validate_generation_normalization,
)
from training.jax_fidelity.integrity import artifact_manifest, sha256_file


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def test_generation_normalization_adds_only_manifest_bound_metadata(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    original = tmp_path / "original"
    normalized = tmp_path / "normalized"
    raw.mkdir()
    original.mkdir()
    (raw / "model.safetensors").write_bytes(b"trained")
    (raw / "config.json").write_bytes(b'{"model_type":"gemma4_text"}\n')
    (original / "model.safetensors").write_bytes(b"base")
    generation = original / "generation_config.json"
    _write_json(generation, {"eos_token_id": [1, 106, 50], "temperature": 1.0})
    receipt_path = tmp_path / "normalization.json"
    prior_sha = "a" * 64

    receipt = normalize_generation_config(
        source_checkpoint=raw,
        original_checkpoint=original,
        destination=normalized,
        receipt_path=receipt_path,
        prior_conversion_completion_sha256=prior_sha,
    )

    assert (normalized / "generation_config.json").read_bytes() == generation.read_bytes()
    assert (normalized / "model.safetensors").read_bytes() == b"trained"
    assert receipt["source_checkpoint_manifest"] == artifact_manifest(raw)
    assert receipt["output_checkpoint_manifest"] == artifact_manifest(normalized)
    assert receipt["original_generation_config"]["sha256"] == sha256_file(generation)
    validate_generation_normalization(
        receipt,
        source_checkpoint_manifest=artifact_manifest(raw),
        original_checkpoint_manifest=artifact_manifest(original),
        normalized_checkpoint=normalized,
        source_generation_config=generation,
        prior_conversion_completion_sha256=prior_sha,
    )


def test_generation_normalization_rejects_changed_or_extra_metadata(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    original = tmp_path / "original"
    raw.mkdir()
    original.mkdir()
    (raw / "model.safetensors").write_bytes(b"trained")
    _write_json(raw / "generation_config.json", {"eos_token_id": [1, 106]})
    _write_json(original / "generation_config.json", {"eos_token_id": [1, 106, 50]})

    with pytest.raises(GenerationNormalizationError, match="different generation metadata"):
        normalize_generation_config(
            source_checkpoint=raw,
            original_checkpoint=original,
            destination=tmp_path / "normalized",
            receipt_path=tmp_path / "normalization.json",
        )


def test_generation_normalization_validation_detects_tampering(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    original = tmp_path / "original"
    normalized = tmp_path / "normalized"
    raw.mkdir()
    original.mkdir()
    (raw / "model.safetensors").write_bytes(b"trained")
    generation = original / "generation_config.json"
    _write_json(generation, {"eos_token_id": [1, 106, 50]})
    receipt = normalize_generation_config(
        source_checkpoint=raw,
        original_checkpoint=original,
        destination=normalized,
        receipt_path=tmp_path / "normalization.json",
    )
    (normalized / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(GenerationNormalizationError, match="inconsistent"):
        validate_generation_normalization(
            receipt,
            source_checkpoint_manifest=artifact_manifest(raw),
            original_checkpoint_manifest=artifact_manifest(original),
            normalized_checkpoint=normalized,
            source_generation_config=generation,
            prior_conversion_completion_sha256=None,
        )
