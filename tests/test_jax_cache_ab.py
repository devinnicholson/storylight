# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.cache_ab import compare_cache_runs


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    run = {
        "run_id": "train-1",
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "stage": "lora-train",
        "metadata": {"inputs": {"prepared_train": {"sha256": "c" * 64}}},
    }
    evidence = {
        "learning": {
            "status": "passed",
            "optimizer_steps": 100,
            "event_stream": {"sha256": "different-per-run"},
            "scalars": {"learning/loss": {"first": 2.0, "last": 0.1}},
        },
        "checkpoint_progression": {
            "changed_model_lora_array_count": 410,
            "changed_model_lora_element_count": 21_568_483,
            "checkpoint_relative_delta": 0.048,
        },
        "learnability_acceptance": {"status": "passed"},
    }
    cache_identity = {
        "schema_version": "bookforge-jax-compilation-cache-v1",
        "status": "complete",
        "cache_directory": "/jax-cache/maxtext-entries-v1",
        "owner_sha256": "d" * 64,
        "environment": {"JAX_ENABLE_COMPILATION_CACHE": "true"},
        "maxtext_entrypoint": {"configured": True},
    }
    paths: dict[str, object] = {
        "cold_run_path": _write(
            tmp_path / "cold-run.json",
            {**run, "created_at": "2026-09-03T00:00:00Z"},
        ),
        "warm_run_path": _write(
            tmp_path / "warm-run.json",
            {**run, "created_at": "2026-09-03T01:00:00Z"},
        ),
        "cold_completion_path": _write(
            tmp_path / "cold-completion.json",
            {"status": "succeeded", "completed_at": "2026-09-03T00:12:18Z", "evidence": evidence},
        ),
        "warm_completion_path": _write(
            tmp_path / "warm-completion.json",
            {
                "status": "succeeded",
                "completed_at": "2026-09-03T01:05:31Z",
                "evidence": {
                    **evidence,
                    "learning": {**evidence["learning"], "event_stream": {"sha256": "warm"}},
                },
            },
        ),
        "cold_cache_path": _write(
            tmp_path / "cold-cache.json",
            {
                **cache_identity,
                "cache_was_warm": False,
                "before": {"bytes": 0, "files": 0},
                "after": {"bytes": 1000, "files": 10},
                "added_bytes": 1000,
                "added_files": 10,
            },
        ),
        "warm_cache_path": _write(
            tmp_path / "warm-cache.json",
            {
                **cache_identity,
                "cache_was_warm": True,
                "before": {"bytes": 1000, "files": 10},
                "after": {"bytes": 1000, "files": 10},
                "added_bytes": 0,
                "added_files": 0,
            },
        ),
        "cold_billing_path": _write(
            tmp_path / "cold-billing.json",
            [{"Object ID": "cold-app", "Cost": "0.60"}],
        ),
        "warm_billing_path": _write(
            tmp_path / "warm-billing.json",
            [{"Object ID": "warm-app", "Cost": "0.30"}],
        ),
        "cold_app_id": "cold-app",
        "warm_app_id": "warm-app",
    }
    return paths


def test_compare_cache_runs_proves_reuse_without_training_drift(tmp_path: Path) -> None:
    receipt = compare_cache_runs(**_fixture(tmp_path))

    assert receipt["status"] == "passed"
    assert receipt["cold"]["duration_seconds"] == 738.0
    assert receipt["warm"]["duration_seconds"] == 331.0
    assert receipt["improvement"]["provider_cost_usd"] == "0.30"
    assert receipt["cache"]["warm_inventory_unchanged"] is True
    assert receipt["invariants"]["learning_evidence_identical"] is True


def test_compare_cache_runs_rejects_new_warm_cache_entries(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    warm_cache_path = fixture["warm_cache_path"]
    assert isinstance(warm_cache_path, Path)
    warm_cache = json.loads(warm_cache_path.read_text(encoding="utf-8"))
    warm_cache["added_files"] = 1
    _write(warm_cache_path, warm_cache)

    with pytest.raises(ValueError, match="wrote unexpected cache entries"):
        compare_cache_runs(**fixture)


def test_compare_cache_runs_rejects_learning_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    warm_completion_path = fixture["warm_completion_path"]
    assert isinstance(warm_completion_path, Path)
    completion = json.loads(warm_completion_path.read_text(encoding="utf-8"))
    completion["evidence"]["learning"]["optimizer_steps"] = 99
    _write(warm_completion_path, completion)

    with pytest.raises(ValueError, match="training semantics changed"):
        compare_cache_runs(**fixture)
