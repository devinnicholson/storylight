from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy import modal_jax_fidelity
from training.jax_fidelity.learning_evidence import (
    REQUIRED_SCALARS,
    LearningEvidenceError,
    summarize_scalar_series,
    verify_v3_terminal_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"


def _thresholds() -> dict[str, object]:
    import json

    return json.loads(V3_CONFIG.read_text())["recovery"]["learnability_acceptance"]


def _v3_series(steps: int = 100) -> dict[str, list[tuple[int, float]]]:
    series = _series(steps)
    series["learning/loss"] = [(step, 2.0 - (0.8 * step / (steps - 1))) for step in range(steps)]
    series["learning/raw_grad_norm"] = [(step, 0.7) for step in range(steps)]
    series["learning/grad_norm"] = [(step, 0.7) for step in range(steps)]
    series["learning/update_norm"] = [(step, 0.01) for step in range(steps)]
    series["learning/changed_trainable_leaves"] = [(step, 205.0) for step in range(steps)]
    return series


def _series(steps: int = 5) -> dict[str, list[tuple[int, float]]]:
    values = {name: [(step, float(step + 1)) for step in range(steps)] for name in REQUIRED_SCALARS}
    values["learning/loss"] = [(step, 2.0 - step * 0.1) for step in range(steps)]
    values["learning/current_learning_rate"] = [
        (step, 0.0 if step == 0 else 1e-4) for step in range(steps)
    ]
    return values


def _checkpoint_progression(adapter: dict[str, object]) -> dict[str, object]:
    pair_count = int(adapter["lora_pair_count"])
    initial_adapter = dict(adapter)
    initial_adapter["checkpoint_step"] = 0
    return {
        "schema_version": "bookforge-jax-lora-checkpoint-progression-v1",
        "status": "passed",
        "comparison_dtype": "float32",
        "accumulation_dtype": "float64",
        "initial_step": 0,
        "terminal_step": 99,
        "model_lora_array_count": pair_count * 2,
        "model_lora_element_count": 1_000_000,
        "changed_model_lora_array_count": pair_count * 2,
        "changed_model_lora_element_count": 500_000,
        "initial_model_lora_l2_norm": 81.0,
        "checkpoint_delta_l2_norm": 2.5,
        "checkpoint_relative_delta": 2.5 / 81.0,
        "minimum_checkpoint_relative_delta": 1e-6,
        "initial_adapter": initial_adapter,
        "terminal_adapter": adapter,
    }


def test_learning_evidence_accepts_complete_nonzero_optimization_trace() -> None:
    evidence = summarize_scalar_series(_series(), expected_steps=5)

    assert evidence["status"] == "passed"
    assert evidence["optimizer_steps"] == 5
    assert evidence["nonzero_raw_gradient_observed"] is True
    assert evidence["completion_tokens_present_every_step"] is True


def test_learning_evidence_rejects_the_v2_zero_gradient_failure() -> None:
    series = _series()
    series["learning/raw_grad_norm"] = [(step, 0.0) for step in range(5)]
    series["learning/grad_norm"] = [(step, 0.0) for step in range(5)]

    with pytest.raises(LearningEvidenceError, match="no nonzero gradients"):
        summarize_scalar_series(series, expected_steps=5)


def test_v3_smoke_enforces_epsilon_and_parameter_update() -> None:
    series = _series(1)
    series["learning/current_learning_rate"] = [(0, 1e-4)]
    series["learning/raw_grad_norm"] = [(0, 1e-13)]
    series["learning/grad_norm"] = [(0, 1e-13)]
    series["learning/update_norm"] = [(0, 0.01)]
    series["learning/changed_trainable_leaves"] = [(0, 1.0)]
    with pytest.raises(LearningEvidenceError, match="no nonzero gradients"):
        summarize_scalar_series(series, expected_steps=1, v3_acceptance=_thresholds())

    series["learning/raw_grad_norm"] = [(0, 0.01)]
    series["learning/grad_norm"] = [(0, 0.01)]
    series["learning/update_norm"] = [(0, 0.0)]
    with pytest.raises(LearningEvidenceError, match="no trainable parameter update"):
        summarize_scalar_series(series, expected_steps=1, v3_acceptance=_thresholds())


def test_learning_evidence_rejects_missing_steps_and_empty_supervision() -> None:
    missing = _series()
    missing["learning/loss"] = missing["learning/loss"][:-1]
    with pytest.raises(LearningEvidenceError, match="every expected optimizer step"):
        summarize_scalar_series(missing, expected_steps=5)

    empty = _series()
    empty["learning/total_weights"][2] = (2, 0.0)
    with pytest.raises(LearningEvidenceError, match="no supervised completion tokens"):
        summarize_scalar_series(empty, expected_steps=5)


def test_v3_full_gate_accepts_persistent_learning() -> None:
    evidence = summarize_scalar_series(
        _v3_series(),
        expected_steps=100,
        v3_acceptance=_thresholds(),
        require_full_v3=True,
    )

    gate = evidence["v3_acceptance"]
    assert gate["mode"] == "full-canary"
    assert gate["nonzero_raw_gradient_steps"] == 100
    assert gate["rolling_loss_relative_reduction"] >= 0.1


def test_v3_full_gate_rejects_constant_loss() -> None:
    series = _v3_series()
    series["learning/loss"] = [(step, 1.5) for step in range(100)]

    with pytest.raises(LearningEvidenceError, match="rolling loss reduction"):
        summarize_scalar_series(
            series,
            expected_steps=100,
            v3_acceptance=_thresholds(),
            require_full_v3=True,
        )


def test_v3_full_gate_rejects_one_isolated_nonzero_gradient() -> None:
    series = _v3_series()
    series["learning/raw_grad_norm"] = [(step, 0.5 if step == 0 else 0.0) for step in range(100)]
    series["learning/grad_norm"] = [(step, 0.5 if step == 0 else 0.0) for step in range(100)]

    with pytest.raises(LearningEvidenceError, match="sustain nonzero gradients"):
        summarize_scalar_series(
            series,
            expected_steps=100,
            v3_acceptance=_thresholds(),
            require_full_v3=True,
        )


def test_v3_full_gate_does_not_confuse_global_norm_with_parameter_identity() -> None:
    series = _v3_series()
    series["learning/param_norm"] = [(step, 10.0) for step in range(100)]

    evidence = summarize_scalar_series(
        series,
        expected_steps=100,
        v3_acceptance=_thresholds(),
        require_full_v3=True,
    )

    assert evidence["v3_acceptance"]["mode"] == "full-canary"


def test_v3_smoke_requires_optimizer_tokens_and_adapter_checkpoint_proof() -> None:
    series = _series(1)
    series["learning/current_learning_rate"] = [(0, 1e-4)]
    series["learning/update_norm"] = [(0, 0.01)]
    series["learning/changed_trainable_leaves"] = [(0, 205.0)]
    learning = summarize_scalar_series(
        series,
        expected_steps=1,
        v3_acceptance=_thresholds(),
    )
    pair_count = 205
    adapter = {
        "schema_version": "1.0",
        "format": "maxtext-orbax-lora-tree",
        "rank": 16,
        "lora_pair_count": pair_count,
        "lora_tensor_count": pair_count * 2,
        "optimizer_lora_tensor_count": pair_count * 4,
        "payload_arrays_restored": True,
        "restored_lora_array_count": pair_count * 6,
        "metadata_sha256": "a" * 64,
        "expected_lora_pair_count": pair_count,
        "checkpoint_step": 0,
        "checkpoint_step_binding": "directory-name-plus-root-step-leaf",
        "approved_maxtext_patch_sha256": "b" * 64,
        "pairs": [{} for _ in range(pair_count)],
    }

    accepted = verify_v3_terminal_acceptance(
        experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
        smoke=True,
        expected_steps=1,
        expected_rank=16,
        expected_lora_pair_count=pair_count,
        approved_maxtext_patch_sha256="b" * 64,
        learning_evidence=learning,
        adapter_evidence=adapter,
    )
    assert accepted is not None
    assert accepted["mode"] == "smoke"
    assert accepted["expected_optimizer_lora_tensor_count"] == pair_count * 4

    adapter["lora_pair_count"] = 1
    with pytest.raises(LearningEvidenceError, match="checkpoint proof"):
        verify_v3_terminal_acceptance(
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=True,
            expected_steps=1,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256="b" * 64,
            learning_evidence=learning,
            adapter_evidence=adapter,
        )

    adapter["lora_pair_count"] = pair_count
    adapter["optimizer_lora_tensor_count"] = pair_count * 4 - 1
    with pytest.raises(LearningEvidenceError, match="checkpoint proof"):
        verify_v3_terminal_acceptance(
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=True,
            expected_steps=1,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256="b" * 64,
            learning_evidence=learning,
            adapter_evidence=adapter,
        )

    adapter["optimizer_lora_tensor_count"] = pair_count * 4 + 1
    with pytest.raises(LearningEvidenceError, match="checkpoint proof"):
        verify_v3_terminal_acceptance(
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=True,
            expected_steps=1,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256="b" * 64,
            learning_evidence=learning,
            adapter_evidence=adapter,
        )


def test_modal_publication_rechecks_the_terminal_v3_acceptance_receipt() -> None:
    learning = summarize_scalar_series(
        _v3_series(),
        expected_steps=100,
        v3_acceptance=_thresholds(),
        require_full_v3=True,
    )
    pair_count = 205
    patch_sha = "b" * 64
    adapter = {
        "schema_version": "1.0",
        "format": "maxtext-orbax-lora-tree",
        "rank": 16,
        "lora_pair_count": pair_count,
        "lora_tensor_count": pair_count * 2,
        "optimizer_lora_tensor_count": pair_count * 4,
        "payload_arrays_restored": True,
        "restored_lora_array_count": pair_count * 6,
        "metadata_sha256": "a" * 64,
        "expected_lora_pair_count": pair_count,
        "checkpoint_step": 99,
        "checkpoint_step_binding": "directory-name-plus-root-step-leaf",
        "approved_maxtext_patch_sha256": patch_sha,
        "pairs": [{} for _ in range(pair_count)],
    }
    acceptance = verify_v3_terminal_acceptance(
        experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
        smoke=False,
        expected_steps=100,
        expected_rank=16,
        expected_lora_pair_count=pair_count,
        approved_maxtext_patch_sha256=patch_sha,
        learning_evidence=learning,
        adapter_evidence=adapter,
        checkpoint_progression_evidence=_checkpoint_progression(adapter),
    )
    with pytest.raises(LearningEvidenceError, match="checkpoint progression proof"):
        verify_v3_terminal_acceptance(
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=False,
            expected_steps=100,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256=patch_sha,
            learning_evidence=learning,
            adapter_evidence=adapter,
        )
    completion = {
        "evidence": {
            "learning": learning,
            "terminal_adapter": adapter,
            "checkpoint_progression": _checkpoint_progression(adapter),
            "learnability_acceptance": acceptance,
        }
    }

    assert (
        modal_jax_fidelity._verify_training_completion_acceptance(
            completion,
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=False,
            expected_steps=100,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256=patch_sha,
        )
        == acceptance
    )

    completion["evidence"]["learnability_acceptance"] = {"status": "passed"}
    with pytest.raises(RuntimeError, match="receipt changed before publication"):
        modal_jax_fidelity._verify_training_completion_acceptance(
            completion,
            experiment_id="bookforge-gemma4-e2b-lora-r16-v3-canary",
            smoke=False,
            expected_steps=100,
            expected_rank=16,
            expected_lora_pair_count=pair_count,
            approved_maxtext_patch_sha256=patch_sha,
        )
