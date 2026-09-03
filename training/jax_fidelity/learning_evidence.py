"""Fail-closed learning evidence for a completed MaxText training process."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .configuration import RECOVERY_EXPERIMENT_ID
from .integrity import sha256_file


class LearningEvidenceError(RuntimeError):
    """A training run did not prove that optimization occurred."""


REQUIRED_SCALARS = (
    "learning/loss",
    "learning/raw_grad_norm",
    "learning/grad_norm",
    "learning/current_learning_rate",
    "learning/total_weights",
)
V3_UPDATE_NORM_SCALAR = "learning/update_norm"
V3_CHANGED_LEAVES_SCALAR = "learning/changed_trainable_leaves"
V3_ACCEPTANCE_SCHEMA = "bookforge-jax-v3-learnability-acceptance-v2"
V3_THRESHOLD_FIELDS = {
    "schema_version",
    "nonzero_gradient_epsilon",
    "minimum_nonzero_gradient_fraction",
    "rolling_loss_window_steps",
    "minimum_rolling_loss_relative_reduction",
    "minimum_checkpoint_lora_relative_delta",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _v3_thresholds(raw: Mapping[str, Any]) -> dict[str, float | int | str]:
    if set(raw) != V3_THRESHOLD_FIELDS or raw.get("schema_version") != V3_ACCEPTANCE_SCHEMA:
        raise LearningEvidenceError("v3 learnability thresholds changed")
    epsilon = raw.get("nonzero_gradient_epsilon")
    fraction = raw.get("minimum_nonzero_gradient_fraction")
    window = raw.get("rolling_loss_window_steps")
    loss_reduction = raw.get("minimum_rolling_loss_relative_reduction")
    checkpoint_delta = raw.get("minimum_checkpoint_lora_relative_delta")
    if (
        type(epsilon) not in (int, float)
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
        or type(fraction) not in (int, float)
        or not math.isfinite(float(fraction))
        or not 0 < float(fraction) <= 1
        or type(window) is not int
        or window < 1
        or type(loss_reduction) not in (int, float)
        or not math.isfinite(float(loss_reduction))
        or not 0 < float(loss_reduction) < 1
        or type(checkpoint_delta) not in (int, float)
        or not math.isfinite(float(checkpoint_delta))
        or float(checkpoint_delta) <= 0
    ):
        raise LearningEvidenceError("v3 learnability thresholds are invalid")
    return {
        "schema_version": V3_ACCEPTANCE_SCHEMA,
        "nonzero_gradient_epsilon": float(epsilon),
        "minimum_nonzero_gradient_fraction": float(fraction),
        "rolling_loss_window_steps": window,
        "minimum_rolling_loss_relative_reduction": float(loss_reduction),
        "minimum_checkpoint_lora_relative_delta": float(checkpoint_delta),
    }


def summarize_scalar_series(
    series: Mapping[str, Sequence[tuple[int, float]]],
    *,
    expected_steps: int,
    v3_acceptance: Mapping[str, Any] | None = None,
    require_full_v3: bool = False,
) -> dict[str, Any]:
    """Validate exact step coverage and summarize optimization-critical scalars."""

    if type(expected_steps) is not int or expected_steps < 1:
        raise LearningEvidenceError("expected_steps must be a positive integer")
    expected_indices = list(range(expected_steps))
    thresholds = _v3_thresholds(v3_acceptance) if v3_acceptance is not None else None
    summaries: dict[str, dict[str, float | int]] = {}
    required_scalars = list(REQUIRED_SCALARS)
    if thresholds is not None:
        required_scalars.extend((V3_UPDATE_NORM_SCALAR, V3_CHANGED_LEAVES_SCALAR))
    if require_full_v3 and v3_acceptance is None:
        raise LearningEvidenceError("full v3 training requires immutable thresholds")
    values_by_name: dict[str, list[float]] = {}
    for name in required_scalars:
        points = list(series.get(name, ()))
        if len(points) != expected_steps or [step for step, _ in points] != expected_indices:
            raise LearningEvidenceError(f"{name} does not cover every expected optimizer step")
        values = [float(value) for _, value in points]
        if not all(math.isfinite(value) for value in values):
            raise LearningEvidenceError(f"{name} contains a non-finite value")
        summaries[name] = {
            "samples": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
        }
        values_by_name[name] = values

    raw_gradients = values_by_name["learning/raw_grad_norm"]
    clipped_gradients = values_by_name["learning/grad_norm"]
    learning_rates = values_by_name["learning/current_learning_rate"]
    supervised_weights = values_by_name["learning/total_weights"]
    if any(value < 0 for value in raw_gradients + clipped_gradients):
        raise LearningEvidenceError("gradient norm cannot be negative")
    epsilon = float(thresholds["nonzero_gradient_epsilon"]) if thresholds else 0.0
    if max(raw_gradients) <= epsilon or max(clipped_gradients) <= epsilon:
        raise LearningEvidenceError("training produced no nonzero gradients")
    if max(learning_rates) <= 0:
        raise LearningEvidenceError("training never used a positive learning rate")
    if min(supervised_weights) <= 0:
        raise LearningEvidenceError("a training step had no supervised completion tokens")
    evidence: dict[str, Any] = {
        "schema_version": "bookforge-jax-learning-evidence-v1",
        "status": "passed",
        "optimizer_steps": expected_steps,
        "all_gradients_finite": True,
        "nonzero_raw_gradient_observed": True,
        "nonzero_clipped_gradient_observed": True,
        "positive_learning_rate_observed": True,
        "completion_tokens_present_every_step": True,
        "scalars": summaries,
    }
    if thresholds is not None:
        update_norms = values_by_name[V3_UPDATE_NORM_SCALAR]
        changed_leaves = values_by_name[V3_CHANGED_LEAVES_SCALAR]
        if any(value < 0 for value in update_norms + changed_leaves):
            raise LearningEvidenceError("update evidence cannot be negative")
        if max(update_norms) <= epsilon or max(changed_leaves) < 1:
            raise LearningEvidenceError("training produced no trainable parameter update")
        evidence["nonzero_update_observed"] = True
        evidence["changed_trainable_leaf_observed"] = True
        acceptance: dict[str, Any] = {
            "schema_version": V3_ACCEPTANCE_SCHEMA,
            "mode": "smoke",
            "thresholds": thresholds,
            "minimum_step_proof": {
                "nonzero_raw_gradient": True,
                "nonzero_clipped_gradient": True,
                "nonzero_parameter_update": True,
                "changed_trainable_leaf": True,
                "positive_learning_rate": True,
                "supervised_completion_tokens": True,
            },
        }
        if require_full_v3:
            window = int(thresholds["rolling_loss_window_steps"])
            if expected_steps < window * 2:
                raise LearningEvidenceError(
                    "full v3 run is too short for disjoint rolling loss windows"
                )
            raw_nonzero = sum(value > epsilon for value in raw_gradients)
            clipped_nonzero = sum(value > epsilon for value in clipped_gradients)
            minimum_nonzero = math.ceil(
                expected_steps * float(thresholds["minimum_nonzero_gradient_fraction"])
            )
            if raw_nonzero < minimum_nonzero or clipped_nonzero < minimum_nonzero:
                raise LearningEvidenceError("full v3 run did not sustain nonzero gradients")

            losses = values_by_name["learning/loss"]
            initial_loss = sum(losses[:window]) / window
            terminal_loss = sum(losses[-window:]) / window
            loss_reduction = (initial_loss - terminal_loss) / max(abs(initial_loss), epsilon)
            if loss_reduction < float(thresholds["minimum_rolling_loss_relative_reduction"]):
                raise LearningEvidenceError(
                    "full v3 rolling loss reduction is below the acceptance threshold"
                )

            acceptance.update(
                {
                    "mode": "full-canary",
                    "nonzero_raw_gradient_steps": raw_nonzero,
                    "nonzero_clipped_gradient_steps": clipped_nonzero,
                    "minimum_nonzero_gradient_steps": minimum_nonzero,
                    "initial_rolling_loss": initial_loss,
                    "terminal_rolling_loss": terminal_loss,
                    "rolling_loss_relative_reduction": loss_reduction,
                }
            )
        evidence["v3_acceptance"] = acceptance
    return evidence


def verify_tensorboard_learning(
    output_directory: Path | str,
    *,
    expected_steps: int,
    v3_acceptance: Mapping[str, Any] | None = None,
    require_full_v3: bool = False,
) -> dict[str, Any]:
    """Read the only TensorBoard event stream and prove that learning occurred."""

    root = Path(output_directory).resolve()
    event_files = sorted(
        path for path in root.rglob("*tfevents*") if path.is_file() and not path.is_symlink()
    )
    if len(event_files) != 1:
        raise LearningEvidenceError(
            f"expected one TensorBoard event stream, found {len(event_files)}"
        )

    from tensorboard.backend.event_processing import event_accumulator

    event_file = event_files[0]
    accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    required = set(REQUIRED_SCALARS)
    if v3_acceptance is not None:
        required.update((V3_UPDATE_NORM_SCALAR, V3_CHANGED_LEAVES_SCALAR))
    missing = sorted(required - available)
    if missing:
        raise LearningEvidenceError(f"TensorBoard is missing required scalars: {missing}")
    series = {
        name: [(int(event.step), float(event.value)) for event in accumulator.Scalars(name)]
        for name in required
    }
    evidence = summarize_scalar_series(
        series,
        expected_steps=expected_steps,
        v3_acceptance=v3_acceptance,
        require_full_v3=require_full_v3,
    )
    evidence["event_stream"] = {
        "path": event_file.relative_to(root).as_posix(),
        "bytes": event_file.stat().st_size,
        "sha256": sha256_file(event_file),
    }
    return evidence


def verify_v3_terminal_acceptance(
    *,
    experiment_id: str,
    smoke: bool,
    expected_steps: int,
    expected_rank: int,
    expected_lora_pair_count: int,
    approved_maxtext_patch_sha256: str,
    learning_evidence: Mapping[str, Any],
    adapter_evidence: Mapping[str, Any],
    checkpoint_progression_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bind optimization metrics to the terminal trainable adapter checkpoint."""

    if experiment_id != RECOVERY_EXPERIMENT_ID:
        return None
    if (
        type(smoke) is not bool
        or type(expected_steps) is not int
        or expected_steps < 1
        or type(expected_rank) is not int
        or expected_rank < 1
        or not isinstance(approved_maxtext_patch_sha256, str)
        or _SHA256.fullmatch(approved_maxtext_patch_sha256) is None
    ):
        raise LearningEvidenceError("v3 terminal acceptance contract is invalid")
    acceptance = learning_evidence.get("v3_acceptance")
    expected_mode = "smoke" if smoke else "full-canary"
    if (
        learning_evidence.get("status") != "passed"
        or learning_evidence.get("optimizer_steps") != expected_steps
        or learning_evidence.get("nonzero_raw_gradient_observed") is not True
        or learning_evidence.get("nonzero_clipped_gradient_observed") is not True
        or learning_evidence.get("nonzero_update_observed") is not True
        or learning_evidence.get("changed_trainable_leaf_observed") is not True
        or learning_evidence.get("positive_learning_rate_observed") is not True
        or learning_evidence.get("completion_tokens_present_every_step") is not True
        or not isinstance(acceptance, Mapping)
        or acceptance.get("schema_version") != V3_ACCEPTANCE_SCHEMA
        or acceptance.get("mode") != expected_mode
    ):
        raise LearningEvidenceError("v3 optimizer evidence is not terminally acceptable")
    if (
        type(expected_lora_pair_count) is not int
        or expected_lora_pair_count < 1
        or adapter_evidence.get("schema_version") != "1.0"
        or adapter_evidence.get("format") != "maxtext-orbax-lora-tree"
        or adapter_evidence.get("rank") != expected_rank
        or adapter_evidence.get("lora_pair_count") != expected_lora_pair_count
        or adapter_evidence.get("lora_tensor_count") != expected_lora_pair_count * 2
        or adapter_evidence.get("optimizer_lora_tensor_count")
        != expected_lora_pair_count * 4
        or adapter_evidence.get("payload_arrays_restored") is not True
        or adapter_evidence.get("restored_lora_array_count")
        != expected_lora_pair_count * 6
        or adapter_evidence.get("expected_lora_pair_count") != expected_lora_pair_count
        or adapter_evidence.get("checkpoint_step") != expected_steps - 1
        or adapter_evidence.get("checkpoint_step_binding") != "directory-name-plus-root-step-leaf"
        or adapter_evidence.get("approved_maxtext_patch_sha256") != approved_maxtext_patch_sha256
        or not isinstance(adapter_evidence.get("metadata_sha256"), str)
        or _SHA256.fullmatch(adapter_evidence["metadata_sha256"]) is None
        or not isinstance(adapter_evidence.get("pairs"), list)
        or len(adapter_evidence["pairs"]) != expected_lora_pair_count
    ):
        raise LearningEvidenceError("v3 terminal adapter checkpoint proof is incomplete")
    if not smoke:
        progression = checkpoint_progression_evidence
        initial_adapter = (
            progression.get("initial_adapter") if isinstance(progression, Mapping) else None
        )
        if (
            not isinstance(progression, Mapping)
            or progression.get("schema_version")
            != "bookforge-jax-lora-checkpoint-progression-v1"
            or progression.get("status") != "passed"
            or progression.get("comparison_dtype") != "float32"
            or progression.get("accumulation_dtype") != "float64"
            or progression.get("initial_step") != 0
            or progression.get("terminal_step") != expected_steps - 1
            or progression.get("model_lora_array_count") != expected_lora_pair_count * 2
            or type(progression.get("model_lora_element_count")) is not int
            or progression["model_lora_element_count"] < expected_lora_pair_count * 2
            or type(progression.get("changed_model_lora_array_count")) is not int
            or progression["changed_model_lora_array_count"] < 1
            or progression["changed_model_lora_array_count"]
            > progression["model_lora_array_count"]
            or type(progression.get("changed_model_lora_element_count")) is not int
            or progression["changed_model_lora_element_count"] < 1
            or progression["changed_model_lora_element_count"]
            > progression["model_lora_element_count"]
            or type(progression.get("initial_model_lora_l2_norm")) not in (int, float)
            or not math.isfinite(float(progression["initial_model_lora_l2_norm"]))
            or float(progression["initial_model_lora_l2_norm"]) < 0
            or type(progression.get("checkpoint_delta_l2_norm")) not in (int, float)
            or not math.isfinite(float(progression["checkpoint_delta_l2_norm"]))
            or float(progression["checkpoint_delta_l2_norm"]) <= 0
            or type(progression.get("checkpoint_relative_delta")) not in (int, float)
            or not math.isfinite(float(progression["checkpoint_relative_delta"]))
            or float(progression["checkpoint_relative_delta"])
            < float(acceptance["thresholds"]["minimum_checkpoint_lora_relative_delta"])
            or progression.get("minimum_checkpoint_relative_delta")
            != acceptance["thresholds"]["minimum_checkpoint_lora_relative_delta"]
            or not isinstance(initial_adapter, Mapping)
            or initial_adapter.get("schema_version") != "1.0"
            or initial_adapter.get("format") != "maxtext-orbax-lora-tree"
            or initial_adapter.get("rank") != expected_rank
            or initial_adapter.get("lora_pair_count") != expected_lora_pair_count
            or initial_adapter.get("lora_tensor_count") != expected_lora_pair_count * 2
            or initial_adapter.get("optimizer_lora_tensor_count")
            != expected_lora_pair_count * 4
            or initial_adapter.get("payload_arrays_restored") is not True
            or initial_adapter.get("restored_lora_array_count")
            != expected_lora_pair_count * 6
            or initial_adapter.get("expected_lora_pair_count") != expected_lora_pair_count
            or initial_adapter.get("checkpoint_step") != 0
            or initial_adapter.get("checkpoint_step_binding")
            != "directory-name-plus-root-step-leaf"
            or initial_adapter.get("approved_maxtext_patch_sha256")
            != approved_maxtext_patch_sha256
            or not isinstance(initial_adapter.get("metadata_sha256"), str)
            or _SHA256.fullmatch(initial_adapter["metadata_sha256"]) is None
            or not isinstance(initial_adapter.get("pairs"), list)
            or len(initial_adapter["pairs"]) != expected_lora_pair_count
            or progression.get("terminal_adapter") != adapter_evidence
        ):
            raise LearningEvidenceError("v3 checkpoint progression proof is incomplete")
    return {
        "schema_version": V3_ACCEPTANCE_SCHEMA,
        "status": "passed",
        "mode": expected_mode,
        "optimizer_steps": expected_steps,
        "expected_lora_pair_count": expected_lora_pair_count,
        "expected_optimizer_lora_tensor_count": expected_lora_pair_count * 4,
        "approved_maxtext_patch_sha256": approved_maxtext_patch_sha256,
        "adapter_metadata_sha256": adapter_evidence["metadata_sha256"],
        "learning_evidence": acceptance,
        "checkpoint_progression": (
            None
            if smoke
            else {
                "schema_version": checkpoint_progression_evidence["schema_version"],
                "initial_step": checkpoint_progression_evidence["initial_step"],
                "terminal_step": checkpoint_progression_evidence["terminal_step"],
                "model_lora_array_count": checkpoint_progression_evidence[
                    "model_lora_array_count"
                ],
                "changed_model_lora_array_count": checkpoint_progression_evidence[
                    "changed_model_lora_array_count"
                ],
                "changed_model_lora_element_count": checkpoint_progression_evidence[
                    "changed_model_lora_element_count"
                ],
                "checkpoint_delta_l2_norm": checkpoint_progression_evidence[
                    "checkpoint_delta_l2_norm"
                ],
                "checkpoint_relative_delta": checkpoint_progression_evidence[
                    "checkpoint_relative_delta"
                ],
            }
        ),
    }
