"""Validate rematerialization performance experiments from trusted releases."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from .configuration import validate_config

REMAT_AB_SCHEMA = "bookforge-jax-remat-ab-v1"
_RELEASE_VOLUME = "bookforge-jax-fidelity-release"
_STEP_TIME_TAG = "perf/step_time_seconds"
_TOKENS_PER_SECOND_TAG = "perf/per_device_tokens_per_sec"


@dataclass(frozen=True)
class RunEvidence:
    """Local receipts for one immutable provider run."""

    label: str
    release_directory: Path
    billing_path: Path
    provider_app_id: str


EventLoader = Callable[[Path], dict[str, list[tuple[int, float, float]]]]


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _billing_cost(path: Path, app_id: str) -> Decimal:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Modal billing report must be a JSON array")
    matches = [row for row in rows if isinstance(row, dict) and row.get("Object ID") == app_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one billing row for {app_id}")
    cost = Decimal(str(matches[0].get("Cost")))
    if cost <= 0:
        raise ValueError("provider cost must be positive")
    return cost


def load_tensorboard_events(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Load only the two performance streams used by the comparison."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - exercised only without the analysis extra
        raise RuntimeError("install the jax-analysis extra to read TensorBoard events") from exc

    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    result: dict[str, list[tuple[int, float, float]]] = {}
    for tag in (_STEP_TIME_TAG, _TOKENS_PER_SECOND_TAG):
        if tag not in accumulator.Tags().get("scalars", []):
            raise ValueError(f"TensorBoard event stream is missing {tag}")
        result[tag] = [
            (sample.step, sample.wall_time, sample.value) for sample in accumulator.Scalars(tag)
        ]
    return result


def _validate_configs(baseline_path: Path, candidate_path: Path) -> tuple[int, str, str]:
    baseline = _json_object(baseline_path)
    candidate = _json_object(candidate_path)
    validate_config(baseline)
    validate_config(candidate)
    baseline_policy = baseline["training"].get("remat_policy", "full")
    candidate_policy = candidate["training"].get("remat_policy", "full")
    if baseline_policy != "full" or candidate_policy != "none":
        raise ValueError("comparison requires full-remat baseline and no-remat candidate")
    normalized_baseline = deepcopy(baseline)
    normalized_candidate = deepcopy(candidate)
    normalized_baseline["training"].pop("remat_policy", None)
    normalized_candidate["training"].pop("remat_policy", None)
    if normalized_baseline != normalized_candidate:
        raise ValueError("experiment configuration changed beyond remat_policy")
    return int(baseline["training"]["steps"]), _sha256(baseline_path), _sha256(candidate_path)


def _series_summary(
    events: dict[str, list[tuple[int, float, float]]],
    *,
    expected_steps: int,
) -> dict[str, float | int]:
    expected = list(range(expected_steps))
    by_tag: dict[str, list[tuple[int, float, float]]] = {}
    for tag in (_STEP_TIME_TAG, _TOKENS_PER_SECOND_TAG):
        samples = events.get(tag)
        if not isinstance(samples, list) or [sample[0] for sample in samples] != expected:
            raise ValueError(f"{tag} must contain every optimizer step exactly once")
        if any(value <= 0 for _, _, value in samples):
            raise ValueError(f"{tag} contains a non-positive value")
        if any(right[1] < left[1] for left, right in zip(samples, samples[1:], strict=False)):
            raise ValueError(f"{tag} wall times are not monotonic")
        by_tag[tag] = samples

    trim = max(1, expected_steps // 10)
    stable_start = trim
    stable_end = expected_steps - trim
    if stable_end - stable_start < 3:
        raise ValueError("run is too short for a stable performance window")
    step_times = [
        value for step, _, value in by_tag[_STEP_TIME_TAG] if stable_start <= step < stable_end
    ]
    tokens = [
        value
        for step, _, value in by_tag[_TOKENS_PER_SECOND_TAG]
        if stable_start <= step < stable_end
    ]
    ordered_step_times = sorted(step_times)
    p90_index = math.ceil(0.9 * len(ordered_step_times)) - 1
    return {
        "first_metric_wall_time": by_tag[_STEP_TIME_TAG][0][1],
        "stable_step_start": stable_start,
        "stable_step_end_exclusive": stable_end,
        "stable_step_samples": len(step_times),
        "median_step_time_seconds": median(step_times),
        "p90_step_time_seconds": ordered_step_times[p90_index],
        "median_tokens_per_second_per_gpu": median(tokens),
    }


def _summarize_run(
    evidence: RunEvidence,
    *,
    expected_config_sha256: str,
    expected_steps: int,
    event_loader: EventLoader,
) -> dict[str, Any]:
    run_path = evidence.release_directory / "training-run.json"
    completion_path = evidence.release_directory / "training-completion.json"
    provider_completion_path = evidence.release_directory / "completion.json"
    event_path = evidence.release_directory / "events.tfevents"
    run = _json_object(run_path)
    completion = _json_object(completion_path)
    provider_completion = _json_object(provider_completion_path)
    provider_run_id = provider_completion.get("run_id")

    if run.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"{evidence.label} run used an unexpected configuration")
    if completion.get("status") != "succeeded" or completion.get("run_id") != run.get("run_id"):
        raise ValueError(f"{evidence.label} training completion is not trusted")
    if completion.get("run_manifest_sha256") != _sha256(run_path):
        raise ValueError(f"{evidence.label} training run hash changed")
    portable_package = provider_completion.get("portable_package")
    provider_files = provider_completion.get("files")
    completion_file = None
    if isinstance(provider_files, list):
        completion_file = next(
            (
                row
                for row in provider_files
                if isinstance(row, dict) and row.get("path") == "training/completion.json"
            ),
            None,
        )
    if (
        provider_completion.get("status") != "succeeded"
        or not isinstance(provider_run_id, str)
        or not provider_run_id
        or provider_completion.get("training_run_id") != run.get("run_id")
        or provider_completion.get("config_sha256") != expected_config_sha256
        or provider_completion.get("source_training_completion_sha256")
        != completion.get("source_training_completion_sha256")
        or not isinstance(portable_package, dict)
        or portable_package.get("training_run_sha256") != _sha256(run_path)
        or portable_package.get("training_completion_sha256") != _sha256(completion_path)
        or not isinstance(completion_file, dict)
        or completion_file.get("sha256") != _sha256(completion_path)
    ):
        raise ValueError(f"{evidence.label} provider completion is not trusted")

    evidence_document = completion.get("evidence")
    if not isinstance(evidence_document, dict):
        raise ValueError(f"{evidence.label} completion has no evidence")
    learning = evidence_document.get("learning")
    acceptance = evidence_document.get("learnability_acceptance")
    progression = evidence_document.get("checkpoint_progression")
    if (
        not isinstance(learning, dict)
        or learning.get("status") != "passed"
        or learning.get("optimizer_steps") != expected_steps
        or not isinstance(acceptance, dict)
        or acceptance.get("status") != "passed"
        or not isinstance(progression, dict)
        or progression.get("changed_model_lora_array_count")
        != progression.get("model_lora_array_count")
        or float(progression.get("checkpoint_relative_delta", 0)) <= 0
    ):
        raise ValueError(f"{evidence.label} did not prove successful learning")
    event_receipt = learning.get("event_stream")
    if not isinstance(event_receipt, dict) or event_receipt.get("sha256") != _sha256(event_path):
        raise ValueError(f"{evidence.label} TensorBoard stream hash changed")

    created_at = _timestamp(run.get("created_at"), "run.created_at")
    completed_at = _timestamp(completion.get("completed_at"), "completion.completed_at")
    duration = (completed_at - created_at).total_seconds()
    if duration <= 0:
        raise ValueError(f"{evidence.label} completion precedes its run")
    performance = _series_summary(event_loader(event_path), expected_steps=expected_steps)
    startup_seconds = float(performance.pop("first_metric_wall_time")) - created_at.timestamp()
    if startup_seconds <= 0:
        raise ValueError(f"{evidence.label} first metric precedes its run")

    source_paths = {
        "training_run": run_path,
        "training_completion": completion_path,
        "provider_completion": provider_completion_path,
        "tensorboard_events": event_path,
        "billing_report": evidence.billing_path,
    }
    return {
        "label": evidence.label,
        "provider_app_id": evidence.provider_app_id,
        "provider_release": {
            "volume": _RELEASE_VOLUME,
            "path": provider_run_id,
        },
        "training_run_id": run["run_id"],
        "config_sha256": expected_config_sha256,
        "dataset_manifest_sha256": run["dataset_manifest_sha256"],
        "inputs": run["metadata"]["inputs"],
        "optimizer_steps": expected_steps,
        "startup_to_first_metric_seconds": startup_seconds,
        "end_to_end_duration_seconds": duration,
        "provider_cost_usd": format(
            _billing_cost(evidence.billing_path, evidence.provider_app_id), "f"
        ),
        "checkpoint_relative_delta": progression["checkpoint_relative_delta"],
        "changed_model_lora_array_count": progression["changed_model_lora_array_count"],
        "performance": performance,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }


def compare_remat_runs(
    *,
    baseline_config_path: Path,
    candidate_config_path: Path,
    baseline: RunEvidence,
    candidates: Sequence[RunEvidence],
    safety_factor: float = 1.1,
    event_loader: EventLoader = load_tensorboard_events,
) -> dict[str, Any]:
    """Compare full remat with repeated no-remat runs and select a safe policy."""

    if len(candidates) < 2:
        raise ValueError("at least two no-remat repetitions are required")
    if not 1 <= safety_factor <= 2:
        raise ValueError("safety_factor must be in [1, 2]")
    expected_steps, baseline_sha, candidate_sha = _validate_configs(
        baseline_config_path, candidate_config_path
    )
    baseline_result = _summarize_run(
        baseline,
        expected_config_sha256=baseline_sha,
        expected_steps=expected_steps,
        event_loader=event_loader,
    )
    candidate_results = [
        _summarize_run(
            candidate,
            expected_config_sha256=candidate_sha,
            expected_steps=expected_steps,
            event_loader=event_loader,
        )
        for candidate in candidates
    ]
    for candidate in candidate_results:
        for field in ("dataset_manifest_sha256", "inputs", "optimizer_steps"):
            if candidate[field] != baseline_result[field]:
                raise ValueError(f"training input changed for {candidate['label']}: {field}")
        if (
            candidate["changed_model_lora_array_count"]
            != baseline_result["changed_model_lora_array_count"]
        ):
            raise ValueError(f"LoRA trainable coverage changed for {candidate['label']}")

    baseline_step = baseline_result["performance"]["median_step_time_seconds"]
    break_even_steps: list[float] = []
    for candidate in candidate_results:
        candidate_step = candidate["performance"]["median_step_time_seconds"]
        saved_per_step = baseline_step - candidate_step
        if saved_per_step <= 0:
            raise ValueError(f"no-remat did not improve steady-state speed: {candidate['label']}")
        startup_penalty = max(
            0.0,
            candidate["startup_to_first_metric_seconds"]
            - baseline_result["startup_to_first_metric_seconds"],
        )
        break_even = startup_penalty / saved_per_step
        break_even_steps.append(break_even)
        candidate["comparison"] = {
            "steady_step_time_reduction_fraction": (baseline_step - candidate_step) / baseline_step,
            "startup_penalty_seconds": startup_penalty,
            "break_even_optimizer_steps": break_even,
        }

    conservative_break_even = max(break_even_steps)
    minimum_no_remat_steps = math.ceil(conservative_break_even * safety_factor)
    observed_steps = [
        candidate["performance"]["median_step_time_seconds"] for candidate in candidate_results
    ]
    selected_policy = "none" if expected_steps >= minimum_no_remat_steps else "full"
    return {
        "schema_version": REMAT_AB_SCHEMA,
        "status": "passed",
        "decision": {
            "canary_optimizer_steps": expected_steps,
            "selected_canary_remat_policy": selected_policy,
            "minimum_optimizer_steps_for_no_remat": minimum_no_remat_steps,
            "conservative_break_even_optimizer_steps": conservative_break_even,
            "safety_factor": safety_factor,
            "reason": (
                "full rematerialization minimizes end-to-end latency for the bounded canary; "
                "no rematerialization is reserved for longer runs"
            ),
        },
        "host_variance": {
            "no_remat_median_step_time_ratio": max(observed_steps) / min(observed_steps),
            "repetitions": len(candidate_results),
        },
        "baseline": baseline_result,
        "candidates": candidate_results,
        "configuration_sources": {
            "baseline": {
                "path": str(baseline_config_path),
                "sha256": baseline_sha,
                "remat_policy": "full",
            },
            "candidate": {
                "path": str(candidate_config_path),
                "sha256": candidate_sha,
                "remat_policy": "none",
            },
        },
    }
