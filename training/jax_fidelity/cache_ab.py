"""Validate and summarize a cold-versus-warm JAX cache experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

CACHE_AB_SCHEMA = "bookforge-jax-cache-ab-v1"
CACHE_RECEIPT_SCHEMA = "bookforge-jax-compilation-cache-v1"


def _load_object(path: Path) -> dict[str, Any]:
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


def _duration_seconds(run: dict[str, Any], completion: dict[str, Any]) -> float:
    started = _timestamp(run.get("created_at"), "run.created_at")
    completed = _timestamp(completion.get("completed_at"), "completion.completed_at")
    duration = (completed - started).total_seconds()
    if duration <= 0:
        raise ValueError("completion must occur after run creation")
    return duration


def _learning_fingerprint(completion: dict[str, Any]) -> dict[str, Any]:
    evidence = completion.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("training completion has no evidence object")
    learning = evidence.get("learning")
    if not isinstance(learning, dict) or learning.get("status") != "passed":
        raise ValueError("training completion has no passed learning evidence")
    learning = dict(learning)
    learning.pop("event_stream", None)
    progression = evidence.get("checkpoint_progression")
    acceptance = evidence.get("learnability_acceptance")
    if not isinstance(progression, dict) or not isinstance(acceptance, dict):
        raise ValueError("training completion lacks checkpoint acceptance evidence")
    return {
        "learning": learning,
        "checkpoint_progression": progression,
        "learnability_acceptance": acceptance,
    }


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


def _validate_cache_pair(cold: dict[str, Any], warm: dict[str, Any]) -> None:
    for label, receipt in (("cold", cold), ("warm", warm)):
        if (
            receipt.get("schema_version") != CACHE_RECEIPT_SCHEMA
            or receipt.get("status") != "complete"
        ):
            raise ValueError(f"{label} cache receipt is not complete")
    identity_fields = ("cache_directory", "owner_sha256", "environment", "maxtext_entrypoint")
    for field in identity_fields:
        if cold.get(field) != warm.get(field):
            raise ValueError(f"cache identity changed between runs: {field}")
    if cold.get("cache_was_warm") is not False:
        raise ValueError("cold run did not start from a cold cache")
    if warm.get("cache_was_warm") is not True:
        raise ValueError("warm run did not report cache reuse")
    if cold.get("before") != {"bytes": 0, "files": 0}:
        raise ValueError("cold run cache was not empty")
    if cold.get("after") != warm.get("before") or warm.get("before") != warm.get("after"):
        raise ValueError("warm cache inventory does not exactly continue the cold run")
    if int(cold.get("added_files", 0)) <= 0 or int(cold.get("added_bytes", 0)) <= 0:
        raise ValueError("cold run populated no persistent MaxText cache entries")
    if warm.get("added_files") != 0 or warm.get("added_bytes") != 0:
        raise ValueError("warm run wrote unexpected cache entries")


def compare_cache_runs(
    *,
    cold_run_path: Path,
    cold_completion_path: Path,
    cold_cache_path: Path,
    cold_billing_path: Path,
    cold_app_id: str,
    warm_run_path: Path,
    warm_completion_path: Path,
    warm_cache_path: Path,
    warm_billing_path: Path,
    warm_app_id: str,
) -> dict[str, Any]:
    """Return a fail-closed A/B receipt from two trusted training releases."""

    cold_run = _load_object(cold_run_path)
    warm_run = _load_object(warm_run_path)
    cold_completion = _load_object(cold_completion_path)
    warm_completion = _load_object(warm_completion_path)
    cold_cache = _load_object(cold_cache_path)
    warm_cache = _load_object(warm_cache_path)

    for label, completion in (("cold", cold_completion), ("warm", warm_completion)):
        if completion.get("status") != "succeeded":
            raise ValueError(f"{label} training completion did not succeed")
    invariant_fields = ("run_id", "config_sha256", "dataset_manifest_sha256", "stage", "metadata")
    for field in invariant_fields:
        if cold_run.get(field) != warm_run.get(field):
            raise ValueError(f"training input changed between runs: {field}")
    if _learning_fingerprint(cold_completion) != _learning_fingerprint(warm_completion):
        raise ValueError("training semantics changed between cold and warm runs")
    _validate_cache_pair(cold_cache, warm_cache)

    cold_duration = _duration_seconds(cold_run, cold_completion)
    warm_duration = _duration_seconds(warm_run, warm_completion)
    cold_cost = _billing_cost(cold_billing_path, cold_app_id)
    warm_cost = _billing_cost(warm_billing_path, warm_app_id)
    if warm_duration >= cold_duration:
        raise ValueError("warm cache did not improve end-to-end training duration")
    if warm_cost >= cold_cost:
        raise ValueError("warm cache did not reduce authoritative provider cost")

    progression = cold_completion["evidence"]["checkpoint_progression"]
    after = cold_cache["after"]
    return {
        "schema_version": CACHE_AB_SCHEMA,
        "status": "passed",
        "invariants": {
            "training_run_id": cold_run["run_id"],
            "config_sha256": cold_run["config_sha256"],
            "dataset_manifest_sha256": cold_run["dataset_manifest_sha256"],
            "inputs": cold_run["metadata"]["inputs"],
            "optimizer_steps": cold_completion["evidence"]["learning"]["optimizer_steps"],
            "changed_model_lora_array_count": progression["changed_model_lora_array_count"],
            "changed_model_lora_element_count": progression["changed_model_lora_element_count"],
            "checkpoint_relative_delta": progression["checkpoint_relative_delta"],
            "learning_evidence_identical": True,
        },
        "cache": {
            "directory": cold_cache["cache_directory"],
            "files": after["files"],
            "bytes": after["bytes"],
            "cold_added_files": cold_cache["added_files"],
            "warm_added_files": warm_cache["added_files"],
            "warm_inventory_unchanged": True,
        },
        "cold": {
            "provider_app_id": cold_app_id,
            "duration_seconds": cold_duration,
            "provider_cost_usd": format(cold_cost, "f"),
        },
        "warm": {
            "provider_app_id": warm_app_id,
            "duration_seconds": warm_duration,
            "provider_cost_usd": format(warm_cost, "f"),
        },
        "improvement": {
            "duration_seconds": cold_duration - warm_duration,
            "duration_fraction": float(
                (Decimal(str(cold_duration)) - Decimal(str(warm_duration)))
                / Decimal(str(cold_duration))
            ),
            "provider_cost_usd": format(cold_cost - warm_cost, "f"),
            "provider_cost_fraction": float((cold_cost - warm_cost) / cold_cost),
        },
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("cold_run", cold_run_path),
                ("cold_completion", cold_completion_path),
                ("cold_cache", cold_cache_path),
                ("cold_billing", cold_billing_path),
                ("warm_run", warm_run_path),
                ("warm_completion", warm_completion_path),
                ("warm_cache", warm_cache_path),
                ("warm_billing", warm_billing_path),
            )
        },
    }
