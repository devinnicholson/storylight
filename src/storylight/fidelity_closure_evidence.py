"""Strict schemas for terminal deployment and no-spend campaign closure."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_COST_SOURCE_PRODUCERS = {
    "gcp": "storylight-gcp-cost-source",
    "modal": "storylight-modal-cost-source",
}
_RESOURCE_SOURCE_PRODUCERS = {
    "vertex_jobs": "storylight-vertex-jobs-snapshot",
    "cloud_run_services": "storylight-cloud-run-services-snapshot",
    "modal_tasks": "storylight-modal-tasks-snapshot",
    "modal_functions": "storylight-modal-functions-snapshot",
}


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def build_cost_reconciliation(
    run_id: str,
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> dict[str, object]:
    if set(sources) != set(_COST_SOURCE_PRODUCERS):
        raise ValueError("cost reconciliation requires exact GCP and Modal sources")
    providers: dict[str, object] = {}
    amounts: list[float] = []
    for provider, expected_producer in _COST_SOURCE_PRODUCERS.items():
        report, source_sha256 = sources[provider]
        _sha(source_sha256, f"{provider} cost source")
        if (
            set(report)
            != {
                "schema_version",
                "producer",
                "status",
                "run_id",
                "provider",
                "currency",
                "gross_cost_usd",
            }
            or report.get("schema_version") != "story-fidelity-provider-cost-v1"
            or report.get("producer") != expected_producer
            or report.get("status") != "final"
            or report.get("run_id") != run_id
            or report.get("provider") != provider
            or report.get("currency") != "USD"
        ):
            raise ValueError(f"{provider} cost source is not authoritative for this run")
        amount = report.get("gross_cost_usd")
        if (
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not math.isfinite(float(amount))
            or amount < 0
        ):
            raise ValueError(f"{provider} gross cost is invalid")
        amounts.append(float(amount))
        providers[provider] = {
            "source_sha256": source_sha256,
            "gross_cost_usd": amount,
            "reconciled": True,
        }
    return {
        "schema_version": "story-fidelity-cost-reconciliation-v1",
        "producer": "storylight-fidelity-cost-reconciler",
        "status": "reconciled",
        "run_id": run_id,
        "currency": "USD",
        "gross_cost_usd": math.fsum(amounts),
        "providers": providers,
    }


def build_paid_resource_inventory(
    run_id: str,
    sources: Mapping[str, tuple[Mapping[str, object], str]],
) -> dict[str, object]:
    if set(sources) != set(_RESOURCE_SOURCE_PRODUCERS):
        raise ValueError("resource inventory requires every supported provider snapshot")
    source_snapshots: dict[str, str] = {}
    counts: dict[str, int] = {}
    for resource, expected_producer in _RESOURCE_SOURCE_PRODUCERS.items():
        snapshot, source_sha256 = sources[resource]
        _sha(source_sha256, f"{resource} source snapshot")
        if (
            set(snapshot)
            != {
                "schema_version",
                "producer",
                "status",
                "run_id",
                "resource_class",
                "active_paid_resources",
            }
            or snapshot.get("schema_version")
            != "story-fidelity-resource-snapshot-v1"
            or snapshot.get("producer") != expected_producer
            or snapshot.get("status") != "observed"
            or snapshot.get("run_id") != run_id
            or snapshot.get("resource_class") != resource
            or type(snapshot.get("active_paid_resources")) is not int
            or snapshot.get("active_paid_resources") != 0
        ):
            raise ValueError(f"{resource} snapshot is not a trusted zero-resource report")
        source_snapshots[resource] = source_sha256
        counts[resource] = 0
    return {
        "schema_version": "story-fidelity-paid-resource-inventory-v1",
        "producer": "storylight-fidelity-resource-reconciler",
        "status": "zero-active-paid-resources",
        "run_id": run_id,
        "source_snapshots": source_snapshots,
        "counts": counts,
        "active_paid_resources": 0,
    }


def validate_terminal_evidence(
    receipt: Mapping[str, object],
    health: Mapping[str, object],
    rollback: Mapping[str, object] | None,
    *,
    run_id: str,
    training_run_id: str,
    candidate_id: str,
    candidate_manifest_sha256: str,
    gate_artifact_sha256: str,
    baseline_engine_sha256: str,
    candidate_engine_sha256: str,
    promoted: bool,
) -> None:
    outcome = "promoted" if promoted else "retained"
    producer = (
        "storylight-trained-planner-promotion"
        if promoted
        else "storylight-baseline-retention"
    )
    receipt_fields = {
        "schema_version",
        "producer",
        "status",
        "outcome",
        "run_id",
        "training_run_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "gate_artifact_sha256",
        "active_engine_sha256",
        "accepted_baseline_engine_sha256",
        "one_purpose_approval_sha256",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema_version") != "story-fidelity-terminal-receipt-v1"
        or receipt.get("producer") != producer
        or receipt.get("status") != "succeeded"
        or receipt.get("outcome") != outcome
        or receipt.get("run_id") != run_id
        or receipt.get("training_run_id") != training_run_id
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or receipt.get("gate_artifact_sha256") != gate_artifact_sha256
        or receipt.get("accepted_baseline_engine_sha256") != baseline_engine_sha256
    ):
        raise ValueError("terminal deployment receipt is not authoritative for this gate")
    active_engine = _sha(receipt.get("active_engine_sha256"), "active engine")
    _sha(receipt.get("one_purpose_approval_sha256"), "one-purpose approval")
    expected_engine = candidate_engine_sha256 if promoted else baseline_engine_sha256
    if active_engine != expected_engine:
        if promoted:
            raise ValueError("promotion receipt did not activate the gated candidate engine")
        raise ValueError("retention receipt did not preserve the accepted baseline engine")

    checks = health.get("checks")
    observations = health.get("observations")
    health_fields = {
        "schema_version",
        "producer",
        "status",
        "outcome",
        "run_id",
        "training_run_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "gate_artifact_sha256",
        "active_engine_sha256",
        "active_model_revision",
        "checks",
        "observations",
    }
    required_checks = {
        "planner_ready",
        "api_ready",
        "controller_ready",
        "kiosk_active",
        "live_scene_passed",
        "output_tokens_bounded",
        "swap_disabled",
    }
    if (
        set(health) != health_fields
        or health.get("schema_version") != "story-fidelity-post-action-health-v1"
        or health.get("producer") != "storylight-terminal-health-recorder"
        or health.get("status") != "passed"
        or health.get("outcome") != outcome
        or health.get("run_id") != run_id
        or health.get("training_run_id") != training_run_id
        or health.get("candidate_id") != candidate_id
        or health.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or health.get("gate_artifact_sha256") != gate_artifact_sha256
        or health.get("active_engine_sha256") != active_engine
        or health.get("active_model_revision") != f"sha256:{active_engine}"
        or not isinstance(checks, dict)
        or set(checks) != required_checks
        or not all(value is True for value in checks.values())
        or not isinstance(observations, dict)
        or set(observations) != {"maximum_output_tokens"}
        or type(observations.get("maximum_output_tokens")) is not int
        or not 1 <= observations["maximum_output_tokens"] <= 64
    ):
        raise ValueError("post-action health evidence is incomplete or failed")

    if promoted:
        rollback_fields = {
            "schema_version",
            "producer",
            "status",
            "run_id",
            "training_run_id",
            "candidate_id",
            "candidate_manifest_sha256",
            "gate_artifact_sha256",
            "accepted_baseline_engine_sha256",
            "backup_config_sha256",
            "restorable",
        }
        if (
            not isinstance(rollback, Mapping)
            or set(rollback) != rollback_fields
            or rollback.get("schema_version") != "story-fidelity-rollback-state-v1"
            or rollback.get("producer") != "storylight-trained-planner-rollback-state"
            or rollback.get("status") != "ready"
            or rollback.get("run_id") != run_id
            or rollback.get("training_run_id") != training_run_id
            or rollback.get("candidate_id") != candidate_id
            or rollback.get("candidate_manifest_sha256") != candidate_manifest_sha256
            or rollback.get("gate_artifact_sha256") != gate_artifact_sha256
            or rollback.get("accepted_baseline_engine_sha256")
            != baseline_engine_sha256
            or rollback.get("restorable") is not True
        ):
            raise ValueError("promotion rollback state is incomplete or not restorable")
        _sha(rollback.get("backup_config_sha256"), "rollback backup config")
    elif rollback is not None:
        raise ValueError("baseline retention must not carry promotion rollback state")


def validate_reconciliation_evidence(
    cost: Mapping[str, object],
    inventory: Mapping[str, object],
    *,
    run_id: str,
) -> None:
    providers = cost.get("providers")
    if (
        set(cost)
        != {
            "schema_version",
            "producer",
            "status",
            "run_id",
            "currency",
            "gross_cost_usd",
            "providers",
        }
        or cost.get("schema_version") != "story-fidelity-cost-reconciliation-v1"
        or cost.get("producer") != "storylight-fidelity-cost-reconciler"
        or cost.get("status") != "reconciled"
        or cost.get("run_id") != run_id
        or cost.get("currency") != "USD"
        or not isinstance(providers, dict)
        or set(providers) != {"gcp", "modal"}
    ):
        raise ValueError("cost reconciliation is not a trusted campaign ledger")
    total = 0.0
    for name, report in providers.items():
        if (
            not isinstance(report, dict)
            or set(report) != {"source_sha256", "gross_cost_usd", "reconciled"}
            or report.get("reconciled") is not True
        ):
            raise ValueError(f"{name} cost reconciliation is malformed")
        _sha(report.get("source_sha256"), f"{name} cost source")
        amount = report.get("gross_cost_usd")
        if (
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not math.isfinite(float(amount))
            or amount < 0
        ):
            raise ValueError(f"{name} gross cost is invalid")
        total += float(amount)
    gross = cost.get("gross_cost_usd")
    if (
        not isinstance(gross, (int, float))
        or isinstance(gross, bool)
        or not math.isfinite(float(gross))
        or not math.isclose(float(gross), total, abs_tol=0.000001)
    ):
        raise ValueError("gross cost does not equal the checksum-bound provider reports")

    snapshots = inventory.get("source_snapshots")
    counts = inventory.get("counts")
    resources = {"vertex_jobs", "cloud_run_services", "modal_tasks", "modal_functions"}
    if (
        set(inventory)
        != {
            "schema_version",
            "producer",
            "status",
            "run_id",
            "source_snapshots",
            "counts",
            "active_paid_resources",
        }
        or inventory.get("schema_version")
        != "story-fidelity-paid-resource-inventory-v1"
        or inventory.get("producer") != "storylight-fidelity-resource-reconciler"
        or inventory.get("status") != "zero-active-paid-resources"
        or inventory.get("run_id") != run_id
        or type(inventory.get("active_paid_resources")) is not int
        or inventory.get("active_paid_resources") != 0
        or not isinstance(snapshots, dict)
        or set(snapshots) != resources
        or not isinstance(counts, dict)
        or set(counts) != resources
        or any(type(value) is not int or value != 0 for value in counts.values())
    ):
        raise ValueError("paid-resource inventory is not a trusted zero-resource snapshot")
    for resource, digest in snapshots.items():
        _sha(digest, f"{resource} source snapshot")
