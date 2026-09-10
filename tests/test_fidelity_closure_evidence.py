from __future__ import annotations

import copy

import pytest

from storylight.fidelity_closure_evidence import (
    build_cost_reconciliation,
    build_paid_resource_inventory,
    validate_reconciliation_evidence,
    validate_terminal_evidence,
)


def _terminal(*, promoted: bool = True):
    engine = "8" * 64 if promoted else "9" * 64
    outcome = "promoted" if promoted else "retained"
    producer = (
        "storylight-trained-planner-promotion"
        if promoted
        else "storylight-baseline-retention"
    )
    common = {
        "run_id": "campaign-v1",
        "training_run_id": "lora-train-abc",
        "candidate_id": "candidate-v1",
        "candidate_manifest_sha256": "7" * 64,
        "gate_artifact_sha256": "6" * 64,
    }
    receipt = {
        "schema_version": "story-fidelity-terminal-receipt-v1",
        "producer": producer,
        "status": "succeeded",
        "outcome": outcome,
        **common,
        "active_engine_sha256": engine,
        "accepted_baseline_engine_sha256": "9" * 64,
        "one_purpose_approval_sha256": "5" * 64,
    }
    health = {
        "schema_version": "story-fidelity-post-action-health-v1",
        "producer": "storylight-terminal-health-recorder",
        "status": "passed",
        "outcome": outcome,
        **common,
        "active_engine_sha256": engine,
        "active_model_revision": f"sha256:{engine}",
        "checks": {
            name: True
            for name in (
                "planner_ready",
                "api_ready",
                "controller_ready",
                "kiosk_active",
                "live_scene_passed",
                "output_tokens_bounded",
                "swap_disabled",
            )
        },
        "observations": {"maximum_output_tokens": 64},
    }
    rollback = (
        {
            "schema_version": "story-fidelity-rollback-state-v1",
            "producer": "storylight-trained-planner-rollback-state",
            "status": "ready",
            **common,
            "accepted_baseline_engine_sha256": "9" * 64,
            "backup_config_sha256": "4" * 64,
            "restorable": True,
        }
        if promoted
        else None
    )
    return receipt, health, rollback, common


def _validate_terminal(receipt, health, rollback, common, *, promoted=True):
    validate_terminal_evidence(
        receipt,
        health,
        rollback,
        **common,
        baseline_engine_sha256="9" * 64,
        candidate_engine_sha256="8" * 64,
        promoted=promoted,
    )


def test_terminal_evidence_requires_authoritative_receipt_health_and_rollback() -> None:
    receipt, health, rollback, common = _terminal()
    _validate_terminal(receipt, health, rollback, common)

    tampered = copy.deepcopy(health)
    tampered["checks"]["live_scene_passed"] = False
    with pytest.raises(ValueError, match="health evidence"):
        _validate_terminal(receipt, tampered, rollback, common)

    with pytest.raises(ValueError, match="rollback state"):
        _validate_terminal(receipt, health, {"restorable": True}, common)


def test_retention_requires_the_accepted_baseline_to_remain_active() -> None:
    receipt, health, rollback, common = _terminal(promoted=False)
    _validate_terminal(receipt, health, rollback, common, promoted=False)
    receipt["active_engine_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="accepted baseline"):
        _validate_terminal(receipt, health, rollback, common, promoted=False)


def test_promotion_requires_the_exact_gated_candidate_engine() -> None:
    receipt, health, rollback, common = _terminal()
    receipt["active_engine_sha256"] = "a" * 64
    health["active_engine_sha256"] = "a" * 64
    health["active_model_revision"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="gated candidate engine"):
        _validate_terminal(receipt, health, rollback, common)


def _reconciliation():
    cost = {
        "schema_version": "story-fidelity-cost-reconciliation-v1",
        "producer": "storylight-fidelity-cost-reconciler",
        "status": "reconciled",
        "run_id": "campaign-v1",
        "currency": "USD",
        "gross_cost_usd": 3.5,
        "providers": {
            "gcp": {
                "source_sha256": "1" * 64,
                "gross_cost_usd": 2.0,
                "reconciled": True,
            },
            "modal": {
                "source_sha256": "2" * 64,
                "gross_cost_usd": 1.5,
                "reconciled": True,
            },
        },
    }
    resources = {"vertex_jobs", "cloud_run_services", "modal_tasks", "modal_functions"}
    inventory = {
        "schema_version": "story-fidelity-paid-resource-inventory-v1",
        "producer": "storylight-fidelity-resource-reconciler",
        "status": "zero-active-paid-resources",
        "run_id": "campaign-v1",
        "source_snapshots": {name: "3" * 64 for name in resources},
        "counts": {name: 0 for name in resources},
        "active_paid_resources": 0,
    }
    return cost, inventory


def test_reconciliation_requires_checksum_bound_provider_and_resource_sources() -> None:
    cost, inventory = _reconciliation()
    validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")

    cost["providers"]["modal"]["source_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="source SHA-256"):
        validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")


def test_reconciliation_rejects_nonfinite_costs() -> None:
    cost, inventory = _reconciliation()
    cost["providers"]["modal"]["gross_cost_usd"] = float("inf")
    cost["gross_cost_usd"] = float("inf")
    with pytest.raises(ValueError, match="gross cost"):
        validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")


def test_reconciliation_rejects_false_zero_resource_claim() -> None:
    cost, inventory = _reconciliation()
    inventory["counts"]["vertex_jobs"] = 1
    with pytest.raises(ValueError, match="zero-resource"):
        validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")


def test_trusted_builders_bind_exact_source_digests() -> None:
    cost_sources = {
        provider: (
            {
                "schema_version": "story-fidelity-provider-cost-v1",
                "producer": f"storylight-{provider}-cost-source",
                "status": "final",
                "run_id": "campaign-v1",
                "provider": provider,
                "currency": "USD",
                "gross_cost_usd": amount,
            },
            digest * 64,
        )
        for provider, amount, digest in (("gcp", 2.0, "1"), ("modal", 1.5, "2"))
    }
    cost = build_cost_reconciliation("campaign-v1", cost_sources)

    producers = {
        "vertex_jobs": "storylight-vertex-jobs-snapshot",
        "cloud_run_services": "storylight-cloud-run-services-snapshot",
        "modal_tasks": "storylight-modal-tasks-snapshot",
        "modal_functions": "storylight-modal-functions-snapshot",
    }
    inventory_sources = {
        resource: (
            {
                "schema_version": "story-fidelity-resource-snapshot-v1",
                "producer": producer,
                "status": "observed",
                "run_id": "campaign-v1",
                "resource_class": resource,
                "active_paid_resources": 0,
            },
            "3" * 64,
        )
        for resource, producer in producers.items()
    }
    inventory = build_paid_resource_inventory("campaign-v1", inventory_sources)
    validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")
    assert cost["providers"]["gcp"]["source_sha256"] == "1" * 64

    inventory_sources["vertex_jobs"][0]["active_paid_resources"] = False
    with pytest.raises(ValueError, match="trusted zero-resource"):
        build_paid_resource_inventory("campaign-v1", inventory_sources)

    inventory["counts"]["vertex_jobs"] = False
    with pytest.raises(ValueError, match="zero-resource"):
        validate_reconciliation_evidence(cost, inventory, run_id="campaign-v1")
