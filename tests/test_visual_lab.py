from pathlib import Path

import pytest

from bookforge.visual_lab import (
    BudgetEnvelope,
    GenerationRecord,
    VisualLabBudgetError,
    VisualLabLedger,
)


def envelope() -> BudgetEnvelope:
    return BudgetEnvelope(
        monthly_credit_usd=30.0,
        usage_before_lab_usd=13.33515451,
        reserve_usd=1.66484549,
        run_cap_usd=15.0,
    )


def record(*, experiment_id: str = "master-001", cost: float = 0.05) -> GenerationRecord:
    return GenerationRecord(
        experiment_id=experiment_id,
        stage="master",
        model="sana",
        model_revision="abc123",
        gpu="L4",
        seed=42,
        prompt="A luminous paper forest",
        artifact_path="master-001.png",
        sha256="a" * 64,
        generation_seconds=2.0,
        estimated_gpu_usd=cost,
        width=1024,
        height=576,
    )


def test_budget_rejects_impossible_or_over_cap_work() -> None:
    with pytest.raises(ValueError, match="exceeds remaining"):
        BudgetEnvelope(
            monthly_credit_usd=30,
            usage_before_lab_usd=20,
            reserve_usd=2,
            run_cap_usd=9,
        )

    ledger = VisualLabLedger(envelope=envelope(), records=[record(cost=14.9)])
    with pytest.raises(VisualLabBudgetError, match="would exceed"):
        ledger.reserve(gpu="H100", maximum_seconds=100)


def test_ledger_is_resumable_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = VisualLabLedger(envelope=envelope(), prior_estimated_usd=0.13317055)
    ledger.add(record())
    ledger.write(path)

    resumed = VisualLabLedger.read(path, envelope=envelope())

    assert resumed.records == ledger.records
    assert resumed.prior_estimated_usd == pytest.approx(0.13317055)
    assert resumed.estimated_usage_usd == pytest.approx(0.18317055)
    with pytest.raises(ValueError, match="duplicate"):
        resumed.add(record())


def test_reservation_is_persisted_and_counts_against_cap(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = VisualLabLedger(envelope=envelope(), prior_estimated_usd=14.93)
    estimate = ledger.reserve(
        gpu="L4",
        maximum_seconds=180,
        reservation_id="master:abc",
    )
    ledger.write(path)

    resumed = VisualLabLedger.read(path, envelope=envelope())

    assert resumed.reservations == {"master:abc": pytest.approx(estimate)}
    with pytest.raises(VisualLabBudgetError, match="would exceed"):
        resumed.reserve(gpu="L4", maximum_seconds=180)
    resumed.release("master:abc")
    assert resumed.estimated_usage_usd == pytest.approx(14.93)


def test_reconcile_billed_total_releases_only_named_failed_reservation() -> None:
    ledger = VisualLabLedger(envelope=envelope())
    ledger.reserve(gpu="L4", maximum_seconds=10, reservation_id="failed-score")
    ledger.reserve(gpu="L4", maximum_seconds=10, reservation_id="other")

    ledger.reconcile_billed_total(0.25, release_reservation_id="failed-score")

    assert "failed-score" not in ledger.reservations
    assert "other" in ledger.reservations
    assert ledger.prior_estimated_usd == pytest.approx(0.25)
    assert ledger.estimated_usage_usd == pytest.approx(0.25222)


def test_reconcile_billed_total_fails_closed_below_recorded_estimates() -> None:
    ledger = VisualLabLedger(envelope=envelope())
    ledger.add(record(cost=0.3))

    with pytest.raises(ValueError, match="cannot be lower"):
        ledger.reconcile_billed_total(0.29)
