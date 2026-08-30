import json
import multiprocessing
from pathlib import Path

import pytest

from bookforge.modal_budget import (
    authorize_and_reserve_modal_budget,
    budget_envelope_from_plan,
    reconcile_failed_modal_budget_reservation,
    release_modal_budget_reservation,
    require_modal_budget_reservation,
    reserve_modal_budget,
    settle_modal_budget,
)
from bookforge.visual_lab import GenerationRecord, VisualLabLedger


def _reserve_worker(
    plan_path: str,
    ledger_path: str,
    experiment_id: str,
    start,
) -> None:
    start.wait()
    reserve_modal_budget(
        plan_path=Path(plan_path),
        ledger_path=Path(ledger_path),
        experiment_id=experiment_id,
        gpu="L4",
        timeout_seconds=100,
        maximum_gpu_usd=0.03,
    )


def _settle_worker(
    plan_path: str,
    ledger_path: str,
    experiment_id: str,
    start,
) -> None:
    start.wait()
    settle_modal_budget(
        plan_path=Path(plan_path),
        ledger_path=Path(ledger_path),
        reservation_id=f"reservation:{experiment_id}",
        record=GenerationRecord(
            experiment_id=experiment_id,
            stage="test",
            model="fixture",
            model_revision="a" * 40,
            gpu="L4",
            seed=1,
            prompt="fixture",
            artifact_path=f"{experiment_id}.png",
            sha256="b" * 64,
            generation_seconds=1,
            estimated_gpu_usd=0.001,
            width=32,
            height=32,
        ),
    )


def _authoritative_reserve_worker(
    plan_path: str,
    ledger_path: str,
    experiment_id: str,
    start,
    results,
) -> None:
    start.wait()
    try:
        authorize_and_reserve_modal_budget(
            plan_path=Path(plan_path),
            ledger_path=Path(ledger_path),
            authoritative_workspace_usd=28.88,
            experiment_id=experiment_id,
            full_call_ceiling_usd=0.10656,
        )
    except ValueError as error:
        results.put((experiment_id, "blocked", str(error)))
    else:
        results.put((experiment_id, "reserved", ""))


def _start_together(target, *, plan: Path, ledger: Path, ids: list[str]) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=target, args=(str(plan), str(ledger), experiment_id, start))
        for experiment_id in ids
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0


def test_concurrent_process_reservations_and_settlements_do_not_lose_updates(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 10.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 19.0,
            }
        )
    )
    identifiers = ["concurrent-a", "concurrent-b"]

    _start_together(
        _reserve_worker,
        plan=plan,
        ledger=ledger_path,
        ids=identifiers,
    )
    envelope, _ = budget_envelope_from_plan(plan)
    reserved = VisualLabLedger.read(ledger_path, envelope=envelope)
    assert set(reserved.reservations) == {
        "reservation:concurrent-a",
        "reservation:concurrent-b",
    }

    _start_together(
        _settle_worker,
        plan=plan,
        ledger=ledger_path,
        ids=identifiers,
    )
    settled = VisualLabLedger.read(ledger_path, envelope=envelope)
    assert settled.reservations == {}
    assert {record.experiment_id for record in settled.records} == set(identifiers)
    assert settled.estimated_usage_usd == 0.002


def test_authoritative_full_call_ceiling_is_atomic_across_processes(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 10.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 19.0,
            }
        )
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_authoritative_reserve_worker,
            args=(str(plan), str(ledger_path), experiment_id, start, results),
        )
        for experiment_id in ("motion-a", "motion-b")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0
    outcomes = [results.get(timeout=2) for _ in processes]

    assert sorted(outcome[1] for outcome in outcomes) == ["blocked", "reserved"]
    assert "hard stop" in next(outcome[2] for outcome in outcomes if outcome[1] == "blocked")
    envelope, _ = budget_envelope_from_plan(plan)
    ledger = VisualLabLedger.read(ledger_path, envelope=envelope)
    assert len(ledger.reservations) == 1
    assert ledger.estimated_usage_usd == 18.98656


def test_external_reservation_is_bound_to_its_exact_experiment(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 10.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 19.0,
            }
        )
    )
    reservation_id = authorize_and_reserve_modal_budget(
        plan_path=plan,
        ledger_path=ledger_path,
        authoritative_workspace_usd=10.0,
        experiment_id="scene-a",
        full_call_ceiling_usd=0.08,
    )

    require_modal_budget_reservation(
        plan_path=plan,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        expected_experiment_id="scene-a",
    )
    with pytest.raises(ValueError, match="does not belong"):
        require_modal_budget_reservation(
            plan_path=plan,
            ledger_path=ledger_path,
            reservation_id=reservation_id,
            expected_experiment_id="scene-b",
        )


def test_unused_authorization_can_be_released_before_any_remote_call(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 10.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 19.0,
            }
        )
    )
    reservation_id = authorize_and_reserve_modal_budget(
        plan_path=plan,
        ledger_path=ledger_path,
        authoritative_workspace_usd=10.0,
        experiment_id="prepared-scene",
        full_call_ceiling_usd=0.08,
    )

    release_modal_budget_reservation(
        plan_path=plan,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )

    envelope, _ = budget_envelope_from_plan(plan)
    ledger = VisualLabLedger.read(ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert ledger.records == []


def test_failed_reservation_reconciles_only_after_authoritative_charge(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    ledger_path = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 10.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 19.0,
            }
        )
    )
    reservation_id = authorize_and_reserve_modal_budget(
        plan_path=plan,
        ledger_path=ledger_path,
        authoritative_workspace_usd=10.0,
        experiment_id="failed-scene",
        full_call_ceiling_usd=0.20,
    )

    reconcile_failed_modal_budget_reservation(
        plan_path=plan,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        authoritative_workspace_usd=10.17,
    )

    envelope, _ = budget_envelope_from_plan(plan)
    ledger = VisualLabLedger.read(ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert ledger.estimated_usage_usd == pytest.approx(0.17)
    with pytest.raises(ValueError, match="unknown reservation_id"):
        reconcile_failed_modal_budget_reservation(
            plan_path=plan,
            ledger_path=ledger_path,
            reservation_id=reservation_id,
            authoritative_workspace_usd=10.17,
        )
