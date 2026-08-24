from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bookforge.visual_lab import (
    BudgetEnvelope,
    GenerationRecord,
    VisualLabLedger,
    worst_case_gpu_cost,
)


def budget_envelope_from_plan(plan_path: Path) -> tuple[BudgetEnvelope, float]:
    plan = json.loads(plan_path.read_text())
    if "budget" in plan:
        return BudgetEnvelope(**plan["budget"]), float(plan.get("ledger_baseline_usd", 0))
    return (
        BudgetEnvelope(
            monthly_credit_usd=float(plan["monthly_credit_usd"]),
            usage_before_lab_usd=float(plan["workspace_usage_before_live_scenes_usd"]),
            reserve_usd=float(plan["billing_delay_reserve_usd"]),
            run_cap_usd=float(plan["maximum_new_spend_usd"]),
        ),
        0.0,
    )


@contextmanager
def locked_modal_budget_ledger(
    *,
    plan_path: Path,
    ledger_path: Path,
) -> Iterator[VisualLabLedger]:
    """Reload and atomically update a Modal ledger under a POSIX process lock."""

    envelope, baseline = budget_envelope_from_plan(plan_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if ledger_path.exists():
            ledger = VisualLabLedger.read(ledger_path, envelope=envelope)
        else:
            ledger = VisualLabLedger(envelope=envelope, prior_estimated_usd=baseline)
        try:
            yield ledger
        except BaseException:
            raise
        else:
            ledger.write(ledger_path)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def reserve_modal_budget(
    *,
    plan_path: Path,
    ledger_path: Path,
    experiment_id: str,
    gpu: str,
    timeout_seconds: int,
    maximum_gpu_usd: float,
) -> str:
    worst_case = worst_case_gpu_cost(gpu=gpu, maximum_seconds=timeout_seconds)
    if maximum_gpu_usd <= 0 or worst_case > maximum_gpu_usd + 1e-9:
        raise ValueError(
            f"${worst_case:.6f} timeout reservation exceeds ${maximum_gpu_usd:.6f} call cap"
        )
    reservation_id = f"reservation:{experiment_id}"
    with locked_modal_budget_ledger(plan_path=plan_path, ledger_path=ledger_path) as ledger:
        if any(record.experiment_id == experiment_id for record in ledger.records):
            raise ValueError(f"generation already recorded: {experiment_id}")
        ledger.reserve(
            gpu=gpu,
            maximum_seconds=timeout_seconds,
            reservation_id=reservation_id,
        )
    return reservation_id


def authorize_and_reserve_modal_budget(
    *,
    plan_path: Path,
    ledger_path: Path,
    authoritative_workspace_usd: float,
    experiment_id: str,
    full_call_ceiling_usd: float,
) -> str:
    """Atomically anchor current spend and reserve a cross-process call ceiling."""

    if not math.isfinite(authoritative_workspace_usd) or authoritative_workspace_usd < 0:
        raise ValueError("authoritative workspace total must be finite and non-negative")
    if not math.isfinite(full_call_ceiling_usd) or full_call_ceiling_usd <= 0:
        raise ValueError("full call ceiling must be finite and positive")
    plan = json.loads(plan_path.read_text())
    envelope, _ = budget_envelope_from_plan(plan_path)
    hard_stop = float(
        plan.get(
            "hard_stop_workspace_total_usd",
            envelope.monthly_credit_usd - envelope.reserve_usd,
        )
    )
    if not math.isfinite(hard_stop) or hard_stop <= 0:
        raise ValueError("hard-stop workspace total must be finite and positive")
    reservation_id = f"reservation:{experiment_id}"
    with locked_modal_budget_ledger(plan_path=plan_path, ledger_path=ledger_path) as ledger:
        if reservation_id in ledger.reservations:
            raise ValueError(f"duplicate reservation_id: {reservation_id}")
        if any(record.experiment_id == experiment_id for record in ledger.records):
            raise ValueError(f"generation already recorded: {experiment_id}")
        recorded = sum(record.estimated_gpu_usd for record in ledger.records)
        reported_phase = max(
            0.0,
            authoritative_workspace_usd - envelope.usage_before_lab_usd,
        )
        existing_settled = ledger.prior_estimated_usd + recorded
        settled_floor = max(reported_phase, recorded, existing_settled)
        ledger.prior_estimated_usd = settled_floor - recorded
        projected_workspace = (
            envelope.usage_before_lab_usd
            + ledger.estimated_usage_usd
            + full_call_ceiling_usd
        )
        if projected_workspace > hard_stop + 1e-9:
            raise ValueError(
                f"authoritative Modal hard stop would be exceeded: "
                f"${projected_workspace:.6f} projected > ${hard_stop:.6f}"
            )
        if ledger.estimated_usage_usd + full_call_ceiling_usd > envelope.run_cap_usd + 1e-9:
            raise ValueError("full call ceiling would exceed the Modal phase run cap")
        ledger.reservations[reservation_id] = full_call_ceiling_usd
    return reservation_id


def require_modal_budget_reservation(
    *,
    plan_path: Path,
    ledger_path: Path,
    reservation_id: str,
    expected_experiment_id: str,
) -> None:
    expected_reservation_id = f"reservation:{expected_experiment_id}"
    if reservation_id != expected_reservation_id:
        raise ValueError(
            "reservation_id does not belong to the expected generation experiment"
        )
    with locked_modal_budget_ledger(plan_path=plan_path, ledger_path=ledger_path) as ledger:
        if reservation_id not in ledger.reservations:
            raise ValueError(f"unknown reservation_id: {reservation_id}")


def release_modal_budget_reservation(
    *,
    plan_path: Path,
    ledger_path: Path,
    reservation_id: str,
) -> None:
    """Release an authorization only when no remote paid call was started."""

    with locked_modal_budget_ledger(plan_path=plan_path, ledger_path=ledger_path) as ledger:
        ledger.release(reservation_id)


def settle_modal_budget(
    *,
    plan_path: Path,
    ledger_path: Path,
    reservation_id: str,
    record: GenerationRecord,
) -> None:
    with locked_modal_budget_ledger(plan_path=plan_path, ledger_path=ledger_path) as ledger:
        ledger.release(reservation_id)
        ledger.add(record)
