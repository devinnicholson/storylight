from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Modal list prices observed on 2026-08-22. The reserve below absorbs CPU, memory,
# storage, downloads, and pricing drift; these rates are intentionally not treated
# as a billing-system replacement.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}


class VisualLabBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetEnvelope:
    monthly_credit_usd: float
    usage_before_lab_usd: float
    reserve_usd: float
    run_cap_usd: float
    authorized_paid_usd: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.monthly_credit_usd,
            self.usage_before_lab_usd,
            self.reserve_usd,
            self.run_cap_usd,
            self.authorized_paid_usd,
        )
        if any(not math.isfinite(value) or value < 0 for value in values) or not math.isfinite(
            self.funding_limit_usd
        ):
            raise ValueError("budget values must be finite and non-negative")
        if self.usage_before_lab_usd + self.reserve_usd + self.run_cap_usd > (
            self.funding_limit_usd + 1e-9
        ):
            raise ValueError("run cap plus reserve exceeds available monthly funding")

    @property
    def funding_limit_usd(self) -> float:
        return self.monthly_credit_usd + self.authorized_paid_usd

    @property
    def remaining_credit_usd(self) -> float:
        return self.monthly_credit_usd - self.usage_before_lab_usd

    def require_capacity(
        self,
        *,
        recorded_estimated_usd: float = 0,
        gpu: str,
        maximum_seconds: int,
        jobs: int = 1,
    ) -> float:
        estimate = worst_case_gpu_cost(gpu=gpu, maximum_seconds=maximum_seconds, jobs=jobs)
        if recorded_estimated_usd + estimate > self.run_cap_usd + 1e-9:
            raise VisualLabBudgetError(
                f"job would exceed ${self.run_cap_usd:.2f} visual-lab cap: "
                f"${recorded_estimated_usd:.4f} recorded + ${estimate:.4f} worst case"
            )
        return estimate


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    experiment_id: str
    stage: str
    model: str
    model_revision: str
    gpu: str
    seed: int
    prompt: str
    artifact_path: str
    sha256: str
    generation_seconds: float
    estimated_gpu_usd: float
    width: int
    height: int
    frames: int = 1
    fps: int = 0


@dataclass(slots=True)
class VisualLabLedger:
    envelope: BudgetEnvelope
    prior_estimated_usd: float = 0
    records: list[GenerationRecord] = field(default_factory=list)
    reservations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.prior_estimated_usd) or self.prior_estimated_usd < 0:
            raise ValueError("prior estimated usage must be finite and non-negative")
        if any(
            not key.strip() or not math.isfinite(value) or value < 0
            for key, value in self.reservations.items()
        ):
            raise ValueError("reservations require names and finite non-negative costs")
        if self.estimated_usage_usd > self.envelope.run_cap_usd:
            raise VisualLabBudgetError("recorded usage exceeds visual-lab cap")

    @property
    def estimated_usage_usd(self) -> float:
        return (
            self.prior_estimated_usd
            + sum(record.estimated_gpu_usd for record in self.records)
            + sum(self.reservations.values())
        )

    def reserve(
        self,
        *,
        gpu: str,
        maximum_seconds: int,
        jobs: int = 1,
        reservation_id: str | None = None,
    ) -> float:
        if reservation_id is not None and reservation_id in self.reservations:
            raise ValueError(f"duplicate reservation_id: {reservation_id}")
        estimate = self.envelope.require_capacity(
            recorded_estimated_usd=self.estimated_usage_usd,
            gpu=gpu,
            maximum_seconds=maximum_seconds,
            jobs=jobs,
        )
        if reservation_id is not None:
            if not reservation_id.strip():
                raise ValueError("reservation_id must not be empty")
            self.reservations[reservation_id] = estimate
        return estimate

    def release(self, reservation_id: str) -> None:
        if reservation_id not in self.reservations:
            raise ValueError(f"unknown reservation_id: {reservation_id}")
        del self.reservations[reservation_id]

    def reconcile_billed_total(
        self,
        billed_total_usd: float,
        *,
        release_reservation_id: str | None = None,
    ) -> None:
        """Anchor the ledger to an authoritative provider total after a stopped run."""
        if not math.isfinite(billed_total_usd) or billed_total_usd < 0:
            raise ValueError("billed total must be finite and non-negative")
        recorded = sum(record.estimated_gpu_usd for record in self.records)
        if billed_total_usd + 1e-9 < recorded:
            raise ValueError("billed total cannot be lower than recorded generation estimates")
        if release_reservation_id is not None:
            self.release(release_reservation_id)
        self.prior_estimated_usd = billed_total_usd - recorded

    def add(self, record: GenerationRecord) -> None:
        if any(existing.experiment_id == record.experiment_id for existing in self.records):
            raise ValueError(f"duplicate experiment_id: {record.experiment_id}")
        if self.estimated_usage_usd + record.estimated_gpu_usd > self.envelope.run_cap_usd:
            raise VisualLabBudgetError("record would exceed visual-lab cap")
        self.records.append(record)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "envelope": asdict(self.envelope),
            "prior_estimated_usd": self.prior_estimated_usd,
            "estimated_usage_usd": round(self.estimated_usage_usd, 8),
            "reservations": self.reservations,
            "records": [asdict(record) for record in self.records],
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path, *, envelope: BudgetEnvelope) -> VisualLabLedger:
        if not path.exists():
            return cls(envelope=envelope)
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported visual-lab ledger schema")
        persisted_envelope = BudgetEnvelope(**payload["envelope"])
        if persisted_envelope != envelope:
            raise ValueError("ledger budget envelope does not match requested envelope")
        return cls(
            envelope=envelope,
            prior_estimated_usd=float(payload.get("prior_estimated_usd", 0)),
            records=[GenerationRecord(**record) for record in payload.get("records", [])],
            reservations={
                str(key): float(value) for key, value in payload.get("reservations", {}).items()
            },
        )


def worst_case_gpu_cost(*, gpu: str, maximum_seconds: int, jobs: int = 1) -> float:
    if gpu not in GPU_USD_PER_SECOND:
        raise ValueError(f"unknown GPU price: {gpu}")
    if maximum_seconds < 1 or jobs < 1:
        raise ValueError("maximum_seconds and jobs must be positive")
    return GPU_USD_PER_SECOND[gpu] * maximum_seconds * jobs


def measured_gpu_cost(*, gpu: str, seconds: float) -> float:
    if gpu not in GPU_USD_PER_SECOND:
        raise ValueError(f"unknown GPU price: {gpu}")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and non-negative")
    return GPU_USD_PER_SECOND[gpu] * seconds
