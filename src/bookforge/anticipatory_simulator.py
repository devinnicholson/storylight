"""Deterministic latency and cost hypothesis for anticipatory story streaming.

This is intentionally labeled simulation evidence. It establishes measurable
promotion gates before any GKE GPU is started; it must never be presented as a
hardware benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Annotated, Literal

from pydantic import Field, model_validator

from bookforge.domain import FrozenStrictModel


class SimulationCase(FrozenStrictModel):
    case_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")]
    mode: Literal["exact_lookahead", "predicted_branch"]
    candidates: Annotated[int, Field(ge=1, le=2)]
    lookahead_ms: Annotated[float, Field(ge=0, le=60_000)]
    render_ms: Annotated[float, Field(gt=0, le=60_000)]
    critic_ms: Annotated[float, Field(gt=0, le=60_000)]
    selected_repair: bool = False
    render_cost_usd: Annotated[float, Field(gt=0, le=0.25)] = 0.004

    @model_validator(mode="after")
    def require_branch_mode(self) -> SimulationCase:
        if self.mode == "exact_lookahead" and self.candidates != 1:
            raise ValueError("exact lookahead must have one candidate")
        return self


class SimulatedCaseResult(FrozenStrictModel):
    case_id: str
    mode: str
    candidates: int
    speculative_hit: bool
    ordinary_activation_ms: float
    anticipatory_activation_ms: float
    latency_saved_ms: float
    render_attempts: int
    wasted_render_attempts: int
    render_cost_usd: float
    wasted_render_cost_usd: float


class SimulationSummary(FrozenStrictModel):
    cases: Annotated[int, Field(ge=1)]
    speculative_hit_rate: Annotated[float, Field(ge=0, le=1)]
    exact_lookahead_hit_rate: Annotated[float, Field(ge=0, le=1)]
    mean_ordinary_activation_ms: Annotated[float, Field(ge=0)]
    mean_anticipatory_activation_ms: Annotated[float, Field(ge=0)]
    p95_anticipatory_activation_ms: Annotated[float, Field(ge=0)]
    mean_latency_saved_ms: Annotated[float, Field(ge=0)]
    render_attempts: Annotated[int, Field(ge=1)]
    wasted_render_attempts: Annotated[int, Field(ge=0)]
    wasted_render_ratio: Annotated[float, Field(ge=0, le=1)]
    render_cost_usd: Annotated[float, Field(ge=0)]
    wasted_render_cost_usd: Annotated[float, Field(ge=0)]


class SimulationGates(FrozenStrictModel):
    exact_hit_rate_at_least_70_percent: bool
    waste_ratio_at_most_20_percent: bool
    p95_activation_below_250_ms: bool
    positive_latency_savings: bool
    passed: bool


class AnticipatorySimulationReport(FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_kind: Literal["deterministic_simulation_not_hardware_measurement"] = (
        "deterministic_simulation_not_hardware_measurement"
    )
    created_at: datetime
    assumptions: dict[str, str | float]
    results: list[SimulatedCaseResult]
    summary: SimulationSummary
    gates: SimulationGates


def simulate(
    cases: list[SimulationCase],
    *,
    committed_swap_ms: float = 10,
) -> AnticipatorySimulationReport:
    if not cases:
        raise ValueError("simulation requires at least one case")
    if not math.isfinite(committed_swap_ms) or not 0 <= committed_swap_ms <= 250:
        raise ValueError("committed_swap_ms must be between 0 and 250")
    results: list[SimulatedCaseResult] = []
    for case in cases:
        selected_ready_ms = case.render_ms + case.critic_ms
        selected_attempts = 1
        if case.selected_repair:
            selected_ready_ms += case.render_ms + case.critic_ms
            selected_attempts = 2
        ordinary_activation_ms = selected_ready_ms
        speculative_hit = selected_ready_ms <= case.lookahead_ms
        anticipatory_activation_ms = (
            committed_swap_ms
            if speculative_hit
            else committed_swap_ms + selected_ready_ms - case.lookahead_ms
        )
        unselected_attempts = case.candidates - 1
        render_attempts = selected_attempts + unselected_attempts
        render_cost_usd = render_attempts * case.render_cost_usd
        wasted_cost = unselected_attempts * case.render_cost_usd
        results.append(
            SimulatedCaseResult(
                case_id=case.case_id,
                mode=case.mode,
                candidates=case.candidates,
                speculative_hit=speculative_hit,
                ordinary_activation_ms=ordinary_activation_ms,
                anticipatory_activation_ms=anticipatory_activation_ms,
                latency_saved_ms=max(0, ordinary_activation_ms - anticipatory_activation_ms),
                render_attempts=render_attempts,
                wasted_render_attempts=unselected_attempts,
                render_cost_usd=render_cost_usd,
                wasted_render_cost_usd=wasted_cost,
            )
        )
    exact = [result for result in results if result.mode == "exact_lookahead"]
    if not exact:
        raise ValueError("simulation requires at least one exact-lookahead case")
    activations = sorted(result.anticipatory_activation_ms for result in results)
    attempts = sum(result.render_attempts for result in results)
    wasted_attempts = sum(result.wasted_render_attempts for result in results)
    summary = SimulationSummary(
        cases=len(results),
        speculative_hit_rate=sum(result.speculative_hit for result in results) / len(results),
        exact_lookahead_hit_rate=sum(result.speculative_hit for result in exact) / len(exact),
        mean_ordinary_activation_ms=mean(result.ordinary_activation_ms for result in results),
        mean_anticipatory_activation_ms=mean(
            result.anticipatory_activation_ms for result in results
        ),
        p95_anticipatory_activation_ms=_percentile(activations, 0.95),
        mean_latency_saved_ms=mean(result.latency_saved_ms for result in results),
        render_attempts=attempts,
        wasted_render_attempts=wasted_attempts,
        wasted_render_ratio=wasted_attempts / attempts,
        render_cost_usd=sum(result.render_cost_usd for result in results),
        wasted_render_cost_usd=sum(result.wasted_render_cost_usd for result in results),
    )
    gate_values = {
        "exact_hit_rate_at_least_70_percent": summary.exact_lookahead_hit_rate >= 0.70,
        "waste_ratio_at_most_20_percent": summary.wasted_render_ratio <= 0.20,
        "p95_activation_below_250_ms": summary.p95_anticipatory_activation_ms < 250,
        "positive_latency_savings": summary.mean_latency_saved_ms > 0,
    }
    gates = SimulationGates(**gate_values, passed=all(gate_values.values()))
    return AnticipatorySimulationReport(
        created_at=datetime.now(UTC),
        assumptions={
            "committed_swap_ms": committed_swap_ms,
            "renderer": "synthetic warm RTX timing hypothesis",
            "critic": "synthetic warm Nemotron timing hypothesis",
            "cost_scope": "renderer requests only; excludes persistent GKE node time",
        },
        results=results,
        summary=summary,
        gates=gates,
    )


def default_cases() -> list[SimulationCase]:
    cases: list[SimulationCase] = []
    timings = [
        ("moon_gate", 4200, 1800, 450, False),
        ("paper_forest", 3800, 1950, 500, False),
        ("lantern_river", 5000, 1700, 430, False),
        ("clockwork_bird", 3400, 2100, 520, False),
        ("cloud_library", 5000, 1850, 470, True),
        ("undersea_window", 3900, 1900, 480, False),
        ("dragon_kite", 4300, 2000, 510, False),
        # The scheduler queues short transitions two beats ahead instead of
        # waiting for the immediately preceding sentence boundary.
        ("short_transition", 3600, 1800, 450, False),
    ]
    for case_id, lookahead, render, critic, repair in timings:
        cases.append(
            SimulationCase(
                case_id=case_id,
                mode="exact_lookahead",
                candidates=1,
                lookahead_ms=lookahead,
                render_ms=render,
                critic_ms=critic,
                selected_repair=repair,
            )
        )
    cases.extend(
        [
            SimulationCase(
                case_id="improvised_stairway",
                mode="predicted_branch",
                candidates=2,
                lookahead_ms=4200,
                render_ms=1900,
                critic_ms=480,
            ),
            SimulationCase(
                case_id="improvised_star_whale",
                mode="predicted_branch",
                candidates=2,
                lookahead_ms=4500,
                render_ms=2050,
                critic_ms=510,
            ),
        ]
    )
    return cases


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    rank = max(0, math.ceil(quantile * len(values)) - 1)
    return values[rank]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.output
    if output.exists():
        raise SystemExit(f"Refusing to overwrite simulation evidence: {output}")
    report = simulate(default_cases())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "evidence_kind": report.evidence_kind,
                "gates_passed": report.gates.passed,
                "speculative_hit_rate": report.summary.speculative_hit_rate,
                "p95_activation_ms": report.summary.p95_anticipatory_activation_ms,
                "wasted_render_ratio": report.summary.wasted_render_ratio,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
