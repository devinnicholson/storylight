"""Bounded live acceptance benchmark for the GKE anticipatory story engine.

Unlike ``anticipatory_simulator``, this command calls the deployed service, waits
for Nemotron promotion, downloads both synthetic assets, and verifies their
digests.  It accepts only pre-sanitized visual contracts and writes an immutable
evidence file so simulated and measured results cannot be confused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Annotated, Literal

from pydantic import Field

from bookforge.anticipatory import (
    AnticipationSource,
    AnticipatoryBatchRequest,
    AnticipatorySceneSpec,
    CandidateRecord,
    CandidateState,
    CommitRequest,
    PrivacyAttestation,
)
from bookforge.anticipatory_edge import AnticipatoryEdgeClient, new_anticipation_session_token
from bookforge.domain import FrozenStrictModel

AUTHORIZATION = "I_UNDERSTAND_THIS_WAKES_BILLABLE_GPUS"
TERMINAL_STATES = {
    CandidateState.READY,
    CandidateState.REJECTED,
    CandidateState.FAILED,
    CandidateState.EXPIRED,
}


class LiveBenchmarkResult(FrozenStrictModel):
    case_id: str
    pass_index: Annotated[int, Field(ge=1, le=2)]
    cache_hit: bool
    accepted: bool
    assets_verified: bool
    submit_ms: Annotated[float, Field(ge=0)]
    ready_ms: Annotated[float, Field(ge=0)]
    commit_and_fetch_ms: Annotated[float, Field(ge=0)]
    render_ms: Annotated[float, Field(ge=0)]
    critic_ms: Annotated[float, Field(ge=0)]
    critic_input_tokens: Annotated[int, Field(ge=0)] = 0
    critic_output_tokens: Annotated[int, Field(ge=0)] = 0
    render_attempts: Annotated[int, Field(ge=0, le=2)]
    repair_attempts: Annotated[int, Field(ge=0, le=1)]
    estimated_gpu_usd: Annotated[float, Field(ge=0, le=0.5)]
    error: str | None = None


class LiveBenchmarkSummary(FrozenStrictModel):
    cases: Annotated[int, Field(ge=1)]
    accepted: Annotated[int, Field(ge=0)]
    assets_verified: Annotated[int, Field(ge=0)]
    cache_hits: Annotated[int, Field(ge=0)]
    p95_ready_ms: Annotated[float, Field(ge=0)]
    p95_commit_and_fetch_ms: Annotated[float, Field(ge=0)]
    mean_ready_ms: Annotated[float, Field(ge=0)]
    total_estimated_gpu_usd: Annotated[float, Field(ge=0)]


class LiveBenchmarkGates(FrozenStrictModel):
    all_candidates_accepted: bool
    all_assets_verified: bool
    replay_cache_hit_rate_100_percent: bool
    p95_replay_ready_below_250_ms: bool
    p95_commit_and_fetch_below_750_ms: bool
    total_render_cost_below_ceiling: bool
    passed: bool


class LiveBenchmarkPrewarm(FrozenStrictModel):
    ready: Literal[True] = True
    wall_ms: Annotated[float, Field(ge=0)]
    detail: Annotated[str, Field(min_length=1, max_length=600)]


class AnticipatoryLiveBenchmarkReport(FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_kind: Literal["measured_gke_cloud_run_nemotron_acceptance"] = (
        "measured_gke_cloud_run_nemotron_acceptance"
    )
    created_at: datetime
    endpoint: str
    privacy_boundary: Literal["sanitized_scene_spec_v1"] = "sanitized_scene_spec_v1"
    source_media_sent: Literal[False] = False
    cost_scope: Literal["incremental_renderer_requests_only"] = "incremental_renderer_requests_only"
    prewarm: LiveBenchmarkPrewarm | None = None
    results: list[LiveBenchmarkResult]
    summary: LiveBenchmarkSummary
    gates: LiveBenchmarkGates


def benchmark_specs(*, not_after: datetime) -> list[tuple[str, AnticipatorySceneSpec]]:
    cases = [
        (
            "moon_gate",
            "A single silver fox steps through a round moon gate into an indigo garden; "
            "gold fireflies rise beside it, full-body subject, layered theatrical depth.",
            ["one silver fox", "one round moon gate", "gold fireflies"],
            7,
        ),
        (
            "paper_forest",
            "One small child in a yellow raincoat follows a blue paper bird through a vast "
            "emerald cut-paper forest, clear silhouettes and deep foreground framing.",
            ["one child in a yellow raincoat", "one blue paper bird"],
            17,
        ),
        (
            "lantern_river",
            "A solitary wooden boat carries one glowing lantern down a midnight river beneath "
            "willow branches, luminous reflections and a calm cinematic composition.",
            ["one wooden boat", "one glowing lantern"],
            29,
        ),
    ]
    return [
        (
            case_id,
            AnticipatorySceneSpec(
                branch_id="known_next",
                sequence=index,
                source=AnticipationSource.EXACT_LOOKAHEAD,
                visual_brief=brief,
                expected_subjects=subjects,
                continuity_sha256=_sha256(f"{case_id}:continuity"),
                seed=seed,
                not_after=not_after,
                max_render_cost_usd=0.012,
                privacy=PrivacyAttestation(edge_gate_revision="hardware-benchmark-v1"),
            ),
        )
        for index, (case_id, brief, subjects, seed) in enumerate(cases, start=1)
    ]


async def run_live_benchmark(
    client: AnticipatoryEdgeClient,
    *,
    poll_seconds: float = 0.1,
    ready_timeout_seconds: float = 120,
    total_cost_ceiling_usd: float = 0.15,
    prewarm: LiveBenchmarkPrewarm | None = None,
) -> AnticipatoryLiveBenchmarkReport:
    if not math.isfinite(poll_seconds) or not 0.05 <= poll_seconds <= 5:
        raise ValueError("poll_seconds must be between 0.05 and 5")
    if not math.isfinite(ready_timeout_seconds) or not 1 <= ready_timeout_seconds <= 600:
        raise ValueError("ready_timeout_seconds must be between 1 and 600")
    if not math.isfinite(total_cost_ceiling_usd) or not 0 < total_cost_ceiling_usd <= 1:
        raise ValueError("total_cost_ceiling_usd must be between 0 and 1")

    cases = benchmark_specs(not_after=datetime.now(UTC) + timedelta(minutes=10))
    results: list[LiveBenchmarkResult] = []
    for pass_index in (1, 2):
        for case_id, template in cases:
            session_token = new_anticipation_session_token()
            sequence = template.sequence + pass_index * 100
            spec = template.model_copy(update={"sequence": sequence})
            started = time.perf_counter()
            status = await client.submit(
                AnticipatoryBatchRequest(
                    session_token=session_token,
                    sequence=sequence,
                    candidates=[spec],
                    session_cost_ceiling_usd=0.024,
                )
            )
            submit_ms = (time.perf_counter() - started) * 1_000
            deadline = time.monotonic() + ready_timeout_seconds
            while status.candidates[0].state not in TERMINAL_STATES:
                if time.monotonic() >= deadline:
                    results.append(
                        _failed_result(
                            case_id=case_id,
                            pass_index=pass_index,
                            submit_ms=submit_ms,
                            ready_ms=(time.perf_counter() - started) * 1_000,
                            error="candidate readiness timeout",
                        )
                    )
                    await client.cancel(session_token, sequence)
                    break
                await asyncio.sleep(poll_seconds)
                status = await client.status(session_token, sequence)
            else:
                candidate = status.candidates[0]
                if candidate.state is not CandidateState.READY or candidate.rendered is None:
                    results.append(
                        _result_from_terminal(
                            case_id=case_id,
                            pass_index=pass_index,
                            candidate=candidate,
                            submit_ms=submit_ms,
                            ready_ms=(time.perf_counter() - started) * 1_000,
                        )
                    )
                    continue
                ready_ms = (time.perf_counter() - started) * 1_000
                activation_started = time.perf_counter()
                committed = await client.commit(
                    CommitRequest(
                        session_token=session_token,
                        sequence=sequence,
                        branch_id=spec.branch_id,
                    )
                )
                selected = committed.candidates[0]
                if selected.rendered is None:
                    results.append(
                        _failed_result(
                            case_id=case_id,
                            pass_index=pass_index,
                            submit_ms=submit_ms,
                            ready_ms=ready_ms,
                            error="committed candidate has no rendered scene",
                        )
                    )
                    continue
                await client.fetch_scene(selected.rendered)
                activation_ms = (time.perf_counter() - activation_started) * 1_000
                results.append(
                    _result_from_terminal(
                        case_id=case_id,
                        pass_index=pass_index,
                        candidate=selected,
                        submit_ms=submit_ms,
                        ready_ms=ready_ms,
                        commit_and_fetch_ms=activation_ms,
                        assets_verified=True,
                    )
                )

    replay = [result for result in results if result.pass_index == 2]
    summary = LiveBenchmarkSummary(
        cases=len(results),
        accepted=sum(result.accepted for result in results),
        assets_verified=sum(result.assets_verified for result in results),
        cache_hits=sum(result.cache_hit for result in results),
        p95_ready_ms=_percentile([result.ready_ms for result in results], 0.95),
        p95_commit_and_fetch_ms=_percentile(
            [result.commit_and_fetch_ms for result in results],
            0.95,
        ),
        mean_ready_ms=mean(result.ready_ms for result in results),
        total_estimated_gpu_usd=sum(result.estimated_gpu_usd for result in results),
    )
    replay_cache_gate = bool(replay) and all(result.cache_hit for result in replay)
    replay_latency_gate = (
        bool(replay)
        and _percentile(
            [result.ready_ms for result in replay],
            0.95,
        )
        < 250
    )
    gate_values = {
        "all_candidates_accepted": summary.accepted == summary.cases,
        "all_assets_verified": summary.assets_verified == summary.cases,
        "replay_cache_hit_rate_100_percent": replay_cache_gate,
        "p95_replay_ready_below_250_ms": replay_latency_gate,
        "p95_commit_and_fetch_below_750_ms": summary.p95_commit_and_fetch_ms < 750,
        "total_render_cost_below_ceiling": (
            summary.total_estimated_gpu_usd <= total_cost_ceiling_usd
        ),
    }
    return AnticipatoryLiveBenchmarkReport(
        created_at=datetime.now(UTC),
        endpoint=client.base_url,
        prewarm=prewarm,
        results=results,
        summary=summary,
        gates=LiveBenchmarkGates(**gate_values, passed=all(gate_values.values())),
    )


def _result_from_terminal(
    *,
    case_id: str,
    pass_index: int,
    candidate: CandidateRecord,
    submit_ms: float,
    ready_ms: float,
    commit_and_fetch_ms: float = 0,
    assets_verified: bool = False,
) -> LiveBenchmarkResult:
    critic_ms = (
        0
        if candidate.cache_hit
        else sum(evidence.latency_ms for evidence in candidate.critic_history)
    )
    return LiveBenchmarkResult(
        case_id=case_id,
        pass_index=pass_index,
        cache_hit=candidate.cache_hit,
        accepted=candidate.state in {CandidateState.READY, CandidateState.COMMITTED},
        assets_verified=assets_verified,
        submit_ms=submit_ms,
        ready_ms=ready_ms,
        commit_and_fetch_ms=commit_and_fetch_ms,
        render_ms=(
            0
            if candidate.cache_hit or candidate.rendered is None
            else candidate.rendered.render_latency_ms
        ),
        critic_ms=critic_ms,
        critic_input_tokens=(
            0
            if candidate.cache_hit
            else sum(evidence.input_tokens for evidence in candidate.critic_history)
        ),
        critic_output_tokens=(
            0
            if candidate.cache_hit
            else sum(evidence.output_tokens for evidence in candidate.critic_history)
        ),
        render_attempts=candidate.render_attempts,
        repair_attempts=candidate.repair_attempts,
        estimated_gpu_usd=candidate.render_cost_usd,
        error=candidate.error,
    )


def _failed_result(
    *,
    case_id: str,
    pass_index: int,
    submit_ms: float,
    ready_ms: float,
    error: str,
) -> LiveBenchmarkResult:
    return LiveBenchmarkResult(
        case_id=case_id,
        pass_index=pass_index,
        cache_hit=False,
        accepted=False,
        assets_verified=False,
        submit_ms=submit_ms,
        ready_ms=ready_ms,
        commit_and_fetch_ms=0,
        render_ms=0,
        critic_ms=0,
        critic_input_tokens=0,
        critic_output_tokens=0,
        render_attempts=0,
        repair_attempts=0,
        estimated_gpu_usd=0,
        error=error,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise SystemExit(f"Authorization must exactly equal {AUTHORIZATION}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite measured evidence: {args.output}")

    async def run() -> AnticipatoryLiveBenchmarkReport:
        client = AnticipatoryEdgeClient(
            base_url=args.base_url,
            allow_loopback_http=True,
        )
        try:
            prewarm_started = time.perf_counter()
            prewarm_result = await client.prewarm_runtime()
            prewarm = LiveBenchmarkPrewarm(
                wall_ms=(time.perf_counter() - prewarm_started) * 1_000,
                detail=str(prewarm_result["detail"]),
            )
            return await run_live_benchmark(
                client,
                ready_timeout_seconds=args.ready_timeout_seconds,
                prewarm=prewarm,
            )
        finally:
            await client.aclose()

    report = asyncio.run(run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evidence_kind": report.evidence_kind,
                "gates_passed": report.gates.passed,
                "prewarm_ms": report.prewarm.wall_ms if report.prewarm else None,
                "p95_ready_ms": report.summary.p95_ready_ms,
                "p95_commit_and_fetch_ms": report.summary.p95_commit_and_fetch_ms,
                "total_estimated_gpu_usd": report.summary.total_estimated_gpu_usd,
            },
            indent=2,
        )
    )
    if not report.gates.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
