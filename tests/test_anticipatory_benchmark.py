from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

from bookforge.anticipatory import (
    AnticipationMetrics,
    AnticipationStatus,
    CandidateRecord,
    CandidateState,
    RenderedScene,
)
from bookforge.anticipatory_benchmark import benchmark_specs, run_live_benchmark
from bookforge.nemotron_critic import (
    NemotronCriticDecision,
    NemotronCriticEvidence,
    NemotronCriticVerdict,
)


class _AcceptedClient:
    base_url = "http://127.0.0.1:18082"

    def __init__(self) -> None:
        self.statuses = {}

    async def submit(self, batch):
        now = datetime.now(UTC)
        spec = batch.candidates[0]
        scene = RenderedScene(
            master_ref=f"asset_{hashlib.sha256(b'master').hexdigest()}",
            depth_ref=f"asset_{hashlib.sha256(b'depth').hexdigest()}",
            master_sha256=hashlib.sha256(b"master").hexdigest(),
            depth_sha256=hashlib.sha256(b"depth").hexdigest(),
            provider="test",
            model="test-renderer",
            model_revision="test-v1",
            render_latency_ms=12,
            estimated_gpu_usd=0.001,
        )
        critic = NemotronCriticEvidence(
            verdict=NemotronCriticVerdict(
                fidelity_score=0.95,
                composition_score=0.94,
                projection_legibility_score=0.96,
                identity_consistent=True,
                unintended_text=False,
                decision=NemotronCriticDecision.ACCEPT,
                reason="The expected subjects are clear and projection-ready.",
            ),
            model="test-nemotron",
            latency_ms=5,
        )
        record = CandidateRecord(
            spec=spec,
            state=CandidateState.READY,
            cache_hit=batch.sequence >= 200,
            render_attempts=0 if batch.sequence >= 200 else 1,
            rendered=scene,
            critic_history=[critic],
            render_cost_usd=0 if batch.sequence >= 200 else 0.001,
            ready_latency_ms=1 if batch.sequence >= 200 else 17,
            created_at=now,
            updated_at=now,
        )
        status = _status(batch.session_token, batch.sequence, record)
        self.statuses[(batch.session_token, batch.sequence)] = status
        return status

    async def status(self, session_token, sequence):  # pragma: no cover
        raise AssertionError("ready submissions should not be polled")

    async def commit(self, request):
        submitted = self.statuses[(request.session_token, request.sequence)]
        record = submitted.candidates[0].model_copy(
            update={"state": CandidateState.COMMITTED, "selected": True}
        )
        return _status(request.session_token, request.sequence, record)

    async def fetch_scene(self, scene):
        return b"master", b"depth"

    async def cancel(self, session_token, sequence):  # pragma: no cover
        raise AssertionError("ready submissions should not be cancelled")


def _status(session_token, sequence, record):
    return AnticipationStatus(
        session_token=session_token,
        sequence=sequence,
        candidates=[record],
        metrics=AnticipationMetrics(
            candidates=1,
            ready=int(record.state is CandidateState.READY),
            committed=int(record.state is CandidateState.COMMITTED),
            cache_hits=int(record.cache_hit),
            cancelled=0,
            rejected=0,
            failed=0,
            repair_attempts=record.repair_attempts,
            render_cost_usd=record.render_cost_usd,
            wasted_render_cost_usd=0,
            mean_ready_ms=record.ready_latency_ms,
        ),
    )


def test_benchmark_contracts_are_sanitized_and_measured_run_passes() -> None:
    specs = benchmark_specs(not_after=datetime.now(UTC))
    serialized = "".join(spec.model_dump_json() for _, spec in specs)
    assert len(specs) == 3
    assert "source_text" not in serialized
    assert '"audio":' not in serialized
    assert '"camera_data":' not in serialized

    report = asyncio.run(run_live_benchmark(_AcceptedClient()))  # type: ignore[arg-type]

    assert report.evidence_kind == "measured_gke_cloud_run_nemotron_acceptance"
    assert report.cost_scope == "incremental_renderer_requests_only"
    assert report.summary.cases == 6
    assert report.summary.cache_hits == 3
    assert report.summary.total_estimated_gpu_usd == 0.003
    replay = [result for result in report.results if result.cache_hit]
    assert all(result.render_ms == 0 for result in replay)
    assert all(result.critic_ms == 0 for result in replay)
    assert all(result.critic_input_tokens == 0 for result in replay)
    assert all(result.critic_output_tokens == 0 for result in replay)
    assert report.gates.passed is True
