from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bookforge.anticipatory import (
    AnticipationSource,
    AnticipatoryBatchRequest,
    AnticipatorySceneOrchestrator,
    AnticipatorySceneSpec,
    CandidateState,
    CommitRequest,
    PrivacyAttestation,
    RenderedScene,
)
from bookforge.nemotron_critic import (
    NemotronCriticDecision,
    NemotronCriticEvidence,
    NemotronCriticVerdict,
)

NOW = datetime(2026, 9, 2, 12, tzinfo=UTC)
SESSION_A = "anticipate_0123456789abcdef01234567"
SESSION_B = "anticipate_89abcdef0123456789abcdef"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _spec(
    branch_id: str = "moon_path",
    *,
    sequence: int = 3,
    source: AnticipationSource = AnticipationSource.PREDICTED_BRANCH,
    seed: int = 7,
    not_after: datetime | None = None,
) -> AnticipatorySceneSpec:
    return AnticipatorySceneSpec(
        branch_id=branch_id,
        sequence=sequence,
        source=source,
        visual_brief=(
            "One silver fox steps through a round moon gate into a luminous indigo garden."
        ),
        expected_subjects=["one silver fox", "round moon gate"],
        continuity_sha256=_sha("fox-at-moon-gate"),
        seed=seed,
        not_after=not_after or NOW + timedelta(minutes=5),
        max_render_cost_usd=0.02,
        privacy=PrivacyAttestation(edge_gate_revision="edge-gate-test-v1"),
    )


def _batch(
    *specs: AnticipatorySceneSpec,
    session: str = SESSION_A,
    sequence: int = 3,
) -> AnticipatoryBatchRequest:
    return AnticipatoryBatchRequest(
        session_token=session,
        sequence=sequence,
        candidates=list(specs),
        session_cost_ceiling_usd=0.08,
    )


def _scene(spec: AnticipatorySceneSpec, attempt: int, *, cost: float = 0.004) -> RenderedScene:
    suffix = f"{spec.cache_key}-{attempt}"
    return RenderedScene(
        master_ref=f"memory://{suffix}/master.jpg",
        depth_ref=f"memory://{suffix}/depth.jpg",
        master_sha256=_sha(f"master-{suffix}"),
        depth_sha256=_sha(f"depth-{suffix}"),
        provider="fake-rtx-renderer",
        model="sana-sprint",
        model_revision="test-revision",
        render_latency_ms=8,
        estimated_gpu_usd=cost,
    )


def _evidence(decision: NemotronCriticDecision) -> NemotronCriticEvidence:
    correction = None
    if decision is not NemotronCriticDecision.ACCEPT:
        correction = "Show exactly one silver fox crossing the moon gate; retain the indigo garden."
    return NemotronCriticEvidence(
        verdict=NemotronCriticVerdict(
            fidelity_score=0.95 if decision is NemotronCriticDecision.ACCEPT else 0.45,
            composition_score=0.92,
            projection_legibility_score=0.9,
            identity_consistent=decision is NemotronCriticDecision.ACCEPT,
            unintended_text=False,
            decision=decision,
            reason="The scene passed."
            if decision is NemotronCriticDecision.ACCEPT
            else "Fox duplicated.",
            correction_visual_brief=correction,
        ),
        model="nemotron-test",
        latency_ms=5,
    )


class FakeRenderer:
    def __init__(self, *, delay: float = 0, cost: float = 0.004) -> None:
        self.delay = delay
        self.cost = cost
        self.calls: list[tuple[str, int, str, str | None]] = []

    async def render(
        self,
        spec: AnticipatorySceneSpec,
        *,
        attempt: int,
        repair_guidance: str | None = None,
    ) -> RenderedScene:
        self.calls.append((spec.branch_id, attempt, spec.visual_brief, repair_guidance))
        if self.delay:
            await asyncio.sleep(self.delay)
        return _scene(spec, attempt, cost=self.cost)


class FakeCritic:
    def __init__(self, *decisions: NemotronCriticDecision) -> None:
        self.decisions = list(decisions) or [NemotronCriticDecision.ACCEPT]
        self.calls = 0

    async def evaluate(
        self,
        spec: AnticipatorySceneSpec,
        scene: RenderedScene,
    ) -> NemotronCriticEvidence:
        del spec, scene
        index = min(self.calls, len(self.decisions) - 1)
        self.calls += 1
        return _evidence(self.decisions[index])


def test_contract_has_no_raw_reader_data_escape_hatch() -> None:
    payload = _spec().model_dump(mode="json")
    with pytest.raises(ValidationError, match="source_text"):
        AnticipatorySceneSpec.model_validate({**payload, "source_text": "The private passage"})
    with pytest.raises(ValidationError, match="audio_removed"):
        PrivacyAttestation.model_validate(
            {
                "edge_gate_revision": "test",
                "audio_removed": False,
            }
        )


def test_accept_then_commit_cancels_the_unused_ready_branch() -> None:
    async def scenario() -> None:
        renderer = FakeRenderer()
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=renderer,
            critic=FakeCritic(),
            now=lambda: NOW,
        )
        await orchestrator.submit(_batch(_spec("left_path"), _spec("right_path", seed=8)))
        await asyncio.gather(
            orchestrator.wait_terminal(SESSION_A, 3, "left_path"),
            orchestrator.wait_terminal(SESSION_A, 3, "right_path"),
        )
        status = await orchestrator.commit(
            CommitRequest(session_token=SESSION_A, sequence=3, branch_id="right_path")
        )
        states = {candidate.spec.branch_id: candidate.state for candidate in status.candidates}
        assert states == {
            "left_path": CandidateState.CANCELLED,
            "right_path": CandidateState.COMMITTED,
        }
        assert status.metrics.committed == 1
        assert status.metrics.cancelled == 1
        assert status.metrics.wasted_render_cost_usd == pytest.approx(0.004)
        await orchestrator.close()

    asyncio.run(scenario())


def test_critic_can_request_exactly_one_bounded_repair() -> None:
    async def scenario() -> None:
        renderer = FakeRenderer()
        critic = FakeCritic(
            NemotronCriticDecision.REFINE,
            NemotronCriticDecision.ACCEPT,
        )
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=renderer,
            critic=critic,
            now=lambda: NOW,
        )
        await orchestrator.submit(_batch(_spec()))
        record = await orchestrator.wait_terminal(SESSION_A, 3, "moon_path")
        assert record.state is CandidateState.READY
        assert record.render_attempts == 2
        assert record.repair_attempts == 1
        assert len(record.critic_history) == 2
        assert len(renderer.calls) == 2
        assert renderer.calls[1][2] == _spec().visual_brief
        assert renderer.calls[1][3] == record.critic_history[0].verdict.correction_visual_brief
        assert record.render_cost_usd == pytest.approx(0.008)
        await orchestrator.close()

    asyncio.run(scenario())


def test_second_critic_failure_rejects_without_a_third_render() -> None:
    async def scenario() -> None:
        renderer = FakeRenderer()
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=renderer,
            critic=FakeCritic(
                NemotronCriticDecision.REFINE,
                NemotronCriticDecision.REJECT,
            ),
            now=lambda: NOW,
        )
        await orchestrator.submit(_batch(_spec()))
        record = await orchestrator.wait_terminal(SESSION_A, 3, "moon_path")
        assert record.state is CandidateState.REJECTED
        assert record.render_attempts == 2
        assert len(renderer.calls) == 2
        await orchestrator.close()

    asyncio.run(scenario())


def test_repair_cannot_weaken_the_original_acceptance_contract() -> None:
    class DriftingCritic:
        def __init__(self) -> None:
            self.requests: list[AnticipatorySceneSpec] = []

        async def evaluate(self, spec, scene):
            self.requests.append(spec)
            # The proposed correction omits the fox, gate, and crossing action.
            if len(self.requests) == 1:
                evidence = _evidence(NemotronCriticDecision.REFINE)
                return evidence.model_copy(update={
                    "verdict": evidence.verdict.model_copy(update={
                        "correction_visual_brief": "A luminous indigo garden under moonlight."
                    })
                })
            decision = (
                NemotronCriticDecision.REJECT
                if "fox" in spec.visual_brief
                else NemotronCriticDecision.ACCEPT
            )
            return _evidence(decision)

    async def scenario() -> None:
        renderer = FakeRenderer()
        critic = DriftingCritic()
        original = _spec()
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=renderer, critic=critic, now=lambda: NOW,
        )
        try:
            await orchestrator.submit(_batch(original))
            record = await orchestrator.wait_terminal(SESSION_A, 3, "moon_path")
            assert record.state is CandidateState.REJECTED
            assert critic.requests == [original, original]
            assert renderer.calls[1][2] == original.visual_brief
            assert renderer.calls[1][3] == "A luminous indigo garden under moonlight."
            assert record.render_attempts == 2
            # A rejected repair must not seed a successful cache entry.
            await orchestrator.submit(_batch(_spec("later"), session=SESSION_B))
            await orchestrator.wait_terminal(SESSION_B, 3, "later")
            assert len(renderer.calls) == 4
        finally:
            await orchestrator.close()

    asyncio.run(scenario())


def test_accepted_scene_is_content_addressed_across_private_sessions() -> None:
    async def scenario() -> None:
        renderer = FakeRenderer()
        critic = FakeCritic()
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=renderer,
            critic=critic,
            now=lambda: NOW,
        )
        await orchestrator.submit(_batch(_spec()))
        first = await orchestrator.wait_terminal(SESSION_A, 3, "moon_path")
        replay_spec = _spec("replayed", sequence=9, not_after=NOW + timedelta(minutes=4))
        replay_batch = _batch(replay_spec, session=SESSION_B, sequence=9)
        replay_status = await orchestrator.submit(replay_batch)
        replay = replay_status.candidates[0]
        assert first.state is CandidateState.READY
        assert replay.state is CandidateState.READY
        assert replay.cache_hit is True
        assert replay.render_attempts == 0
        assert replay.render_cost_usd == 0
        assert len(renderer.calls) == 1
        assert critic.calls == 1
        await orchestrator.close()

    asyncio.run(scenario())


def test_renderer_cost_overrun_fails_closed() -> None:
    async def scenario() -> None:
        orchestrator = AnticipatorySceneOrchestrator(
            renderer=FakeRenderer(cost=0.03),
            critic=FakeCritic(),
            now=lambda: NOW,
        )
        await orchestrator.submit(_batch(_spec()))
        record = await orchestrator.wait_terminal(SESSION_A, 3, "moon_path")
        assert record.state is CandidateState.FAILED
        assert "cost ceiling" in (record.error or "")
        assert not record.critic_history
        await orchestrator.close()

    asyncio.run(scenario())
