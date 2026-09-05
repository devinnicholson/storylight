from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from bookforge.anticipatory import (
    AnticipationMetrics,
    AnticipationSource,
    AnticipationStatus,
    AnticipatoryBatchRequest,
    CandidateRecord,
    CandidateState,
)
from bookforge.anticipatory_edge import (
    AnticipatoryEdgeClient,
    AnticipatoryEdgeCoordinator,
    AnticipatoryEdgeError,
    LocalAnticipationCandidate,
    LocalAnticipationPrepareRequest,
    scene_spec_from_local_plan,
)
from bookforge.domain import ModelMetrics
from bookforge.live_scene_planner import (
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlanningResult,
)

NOW = datetime.now(UTC)
SESSION = "anticipate_0123456789abcdef01234567"


def _plan() -> LiveScenePlan:
    return LiveScenePlan(
        scene_summary="A silver fox enters a moonlit garden",
        art_direction="Layered indigo paper theater with luminous gold edges",
        camera_motion="slow_push",
        background_prompt="An indigo garden behind a round moon gate",
        focus_label="silver fox",
        focus=LiveScenePlacedLayerPlan(
            kind="character",
            prompt="one silver fox stepping through the gate",
            anchor=(0.2, 0.35, 0.45, 0.55),
            depth=4,
            motion="drift",
        ),
        accent=LiveScenePlacedLayerPlan(
            kind="effect",
            prompt="gold fireflies rising beside the gate",
            anchor=(0.6, 0.2, 0.3, 0.45),
            depth=8,
            motion="float",
        ),
        ambience=["fireflies"],
    )


def _spec():
    return scene_spec_from_local_plan(
        _plan(),
        source_text="Mira whispered that the hidden garden had finally opened.",
        branch_id="known_next",
        sequence=1,
        source=AnticipationSource.EXACT_LOOKAHEAD,
        visual_style="luminous watercolor paper theater",
        seed=4,
        not_after=NOW + timedelta(minutes=5),
        edge_gate_revision="semantic-v18-test",
    )


def _status() -> AnticipationStatus:
    spec = _spec()
    record = CandidateRecord(
        spec=spec,
        state=CandidateState.QUEUED,
        created_at=NOW,
        updated_at=NOW,
    )
    return AnticipationStatus(
        session_token=SESSION,
        sequence=1,
        candidates=[record],
        metrics=AnticipationMetrics(
            candidates=1,
            ready=0,
            committed=0,
            cache_hits=0,
            cancelled=0,
            rejected=0,
            failed=0,
            repair_attempts=0,
            render_cost_usd=0,
            wasted_render_cost_usd=0,
        ),
    )


def test_local_plan_bridge_runs_privacy_gate_and_discards_source_text() -> None:
    spec = _spec()
    serialized = spec.model_dump_json()
    assert spec.privacy.raw_text_removed is True
    assert "Mira" not in serialized
    assert "whispered" not in serialized
    assert "source_text" not in serialized
    assert spec.expected_subjects == [
        "one silver fox stepping through the gate",
        "gold fireflies rising beside the gate",
    ]


def test_edge_client_fetches_fixed_path_assets_and_verifies_both_checksums() -> None:
    master = b"\xff\xd8\xffmaster"
    depth = b"\x89PNG\r\n\x1a\ndepth"
    master_sha = hashlib.sha256(master).hexdigest()
    depth_sha = hashlib.sha256(depth).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(master_sha):
            return httpx.Response(200, content=master, headers={"X-Content-SHA256": master_sha})
        return httpx.Response(200, content=depth, headers={"X-Content-SHA256": depth_sha})

    async def scenario() -> None:
        client = AnticipatoryEdgeClient(
            base_url="http://127.0.0.1:18082",
            allow_loopback_http=True,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        from bookforge.anticipatory import RenderedScene

        scene = RenderedScene(
            master_ref=f"asset_{master_sha}",
            depth_ref=f"asset_{depth_sha}",
            master_sha256=master_sha,
            depth_sha256=depth_sha,
            provider="test",
            model="test",
            model_revision="test",
            render_latency_ms=1,
            estimated_gpu_usd=0,
        )
        fetched = await client.fetch_scene(scene)
        assert fetched == (master, depth)
        await client.aclose()

    asyncio.run(scenario())


def test_edge_client_rejects_insecure_remote_url_and_checksum_mismatch() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        AnticipatoryEdgeClient(base_url="http://anticipatory.example")

    content = b"\xff\xd8\xffchanged"
    expected = hashlib.sha256(b"expected").hexdigest()

    async def scenario() -> None:
        client = AnticipatoryEdgeClient(
            base_url="http://127.0.0.1:18082",
            allow_loopback_http=True,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        content=content,
                        headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()},
                    )
                ),
                **kwargs,
            ),
        )
        with pytest.raises(AnticipatoryEdgeError, match="checksum"):
            await client.fetch_asset(f"asset_{expected}", expected_sha256=expected)
        await client.aclose()

    asyncio.run(scenario())


def test_edge_coordinator_plans_private_text_locally_and_submits_only_sanitized_spec() -> None:
    observed_batches: list[AnticipatoryBatchRequest] = []

    class Planner:
        async def plan(self, *, text: str, visual_style: str, seed: int):
            assert text == "Mira whispered that the hidden garden had finally opened."
            assert visual_style == "luminous watercolor paper theater"
            assert seed == 4
            return LiveScenePlanningResult(
                plan=_plan(),
                metrics=ModelMetrics(backend="tensorrt", model="gemma-edge", total_ms=12),
                model_revision="sha256:edge-planner",
                wall_ms=12,
                cache_hit=False,
            )

    class Remote:
        async def submit(self, batch: AnticipatoryBatchRequest) -> AnticipationStatus:
            observed_batches.append(batch)
            return _status().model_copy(
                update={
                    "session_token": batch.session_token,
                    "sequence": batch.sequence,
                    "candidates": [
                        _status().candidates[0].model_copy(update={"spec": batch.candidates[0]})
                    ],
                }
            )

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        coordinator = AnticipatoryEdgeCoordinator(
            planner=Planner(),
            client=Remote(),  # type: ignore[arg-type]
            edge_gate_revision="semantic-v18-test",
            now=lambda: NOW,
        )
        result = await coordinator.prepare(
            LocalAnticipationPrepareRequest(
                session_token=SESSION,
                sequence=1,
                candidates=[
                    LocalAnticipationCandidate(
                        branch_id="known_next",
                        text="Mira whispered that the hidden garden had finally opened.",
                        source=AnticipationSource.EXACT_LOOKAHEAD,
                        seed=4,
                    )
                ],
                session_cost_ceiling_usd=0.04,
            )
        )
        assert result.planning[0].model == "gemma-edge"
        assert result.privacy_boundary == "sanitized_scene_spec_v1"
        await coordinator.aclose()

    asyncio.run(scenario())
    payload = observed_batches[0].model_dump_json()
    assert "Mira" not in payload
    assert "whispered" not in payload
    assert "source_text" not in payload
