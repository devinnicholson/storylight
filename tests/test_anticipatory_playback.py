from __future__ import annotations

import asyncio
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from PIL import Image

from bookforge.anticipatory import (
    AnticipationMetrics,
    AnticipationStatus,
    AnticipatoryBatchRequest,
    CandidateRecord,
    CandidateState,
    RenderedScene,
)
from bookforge.anticipatory_edge import AnticipatoryEdgeClient, AnticipatoryEdgeCoordinator
from bookforge.anticipatory_playback import (
    ActivateProjectionRequest,
    AnticipatoryPlayback,
    PrepareProjectionRequest,
)
from bookforge.asset_cache import AssetCache
from bookforge.domain import ModelMetrics
from bookforge.live_scene import (
    DeterministicFakeLiveSceneProvider,
    LiveSceneConflictError,
    LiveSceneJobRegistry,
    LiveSceneNotFoundError,
)
from bookforge.live_scene_planner import (
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlanningResult,
)
from bookforge.nemotron_critic import NemotronCriticEvidence, NemotronCriticVerdict
from bookforge.story_store import StoryPackStore

REQUEST = PrepareProjectionRequest(
    text="Mira whispered that the fox had finally found the hidden garden.",
    visual_style="luminous watercolor paper theater",
    session_id="next-page-test",
    seed=42,
)


def test_playback_wait_reaches_cloud_and_keeps_transport_timeout_above_wait(tmp_path):
    async def scenario():
        rig = Rig(tmp_path)
        await rig.initialize()
        try:
            page = await rig.playback.prepare(REQUEST)
            await rig.playback.status(page.prepared_id, wait_seconds=20)
            request = rig.requests[-1]
            assert request.url.params["wait_seconds"] == "20"
            assert request.url.params["branch_id"] == "known_next"
            assert request.extensions["timeout"]["read"] >= 25
            await rig.playback.stage(page.prepared_id)
            calls = len(rig.requests)
            await rig.playback.status(page.prepared_id, wait_seconds=20)
            assert len(rig.requests) == calls
        finally:
            await rig.close()

    asyncio.run(scenario())


class Planner:
    calls = 0

    async def plan(self, **kwargs):
        self.calls += 1
        return LiveScenePlanningResult(
            plan=LiveScenePlan(
                scene_summary="A silver fox enters a moonlit garden",
                art_direction="Indigo paper theater with luminous gold edges",
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
            ),
            metrics=ModelMetrics(backend="tensorrt", model="gemma-edge-test", total_ms=12),
            model_revision="sha256:edge-planner",
            wall_ms=12,
        )


class Rig:
    def __init__(self, path: Path):
        self.cache = AssetCache(path / "cache")
        self.store = StoryPackStore(path / "packs")
        self.registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(cache=self.cache),
            completed_pack_validator=lambda pack: self.cache.install_pack(pack, Path(".")),
            completed_pack_sink=self.store.save,
        )
        self.planner = Planner()
        self.offline = False
        self.bad_digest = False
        self.bad_identity = False
        self.critic_ready = True
        self.state = CandidateState.READY
        self.requests = []
        self.records = {}
        self.images = {}
        for name, color in [("master", "teal"), ("depth", "gray")]:
            output = io.BytesIO()
            Image.new("RGB", (1024, 576), color).save(output, format="PNG")
            self.images[name] = output.getvalue()
        self.client = AnticipatoryEdgeClient(
            base_url="http://127.0.0.1:18082",
            allow_loopback_http=True,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(self.handle),
                **kwargs,
            ),
        )
        self.edge = AnticipatoryEdgeCoordinator(
            planner=self.planner,
            client=self.client,
            edge_gate_revision="test-v1",
        )
        self.playback = AnticipatoryPlayback(
            edge=self.edge,
            cache=self.cache,
            registry=self.registry,
        )

    async def initialize(self):
        await self.cache.initialize()
        await self.store.initialize()

    async def close(self):
        await self.edge.aclose()
        await self.registry.close()

    def handle(self, request):
        assert not self.offline, "Offline activation must not contact the cloud"
        self.requests.append(request)
        path = request.url.path
        if path == "/ready":
            return httpx.Response(
                200 if self.critic_ready else 503,
                json={"ready": self.critic_ready, "critic": {"ready": self.critic_ready}},
            )
        if path.startswith("/v1/assets/"):
            for content in self.images.values():
                digest = hashlib.sha256(content).hexdigest()
                if path.endswith(digest):
                    return httpx.Response(
                        200,
                        content=content if not self.bad_digest else b"bad",
                        headers={"X-Content-SHA256": digest},
                    )
            return httpx.Response(404)
        if path == "/v1/anticipations":
            batch = AnticipatoryBatchRequest.model_validate_json(request.content)
            now = datetime.now(UTC)
            master, depth = [
                hashlib.sha256(self.images[key]).hexdigest() for key in ("master", "depth")
            ]
            candidate = CandidateRecord(
                spec=batch.candidates[0],
                state=self.state,
                render_attempts=1,
                rendered=RenderedScene(
                    master_ref=f"asset_{master}",
                    depth_ref=f"asset_{depth}",
                    master_sha256=master,
                    depth_sha256=depth,
                    provider="gcp-cloud-run",
                    model="sana-sprint",
                    model_revision="fixture-v1",
                    render_latency_ms=400,
                    estimated_gpu_usd=0.002,
                ),
                critic_history=[
                    NemotronCriticEvidence(
                        verdict=NemotronCriticVerdict(
                            fidelity_score=0.95,
                            composition_score=0.9,
                            projection_legibility_score=0.9,
                            identity_consistent=True,
                            unintended_text=False,
                            decision="accept",
                            reason="Matches the brief.",
                        ),
                        model="nemotron-fixture",
                        latency_ms=6200,
                    )
                ],
                ready_latency_ms=6700,
                created_at=now,
                updated_at=now,
                error="Rejected by fixture" if self.state is CandidateState.REJECTED else None,
            )
            token = batch.session_token
            self.records[token] = AnticipationStatus(
                session_token=token,
                sequence=1,
                candidates=[candidate],
                metrics=AnticipationMetrics(
                    candidates=1,
                    ready=1,
                    committed=0,
                    cache_hits=0,
                    cancelled=0,
                    rejected=0,
                    failed=0,
                    repair_attempts=0,
                    render_cost_usd=0.002,
                    wasted_render_cost_usd=0,
                ),
            )
        elif path == "/v1/anticipations:commit":
            token = json.loads(request.content)["session_token"]
            status = self.records[token]
            self.records[token] = status.model_copy(
                update={
                    "candidates": [
                        status.candidates[0].model_copy(
                            update={"state": CandidateState.COMMITTED, "selected": True}
                        )
                    ]
                }
            )
        else:
            token = path.split("/")[-2]
        result = self.records[token]
        if self.bad_identity:
            result = result.model_copy(update={"session_token": "anticipate_" + "f" * 24})
        return httpx.Response(200, json=result.model_dump(mode="json"))


def test_prepare_stage_and_offline_activation_preserve_current_projection(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        try:
            subscription = await rig.registry.subscribe_session(REQUEST.session_id)
            await subscription.receive()  # Initial empty pointer.
            prepared = await rig.playback.prepare(REQUEST)
            assert prepared.state == "approved"
            staged = await rig.playback.stage(prepared.prepared_id)
            assert staged.state == "staged" and staged.assets_verified
            assert staged.render_ms == 400 and staged.critic_ms == 6200
            assert subscription._queue.empty()
            with pytest.raises(LiveSceneNotFoundError):
                await rig.registry.get_session(REQUEST.session_id)
            assert rig.planner.calls == 1
            assert all("Mira" not in request.content.decode() for request in rig.requests)
            rig.offline = True
            assert (await rig.playback.status(staged.prepared_id)) == staged
            assert (await rig.playback.stage(staged.prepared_id)) == staged
            activation = ActivateProjectionRequest(
                prepared_id=staged.prepared_id,
                expected_server_instance_id=rig.registry.server_instance_id,
                expected_session_revision=0,
            )
            pointer = await rig.playback.activate(activation)
            assert pointer.job.complete and pointer.job.stage == "master_ready"
            assert pointer.job.metrics.provider_ms == 0
            assert pointer.job.story_pack.pages[0].source_text == REQUEST.text
            assert {a.kind.value for a in pointer.job.artifacts} == {"master", "depth"}
            assert (await rig.store.latest()).story_id == pointer.job.story_pack.story_id
            assert (await subscription.receive()).job == pointer.job
            assert (await rig.playback.activate(activation)) == pointer
            assert subscription._queue.empty()
            await rig.playback.discard(staged.prepared_id)
            assert (await rig.registry.get_session(REQUEST.session_id)) == pointer
        finally:
            await rig.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["digest", "dimensions", "invalid_image", "rejected", "identity"]
)
def test_unsafe_scenes_never_reach_the_projector(tmp_path, failure):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        try:
            if failure == "dimensions":
                output = io.BytesIO()
                Image.new("RGB", (512, 512)).save(output, format="PNG")
                rig.images["master"] = output.getvalue()
            if failure == "invalid_image":
                rig.images["master"] = b"\x89PNG\r\n\x1a\nfake"
            if failure == "rejected":
                rig.state = CandidateState.REJECTED
            prepared = await rig.playback.prepare(REQUEST)
            rig.bad_digest = failure == "digest"
            rig.bad_identity = failure == "identity"
            with pytest.raises(RuntimeError):
                await rig.playback.stage(prepared.prepared_id)
            with pytest.raises(LiveSceneNotFoundError):
                await rig.registry.get_session(REQUEST.session_id)
        finally:
            await rig.close()

    asyncio.run(run())


def test_stale_epoch_revision_and_old_replay_cannot_override_newer_scene(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        try:
            first = await rig.playback.prepare(REQUEST)
            second = await rig.playback.prepare(REQUEST.model_copy(update={"seed": 43}))
            await rig.playback.stage(first.prepared_id)
            await rig.playback.stage(second.prepared_id)
            args = dict(
                expected_server_instance_id=rig.registry.server_instance_id,
                expected_session_revision=0,
            )
            for update in (
                {"expected_server_instance_id": "server_" + "0" * 32},
                {"expected_session_revision": 1},
            ):
                with pytest.raises(LiveSceneConflictError):
                    await rig.playback.activate(
                        ActivateProjectionRequest(prepared_id=first.prepared_id, **(args | update))
                    )
            pointer = await rig.playback.activate(
                ActivateProjectionRequest(prepared_id=first.prepared_id, **args)
            )
            with pytest.raises(LiveSceneConflictError):
                await rig.playback.activate(
                    ActivateProjectionRequest(prepared_id=second.prepared_id, **args)
                )
            args["expected_session_revision"] = pointer.session_revision
            newer = await rig.playback.activate(
                ActivateProjectionRequest(prepared_id=second.prepared_id, **args)
            )
            args["expected_session_revision"] = newer.session_revision
            with pytest.raises(LiveSceneConflictError):
                await rig.playback.activate(
                    ActivateProjectionRequest(prepared_id=first.prepared_id, **args)
                )
        finally:
            await rig.close()

    asyncio.run(run())


def test_capacity_expiry_and_discard_are_bounded_before_cloud_work(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        try:
            prepared = [await rig.playback.prepare(REQUEST) for _ in range(8)]
            with pytest.raises(RuntimeError, match="capacity"):
                await rig.playback.prepare(REQUEST)
            assert rig.planner.calls == 8
            first = rig.playback._pages[prepared[0].prepared_id]
            first.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            with pytest.raises(RuntimeError, match="expired"):
                await rig.playback.status(prepared[0].prepared_id)
            await rig.playback.discard(prepared[0].prepared_id)
            await rig.playback.discard(prepared[0].prepared_id)
            await rig.playback.prepare(REQUEST)
            assert rig.planner.calls == 9
        finally:
            await rig.close()

    asyncio.run(run())


def test_offline_activation_does_not_wait_for_another_page_planner(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        started = asyncio.Event()
        hold = asyncio.Event()
        try:
            prepared = await rig.playback.prepare(REQUEST)
            await rig.playback.stage(prepared.prepared_id)

            async def slow_plan(**kwargs):
                started.set()
                await hold.wait()

            rig.planner.plan = slow_plan
            pending = asyncio.create_task(rig.playback.prepare(REQUEST))
            await started.wait()
            pointer = await asyncio.wait_for(
                rig.playback.activate(
                    ActivateProjectionRequest(
                        prepared_id=prepared.prepared_id,
                        expected_server_instance_id=rig.registry.server_instance_id,
                        expected_session_revision=0,
                    )
                ),
                timeout=1,
            )
            assert pointer.job.complete
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert rig.playback._preparing == 0
        finally:
            await rig.close()

    asyncio.run(run())


def test_prepared_projection_http_flow_and_privacy_boundary(tmp_path):
    from fastapi.testclient import TestClient

    from bookforge.api import app

    with TestClient(app) as client:
        assert client.get("/v1/prepared-projections/runtime").json()["enabled"] is False
        assert client.post("/v1/prepared-projections", json=REQUEST.model_dump()).status_code == 409
        rig = Rig(tmp_path)
        client.portal.call(rig.initialize)
        rig.playback.registry = app.state.live_scenes
        rig.playback.cache = app.state.asset_cache
        app.state.anticipatory_playback = rig.playback
        try:
            runtime = client.get("/v1/prepared-projections/runtime")
            assert runtime.json()["enabled"] is True
            assert not rig.requests
            assert client.post("/v1/prepared-projections:prewarm", json={}).status_code == 422
            assert (
                client.get(
                    "/v1/prepared-projections/runtime",
                    headers={
                        "X-Forwarded-For": "203.0.113.4",
                    },
                ).status_code
                == 403
            )
            response = client.post("/v1/prepared-projections", json=REQUEST.model_dump())
            assert response.status_code == 202
            path = "/v1/prepared-projections/" + response.json()["prepared_id"]
            assert client.get(path).json()["state"] == "approved"
            assert client.post(path + "/stage").json()["assets_verified"] is True
            assert client.get(f"/v1/live-scene-sessions/{REQUEST.session_id}").status_code == 404
            rig.offline = True
            result = client.post(
                "/v1/prepared-projections:activate",
                json={
                    "prepared_id": response.json()["prepared_id"],
                    "expected_server_instance_id": runtime.json()["server_instance_id"],
                    "expected_session_revision": 0,
                },
            )
            assert result.status_code == 200, result.text
            for asset in result.json()["job"]["story_pack"]["assets"]:
                content = client.get(asset["local_uri"])
                assert content.status_code == 200
                assert hashlib.sha256(content.content).hexdigest() == asset["checksum_sha256"]
            assert client.delete(path).status_code == 204
        finally:
            client.portal.call(rig.close)
            app.state.anticipatory_playback = None


def test_capacity_replacement_publishes_no_empty_projection_event(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        rig.registry.max_retained_jobs = 1
        try:
            first = await rig.playback.prepare(REQUEST)
            second = await rig.playback.prepare(REQUEST.model_copy(update={"seed": 51}))
            await rig.playback.stage(first.prepared_id)
            await rig.playback.stage(second.prepared_id)
            subscription = await rig.registry.subscribe_session(REQUEST.session_id)
            await subscription.receive()
            one = await rig.playback.activate(
                ActivateProjectionRequest(
                    prepared_id=first.prepared_id,
                    expected_server_instance_id=rig.registry.server_instance_id,
                    expected_session_revision=0,
                )
            )
            await subscription.receive()
            two = await rig.playback.activate(
                ActivateProjectionRequest(
                    prepared_id=second.prepared_id,
                    expected_server_instance_id=rig.registry.server_instance_id,
                    expected_session_revision=one.session_revision,
                )
            )
            assert (await subscription.receive()).job == two.job
            assert subscription._queue.empty()
        finally:
            await rig.close()

    asyncio.run(run())


def test_nim_off_blocks_planning_generation_and_renderer_warmup(tmp_path):
    async def run():
        rig = Rig(tmp_path)
        await rig.initialize()
        rig.critic_ready = False
        try:
            with pytest.raises(RuntimeError, match="not ready"):
                await rig.playback.prepare(REQUEST)
            with pytest.raises(RuntimeError, match="not ready"):
                await rig.client.prewarm_runtime()
            assert rig.planner.calls == 0
            assert rig.playback._preparing == 0
            assert not rig.playback._pages
            assert [(request.method, request.url.path) for request in rig.requests] == [
                ("GET", "/ready"),
                ("GET", "/ready"),
            ]
        finally:
            await rig.close()

    asyncio.run(run())
