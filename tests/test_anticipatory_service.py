from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from bookforge.anticipatory import (
    AnticipatorySceneOrchestrator,
    AnticipatorySceneSpec,
    RenderedScene,
)
from bookforge.anticipatory_gcp import MemorySceneAssetStore
from bookforge.anticipatory_service import (
    AnticipatoryRuntime,
    SingleFlightPrewarm,
    create_anticipatory_service,
)
from bookforge.nemotron_critic import (
    NemotronCriticDecision,
    NemotronCriticEvidence,
    NemotronCriticVerdict,
)

SESSION = "anticipate_0123456789abcdef01234567"


class Renderer:
    async def render(
        self,
        spec: AnticipatorySceneSpec,
        *,
        attempt: int,
    ) -> RenderedScene:
        digest = hashlib.sha256(f"{spec.cache_key}-{attempt}".encode()).hexdigest()
        return RenderedScene(
            master_ref=f"asset_{digest}",
            depth_ref=f"asset_{digest}",
            master_sha256=digest,
            depth_sha256=digest,
            provider="test-renderer",
            model="test-model",
            model_revision="test-revision",
            render_latency_ms=1,
            estimated_gpu_usd=0.001,
        )


class Critic:
    async def evaluate(self, spec, scene) -> NemotronCriticEvidence:
        del spec, scene
        return NemotronCriticEvidence(
            verdict=NemotronCriticVerdict(
                fidelity_score=0.95,
                composition_score=0.9,
                projection_legibility_score=0.9,
                identity_consistent=True,
                unintended_text=False,
                decision=NemotronCriticDecision.ACCEPT,
                reason="The bounded scene direction is represented in the synthetic plate.",
            ),
            model="nemotron-test",
            latency_ms=1,
        )


def _payload() -> dict:
    future = datetime.now(UTC) + timedelta(minutes=5)
    return {
        "session_token": SESSION,
        "sequence": 1,
        "candidates": [
            {
                "branch_id": "known_next",
                "sequence": 1,
                "source": "exact_lookahead",
                "visual_brief": "One silver fox crosses a moonlit bridge into an indigo garden.",
                "expected_subjects": ["one silver fox", "moonlit bridge"],
                "continuity_sha256": hashlib.sha256(b"continuity").hexdigest(),
                "seed": 3,
                "not_after": future.isoformat(),
                "max_render_cost_usd": 0.02,
                "privacy": {"edge_gate_revision": "edge-test-v1"},
            }
        ],
        "session_cost_ceiling_usd": 0.04,
    }


def test_http_service_accepts_only_scene_contract_and_commits_candidate() -> None:
    store = MemorySceneAssetStore()
    orchestrator = AnticipatorySceneOrchestrator(
        renderer=Renderer(),
        critic=Critic(),
    )

    async def probe() -> tuple[bool, str]:
        return True, "ready"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(orchestrator, store, probe, probe, probe)
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["privacy_boundary"] == "sanitized_scene_spec_v1"
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["renderer"]["ready"] is None
        renderer_probe = client.post(
            "/v1/renderer:probe",
            json={"authorization": "I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU"},
        )
        assert renderer_probe.status_code == 200
        runtime_prewarm = client.post(
            "/v1/runtime:prewarm",
            json={"authorization": "I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU"},
        )
        assert runtime_prewarm.status_code == 200
        assert runtime_prewarm.json() == {"ready": True, "detail": "ready"}

        unsafe = _payload()
        unsafe["candidates"][0]["source_text"] = "private passage"
        assert client.post("/v1/anticipations", json=unsafe).status_code == 422

        created = client.post("/v1/anticipations", json=_payload())
        assert created.status_code == 202
        deadline = time.monotonic() + 1
        snapshot = created.json()
        while snapshot["candidates"][0]["state"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
            snapshot = client.get(f"/v1/anticipations/{SESSION}/1").json()
        committed = client.post(
            "/v1/anticipations:commit",
            json={"session_token": SESSION, "sequence": 1, "branch_id": "known_next"},
        )
        assert committed.status_code == 200
        assert committed.json()["candidates"][0]["state"] == "committed"


def test_long_poll_wakes_on_completion_and_timeout_does_not_cancel_generation():
    class SlowCritic(Critic):
        calls = 0

        async def evaluate(self, spec, scene):
            self.calls += 1
            await asyncio.sleep(0.1)
            return await super().evaluate(spec, scene)

    critic = SlowCritic()
    orchestrator = AnticipatorySceneOrchestrator(renderer=Renderer(), critic=critic)

    async def probe():
        return True, "ready"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(
            orchestrator,
            MemorySceneAssetStore(),
            probe,
            probe,
        )
    )
    with TestClient(app) as client:
        assert client.post("/v1/anticipations", json=_payload()).status_code == 202
        path = f"/v1/anticipations/{SESSION}/1"
        first = client.get(path, params={"wait_seconds": 0.001})
        assert first.status_code == 200
        assert first.json()["candidates"][0]["state"] not in {"cancelled", "failed"}
        ready = client.get(path, params={"wait_seconds": 1})
        assert ready.status_code == 200
        assert ready.json()["candidates"][0]["state"] == "ready"
        assert critic.calls == 1
        missing = client.get(path, params={"wait_seconds": 1, "branch_id": "absent"})
        assert missing.status_code == 404


@pytest.mark.parametrize("wait", ["-1", "21", "NaN", "inf"])
def test_long_poll_rejects_unbounded_wait(wait):
    async def probe():
        return True, "ready"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(
            AnticipatorySceneOrchestrator(renderer=Renderer(), critic=Critic()),
            MemorySceneAssetStore(),
            probe,
            probe,
        )
    )
    with TestClient(app) as client:
        assert (
            client.get(f"/v1/anticipations/{SESSION}/1", params={"wait_seconds": wait}).status_code
            == 422
        )


def test_asset_endpoint_is_private_cache_content_with_checksum() -> None:
    store = MemorySceneAssetStore()
    orchestrator = AnticipatorySceneOrchestrator(renderer=Renderer(), critic=Critic())

    async def probe() -> tuple[bool, str]:
        return True, "ready"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(orchestrator, store, probe, probe)
    )
    content = b"\xff\xd8\xffsynthetic-scene"
    with TestClient(app) as client:
        ref = asyncio.run(
            store.put(
                content,
                media_type="image/jpeg",
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        response = client.get(f"/v1/assets/{ref}")
        assert response.status_code == 200
        assert response.content == content
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-sha256"] == hashlib.sha256(content).hexdigest()


def test_readiness_fails_when_either_gpu_dependency_is_unavailable() -> None:
    store = MemorySceneAssetStore()
    orchestrator = AnticipatorySceneOrchestrator(renderer=Renderer(), critic=Critic())

    renderer_calls = 0

    async def renderer_probe() -> tuple[bool, str]:
        nonlocal renderer_calls
        renderer_calls += 1
        return False, "renderer asleep"

    async def critic_probe() -> tuple[bool, str]:
        return False, "critic loading"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(orchestrator, store, renderer_probe, critic_probe)
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["critic"] == {"ready": False, "detail": "critic loading"}
        assert renderer_calls == 0


def test_runtime_prewarm_requires_explicit_authorization_and_configuration() -> None:
    store = MemorySceneAssetStore()
    orchestrator = AnticipatorySceneOrchestrator(renderer=Renderer(), critic=Critic())

    async def probe() -> tuple[bool, str]:
        return True, "ready"

    app = create_anticipatory_service(
        lambda: AnticipatoryRuntime(orchestrator, store, probe, probe)
    )
    with TestClient(app) as client:
        assert client.post("/v1/runtime:prewarm", json={}).status_code == 422
        response = client.post(
            "/v1/runtime:prewarm",
            json={"authorization": "I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU"},
        )
        assert response.status_code == 501


def test_paid_runtime_prewarm_is_single_flight_and_cooldown_bounded() -> None:
    async def scenario() -> None:
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()
        now = 10.0

        async def operation() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return True, "both GPU paths warm"

        gate = SingleFlightPrewarm(
            operation,
            cooldown_seconds=60,
            clock=lambda: now,
        )
        first = asyncio.create_task(gate())
        await started.wait()
        second = asyncio.create_task(gate())
        release.set()
        results = await asyncio.gather(first, second)
        assert calls == 1
        assert results[0] == (True, "both GPU paths warm")
        assert results[1][0] is True
        assert "recent result reused" in results[1][1]

    asyncio.run(scenario())
