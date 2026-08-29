import hashlib
import json
import os
from types import SimpleNamespace

import pytest

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"
os.environ["BOOKFORGE_ASSET_BACKEND"] = "fake"
os.environ["BOOKFORGE_DATA_DIR"] = "/tmp/bookforge-live-scene-api-tests/data"
os.environ["BOOKFORGE_CACHE_DIR"] = "/tmp/bookforge-live-scene-api-tests/cache"

from fastapi.testclient import TestClient  # noqa: E402

from bookforge.api import _completed_pack_matches_planner_mode, app  # noqa: E402
from bookforge.config import Settings  # noqa: E402
from bookforge.finite_modal_provider import (  # noqa: E402
    WarmPrewarmReport,
    WarmProviderStatus,
)
from bookforge.live_scene import DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL  # noqa: E402
from bookforge.nemotron_critic import (  # noqa: E402
    NemotronCriticEvidence,
    NemotronCriticVerdict,
)


@pytest.fixture(autouse=True)
def _isolated_live_scene_settings(monkeypatch, tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        model_backend="fake",
        model_name="fake",
        asset_backend="fake",
        live_scene_enable_motion=True,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr("bookforge.api.get_settings", lambda: settings)


def _payload() -> dict[str, object]:
    return {
        "text": "A silver fox opens a book and glowing letters fill the room.",
        "visual_style": "luminous watercolor paper theater",
        "seed": 90210,
        "session_id": "typed-scene-demo",
    }


def _sse_data(response_text: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def test_model_planner_rejects_completed_deterministic_fallback_cache() -> None:
    stale_deterministic = SimpleNamespace(compiler_model="deterministic-live-scene-planner-v1")
    current_deterministic = SimpleNamespace(compiler_model=DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL)
    model_planned = SimpleNamespace(compiler_model="gemma3:1b-it-q4_K_M")

    assert not _completed_pack_matches_planner_mode(stale_deterministic, planner_mode="model")
    assert _completed_pack_matches_planner_mode(model_planned, planner_mode="model")
    assert not _completed_pack_matches_planner_mode(
        stale_deterministic,
        planner_mode="deterministic",
    )
    assert _completed_pack_matches_planner_mode(
        current_deterministic,
        planner_mode="deterministic",
    )


def test_live_scene_api_progresses_to_resolvable_motion_scene() -> None:
    with TestClient(app) as client:
        created = client.post("/v1/live-scenes", json=_payload())
        assert created.status_code == 202
        assert created.headers["x-bookforge-server-instance-id"]
        assert int(created.headers["x-bookforge-session-revision"]) >= 1
        job_id = created.json()["job_id"]

        events = client.get(f"/v1/live-scenes/{job_id}/events")
        status = client.get(f"/v1/live-scenes/{job_id}")

        snapshots = _sse_data(events.text)
        assert events.status_code == 200
        assert events.headers["content-type"].startswith("text/event-stream")
        assert "event: scene.job" in events.text
        assert snapshots
        assert snapshots[0]["revision"] >= created.json()["revision"]
        assert snapshots[-1]["stage"] == "motion_ready"
        assert snapshots[-1]["complete"] is True
        assert status.json() == snapshots[-1]

        metrics = snapshots[-1]["metrics"]
        assert set(metrics) == {
            "elapsed_ms",
            "provider_ms",
            "inference_ms",
            "cache_ms",
            "overhead_ms",
            "packaging_ms",
            "planning_ms",
            "preparation_ms",
            "planning_status",
            "planning_cache_hit",
            "scene_cache_hit",
            "warm_state",
            "gpu",
            "estimated_gpu_usd",
            "cost_source",
            "models",
            "milestones_ms",
        }
        if metrics["scene_cache_hit"]:
            assert metrics["provider_ms"] == 0
            assert metrics["inference_ms"] == 0
            assert metrics["planning_ms"] == 0
            assert metrics["cost_source"] == "unavailable"
        else:
            assert metrics["provider_ms"] == 30
            assert metrics["inference_ms"] == 22
            assert metrics["planning_ms"] == 2
            assert metrics["cost_source"] == "fixture"
        assert metrics["preparation_ms"] == 0
        assert metrics["planning_status"] == "deterministic"
        assert metrics["planning_cache_hit"] is False
        assert metrics["milestones_ms"]["motion_ready"] == metrics["elapsed_ms"]
        assert [model["role"] for model in metrics["models"]] == [
            "scene_plan",
            "master",
            "depth",
            "motion",
        ]
        assert all("metrics" in snapshot for snapshot in snapshots)

        artifacts = snapshots[-1]["artifacts"]
        assert isinstance(artifacts, list)
        assert {artifact["kind"] for artifact in artifacts} == {
            "master",
            "depth",
            "motion",
        }
        for artifact in artifacts:
            response = client.get(artifact["uri"])
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == artifact["checksum_sha256"]


def test_live_scene_sse_replays_terminal_snapshot_on_reconnect() -> None:
    with TestClient(app) as client:
        created = client.post("/v1/live-scenes", json=_payload()).json()
        first = client.get(f"/v1/live-scenes/{created['job_id']}/events")
        terminal = _sse_data(first.text)[-1]
        replay = client.get(
            f"/v1/live-scenes/{created['job_id']}/events",
            headers={"Last-Event-ID": str(terminal["revision"])},
        )

    assert _sse_data(replay.text) == [terminal]


def test_live_scene_api_reuses_exact_verified_completed_scene() -> None:
    payload = {
        **_payload(),
        "text": "A clockwork sparrow folds a map beneath violet stars.",
        "seed": 771_204,
        "session_id": "exact-replay-api-test",
    }
    with TestClient(app) as client:
        generated = client.post("/v1/live-scenes", json=payload).json()
        generated_terminal = _sse_data(
            client.get(f"/v1/live-scenes/{generated['job_id']}/events").text
        )[-1]
        replay = client.post("/v1/live-scenes", json=payload).json()
        replay_terminal = _sse_data(client.get(f"/v1/live-scenes/{replay['job_id']}/events").text)[
            -1
        ]

    assert generated_terminal["complete"] is True
    assert replay_terminal["complete"] is True
    assert replay_terminal["metrics"]["scene_cache_hit"] is True
    assert replay_terminal["metrics"]["provider_ms"] == 0
    assert replay_terminal["metrics"]["inference_ms"] == 0
    assert replay_terminal["metrics"]["estimated_gpu_usd"] == 0
    assert replay_terminal["revision"] == 3
    assert "draft_ready" not in replay_terminal["metrics"]["milestones_ms"]
    assert "master_ready" not in replay_terminal["metrics"]["milestones_ms"]
    assert [artifact["checksum_sha256"] for artifact in replay_terminal["artifacts"]] == [
        artifact["checksum_sha256"] for artifact in generated_terminal["artifacts"]
    ]


def test_live_scene_api_rejects_raw_media_unknown_jobs_and_remote_clients() -> None:
    payload = {**_payload(), "raw_audio": "not-allowed"}
    unknown_id = "scene_000000000000000000000000"
    with TestClient(app) as client:
        raw_media = client.post("/v1/live-scenes", json=payload)
        missing = client.get(f"/v1/live-scenes/{unknown_id}")
        missing_session = client.get("/v1/live-scene-sessions/no-job")
        invalid_cursor = client.get(
            f"/v1/live-scenes/{unknown_id}/events",
            headers={"Last-Event-ID": "not-an-integer"},
        )
        fake_warm_status = client.get("/v1/live-scene-provider/warm-status")
        fake_prewarm = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "demo-prewarm", "include_motion": False},
        )
        fake_planner_prepare = client.post(
            "/v1/live-scene-planner/prepare",
            json={"text": "A fox waits.", "visual_style": "paper theater"},
        )
        disabled_planner_warmup = client.post("/v1/live-scene-planner/warmup")
        disabled_critic = client.post(f"/v1/live-scenes/{unknown_id}:critique")
    with TestClient(app, client=("203.0.113.8", 50000)) as remote:
        remote_create = remote.post("/v1/live-scenes", json=_payload())
        remote_status = remote.get(f"/v1/live-scenes/{unknown_id}")
        remote_events = remote.get(f"/v1/live-scenes/{unknown_id}/events")
        remote_session = remote.get("/v1/live-scene-sessions/no-job")
        remote_session_events = remote.get("/v1/live-scene-sessions/no-job/events")
        remote_prewarm = remote.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "demo-prewarm", "include_motion": False},
        )
        remote_planner_prepare = remote.post(
            "/v1/live-scene-planner/prepare",
            json={"text": "A fox waits.", "visual_style": "paper theater"},
        )
        remote_planner_warmup = remote.post("/v1/live-scene-planner/warmup")
        remote_critic = remote.post(f"/v1/live-scenes/{unknown_id}:critique")

    assert raw_media.status_code == 422
    assert missing.status_code == 404
    assert missing_session.status_code == 404
    assert invalid_cursor.status_code == 400
    assert fake_warm_status.status_code == 409
    assert fake_prewarm.status_code == 409
    assert fake_planner_prepare.status_code == 409
    assert disabled_planner_warmup.status_code == 200
    assert disabled_planner_warmup.json() == {
        "ready": False,
        "warmup_ms": 0.0,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert disabled_critic.status_code == 409
    assert remote_create.status_code == 403
    assert remote_status.status_code == 403
    assert remote_events.status_code == 403
    assert remote_session.status_code == 403
    assert remote_session_events.status_code == 403
    assert remote_prewarm.status_code == 403
    assert remote_planner_prepare.status_code == 403
    assert remote_planner_warmup.status_code == 403
    assert remote_critic.status_code == 403


def test_nemotron_critic_rejects_non_privacy_gated_fallback_scene() -> None:
    calls: list[dict[str, object]] = []

    class StubCritic:
        async def evaluate(self, request, *, image_bytes: bytes, media_type: str):
            calls.append(
                {
                    "request": request,
                    "image_bytes": image_bytes,
                    "media_type": media_type,
                }
            )
            return NemotronCriticEvidence(
                verdict=NemotronCriticVerdict(
                    fidelity_score=0.91,
                    composition_score=0.88,
                    projection_legibility_score=0.86,
                    identity_consistent=True,
                    unintended_text=False,
                    decision="accept",
                    reason="The expected visual subjects are present and projection-readable.",
                ),
                model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                latency_ms=81.5,
            )

    secret_passage = "Quenlora whispered the private amber sentence beside a folded map."
    payload = {
        **_payload(),
        "text": secret_passage,
        "seed": 814_225,
        "session_id": "nemotron-critic-api-test",
    }
    with TestClient(app) as client:
        created = client.post("/v1/live-scenes", json=payload).json()
        terminal = _sse_data(client.get(f"/v1/live-scenes/{created['job_id']}/events").text)[-1]
        client.app.state.live_scene_critic = StubCritic()
        response = client.post(f"/v1/live-scenes/{created['job_id']}:critique")

    assert terminal["complete"] is True
    assert response.status_code == 409
    assert response.json()["detail"] == "Nemotron requires a locally privacy-gated model scene plan"
    assert calls == []


def test_live_scene_planner_prepare_primes_only_the_local_planner() -> None:
    calls: list[dict[str, object]] = []

    class StubPlanner:
        async def plan(self, *, text: str, visual_style: str, seed: int):
            calls.append({"text": text, "visual_style": visual_style, "seed": seed})
            return SimpleNamespace(
                wall_ms=8_750.5,
                cache_hit=False,
                model_revision="ollama-manifest-sha256:test",
                metrics=SimpleNamespace(
                    model="gemma3:1b-it-q4_K_M",
                    input_tokens=537,
                    output_tokens=172,
                ),
            )

    with TestClient(app) as client:
        client.app.state.live_scenes.provider = SimpleNamespace(planner=StubPlanner())
        response = client.post(
            "/v1/live-scene-planner/prepare",
            json={
                "text": "A silver fox waits below the cedar trees.",
                "visual_style": "luminous paper theater",
                "session_id": "projector-yield",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "planning_ms": 8_750.5,
        "cache_hit": False,
        "model": "gemma3:1b-it-q4_K_M",
        "revision": "ollama-manifest-sha256:test",
        "input_tokens": 537,
        "output_tokens": 172,
    }
    assert calls == [
        {
            "text": "A silver fox waits below the cedar trees.",
            "visual_style": "luminous paper theater",
            "seed": 0,
        }
    ]


def test_live_scene_planner_warmup_uses_no_request_body_or_story_text() -> None:
    calls = 0

    class StubPlanner:
        async def warmup(self):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                wall_ms=4_410.5,
                metrics=SimpleNamespace(
                    model="gemma3:1b-it-q4_K_M",
                    input_tokens=24,
                    output_tokens=5,
                ),
            )

    with TestClient(app) as client:
        client.app.state.settings.live_scene_planner_auto_warmup = True
        client.app.state.live_scenes.provider = SimpleNamespace(planner=StubPlanner())
        response = client.post("/v1/live-scene-planner/warmup")
        client.app.state.settings.live_scene_planner_auto_warmup = False

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "warmup_ms": 4_410.5,
        "model": "gemma3:1b-it-q4_K_M",
        "input_tokens": 24,
        "output_tokens": 5,
    }
    assert calls == 1


def test_explicit_warm_provider_routes_are_strict_by_default() -> None:
    class StubWarmProvider:
        async def warm_status(self) -> WarmProviderStatus:
            return WarmProviderStatus(
                ready=True,
                detail="deployed classes reachable",
                state="idle",
            )

        async def prewarm(
            self,
            *,
            prewarm_id: str,
            include_motion: bool,
            scaledown_window_seconds: int,
        ) -> WarmPrewarmReport:
            return WarmPrewarmReport(
                prewarm_id=prewarm_id,
                reservation_id="reservation-123",
                include_motion=include_motion,
                fast_remote_seconds=2.5,
                motion_remote_seconds=0,
                full_session_ceiling_usd=0.2,
                fast_model_load_seconds=2,
                motion_model_load_seconds=0,
                expires_in_seconds=30,
                scaledown_window_seconds=scaledown_window_seconds,
            )

    with TestClient(app) as client:
        client.app.state.live_scenes.provider = SimpleNamespace(
            provider=StubWarmProvider(),
            enable_motion=False,
        )
        status_response = client.get("/v1/live-scene-provider/warm-status")
        prewarm_response = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "demo-prewarm", "include_motion": False},
        )
        presentation_response = client.post(
            "/v1/live-scene-provider/prewarm",
            json={
                "prewarm_id": "demo-presentation",
                "include_motion": False,
                "scaledown_window_seconds": 600,
            },
        )
        invalid = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "x", "include_motion": False},
        )
        unused_motion = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "motion-prewarm", "include_motion": True},
        )
        excessive_window = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "too-long", "scaledown_window_seconds": 901},
        )

    assert status_response.status_code == 200
    assert status_response.json() == {
        "ready": True,
        "detail": "deployed classes reachable",
        "state": "idle",
        "prewarm_id": None,
        "include_motion": False,
        "expires_in_seconds": 0.0,
        "scaledown_window_seconds": 90,
    }
    assert prewarm_response.status_code == 200
    assert prewarm_response.json()["prewarm_id"] == "demo-prewarm"
    assert prewarm_response.json()["expires_in_seconds"] == 30
    assert presentation_response.status_code == 200
    assert presentation_response.json()["scaledown_window_seconds"] == 600
    assert invalid.status_code == 422
    assert unused_motion.status_code == 409
    assert excessive_window.status_code == 422


def test_live_scene_session_endpoint_recovers_latest_job_and_advances_monotonically() -> None:
    first_payload = {**_payload(), "text": "The first browser scene."}
    second_payload = {**_payload(), "text": "The replacement browser scene."}
    with TestClient(app) as client:
        first = client.post("/v1/live-scenes", json=first_payload).json()
        refreshed_mid_job = client.get("/v1/live-scene-sessions/typed-scene-demo")
        second = client.post("/v1/live-scenes", json=second_payload).json()
        latest = client.get("/v1/live-scene-sessions/typed-scene-demo")

    assert refreshed_mid_job.status_code == 200
    first_pointer = refreshed_mid_job.json()
    second_pointer = latest.json()
    assert first_pointer["session_id"] == "typed-scene-demo"
    assert first_pointer["server_instance_id"].startswith("server_")
    assert first_pointer["job"]["job_id"] == first["job_id"]
    assert second_pointer["job"]["job_id"] == second["job_id"]
    assert second_pointer["session_revision"] > first_pointer["session_revision"]
    assert second_pointer["server_instance_id"] == first_pointer["server_instance_id"]


def test_live_scene_session_endpoint_keeps_changed_sessions_isolated() -> None:
    with TestClient(app) as client:
        alpha = client.post(
            "/v1/live-scenes",
            json={**_payload(), "session_id": "alpha-session"},
        ).json()
        beta = client.post(
            "/v1/live-scenes",
            json={**_payload(), "session_id": "beta-session"},
        ).json()
        alpha_pointer = client.get("/v1/live-scene-sessions/alpha-session").json()
        beta_pointer = client.get("/v1/live-scene-sessions/beta-session").json()

    assert alpha_pointer["job"]["job_id"] == alpha["job_id"]
    assert beta_pointer["job"]["job_id"] == beta["job_id"]
