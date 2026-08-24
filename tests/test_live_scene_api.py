import hashlib
import json
import os
from types import SimpleNamespace

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"
os.environ["BOOKFORGE_ASSET_BACKEND"] = "fake"
os.environ["BOOKFORGE_DATA_DIR"] = "/tmp/bookforge-live-scene-api-tests/data"
os.environ["BOOKFORGE_CACHE_DIR"] = "/tmp/bookforge-live-scene-api-tests/cache"

from fastapi.testclient import TestClient  # noqa: E402

from bookforge.api import app  # noqa: E402
from bookforge.finite_modal_provider import (  # noqa: E402
    WarmPrewarmReport,
    WarmProviderStatus,
)


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


def test_live_scene_api_progresses_to_resolvable_motion_scene() -> None:
    with TestClient(app) as client:
        created = client.post("/v1/live-scenes", json=_payload())
        assert created.status_code == 202
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
            "planning_ms",
            "planning_status",
            "warm_state",
            "gpu",
            "estimated_gpu_usd",
            "cost_source",
            "models",
            "milestones_ms",
        }
        assert metrics["provider_ms"] == 30
        assert metrics["inference_ms"] == 22
        assert metrics["planning_ms"] == 2
        assert metrics["planning_status"] == "deterministic"
        assert metrics["cost_source"] == "fixture"
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

    assert raw_media.status_code == 422
    assert missing.status_code == 404
    assert missing_session.status_code == 404
    assert invalid_cursor.status_code == 400
    assert fake_warm_status.status_code == 409
    assert fake_prewarm.status_code == 409
    assert remote_create.status_code == 403
    assert remote_status.status_code == 403
    assert remote_events.status_code == 403
    assert remote_session.status_code == 403
    assert remote_session_events.status_code == 403
    assert remote_prewarm.status_code == 403


def test_explicit_warm_provider_routes_are_strict_and_never_automatic() -> None:
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
        invalid = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "x", "include_motion": False},
        )
        unused_motion = client.post(
            "/v1/live-scene-provider/prewarm",
            json={"prewarm_id": "motion-prewarm", "include_motion": True},
        )

    assert status_response.status_code == 200
    assert status_response.json() == {
        "ready": True,
        "detail": "deployed classes reachable",
        "state": "idle",
        "prewarm_id": None,
        "include_motion": False,
        "expires_in_seconds": 0.0,
    }
    assert prewarm_response.status_code == 200
    assert prewarm_response.json()["prewarm_id"] == "demo-prewarm"
    assert prewarm_response.json()["expires_in_seconds"] == 30
    assert invalid.status_code == 422
    assert unused_motion.status_code == 409


def test_live_scene_session_endpoint_recovers_latest_job_and_advances_monotonically() -> None:
    first_payload = {**_payload(), "text": "The first browser scene."}
    second_payload = {**_payload(), "text": "The replacement browser scene."}
    with TestClient(app) as client:
        first = client.post("/v1/live-scenes", json=first_payload).json()
        refreshed_mid_job = client.get(
            "/v1/live-scene-sessions/typed-scene-demo"
        )
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
