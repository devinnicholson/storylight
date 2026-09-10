import json
from uuid import uuid4

from fastapi.testclient import TestClient

from storylight.api import app
from storylight.config import Settings


def test_prepared_cue_activation_without_generation(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, model_backend="fake", asset_backend="fake", live_scene_backend="fake",
        asr_backend="disabled", live_scene_enable_motion=False,
        data_dir=tmp_path / "data", cache_dir=tmp_path / "cache",
        demo_cues_path=tmp_path / "cues.json",
    )
    monkeypatch.setattr("storylight.api.get_settings", lambda: settings)
    with TestClient(app) as client:
        generated = client.post("/v1/live-scenes", json={
            "text": "A white rabbit holds a golden watch.", "seed": 42,
            "display_when_complete": True,
        }).json()
        client.get(f"/v1/live-scenes/{generated['job_id']}/events")
        job = client.get(f"/v1/live-scenes/{generated['job_id']}").json()
        assert job["complete"] and job["stage"] != "failed"
        settings.demo_cues_path.write_text(json.dumps({"scenes": [{
            "scene_id": "rabbit", "title": "Rabbit", "cues": ["white rabbit"],
            "request": job["request"], "pack": job["story_pack"], "provider": job["provider"],
        }]}))
        assert client.get("/v1/demo-cues").json()["scenes"][0]["cues"] == ["white rabbit"]

        async def forbidden(*args, **kwargs):
            raise AssertionError("Cue playback must never call a provider")

        monkeypatch.setattr(app.state.live_scenes, "submit", forbidden)
        empty = client.get("/v1/live-scene-sessions/cue-test")
        payload = {"scene_id": "rabbit", "session_id": "cue-test",
                   "activation_id": str(uuid4()), "session_revision": 0,
                   "server_instance_id": empty.headers["x-storylight-server-instance-id"]}
        response = client.post("/v1/demo-cues/activate", json=payload)
        assert response.status_code == 200, response.text
        shown = response.json()
        assert shown["job"]["complete"] and not shown["job"]["request"]["defer_presentation"]
        assert client.post("/v1/demo-cues/activate", json=payload).json() == shown
        stale = {**payload, "activation_id": str(uuid4())}
        assert client.post("/v1/demo-cues/activate", json=stale).status_code == 409
        assert client.post("/v1/demo-cues/activate", json={
            **stale, "scene_id": "unknown",
        }).status_code == 404
