import os
from hashlib import sha256
from pathlib import Path

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"
os.environ["BOOKFORGE_ASSET_BACKEND"] = "fake"
os.environ["BOOKFORGE_DATA_DIR"] = "/tmp/bookforge-api-tests/data"
os.environ["BOOKFORGE_CACHE_DIR"] = "/tmp/bookforge-api-tests/cache"

from fastapi.testclient import TestClient  # noqa: E402

from bookforge.api import _jetson_writable_runtime_path, app  # noqa: E402
from bookforge.asr_backend import DisabledAsrBackend  # noqa: E402
from bookforge.domain import TranscriptionResponse  # noqa: E402


class FakeTranscriber:
    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        assert audio == b"recorded-audio"
        assert content_type == "audio/webm"
        return TranscriptionResponse(
            text="The moon gate opened.",
            language="en",
            model="fake-whisper",
            total_ms=12.5,
            audio_bytes=len(audio),
        )


def test_jetson_relative_runtime_paths_resolve_under_writable_data_dir() -> None:
    data_dir = Path("/var/lib/bookforge")
    relative = Path("artifacts/live-scenes/modal-ledger.json")

    assert _jetson_writable_runtime_path("jetson", data_dir, relative) == data_dir / relative
    assert _jetson_writable_runtime_path("development", data_dir, relative) == relative
    assert _jetson_writable_runtime_path("jetson", data_dir, Path("/tmp/ledger.json")) == Path(
        "/tmp/ledger.json"
    )


def test_health_and_model_probe() -> None:
    with TestClient(app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        runtime = client.get("/v1/runtime:status")
        probe = client.get("/v1/models:probe")
        workbench = client.get("/workbench")
        projector = client.get("/projector")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json() == {"ready": True}
    assert runtime.status_code == 200
    assert runtime.json()["storage"]["ready"] is True
    assert runtime.json()["loopback_only"] is True
    assert probe.status_code == 200
    assert probe.json()["ready"] is True
    assert workbench.status_code == 200
    assert "Make the story react as you read" in workbench.text
    assert projector.status_code == 200
    assert "Bookforge projection stage" in projector.text


def test_compile_demo_contract() -> None:
    payload = {
        "story_id": "moon-gate-demo",
        "title": "The Moon Gate",
        "reading_level": 2,
        "visual_style": "luminous paper theater",
        "pages": [
            {
                "page_id": "page-01",
                "text": "The small moth went through the red gate.",
                "art_direction": "A moonlit hill.",
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/story-packs:compile", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["story_pack"]["story_id"] == "moon-gate-demo"
    assert body["story_pack"]["pages"][0]["page_id"] == "page-01"

    with TestClient(app) as client:
        latest = client.get("/v1/story-packs/latest")
    assert latest.status_code == 200
    assert latest.json()["story_id"] == "moon-gate-demo"


def test_build_story_pack_generates_and_stores_ready_assets() -> None:
    payload = {
        "story_id": "silver-fox-api",
        "title": "The Silver Fox",
        "reading_level": 2,
        "visual_style": "layered watercolor paper theater",
        "pages": [
            {
                "page_id": "page-01",
                "text": "A silver fox found a lantern under the old cedar tree.",
                "art_direction": "Blue hour with warm lantern light.",
            }
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/story-packs:build", json=payload)
        latest = client.get("/v1/story-packs/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["story_pack"]["story_id"] == "silver-fox-api"
    assert body["generation_metrics"]["generated_assets"] == 2
    assert {asset["role"] for asset in body["story_pack"]["assets"]} == {"master", "depth"}
    assert all(asset["state"] == "ready" for asset in body["story_pack"]["assets"])
    assert latest.json()["story_id"] == "silver-fox-api"


def test_local_audio_transcription_contract() -> None:
    with TestClient(app) as client:
        app.state.transcriber = FakeTranscriber()
        response = client.post(
            "/v1/audio:transcribe",
            content=b"recorded-audio",
            headers={"Content-Type": "audio/webm"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "The moon gate opened."
    assert response.json()["model"] == "fake-whisper"


def test_disabled_audio_backend_fails_with_service_unavailable() -> None:
    with TestClient(app) as client:
        app.state.transcriber = DisabledAsrBackend(reason="ASR is disabled for device bring-up")
        response = client.post(
            "/v1/audio:transcribe",
            content=b"recorded-audio",
            headers={"Content-Type": "audio/webm"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "ASR is disabled for device bring-up"}


def test_audio_limit_is_enforced_before_the_backend() -> None:
    with TestClient(app) as client:
        original_limit = app.state.settings.asr_max_audio_mb
        app.state.settings.asr_max_audio_mb = 1
        response = client.post(
            "/v1/audio:transcribe",
            content=b"x" * (1024 * 1024 + 1),
            headers={"Content-Type": "audio/webm"},
        )
        app.state.settings.asr_max_audio_mb = original_limit

    assert response.status_code == 413


def test_sensitive_api_routes_reject_remote_clients() -> None:
    with TestClient(app, client=("203.0.113.4", 50000)) as client:
        runtime = client.get("/v1/runtime:status")
        audio = client.post(
            "/v1/audio:transcribe",
            content=b"audio",
            headers={"Content-Type": "audio/webm"},
        )
        compile_response = client.post(
            "/v1/story-packs:compile",
            json={
                "story_id": "remote",
                "title": "Remote",
                "reading_level": 2,
                "visual_style": "paper",
                "pages": [
                    {
                        "page_id": "page-01",
                        "text": "Remote text.",
                        "art_direction": "None.",
                    }
                ],
            },
        )

    assert runtime.status_code == 403
    assert audio.status_code == 403
    assert compile_response.status_code == 403


def test_cached_asset_route_serves_only_validated_cache_paths() -> None:
    content = b"cached-image"
    checksum = sha256(content).hexdigest()
    with TestClient(app) as client:
        directory = app.state.asset_cache.root / checksum
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sky.png").write_bytes(content)

        response = client.get(f"/v1/assets/{checksum}/sky.png")
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["etag"] == f'"{checksum}"'
        traversal = client.get(f"/v1/assets/{checksum}/..%2Fsecret")

    assert response.status_code == 200
    assert response.content == content
    assert traversal.status_code in {404, 422}
