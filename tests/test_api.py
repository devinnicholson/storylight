import os

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402

from bookforge.api import app  # noqa: E402
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


def test_health_and_model_probe() -> None:
    with TestClient(app) as client:
        health = client.get("/healthz")
        probe = client.get("/v1/models:probe")
        workbench = client.get("/workbench")
        projector = client.get("/projector")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert probe.status_code == 200
    assert probe.json()["ready"] is True
    assert workbench.status_code == 200
    assert "Speak a story into a world" in workbench.text
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
