import asyncio
import os

import pytest

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from bookforge.api import app  # noqa: E402
from bookforge.event_hub import ReaderEventHub, ReaderEventPublishRequest  # noqa: E402


def test_reader_event_fans_out_to_all_session_subscribers() -> None:
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/reader-sessions/read-1/events") as first,
        client.websocket_connect("/v1/reader-sessions/read-1/events") as second,
    ):
        response = client.post(
            "/v1/reader-sessions/read-1/events:publish",
            json={
                "type": "transcript.partial",
                "page_id": "page-1",
                "transcript": "The moon",
            },
        )
        first_event = first.receive_json()
        second_event = second.receive_json()

    assert response.status_code == 200
    body = response.json()
    assert body["subscriber_count"] == 2
    assert first_event == second_event == body["event"]
    assert first_event["type"] == "transcript.partial"
    assert first_event["payload"] == {
        "page_id": "page-1",
        "transcript": "The moon",
    }
    assert first_event["session_id"] == "read-1"
    assert first_event["sequence"] == 1


def test_word_reached_contract_is_ready_for_reader_aligner() -> None:
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/reader-sessions/read-2/events") as websocket,
    ):
        response = client.post(
            "/v1/reader-sessions/read-2/events:publish",
            json={
                "type": "word.reached",
                "page_id": "page-7",
                "index": 4,
                "word": "opened",
            },
        )
        event = websocket.receive_json()

    assert response.status_code == 200
    assert event["type"] == "word.reached"
    assert event["payload"] == {"page_id": "page-7", "index": 4, "word": "opened"}


def test_disconnected_subscriber_is_removed() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/v1/reader-sessions/read-3/events"):
            connected = client.post(
                "/v1/reader-sessions/read-3/events:publish",
                json={"type": "transcript.partial", "transcript": "still here"},
            )

        disconnected = client.post(
            "/v1/reader-sessions/read-3/events:publish",
            json={"type": "transcript.partial", "transcript": "gone"},
        )

    assert connected.json()["subscriber_count"] == 1
    assert disconnected.status_code == 200
    assert disconnected.json()["subscriber_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "transcript.partial"},
        {"type": "word.reached", "page_id": "page-1", "word": "moon"},
        {"type": "word.reached", "page_id": "page-1", "index": 0, "word": "moon", "extra": 1},
        {"type": "unknown", "transcript": "text"},
    ],
)
def test_publish_rejects_bad_payload(payload: dict[str, object]) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/reader-sessions/read-4/events:publish",
            json=payload,
        )

    assert response.status_code == 422


def test_publish_model_rejects_cross_type_fields() -> None:
    with pytest.raises(ValidationError):
        ReaderEventPublishRequest(
            type="transcript.partial",
            transcript="The moon",
            index=2,
        )


def test_slow_subscriber_queue_keeps_latest_event() -> None:
    async def exercise_hub() -> None:
        hub = ReaderEventHub(queue_size=1)
        subscription = await hub.subscribe("bounded-session")
        await hub.publish(
            "bounded-session",
            "transcript.partial",
            {"transcript": "first"},
        )
        await hub.publish(
            "bounded-session",
            "transcript.partial",
            {"transcript": "latest"},
        )

        event = await subscription.receive()
        assert event.sequence == 2
        assert event.payload == {"transcript": "latest"}
        await subscription.close()
        await hub.close()

    asyncio.run(exercise_hub())


def test_publish_rejects_non_local_client() -> None:
    with TestClient(app, client=("203.0.113.4", 50000)) as client:
        response = client.post(
            "/v1/reader-sessions/read-5/events:publish",
            json={"type": "transcript.partial", "transcript": "remote"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Reader session events are local-only"


def test_configured_transcript_emits_monotonic_word_events() -> None:
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/reader-sessions/moon-live/events") as websocket,
    ):
        configured = client.put(
            "/v1/reader-sessions/moon-live",
            json={
                "page_id": "page-01",
                "page_text": "The small moth went through the red gate.",
            },
        )
        partial = client.post(
            "/v1/reader-sessions/moon-live/transcripts:simulate",
            json={
                "source": "typed",
                "text": "The small moth",
                "page_id": "page-01",
                "language": "en",
                "is_final": False,
            },
        )
        events = [websocket.receive_json() for _ in range(4)]

    assert configured.status_code == 200
    assert configured.json()["next_word"] == "the"
    assert partial.status_code == 200
    assert [event["type"] for event in events] == [
        "transcript.partial",
        "word.reached",
        "word.reached",
        "word.reached",
    ]
    assert [event["payload"].get("index") for event in events[1:]] == [0, 1, 2]
    assert partial.json()["status"]["next_word"] == "went"


def test_transcript_requires_configured_matching_trusted_page() -> None:
    with TestClient(app) as client:
        unconfigured = client.post(
            "/v1/reader-sessions/not-ready/transcripts:simulate",
            json={"source": "typed", "text": "The moth"},
        )
        client.put(
            "/v1/reader-sessions/page-bound",
            json={"page_id": "page-01", "page_text": "The moth"},
        )
        mismatch = client.post(
            "/v1/reader-sessions/page-bound/transcripts:simulate",
            json={"source": "asr", "text": "The moth", "page_id": "page-02"},
        )

    assert unconfigured.status_code == 409
    assert "trusted page text" in unconfigured.json()["detail"]
    assert mismatch.status_code == 409
    assert "page-01" in mismatch.json()["detail"]
