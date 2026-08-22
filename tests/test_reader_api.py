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
                "generation": 1,
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
        "generation": 1,
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
                "generation": 3,
            },
        )
        event = websocket.receive_json()

    assert response.status_code == 200
    assert event["type"] == "word.reached"
    assert event["payload"] == {
        "page_id": "page-7",
        "index": 4,
        "word": "opened",
        "generation": 3,
    }


def test_disconnected_subscriber_is_removed() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/v1/reader-sessions/read-3/events"):
            connected = client.post(
                "/v1/reader-sessions/read-3/events:publish",
                json={
                    "type": "transcript.partial",
                    "page_id": "page-1",
                    "transcript": "still here",
                    "generation": 1,
                },
            )

        disconnected = client.post(
            "/v1/reader-sessions/read-3/events:publish",
            json={
                "type": "transcript.partial",
                "page_id": "page-1",
                "transcript": "gone",
                "generation": 1,
            },
        )

    assert connected.json()["subscriber_count"] == 1
    assert disconnected.status_code == 200
    assert disconnected.json()["subscriber_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "transcript.partial"},
        {"type": "transcript.partial", "transcript": "moon", "generation": 1},
        {"type": "transcript.partial", "transcript": "moon", "page_id": "page-1"},
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
            {"transcript": "first", "page_id": "page-1", "generation": 1},
        )
        await hub.publish(
            "bounded-session",
            "transcript.partial",
            {"transcript": "latest", "page_id": "page-1", "generation": 1},
        )

        event = await subscription.receive()
        assert event.sequence == 2
        assert event.payload == {
            "transcript": "latest",
            "page_id": "page-1",
            "generation": 1,
        }
        assert event.dropped_before_sequence == 1
        await subscription.close()
        await hub.close()

    asyncio.run(exercise_hub())


def test_queue_overflow_marks_a_dropped_reset_for_client_resynchronization() -> None:
    async def exercise_hub() -> None:
        hub = ReaderEventHub(queue_size=1)
        subscription = await hub.subscribe("reset-gap-session")
        await hub.publish(
            "reset-gap-session",
            "session.reset",
            {"page_id": "page-1", "page_text": "The moon", "generation": 2},
        )
        await hub.publish_word_reached(
            "reset-gap-session",
            page_id="page-1",
            index=0,
            word="The",
            generation=2,
        )
        await hub.publish_word_reached(
            "reset-gap-session",
            page_id="page-1",
            index=1,
            word="moon",
            generation=2,
        )

        event = await subscription.receive()
        assert event.type == "word.reached"
        assert event.sequence == 3
        assert event.payload["index"] == 1
        assert event.payload["generation"] == 2
        assert event.dropped_before_sequence == 1
        await subscription.close()
        await hub.close()

    asyncio.run(exercise_hub())


def test_publish_rejects_non_local_client() -> None:
    with TestClient(app, client=("203.0.113.4", 50000)) as client:
        response = client.post(
            "/v1/reader-sessions/read-5/events:publish",
            json={
                "type": "transcript.partial",
                "page_id": "page-1",
                "transcript": "remote",
                "generation": 1,
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Reader session events are local-only"


def test_direct_event_publish_is_disabled_in_jetson_mode() -> None:
    with TestClient(app) as client:
        original = app.state.settings.environment
        app.state.settings.environment = "jetson"
        response = client.post(
            "/v1/reader-sessions/read-5/events:publish",
            json={
                "type": "transcript.partial",
                "page_id": "page-1",
                "transcript": "injected",
                "generation": 1,
            },
        )
        app.state.settings.environment = original

    assert response.status_code == 403


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
                "generation": configured.json()["generation"],
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
    assert events[0]["payload"]["page_id"] == "page-01"
    assert events[0]["payload"]["generation"] == configured.json()["generation"]
    assert [event["payload"].get("index") for event in events[1:]] == [0, 1, 2]
    assert partial.json()["status"]["next_word"] == "went"


def test_reset_allows_the_same_page_to_be_read_again() -> None:
    session_url = "/v1/reader-sessions/repeat-reading"
    transcript_url = f"{session_url}/transcripts:simulate"
    with (
        TestClient(app) as client,
        client.websocket_connect(f"{session_url}/events") as websocket,
    ):
        configured = client.put(
            session_url,
            json={"page_id": "page-01", "page_text": "The moth"},
        )
        first = client.post(
            transcript_url,
            json={
                "source": "typed",
                "text": "The",
                "page_id": "page-01",
                "generation": configured.json()["generation"],
            },
        )
        first_events = [websocket.receive_json() for _ in range(2)]
        reset = client.post(f"{session_url}:reset")
        reset_event = websocket.receive_json()
        second = client.post(
            transcript_url,
            json={
                "source": "typed",
                "text": "The",
                "page_id": "page-01",
                "generation": reset.json()["generation"],
            },
        )
        second_events = [websocket.receive_json() for _ in range(2)]

    assert first.status_code == 200
    assert reset.status_code == 200
    assert reset.json()["next_word"] == "the"
    assert reset_event["type"] == "session.reset"
    assert reset_event["payload"] == {
        "page_id": "page-01",
        "page_text": "The moth",
        "generation": reset.json()["generation"],
    }
    assert second.status_code == 200
    assert first_events[1]["payload"]["index"] == 0
    assert second_events[1]["payload"]["index"] == 0


def test_stale_generation_cannot_cross_a_reset() -> None:
    session_url = "/v1/reader-sessions/stale-reading"
    with TestClient(app) as client:
        configured = client.put(
            session_url,
            json={"page_id": "page-01", "page_text": "The moth"},
        ).json()
        reset = client.post(f"{session_url}:reset").json()
        stale = client.post(
            f"{session_url}/transcripts:simulate",
            json={
                "source": "asr",
                "text": "The moth",
                "page_id": "page-01",
                "generation": configured["generation"],
            },
        )
        status = client.get(session_url)

    assert stale.status_code == 409
    assert "generation" in stale.json()["detail"]
    assert status.status_code == 200
    assert status.json()["generation"] == reset["generation"]
    assert status.json()["last_reached_index"] is None


@pytest.mark.parametrize(
    ("configure_payload", "transcript_payload"),
    [
        ({"page_id": "page-01", "page_text": "..."}, None),
        (
            {"page_id": "page-01", "page_text": "The moth"},
            {
                "source": "asr",
                "text": "The",
                "page_id": "page-01",
                "started_at_ms": 20,
                "ended_at_ms": 10,
            },
        ),
    ],
)
def test_reader_validation_returns_422(
    configure_payload: dict[str, object], transcript_payload: dict[str, object] | None
) -> None:
    with TestClient(app) as client:
        configured = client.put("/v1/reader-sessions/invalid-reader", json=configure_payload)
        if transcript_payload is None:
            response = configured
        else:
            payload = {**transcript_payload, "generation": configured.json()["generation"]}
            response = client.post(
                "/v1/reader-sessions/invalid-reader/transcripts:simulate", json=payload
            )

    assert response.status_code == 422


def test_transcript_requires_configured_matching_trusted_page() -> None:
    with TestClient(app) as client:
        unconfigured = client.post(
            "/v1/reader-sessions/not-ready/transcripts:simulate",
            json={"source": "typed", "text": "The moth", "generation": 1},
        )
        configured = client.put(
            "/v1/reader-sessions/page-bound",
            json={"page_id": "page-01", "page_text": "The moth"},
        )
        mismatch = client.post(
            "/v1/reader-sessions/page-bound/transcripts:simulate",
            json={
                "source": "asr",
                "text": "The moth",
                "page_id": "page-02",
                "generation": configured.json()["generation"],
            },
        )

    assert unconfigured.status_code == 409
    assert "trusted page text" in unconfigured.json()["detail"]
    assert mismatch.status_code == 409
    assert "page-01" in mismatch.json()["detail"]
