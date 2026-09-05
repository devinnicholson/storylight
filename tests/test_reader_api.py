import asyncio
import os

os.environ["BOOKFORGE_MODEL_BACKEND"] = "fake"
os.environ["BOOKFORGE_MODEL_NAME"] = "fake"

from fastapi.testclient import TestClient  # noqa: E402

from bookforge.api import app  # noqa: E402
from bookforge.event_hub import ReaderEventHub  # noqa: E402


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
