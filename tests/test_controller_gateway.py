from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.testclient import TestClient

from bookforge.controller_gateway import (
    ControllerGatewaySettings,
    _route_allowed,
    create_controller_gateway,
)

PAIRING_TOKEN = "0123456789abcdef" * 4


def _upstream_app() -> FastAPI:
    upstream = FastAPI()

    @upstream.get("/readyz")
    async def readyz() -> dict[str, bool]:
        return {"ready": True}

    @upstream.get("/workbench")
    async def workbench() -> HTMLResponse:
        return HTMLResponse("<h1>Private workbench</h1>")

    @upstream.post("/v1/live-scenes")
    async def live_scene(request: Request) -> dict[str, object]:
        return {
            "payload": await request.json(),
            "forwarded": {
                name: request.headers.get(name)
                for name in ("forwarded", "x-forwarded-for", "x-real-ip")
            },
        }

    @upstream.get("/v1/live-scene-sessions/{session_id}/events")
    async def events(session_id: str) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield f"event: scene.session\ndata: {session_id}\n\n".encode()

        return StreamingResponse(body(), media_type="text/event-stream")

    @upstream.post("/v1/audio:transcribe")
    async def audio() -> dict[str, bool]:
        return {"unexpected": True}

    return upstream


def _gateway() -> FastAPI:
    upstream = _upstream_app()
    return create_controller_gateway(
        ControllerGatewaySettings(
            pairing_token=PAIRING_TOKEN,
            backend_url="http://127.0.0.1:8080",
            session_ttl_seconds=600,
        ),
        transport=httpx.ASGITransport(app=upstream),
    )


def _pair(client: TestClient) -> None:
    response = client.post("/pair-session", json={"token": PAIRING_TOKEN})
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "redirect": "/workbench?session=bookforge-live",
    }
    cookie = response.headers["set-cookie"]
    assert "bookforge_controller=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie


def test_gateway_requires_pairing_and_keeps_token_out_of_pair_page() -> None:
    with TestClient(_gateway(), follow_redirects=False) as client:
        health = client.get("/healthz")
        workbench = client.get("/workbench")
        api = client.post("/v1/live-scenes", json={"text": "private"})
        page = client.get("/pair")
        rejected = client.post("/pair-session", json={"token": "wrong"})

    assert health.status_code == 200
    assert health.json() == {"ready": True, "paired_sessions": 0}
    assert workbench.status_code == 303
    assert workbench.headers["location"] == "/pair"
    assert api.status_code == 401
    assert page.status_code == 200
    assert PAIRING_TOKEN not in page.text
    assert "location.hash.slice(1)" in page.text
    assert "Referrer-Policy" in page.headers
    assert rejected.status_code == 403


def test_paired_gateway_proxies_only_controller_routes_and_strips_forwarding_headers() -> None:
    with TestClient(_gateway()) as client:
        _pair(client)
        workbench = client.get("/workbench")
        generated = client.post(
            "/v1/live-scenes",
            json={"text": "A fox reads."},
            headers={
                "Forwarded": "for=203.0.113.5",
                "X-Forwarded-For": "203.0.113.5",
                "X-Real-IP": "203.0.113.5",
            },
        )
        blocked = client.post("/v1/audio:transcribe", content=b"private audio")

    assert workbench.status_code == 200
    assert "Private workbench" in workbench.text
    assert generated.status_code == 200
    assert generated.json()["payload"] == {"text": "A fox reads."}
    assert generated.json()["forwarded"] == {
        "forwarded": None,
        "x-forwarded-for": None,
        "x-real-ip": None,
    }
    assert blocked.status_code == 403


def test_gateway_streams_session_events_and_enforces_request_limit() -> None:
    with TestClient(_gateway()) as client:
        _pair(client)
        events = client.get("/v1/live-scene-sessions/bookforge-live/events")
        oversized = client.post("/v1/live-scenes", content=b"x" * (64 * 1024 + 1))

    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: scene.session" in events.text
    assert "data: bookforge-live" in events.text
    assert oversized.status_code == 413


def test_gateway_configuration_requires_loopback_backend_and_long_token() -> None:
    for backend in (
        "http://192.168.1.20:8080",
        "https://127.0.0.1:8080",
        "http://127.0.0.1:8080/private",
    ):
        try:
            ControllerGatewaySettings(pairing_token=PAIRING_TOKEN, backend_url=backend)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion gives a clearer failed candidate
            raise AssertionError(f"unsafe backend was accepted: {backend}")

    try:
        ControllerGatewaySettings(pairing_token="short")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("short pairing token was accepted")


def test_gateway_exposes_only_the_bounded_anticipatory_control_paths() -> None:
    assert _route_allowed("POST", "/v1/anticipations:prepare") is True
    assert _route_allowed("POST", "/v1/anticipations:commit") is True
    assert _route_allowed("GET", "/v1/anticipations/anticipate_abc/1") is True
    assert _route_allowed("DELETE", "/v1/anticipations/anticipate_abc/1") is True
    assert _route_allowed("PUT", "/v1/anticipations/anticipate_abc/1") is False
