from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from storylight.voice_gateway import create_voice_gateway


class Stream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content

    async def aclose(self):
        self.closed = True


def test_local_assets_asr_scene_headers_and_sse_cleanup():
    calls, streams = [], []

    def handle(request):
        calls.append(request)
        assert all(
            name not in request.headers
            for name in (
                "authorization",
                "cookie",
                "forwarded",
                "x-forwarded-for",
                "origin",
                "x-drop",
            )
        )
        content = b'event: scene.session\ndata: {"ready":true}\n\n'
        cached = request.url.path == "/v1/assets/cached"
        if cached:
            content = b""
        stream = Stream(content)
        streams.append(stream)
        return httpx.Response(
            304 if cached else (202 if request.method == "POST" else 200),
            headers={
                "content-type": "text/event-stream",
                "set-cookie": "private=value",
                "x-storylight-server-instance-id": "instance-a",
                "x-storylight-session-revision": "2",
            },
            stream=stream,
        )

    app = create_voice_gateway(transport=httpx.MockTransport(handle))
    with TestClient(app, base_url="http://127.0.0.1:18767", client=("127.0.0.1", 50000)) as client:
        page = client.get("/workbench?voice=1")
        assert page.status_code == 200 and 'id="micButton"' in page.text
        assert client.get("/workbench-assets/workbench.js").status_code == 200
        assert '/workbench-assets/voice-timing.js?v=1' in page.text
        assert client.get("/workbench-assets/voice-timing.js").status_code == 200
        assert client.get("/projector?reader=0").status_code == 200
        demo = client.get("/workbench?demo=1&redirect=https://evil.example", follow_redirects=False)
        assert demo.status_code == 303
        assert demo.headers["location"] == "http://127.0.0.1:18766/workbench?demo=1"
        assert not calls
        audio = client.post(
            "/v1/audio:transcribe",
            content=b"audio",
            headers={
                "content-type": "audio/webm",
                "origin": "http://127.0.0.1:18767",
                "authorization": "private",
                "cookie": "secret=value",
                "forwarded": "host=evil",
                "x-forwarded-for": "1.2.3.4",
                "connection": "x-drop",
                "x-drop": "private",
            },
        )
        assert audio.status_code == 202
        assert str(calls[-1].url) == "http://127.0.0.1:18766/v1/audio:transcribe"
        assert calls[-1].content == b"audio"
        assert client.get("/v1/runtime:status").status_code == 200
        assert calls[-1].url.port == 18766
        assert client.get("/v1/demo-cues").status_code == 200
        assert calls[-1].url.port == 18768
        assert client.post("/v1/demo-cues/activate", json={}).status_code == 202
        assert calls[-1].url.port == 18768
        response = client.post("/v1/live-scenes", json={"text": "public synthetic scene"})
        assert response.status_code == 202 and calls[-1].url.port == 18768
        assert response.headers["x-storylight-server-instance-id"] == "instance-a"
        assert response.headers["x-storylight-session-revision"] == "2"
        assert "set-cookie" not in response.headers
        event = client.get("/v1/live-scene-sessions/demo/events?after=2")
        assert event.status_code == 200 and event.content.startswith(b"event:")
        assert calls[-1].url.query == b"after=2"
        assert client.get("/v1/assets/cached").status_code == 304
        assert all(stream.closed for stream in streams)


def test_loopback_scope_limits_and_single_attempt_errors():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("PRIVATE_UPSTREAM_DETAIL", request=request)

    app = create_voice_gateway(transport=httpx.MockTransport(handle))
    with TestClient(app, base_url="http://127.0.0.1:18767", client=("127.0.0.1", 50000)) as client:
        for headers in (
            {"host": "attacker.example"},
            {"origin": "https://attacker.example"},
            {"origin": "null"},
            {"sec-fetch-site": "cross-site"},
        ):
            assert client.post("/v1/live-scenes", json={}, headers=headers).status_code == 403
        for method, path in (
            ("POST", "/v1/story-packs:compile"),
            ("GET", "/openapi.json"),
            ("POST", "/v1/demo-cues"),
            ("GET", "/v1/demo-cues/activate"),
            ("POST", "/v1/audio:transcribe/extra"),
            ("POST", "/v1/live-scene-provider/prewarm"),
            ("GET", "/v1/assets/%252e%252e/config"),
        ):
            assert client.request(method, path).status_code == 403
        assert client.post("/v1/live-scenes", content=b"x" * (65536 + 1)).status_code == 413
        assert (
            client.post(
                "/v1/audio:transcribe", headers={"content-length": str(20 * 1024**2 + 1)}
            ).status_code
            == 413
        )
        assert not calls
        response = client.post("/v1/live-scenes", json={})
        assert response.status_code == 502 and len(calls) == 1
        assert "PRIVATE_UPSTREAM_DETAIL" not in response.text
    with TestClient(
        app, base_url="http://127.0.0.1:18767", client=("192.168.1.3", 50000)
    ) as client:
        assert client.get("/workbench").status_code == 403
    for url in (
        "https://127.0.0.1:18768",
        "http://example.com:18768",
        "http://127.0.0.1:18768/#fragment",
        "http://user@127.0.0.1:18768",
    ):
        with pytest.raises(ValueError):
            create_voice_gateway(scene_url=url)


def test_preconnect_only_routes_empty_request_to_scene_api_without_inference():
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "http://127.0.0.1:18768/v1/live-scene-provider/preconnect"
        assert request.method == "POST" and request.content == b"{}"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=Stream(
                b'{"supported":true,"ready":true,"expires_in_seconds":300,'
                b'"inference_started":false}'
            ),
        )

    app = create_voice_gateway(transport=httpx.MockTransport(handle))
    with TestClient(app, base_url="http://127.0.0.1:18767", client=("127.0.0.1", 50000)) as client:
        response = client.post("/v1/live-scene-provider/preconnect", json={})
        assert response.status_code == 200 and response.json()["inference_started"] is False
        for method, path in (
            ("GET", "/v1/live-scene-provider/preconnect"),
            ("POST", "/v1/live-scene-provider/preconnect/extra"),
            ("POST", "/v1/live-scene-provider/prewarm"),
            ("POST", "/v1/prepared-projections:prewarm"),
        ):
            assert client.request(method, path).status_code == 403
        assert len(calls) == 1
