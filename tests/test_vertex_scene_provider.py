import asyncio
import base64
import json
import struct
from pathlib import Path

import httpx
import pytest

from storylight.finite_modal_provider import FastSceneRequest
from storylight.provider_router import SafeProviderFallbackError, SafeProviderRateLimitError
from storylight.vertex_scene_provider import (
    DEPTH_MODEL,
    PROVIDER_NAME,
    REQUEST_CONTRACT_REVISION,
    VertexGeminiImageSceneProvider,
    VertexSceneAmbiguousError,
    VertexSceneProviderError,
    _request_payload,
    _retry_after_seconds,
)


def _jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x0b\x08"
        + struct.pack(">HH", height, width)
        + b"\x01\x01\x11\x00"
        + b"\xff\xd9"
        + bytes(64)
    )


def _response(width: int = 1024, height: int = 576) -> dict[str, object]:
    return {
        "responseId": "vertex-response-1",
        "modelVersion": "gemini-3.1-flash-lite-image",
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is the generated scene."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64.b64encode(_jpeg(width, height)).decode(),
                            }
                        },
                    ]
                }
            }
        ],
    }


@pytest.mark.parametrize("seed", [2810313968])
def test_vertex_provider_generates_checksum_bound_master_and_local_depth(
    tmp_path: Path,
    seed: int,
) -> None:
    requests: list[httpx.Request] = []
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(404)
        return httpx.Response(200, json=_response())

    async def token_source() -> str:
        nonlocal token_calls
        token_calls += 1
        return "vertex-token"

    def client_factory(**kwargs):
        assert kwargs["http2"] is False
        assert kwargs["follow_redirects"] is False
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project",
        token_source=token_source,
        client_factory=client_factory,
    )
    request = FastSceneRequest(
        scene_id="vertex-scene",
        prompt="One silver fox raises a lantern beneath a moon gate.",
        seed=seed,
    )

    async def exercise():
        ready, detail = await provider.probe()
        assert ready is True
        assert "HTTP/1.1" in detail
        return await provider.generate_fast(request, output_dir=tmp_path / "vertex")

    bundle = asyncio.run(exercise())

    assert [item.method for item in requests] == ["HEAD", "POST"]
    assert requests[-1].headers["authorization"] == "Bearer vertex-token"
    assert token_calls == 1
    assert requests[-1].url.path.endswith(
        "/publishers/google/models/gemini-3.1-flash-lite-image:generateContent"
    )
    body = json.loads(requests[-1].content)
    assert body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert body["generationConfig"]["seed"] == seed & 0x7FFFFFFF
    assert bundle.manifest["request"]["seed"] == seed
    assert bundle.manifest["request"]["provider_seed"] == body["generationConfig"]["seed"]
    assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    contract = body["contents"][0]["parts"][0]["text"]
    assert "NON-NEGOTIABLE VISUAL CONTRACT" in contract
    assert "body pose and physical contact" in contract
    assert "Preserve exact counts and directions" in contract
    assert bundle.manifest["provider"] == PROVIDER_NAME
    assert bundle.master.mime_type == "image/jpeg"
    assert (bundle.master.width, bundle.master.height) == (1024, 576)
    assert bundle.depth.mime_type == "image/png"
    assert bundle.depth.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert bundle.manifest["stages"]["fast"]["additional_models"][0]["model"] == DEPTH_MODEL
    assert bundle.manifest["request"]["prompt_sha256"]
    assert bundle.manifest["request"]["contract_revision"] == REQUEST_CONTRACT_REVISION
    assert "One silver fox" not in bundle.manifest_path.read_text()
    assert bundle.estimated_gpu_usd == pytest.approx(0.034)


@pytest.mark.parametrize("error_type", [httpx.RemoteProtocolError, httpx.ReadTimeout])
def test_vertex_ambiguous_transport_failure_is_never_retried(
    tmp_path: Path, error_type: type[httpx.TransportError],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise error_type("connection ended after request dispatch", request=request)

    async def token_source() -> str:
        return "token"

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project", session_cost_cap_usd=0.05,
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs,
        ),
    )

    async def exercise():
        request = FastSceneRequest(scene_id="ambiguous", prompt="A paper forest.")
        with pytest.raises(VertexSceneAmbiguousError) as caught:
            await provider.generate_fast(request, output_dir=tmp_path / "ambiguous")
        assert isinstance(caught.value.__cause__, error_type)
        assert not isinstance(caught.value, SafeProviderFallbackError)
        assert provider._reserved_usd == pytest.approx(0.034)
        with pytest.raises(VertexSceneProviderError, match="session estimate"):
            await provider.generate_fast(request, output_dir=tmp_path / "blocked")
        await provider.aclose()

    asyncio.run(exercise())
    assert [request.method for request in requests] == ["POST"]
    assert not (tmp_path / "ambiguous").exists()


def test_vertex_request_reinforces_sanitized_exact_counts() -> None:
    payload = _request_payload(
        FastSceneRequest(
            scene_id="count-lock-scene",
            prompt=(
                "A rabbit under a bridge. Required supporting visual: 3 paper lanterns "
                "overhead and 2 silver comet-fish arcing through the sky."
            ),
            seed=79,
        )
    )

    contract = payload["contents"][0]["parts"][0]["text"]
    assert "Show exactly 3 paper lanterns total across the entire frame" in contract
    assert "Show exactly 2 silver comet-fish total across the entire frame" in contract
    assert "show no additional paper lanterns" in contract


def test_vertex_probe_fails_closed_when_preconnect_rejects_credentials() -> None:
    async def token_source() -> str:
        return "expired-token"

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={"error": "unauthorized"})
            ),
            **kwargs,
        ),
    )

    ready, detail = asyncio.run(provider.probe())

    assert ready is False
    assert "rejected credentials with HTTP 401" in detail


def test_vertex_explicit_http_rejection_is_safe_to_fallback(tmp_path: Path) -> None:
    async def token_source() -> str:
        return "token"

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    403,
                    json={"error": {"message": "Vertex API is not enabled"}},
                )
            ),
            **kwargs,
        ),
    )

    with pytest.raises(SafeProviderFallbackError, match="HTTP 403"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(scene_id="rejected", prompt="A paper forest."),
                output_dir=tmp_path / "rejected",
            )
        )

    assert provider._reserved_usd == 0


def test_vertex_rate_limit_preserves_recovery_delay_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("storylight.vertex_scene_provider.time.time", lambda: 1788849600)
    assert _retry_after_seconds("Tue, 08 Sep 2026 06:40:45 GMT") == 45
    assert _retry_after_seconds("Tue, 08 Sep 2026 06:39:00 GMT") == 0
    for header in (None, "-1", "nan", "inf", "9" * 400, "bad header", "1.5"):
        assert _retry_after_seconds(header) is None
    assert _retry_after_seconds("7200") == 7200

    calls = []

    async def token_source():
        return "token"

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"}, json={"error": {}})

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project", token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs),
    )

    async def exercise():
        with pytest.raises(SafeProviderRateLimitError) as error:
            await provider.generate_fast(
                FastSceneRequest(scene_id="limited", prompt="A paper forest."),
                output_dir=tmp_path / "limited",
            )
        assert error.value.retry_after_seconds == 120
        await provider.aclose()

    asyncio.run(exercise())
    assert len(calls) == 1
    assert provider._reserved_usd == 0
    assert not (tmp_path / "limited").exists()


def test_vertex_billable_invalid_response_fails_closed_and_reserves_cost(
    tmp_path: Path,
) -> None:
    async def token_source() -> str:
        return "token"

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project",
        session_cost_cap_usd=0.05,
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"candidates": []})
            ),
            **kwargs,
        ),
    )

    with pytest.raises(VertexSceneAmbiguousError, match="without a valid image"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(scene_id="invalid", prompt="A paper forest."),
                output_dir=tmp_path / "invalid",
            )
        )
    assert provider._reserved_usd == pytest.approx(0.034)
    with pytest.raises(Exception, match="session estimate"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(scene_id="blocked", prompt="A paper forest."),
                output_dir=tmp_path / "blocked",
            )
        )
