import asyncio
import base64
import json
import struct
from pathlib import Path

import httpx
import pytest

from bookforge.finite_modal_provider import FastSceneRequest
from bookforge.provider_router import SafeProviderFallbackError
from bookforge.vertex_scene_provider import (
    DEPTH_MODEL,
    PROVIDER_NAME,
    VertexGeminiImageSceneProvider,
    VertexSceneAmbiguousError,
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


def test_vertex_provider_generates_checksum_bound_master_and_local_depth(
    tmp_path: Path,
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

    provider = VertexGeminiImageSceneProvider(
        project_id="your-gcp-project",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    request = FastSceneRequest(
        scene_id="vertex-scene",
        prompt="One silver fox raises a lantern beneath a moon gate.",
        seed=77,
    )
    async def exercise():
        assert (await provider.probe())[0] is True
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
    assert "One silver fox" not in bundle.manifest_path.read_text()
    assert bundle.estimated_gpu_usd == pytest.approx(0.034)


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
