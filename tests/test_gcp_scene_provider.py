import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest

from bookforge.finite_modal_provider import FastSceneRequest
from bookforge.gcp_scene_provider import (
    DEPTH_MODEL,
    DEPTH_MODEL_REVISION,
    FAST_MODEL,
    FAST_MODEL_REVISION,
    GcpCloudRunSceneProvider,
    GcpSceneProviderError,
    GoogleImpersonatedIdentityTokenSource,
    _jwt_expiration,
)


def _payload(scene_id: str = "gcp-scene") -> dict[str, object]:
    master = b"valid-master-jpeg-fixture"
    depth = b"valid-depth-jpeg-fixture"
    return {
        "provider": "gcp-cloud-run",
        "gpu": "L4",
        "fast_model": FAST_MODEL,
        "fast_model_revision": FAST_MODEL_REVISION,
        "depth_model": DEPTH_MODEL,
        "depth_model_revision": DEPTH_MODEL_REVISION,
        "depth_dtype": "float16",
        "scene_id": scene_id,
        "model_load_seconds": 8.0,
        "container_age_seconds": 20.0,
        "warm_state": "warm",
        "image_seconds": 0.4,
        "depth_seconds": 0.1,
        "image_gpu_ms": 375.0,
        "depth_gpu_ms": 82.0,
        "packaging_seconds": 0.02,
        "master_b64": base64.b64encode(master).decode(),
        "master_sha256": hashlib.sha256(master).hexdigest(),
        "master_media_type": "image/jpeg",
        "master_width": 1024,
        "master_height": 576,
        "depth_b64": base64.b64encode(depth).decode(),
        "depth_sha256": hashlib.sha256(depth).hexdigest(),
        "depth_media_type": "image/jpeg",
        "depth_width": 1024,
        "depth_height": 576,
    }


def test_private_cloud_run_provider_writes_checksum_bound_bundle(tmp_path: Path) -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        assert request.headers["authorization"] == "Bearer test-identity-token"
        return httpx.Response(200, json=_payload())

    async def token_source(audience: str) -> str:
        assert audience == "https://renderer.example.run.app"
        return "test-identity-token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app/",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    bundle = asyncio.run(
        provider.generate_fast(
            FastSceneRequest(scene_id="gcp-scene", prompt="A luminous paper fox."),
            output_dir=tmp_path / "scene",
        )
    )

    assert [request.url.path for request in observed] == ["/v1/generate"]
    assert bundle.scene_id == "gcp-scene"
    assert bundle.master.path.read_bytes() == b"valid-master-jpeg-fixture"
    assert bundle.depth.path.read_bytes() == b"valid-depth-jpeg-fixture"
    assert bundle.manifest["provider"] == "gcp-cloud-run"
    assert bundle.manifest["policy"]["private_iam_endpoint"] is True
    assert bundle.manifest["stages"]["fast"]["gpu"] == "L4"
    assert bundle.manifest["stages"]["fast"]["image_gpu_ms"] == 375.0
    assert bundle.manifest["stages"]["fast"]["depth_gpu_ms"] == 82.0
    assert bundle.estimated_gpu_usd > 0
    assert asyncio.run(provider.is_prewarmed()) is False
    assert asyncio.run(provider.is_renderer_likely_warm()) is True


def test_successful_generation_reuses_recent_cloud_run_warm_hint(tmp_path: Path) -> None:
    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=_payload(scene_id="warm-hint"))
            ),
            **kwargs,
        ),
    )

    async def exercise() -> tuple[bool, bool, bool]:
        before = await provider.is_renderer_likely_warm()
        await provider.generate_fast(
            FastSceneRequest(scene_id="warm-hint", prompt="A luminous paper fox."),
            output_dir=tmp_path / "warm-hint",
        )
        after = await provider.is_renderer_likely_warm()
        provider._remote_warm_deadline = time.monotonic() - 1
        expired = await provider.is_renderer_likely_warm()
        return before, after, expired

    assert asyncio.run(exercise()) == (False, True, False)


def test_private_cloud_run_probe_avoids_reserved_healthz_path() -> None:
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "provider": "gcp-cloud-run",
                "fast_model_revision": FAST_MODEL_REVISION,
                "depth_model_revision": DEPTH_MODEL_REVISION,
                "loaded": False,
            },
        )

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    assert asyncio.run(provider.probe())[0] is True
    assert observed == ["/health"]


def test_cloud_run_provider_accepts_legacy_worker_without_cuda_event_metrics(
    tmp_path: Path,
) -> None:
    payload = _payload(scene_id="legacy-worker")
    del payload["image_gpu_ms"]
    del payload["depth_gpu_ms"]

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            **kwargs,
        ),
    )
    bundle = asyncio.run(
        provider.generate_fast(
            FastSceneRequest(scene_id="legacy-worker", prompt="A luminous paper fox."),
            output_dir=tmp_path / "legacy-worker",
        )
    )

    assert bundle.manifest["stages"]["fast"]["image_gpu_ms"] is None
    assert bundle.manifest["stages"]["fast"]["depth_gpu_ms"] is None


def test_cloud_run_provider_reuses_and_closes_one_http_client() -> None:
    clients: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "provider": "gcp-cloud-run",
                "fast_model_revision": FAST_MODEL_REVISION,
                "depth_model_revision": DEPTH_MODEL_REVISION,
                "loaded": True,
            },
        )

    def client_factory(**kwargs: object) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=client_factory,
    )

    async def exercise() -> None:
        assert (await provider.probe())[0] is True
        assert (await provider.probe())[0] is True
        assert len(clients) == 1
        assert clients[0].is_closed is False
        await provider.aclose()

    asyncio.run(exercise())
    assert clients[0].is_closed is True


def test_private_cloud_run_provider_attests_rtx_pro_6000(tmp_path: Path) -> None:
    payload = _payload(scene_id="rtx-scene")
    payload["gpu"] = "RTX_PRO_6000"

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        gpu="RTX_PRO_6000",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            **kwargs,
        ),
    )
    bundle = asyncio.run(
        provider.generate_fast(
            FastSceneRequest(scene_id="rtx-scene", prompt="A luminous paper fox."),
            output_dir=tmp_path / "rtx-scene",
        )
    )

    assert bundle.manifest["stages"]["fast"]["gpu"] == "RTX_PRO_6000"
    assert bundle.estimated_gpu_usd > 0


def test_cloud_run_provider_rejects_non_native_sana_steps_without_remote_call(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_payload())

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    with pytest.raises(GcpSceneProviderError, match="exactly 2 inference steps"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(
                    scene_id="invalid-steps",
                    prompt="A luminous paper fox.",
                    steps=4,
                ),
                output_dir=tmp_path / "invalid-steps",
            )
        )

    assert calls == 0


def test_cloud_run_provider_rejects_tampered_artifacts_without_writing(tmp_path: Path) -> None:
    payload = _payload()
    payload["master_sha256"] = "0" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(GcpSceneProviderError, match="unexpected master_sha256"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(scene_id="gcp-scene", prompt="A luminous paper fox."),
                output_dir=tmp_path / "scene",
            )
        )
    assert not (tmp_path / "scene").exists()


def test_cloud_run_provider_requires_https_and_enforces_session_cost_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        GcpCloudRunSceneProvider(
            base_url="http://renderer.example",
            audience="https://renderer.example",
        )

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        session_gpu_cap_usd=0.000000000001,
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_payload())),
            **kwargs,
        ),
    )
    with pytest.raises(GcpSceneProviderError, match="request ceiling"):
        asyncio.run(
            provider.generate_fast(
                FastSceneRequest(scene_id="gcp-scene", prompt="A luminous paper fox."),
                output_dir=tmp_path / "unused",
            )
        )


def test_cloud_run_provider_configures_keyless_impersonation() -> None:
    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        impersonate_service_account="renderer@example.iam.gserviceaccount.com",
    )
    assert isinstance(provider._token_source, GoogleImpersonatedIdentityTokenSource)

    with pytest.raises(ValueError, match="service-account email"):
        GcpCloudRunSceneProvider(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            impersonate_service_account="not-an-account",
        )

    async def token_source(audience: str) -> str:
        return "token"

    with pytest.raises(ValueError, match="either GCP service-account impersonation"):
        GcpCloudRunSceneProvider(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            impersonate_service_account="renderer@example.iam.gserviceaccount.com",
            token_source=token_source,
        )


def test_google_identity_token_expiration_is_validated() -> None:
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 3600}).encode())
        .decode()
        .rstrip("=")
    )
    assert _jwt_expiration(f"header.{payload}.signature") > time.time()

    with pytest.raises(ValueError, match="valid expiration"):
        expired = base64.urlsafe_b64encode(b'{"exp":1}').decode().rstrip("=")
        _jwt_expiration(f"header.{expired}.signature")


def test_google_impersonated_identity_token_is_minted_once_and_cached() -> None:
    encoded = (
        base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 3600}).encode())
        .decode()
        .rstrip("=")
    )
    token = f"header.{encoded}.signature"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path.endswith(
            "/renderer@example.iam.gserviceaccount.com:generateIdToken"
        )
        assert request.headers["authorization"] == "Bearer source-access-token"
        assert json.loads(request.content) == {
            "audience": "https://renderer.example.run.app",
            "includeEmail": True,
        }
        return httpx.Response(200, json={"token": token})

    token_source = GoogleImpersonatedIdentityTokenSource(
        "renderer@example.iam.gserviceaccount.com",
        source_access_token=lambda: "source-access-token",
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    async def fetch_twice() -> tuple[str, str]:
        return (
            await token_source("https://renderer.example.run.app"),
            await token_source("https://renderer.example.run.app"),
        )

    assert asyncio.run(fetch_twice()) == (token, token)
    assert calls == 1


def test_failed_remote_call_retains_worst_case_reservation_and_blocks_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"detail": "container failed"})

    async def token_source(audience: str) -> str:
        return "token"

    provider = GcpCloudRunSceneProvider(
        base_url="https://renderer.example.run.app",
        audience="https://renderer.example.run.app",
        timeout_seconds=1,
        session_gpu_cap_usd=0.0002,
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    request = FastSceneRequest(scene_id="gcp-scene", prompt="A luminous paper fox.")
    with pytest.raises(RuntimeError, match="Cloud Run scene request failed"):
        asyncio.run(provider.generate_fast(request, output_dir=tmp_path / "first"))
    with pytest.raises(GcpSceneProviderError, match="request ceiling"):
        asyncio.run(provider.generate_fast(request, output_dir=tmp_path / "second"))

    assert calls == 1
