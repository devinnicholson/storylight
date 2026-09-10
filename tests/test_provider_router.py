import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from storylight.finite_modal_provider import (
    FastSceneRequest,
    FiniteModalUnavailableError,
    FiniteSceneBundle,
    SceneArtifact,
)
from storylight.provider_router import (
    ProviderRoute,
    ResilientFastSceneProvider,
    SafeProviderFallbackError,
    SafeProviderRateLimitError,
)


class StubProvider:
    def __init__(
        self,
        name: str,
        *,
        ready: bool = True,
        generation_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.ready = ready
        self.generation_error = generation_error
        self.probes = 0
        self.generations = 0

    async def probe(self) -> tuple[bool, str]:
        self.probes += 1
        return self.ready, f"{self.name} {'ready' if self.ready else 'down'}"

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        self.generations += 1
        if self.generation_error is not None:
            raise self.generation_error
        output_dir.mkdir(parents=True)
        master = output_dir / "master.jpg"
        depth = output_dir / "depth.jpg"
        master.write_bytes(b"master")
        depth.write_bytes(b"depth")
        manifest = {
            "schema_version": "1.0",
            "provider": self.name,
            "scene_id": request.scene_id,
            "request": {
                "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
                "negative_prompt_sha256": hashlib.sha256(
                    request.negative_prompt.encode()
                ).hexdigest(),
                "seed": request.seed,
                "requested_width": request.width,
                "requested_height": request.height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
            },
            "stages": {
                "fast": {
                    "model": "stub",
                    "model_revision": "v1",
                    "estimated_gpu_usd": 0,
                }
            },
            "artifacts": {
                "master": {
                    "path": "master.jpg",
                    "sha256": hashlib.sha256(b"master").hexdigest(),
                    "mime_type": "image/jpeg",
                    "width": 1024,
                    "height": 576,
                    "duration_ms": 0,
                    "frames": 1,
                    "fps": 0,
                },
                "depth": {
                    "path": "depth.jpg",
                    "sha256": hashlib.sha256(b"depth").hexdigest(),
                    "mime_type": "image/jpeg",
                    "width": 1024,
                    "height": 576,
                    "duration_ms": 0,
                    "frames": 1,
                    "fps": 0,
                },
            },
        }
        manifest_path = output_dir / "scene.manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return FiniteSceneBundle(
            manifest_path=manifest_path,
            scene_id=request.scene_id,
            artifacts={
                "master": SceneArtifact(
                    role="master",
                    path=master,
                    sha256=hashlib.sha256(b"master").hexdigest(),
                    mime_type="image/jpeg",
                    width=1024,
                    height=576,
                ),
                "depth": SceneArtifact(
                    role="depth",
                    path=depth,
                    sha256=hashlib.sha256(b"depth").hexdigest(),
                    mime_type="image/jpeg",
                    width=1024,
                    height=576,
                ),
            },
            manifest=manifest,
        )


def _request(scene_id: str) -> FastSceneRequest:
    return FastSceneRequest(scene_id=scene_id, prompt="One luminous paper fox.")


def test_router_skips_failed_probe_and_records_selected_fallback(tmp_path: Path) -> None:
    primary = StubProvider("gcp-cloud-run", ready=False)
    fallback = StubProvider("modal-finite")
    router = ResilientFastSceneProvider(
        [ProviderRoute("gcp", primary), ProviderRoute("modal", fallback)],
        failure_cooldown_seconds=60,
    )

    bundle = asyncio.run(
        router.generate_fast(_request("routed-scene"), output_dir=tmp_path / "routed")
    )

    assert primary.probes == 1
    assert primary.generations == 0
    assert fallback.generations == 1
    assert bundle.manifest["provider"] == "modal-finite"
    assert bundle.manifest["routing"]["selected_route"] == "modal"
    assert bundle.manifest["routing"]["attempts"][0]["outcome"] == "probe_unavailable"
    assert json.loads(bundle.manifest_path.read_text())["routing"] == bundle.manifest["routing"]


def test_router_status_selects_healthy_route_without_generation() -> None:
    vertex = StubProvider("vertex")
    modal = StubProvider("modal")
    router = ResilientFastSceneProvider(
        [ProviderRoute("vertex", vertex), ProviderRoute("modal", modal)]
    )

    status = asyncio.run(router.warm_status())

    assert status.ready is True
    assert status.state == "idle"
    assert "resilient route selected vertex" in status.detail
    assert vertex.probes == 1
    assert vertex.generations == 0
    assert modal.probes == 0
    assert modal.generations == 0


def test_router_recovers_exact_paid_bundle_without_second_provider_call(
    tmp_path: Path,
) -> None:
    vertex = StubProvider("vertex")
    router = ResilientFastSceneProvider([ProviderRoute("vertex", vertex)])
    first_request = _request("first-scene")
    first = asyncio.run(
        router.generate_fast(first_request, output_dir=tmp_path / "first-scene")
    )
    recovered_request = FastSceneRequest(
        scene_id="recovered-scene",
        prompt=first_request.prompt,
    )

    recovered = asyncio.run(
        router.generate_fast(recovered_request, output_dir=tmp_path / "recovered-scene")
    )

    assert vertex.generations == 1
    assert recovered.scene_id == "recovered-scene"
    assert recovered.master.sha256 == first.master.sha256
    assert recovered.depth.sha256 == first.depth.sha256
    assert recovered.manifest["routing"]["recovery"] == {
        "source_scene_id": "first-scene",
        "provider_call": False,
        "policy": "exact-hashed-request-and-checksum-v1",
    }
    assert recovered.manifest["stages"]["fast"]["estimated_gpu_usd"] == 0
    assert recovered.manifest["stages"]["fast"]["remote_seconds"] == 0


def test_router_advances_after_explicit_nonbillable_rejection(tmp_path: Path) -> None:
    primary = StubProvider(
        "vertex",
        generation_error=SafeProviderFallbackError("HTTP 403 before a billable result"),
    )
    fallback = StubProvider("modal-finite")
    router = ResilientFastSceneProvider(
        [ProviderRoute("vertex", primary), ProviderRoute("modal", fallback)]
    )

    bundle = asyncio.run(
        router.generate_fast(_request("safe-fallback"), output_dir=tmp_path / "safe")
    )

    assert primary.generations == 1
    assert fallback.generations == 1
    assert bundle.manifest["routing"]["attempts"][0]["outcome"] == (
        "safe_generation_rejection"
    )


def test_router_never_duplicates_an_ambiguous_paid_request(tmp_path: Path) -> None:
    primary = StubProvider("gcp", generation_error=RuntimeError("connection reset after POST"))
    fallback = StubProvider("modal")
    router = ResilientFastSceneProvider(
        [ProviderRoute("gcp", primary), ProviderRoute("modal", fallback)]
    )

    with pytest.raises(RuntimeError, match="connection reset"):
        asyncio.run(
            router.generate_fast(_request("ambiguous"), output_dir=tmp_path / "ambiguous")
        )

    assert primary.generations == 1
    assert fallback.generations == 0


@pytest.mark.parametrize("error, cooldown", [
    (SafeProviderRateLimitError("HTTP 429"), 30),
    (SafeProviderRateLimitError("HTTP 429", retry_after_seconds=120), 120),
    (SafeProviderFallbackError("HTTP 403"), 300),
])
def test_router_rechecks_only_new_requests_after_rejection_cooldown(
    tmp_path: Path, monkeypatch, error: SafeProviderFallbackError, cooldown: float,
) -> None:
    now = 1000.0
    monkeypatch.setattr("storylight.provider_router.time.monotonic", lambda: now)
    provider = StubProvider("vertex", generation_error=error)
    router = ResilientFastSceneProvider([ProviderRoute("vertex", provider)])

    async def exercise():
        nonlocal now
        with pytest.raises(FiniteModalUnavailableError):
            await router.generate_fast(_request("rejected"), output_dir=tmp_path / "rejected")
        assert provider.generations == 1
        assert router._unavailable_until["vertex"] == now + cooldown
        provider.generation_error = None
        now += cooldown - 0.01
        with pytest.raises(FiniteModalUnavailableError):
            await router.generate_fast(_request("too-early"), output_dir=tmp_path / "too-early")
        assert provider.generations == provider.probes == 1
        now += 0.01
        bundle = await router.generate_fast(_request("recovered"), output_dir=tmp_path / "ok")
        assert bundle.scene_id == "recovered"
        assert provider.generations == provider.probes == 2

    asyncio.run(exercise())


def test_rate_limit_delay_validation_preserves_server_delay() -> None:
    for delay in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            SafeProviderRateLimitError("HTTP 429", retry_after_seconds=delay)
    for delay in [0, 7200]:
        error = SafeProviderRateLimitError("HTTP 429", retry_after_seconds=delay)
        assert error.retry_after_seconds == delay
        assert isinstance(error, SafeProviderFallbackError)
