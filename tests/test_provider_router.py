import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from bookforge.finite_modal_provider import (
    FastSceneRequest,
    FiniteModalUnavailableError,
    FiniteSceneBundle,
    SceneArtifact,
)
from bookforge.provider_router import (
    ProviderRoute,
    ResilientFastSceneProvider,
    SafeProviderFallbackError,
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
            "stages": {
                "fast": {
                    "model": "stub",
                    "model_revision": "v1",
                    "estimated_gpu_usd": 0,
                }
            },
            "artifacts": {},
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


def test_router_converts_legacy_unavailable_error_after_paid_boundary(tmp_path: Path) -> None:
    primary = StubProvider(
        "gcp",
        generation_error=FiniteModalUnavailableError("transport lost after POST"),
    )
    fallback = StubProvider("modal")
    router = ResilientFastSceneProvider(
        [ProviderRoute("gcp", primary), ProviderRoute("modal", fallback)]
    )

    with pytest.raises(Exception, match="automatic fallback and retry are suppressed") as caught:
        asyncio.run(
            router.generate_fast(
                _request("legacy-ambiguous"),
                output_dir=tmp_path / "legacy-ambiguous",
            )
        )

    assert not isinstance(caught.value, FiniteModalUnavailableError)
    assert fallback.generations == 0


def test_failed_probe_opens_circuit_for_later_scenes(tmp_path: Path) -> None:
    primary = StubProvider("gcp", ready=False)
    fallback = StubProvider("modal")
    router = ResilientFastSceneProvider(
        [ProviderRoute("gcp", primary), ProviderRoute("modal", fallback)],
        failure_cooldown_seconds=60,
    )

    async def exercise() -> None:
        await router.generate_fast(_request("first"), output_dir=tmp_path / "first")
        await router.generate_fast(_request("second"), output_dir=tmp_path / "second")

    asyncio.run(exercise())

    assert primary.probes == 1
    assert fallback.probes == 1
    assert fallback.generations == 2
