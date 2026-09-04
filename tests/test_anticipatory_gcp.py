from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from PIL import Image

from bookforge.anticipatory import (
    AnticipationSource,
    AnticipatorySceneSpec,
    PrivacyAttestation,
    RenderedScene,
)
from bookforge.anticipatory_gcp import (
    AnticipatoryGcpError,
    AssetNotFoundError,
    CloudRunAnticipatoryRenderer,
    MemorySceneAssetStore,
    StoredAssetNemotronCritic,
)
from bookforge.gcp_scene_provider import DEPTH_MODEL_REVISION, FAST_MODEL, FAST_MODEL_REVISION
from bookforge.nemotron_critic import (
    NemotronCriticDecision,
    NemotronCriticRequest,
    NemotronCriticVerdict,
)

NOW = datetime(2026, 9, 2, 12, tzinfo=UTC)
JPEG = b"\xff\xd8\xffsynthetic-jpeg"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _spec(*, max_render_cost_usd: float = 0.02) -> AnticipatorySceneSpec:
    return AnticipatorySceneSpec(
        branch_id="moon_path",
        sequence=3,
        source=AnticipationSource.EXACT_LOOKAHEAD,
        visual_brief="One silver fox steps through a moon gate into an indigo garden.",
        expected_subjects=["one silver fox", "moon gate"],
        continuity_sha256=hashlib.sha256(b"continuity").hexdigest(),
        seed=9,
        not_after=NOW + timedelta(minutes=5),
        max_render_cost_usd=max_render_cost_usd,
        privacy=PrivacyAttestation(edge_gate_revision="edge-test-v1"),
    )


def _render_payload(*, master: bytes = JPEG, depth: bytes = JPEG + b"-depth") -> dict:
    return {
        "provider": "gcp-cloud-run",
        "gpu": "RTX_PRO_6000",
        "fast_model": FAST_MODEL,
        "fast_model_revision": FAST_MODEL_REVISION,
        "depth_model_revision": DEPTH_MODEL_REVISION,
        "master_b64": base64.b64encode(master).decode(),
        "master_sha256": _sha(master),
        "master_width": 1024,
        "master_height": 576,
        "master_media_type": "image/jpeg",
        "depth_b64": base64.b64encode(depth).decode(),
        "depth_sha256": _sha(depth),
        "depth_width": 1024,
        "depth_height": 576,
        "depth_media_type": "image/jpeg",
    }


def _valid_jpeg(*, width: int = 1024, height: int = 576) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (25, 35, 70)).save(output, format="JPEG", quality=90)
    return output.getvalue()


def test_memory_asset_store_is_content_addressed_bounded_and_expiring() -> None:
    async def scenario() -> None:
        store = MemorySceneAssetStore(max_bytes=1024 * 1024)
        first = await store.put(
            JPEG, media_type="image/jpeg", expires_at=NOW + timedelta(seconds=5)
        )
        replay = await store.put(
            JPEG,
            media_type="image/jpeg",
            expires_at=NOW + timedelta(seconds=10),
        )
        assert first == replay
        assert (await store.stats()) == {"assets": 1, "bytes": len(JPEG)}
        assert (await store.get(first, now=NOW)).content == JPEG
        with pytest.raises(AssetNotFoundError, match="expired"):
            await store.get(first, now=NOW + timedelta(seconds=11))

    asyncio.run(scenario())


def test_renderer_cache_validation_fails_closed_after_asset_eviction() -> None:
    async def scenario() -> None:
        store = MemorySceneAssetStore(max_bytes=1024 * 1024)
        master_ref = await store.put(
            JPEG,
            media_type="image/jpeg",
            expires_at=NOW + timedelta(minutes=1),
        )
        depth = JPEG + b"-depth"
        depth_ref = await store.put(
            depth,
            media_type="image/jpeg",
            expires_at=NOW + timedelta(minutes=1),
        )
        renderer = CloudRunAnticipatoryRenderer(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            asset_store=store,
            now=lambda: NOW,
        )
        scene = RenderedScene(
            master_ref=master_ref,
            depth_ref=depth_ref,
            master_sha256=_sha(JPEG),
            depth_sha256=_sha(depth),
            provider="test",
            model="test",
            model_revision="test",
            render_latency_ms=1,
            estimated_gpu_usd=0,
        )
        assert await renderer.cached_result_available(scene) is True

        await store.put(
            b"\xff\xd8\xff" + b"x" * (1024 * 1024 - 3),
            media_type="image/jpeg",
            expires_at=NOW + timedelta(minutes=1),
        )
        assert await renderer.cached_result_available(scene) is False

    asyncio.run(scenario())


@pytest.mark.parametrize("guidance", [None, "Keep one fox, not two. Preserve the moon gate."])
def test_renderer_sends_only_sanitized_scene_direction_and_validates_identity(guidance) -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json=_render_payload())

    async def token_source(audience: str) -> str:
        assert audience == "https://renderer.example.run.app"
        return "workload-identity-token"

    async def scenario() -> None:
        store = MemorySceneAssetStore()
        renderer = CloudRunAnticipatoryRenderer(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            asset_store=store,
            token_source=token_source,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        result = await renderer.render(
            _spec(), attempt=2 if guidance else 1, repair_guidance=guidance
        )
        assert result.provider == "gcp-cloud-run"
        assert result.master_sha256 == _sha(JPEG)
        assert (await store.get(result.master_ref, now=NOW)).content == JPEG
        await renderer.aclose()

    asyncio.run(scenario())
    assert len(observed) == 1
    assert observed[0].headers["authorization"] == "Bearer workload-identity-token"
    payload = json.loads(observed[0].content)
    assert _spec().visual_brief in payload["prompt"]
    if guidance:
        assert payload["prompt"].endswith(guidance)
    else:
        assert payload["prompt"].endswith(_spec().visual_brief)
    serialized = observed[0].content.decode()
    assert "source_text" not in serialized
    assert "transcript" not in serialized
    assert "audio" not in serialized
    assert "camera" not in serialized


def test_renderer_cost_reservation_fails_before_identity_or_network() -> None:
    called = False

    async def token_source(audience: str) -> str:
        nonlocal called
        del audience
        called = True
        return "token"

    async def scenario() -> None:
        renderer = CloudRunAnticipatoryRenderer(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            asset_store=MemorySceneAssetStore(),
            timeout_seconds=120,
            token_source=token_source,
        )
        with pytest.raises(AnticipatoryGcpError, match="timeout reservation"):
            await renderer.render(_spec(max_render_cost_usd=0.02), attempt=1)

    asyncio.run(scenario())
    assert called is False


def test_explicit_renderer_prewarm_loads_and_exercises_the_private_worker() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={**_render_payload(), "ready": True})

    async def token_source(audience: str) -> str:
        assert audience == "https://renderer.example.run.app"
        return "workload-identity-token"

    async def scenario() -> None:
        renderer = CloudRunAnticipatoryRenderer(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            asset_store=MemorySceneAssetStore(),
            token_source=token_source,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        ready, detail = await renderer.prewarm()
        assert ready is True
        assert "prewarmed" in detail
        await renderer.aclose()

    asyncio.run(scenario())
    assert len(observed) == 1
    assert observed[0].url.path == "/v1/prewarm"
    assert json.loads(observed[0].content)["prewarm_id"].startswith("anticipatory-")


def test_renderer_rejects_checksum_and_media_mismatch() -> None:
    payload = _render_payload()
    payload["master_sha256"] = "0" * 64

    async def token_source(audience: str) -> str:
        del audience
        return "token"

    async def scenario() -> None:
        renderer = CloudRunAnticipatoryRenderer(
            base_url="https://renderer.example.run.app",
            audience="https://renderer.example.run.app",
            asset_store=MemorySceneAssetStore(),
            token_source=token_source,
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
                **kwargs,
            ),
        )
        with pytest.raises(AnticipatoryGcpError, match="master_sha256"):
            await renderer.render(_spec(), attempt=1)
        await renderer.aclose()

    asyncio.run(scenario())


def test_nemotron_adapter_reads_only_the_synthetic_master() -> None:
    observed: list[tuple[NemotronCriticRequest, bytes, str]] = []

    class FakeNemotron:
        async def evaluate(self, request, *, image_bytes: bytes, media_type: str):
            observed.append((request, image_bytes, media_type))
            from bookforge.nemotron_critic import NemotronCriticEvidence

            return NemotronCriticEvidence(
                verdict=NemotronCriticVerdict(
                    fidelity_score=0.95,
                    composition_score=0.9,
                    projection_legibility_score=0.9,
                    identity_consistent=True,
                    unintended_text=False,
                    decision=NemotronCriticDecision.ACCEPT,
                    reason="The synthetic plate matches the bounded visual contract.",
                ),
                model="nemotron-test",
                latency_ms=3,
            )

    async def scenario() -> None:
        store = MemorySceneAssetStore()
        master = _valid_jpeg()
        ref = await store.put(
            master,
            media_type="image/jpeg",
            expires_at=NOW + timedelta(minutes=1),
        )
        adapter = StoredAssetNemotronCritic(
            critic=FakeNemotron(),  # type: ignore[arg-type]
            asset_store=store,
            now=lambda: NOW,
        )
        scene = RenderedScene(
            master_ref=ref,
            depth_ref=ref,
            master_sha256=_sha(JPEG),
            depth_sha256=_sha(JPEG),
            provider="test",
            model="test",
            model_revision="test",
            render_latency_ms=1,
            estimated_gpu_usd=0,
        )
        evidence = await adapter.evaluate(_spec(), scene)
        assert evidence.verdict.decision is NemotronCriticDecision.ACCEPT

    asyncio.run(scenario())
    request, content, media_type = observed[0]
    assert request.visual_brief == _spec().visual_brief
    assert content.startswith(b"\xff\xd8\xff")


def test_nemotron_adapter_prewarm_exercises_a_bounded_synthetic_review() -> None:
    observed: list[tuple[NemotronCriticRequest, bytes, str, float | None]] = []

    class FakeNemotron:
        async def evaluate(
            self,
            request,
            *,
            image_bytes: bytes,
            media_type: str,
            timeout_seconds: float | None = None,
        ):
            observed.append((request, image_bytes, media_type, timeout_seconds))
            from bookforge.nemotron_critic import NemotronCriticEvidence

            return NemotronCriticEvidence(
                verdict=NemotronCriticVerdict(
                    fidelity_score=0.95,
                    composition_score=0.9,
                    projection_legibility_score=0.9,
                    identity_consistent=True,
                    unintended_text=False,
                    decision=NemotronCriticDecision.ACCEPT,
                    reason="The deterministic warmup image matches its bounded contract.",
                ),
                model="nemotron-test",
                latency_ms=3,
            )

    adapter = StoredAssetNemotronCritic(
        critic=FakeNemotron(),  # type: ignore[arg-type]
        asset_store=MemorySceneAssetStore(),
        now=lambda: NOW,
    )
    ready, detail = asyncio.run(adapter.prewarm())
    assert ready is True
    assert "prewarmed" in detail
    request, content, media_type, timeout = observed[0]
    assert "gold circle" in request.visual_brief
    assert content.startswith(b"\xff\xd8\xff")
    assert media_type == "image/jpeg"
    assert timeout == 90
    with Image.open(io.BytesIO(content)) as review:
        assert max(review.size) == 512
