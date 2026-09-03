"""GCP adapters for the anticipatory scene orchestrator.

The renderer calls the existing private Cloud Run GPU service with a Workload
Identity token. Synthetic artifacts remain in a bounded memory store long enough
for Nemotron and the edge client to retrieve them. A process restart is a cache
miss, never a loss of the source book or live reader state.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from bookforge.anticipatory import AnticipatorySceneSpec, RenderedScene
from bookforge.gcp_scene_provider import (
    DEPTH_MODEL_REVISION,
    FAST_MODEL,
    FAST_MODEL_REVISION,
    GPU_USD_PER_SECOND,
    google_identity_token,
)
from bookforge.nemotron_critic import (
    NemotronCriticEvidence,
    NemotronCriticRequest,
    NemotronVisionCritic,
)

MAX_REMOTE_ASSET_BYTES = 8 * 1024 * 1024
EXPECTED_PROVIDER = "gcp-cloud-run"

IdentityTokenSource = Callable[[str], Awaitable[str]]


class AnticipatoryGcpError(RuntimeError):
    pass


class AssetNotFoundError(AnticipatoryGcpError, LookupError):
    pass


@dataclass(frozen=True, slots=True)
class StoredAsset:
    content: bytes
    media_type: Literal["image/jpeg", "image/png"]
    sha256: str
    expires_at: datetime


class MemorySceneAssetStore:
    """Content-addressed, expiring storage for synthetic output only."""

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        if not 1024 * 1024 <= max_bytes <= 1024 * 1024 * 1024:
            raise ValueError("asset store max_bytes must be between 1 MiB and 1 GiB")
        self.max_bytes = max_bytes
        self._assets: OrderedDict[str, StoredAsset] = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()

    async def put(
        self,
        content: bytes,
        *,
        media_type: Literal["image/jpeg", "image/png"],
        expires_at: datetime,
    ) -> str:
        if not content:
            raise ValueError("scene asset cannot be empty")
        if len(content) > MAX_REMOTE_ASSET_BYTES:
            raise ValueError("scene asset exceeds the 8 MiB response limit")
        if expires_at.tzinfo is None:
            raise ValueError("scene asset expiry must be timezone-aware")
        digest = hashlib.sha256(content).hexdigest()
        ref = f"asset_{digest}"
        async with self._lock:
            existing = self._assets.get(ref)
            if existing is not None:
                extended = StoredAsset(
                    content=existing.content,
                    media_type=existing.media_type,
                    sha256=existing.sha256,
                    expires_at=max(existing.expires_at, expires_at),
                )
                self._assets[ref] = extended
                self._assets.move_to_end(ref)
                return ref
            while self._assets and self._total_bytes + len(content) > self.max_bytes:
                _, evicted = self._assets.popitem(last=False)
                self._total_bytes -= len(evicted.content)
            if self._total_bytes + len(content) > self.max_bytes:
                raise AnticipatoryGcpError("scene asset cannot fit in the bounded memory store")
            self._assets[ref] = StoredAsset(content, media_type, digest, expires_at)
            self._total_bytes += len(content)
        return ref

    async def get(self, ref: str, *, now: datetime) -> StoredAsset:
        async with self._lock:
            asset = self._assets.get(ref)
            if asset is None:
                raise AssetNotFoundError("synthetic scene asset was not found")
            if asset.expires_at <= now:
                self._assets.pop(ref)
                self._total_bytes -= len(asset.content)
                raise AssetNotFoundError("synthetic scene asset expired")
            self._assets.move_to_end(ref)
            return asset

    async def contains(self, ref: str, *, now: datetime) -> bool:
        try:
            await self.get(ref, now=now)
        except AssetNotFoundError:
            return False
        return True

    async def purge_expired(self, *, now: datetime) -> int:
        async with self._lock:
            expired = [ref for ref, asset in self._assets.items() if asset.expires_at <= now]
            for ref in expired:
                asset = self._assets.pop(ref)
                self._total_bytes -= len(asset.content)
            return len(expired)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {"assets": len(self._assets), "bytes": self._total_bytes}


class CloudRunAnticipatoryRenderer:
    """Parallel-safe client for the private SANA Sprint and depth worker."""

    def __init__(
        self,
        *,
        base_url: str,
        audience: str,
        asset_store: MemorySceneAssetStore,
        expected_gpu: Literal["L4", "RTX_PRO_6000"] = "RTX_PRO_6000",
        timeout_seconds: float = 30,
        token_source: IdentityTokenSource = google_identity_token,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.audience = audience.strip().rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ValueError("Cloud Run renderer URL must use HTTPS")
        if not self.audience.startswith("https://"):
            raise ValueError("Cloud Run renderer audience must use HTTPS")
        if expected_gpu not in GPU_USD_PER_SECOND:
            raise ValueError("expected GPU must be L4 or RTX_PRO_6000")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 120:
            raise ValueError("anticipatory renderer timeout must be 1-120 seconds")
        self.asset_store = asset_store
        self.expected_gpu = expected_gpu
        self.timeout_seconds = timeout_seconds
        self.token_source = token_source
        self.client_factory = client_factory
        self._now = now or (lambda: datetime.now(UTC))
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def probe(self) -> tuple[bool, str]:
        try:
            payload, _ = await self._request("GET", "/health")
            _require_equal(payload, "provider", EXPECTED_PROVIDER)
            _require_equal(payload, "fast_model_revision", FAST_MODEL_REVISION)
            _require_equal(payload, "depth_model_revision", DEPTH_MODEL_REVISION)
        except Exception as error:
            return False, f"Cloud Run renderer is unreachable: {_bounded_error(error)}"
        return True, "private Cloud Run renderer is reachable"

    async def cached_result_available(self, scene: RenderedScene) -> bool:
        """Reject a metadata cache hit after either bounded asset was evicted."""

        now = self._now()
        master, depth = await asyncio.gather(
            self.asset_store.contains(scene.master_ref, now=now),
            self.asset_store.contains(scene.depth_ref, now=now),
        )
        return master and depth

    async def render(
        self,
        spec: AnticipatorySceneSpec,
        *,
        attempt: Literal[1, 2],
    ) -> RenderedScene:
        worst_case_usd = self.timeout_seconds * GPU_USD_PER_SECOND[self.expected_gpu]
        if worst_case_usd > spec.max_render_cost_usd + 1e-9:
            raise AnticipatoryGcpError(
                "renderer timeout reservation exceeds the candidate cost ceiling"
            )
        prompt = f"{spec.visual_style}. {spec.visual_brief}"
        started = time.perf_counter()
        payload, wall_seconds = await self._request(
            "POST",
            "/v1/generate",
            json_body={
                "scene_id": f"anticipate-{spec.cache_key[:32]}-a{attempt}",
                "prompt": prompt,
                "negative_prompt": spec.negative_prompt,
                "seed": spec.seed,
                "width": spec.width,
                "height": spec.height,
                "steps": 2,
                "guidance_scale": 4.5,
            },
        )
        _validate_identity(payload, expected_gpu=self.expected_gpu)
        master = _decode_asset(payload, "master", width=spec.width, height=spec.height)
        depth = _decode_asset(payload, "depth", width=spec.width, height=spec.height)
        master_ref, depth_ref = await asyncio.gather(
            self.asset_store.put(
                master.content,
                media_type=master.media_type,
                expires_at=spec.not_after,
            ),
            self.asset_store.put(
                depth.content,
                media_type=depth.media_type,
                expires_at=spec.not_after,
            ),
        )
        return RenderedScene(
            master_ref=master_ref,
            depth_ref=depth_ref,
            master_sha256=master.sha256,
            depth_sha256=depth.sha256,
            provider=EXPECTED_PROVIDER,
            model=FAST_MODEL,
            model_revision=FAST_MODEL_REVISION,
            render_latency_ms=max((time.perf_counter() - started) * 1_000, wall_seconds * 1_000),
            estimated_gpu_usd=wall_seconds * GPU_USD_PER_SECOND[self.expected_gpu],
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        token = await self.token_source(self.audience)
        if not token:
            raise AnticipatoryGcpError("Google identity token source returned no token")
        client = await self._get_client()
        started = time.perf_counter()
        try:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=dict(json_body) if json_body is not None else None,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            raise AnticipatoryGcpError(
                f"Cloud Run scene request failed: {_bounded_error(error)}"
            ) from error
        if not isinstance(payload, dict):
            raise AnticipatoryGcpError("Cloud Run scene response must be an object")
        return payload, time.perf_counter() - started

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = self.client_factory(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    follow_redirects=False,
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=2,
                        max_keepalive_connections=2,
                        keepalive_expiry=300,
                    ),
                )
            return self._client

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()


class StoredAssetNemotronCritic:
    """Translate a scene contract and stored synthetic plate into a NIM request."""

    def __init__(
        self,
        *,
        critic: NemotronVisionCritic,
        asset_store: MemorySceneAssetStore,
        now: Callable[[], datetime],
    ) -> None:
        self.critic = critic
        self.asset_store = asset_store
        self.now = now

    async def probe(self) -> tuple[bool, str]:
        return await self.critic.probe()

    async def evaluate(
        self,
        spec: AnticipatorySceneSpec,
        scene: RenderedScene,
    ) -> NemotronCriticEvidence:
        master = await self.asset_store.get(scene.master_ref, now=self.now())
        request = NemotronCriticRequest(
            visual_brief=spec.visual_brief,
            expected_subjects=spec.expected_subjects,
            forbidden_content=spec.forbidden_content,
        )
        return await self.critic.evaluate(
            request,
            image_bytes=master.content,
            media_type=master.media_type,
        )

    async def aclose(self) -> None:
        await self.critic.aclose()


@dataclass(frozen=True, slots=True)
class _DecodedAsset:
    content: bytes
    media_type: Literal["image/jpeg", "image/png"]
    sha256: str


def _decode_asset(
    payload: Mapping[str, Any],
    role: str,
    *,
    width: int,
    height: int,
) -> _DecodedAsset:
    encoded = payload.get(f"{role}_b64")
    if not isinstance(encoded, str):
        raise AnticipatoryGcpError(f"Cloud Run response omitted {role}_b64")
    try:
        content = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise AnticipatoryGcpError(f"Cloud Run returned invalid {role} base64") from error
    if not content or len(content) > MAX_REMOTE_ASSET_BYTES:
        raise AnticipatoryGcpError(f"Cloud Run returned invalid {role} byte length")
    digest = hashlib.sha256(content).hexdigest()
    _require_equal(payload, f"{role}_sha256", digest)
    _require_equal(payload, f"{role}_width", width)
    _require_equal(payload, f"{role}_height", height)
    media_type = payload.get(f"{role}_media_type")
    if media_type not in {"image/jpeg", "image/png"}:
        raise AnticipatoryGcpError(f"Cloud Run returned unsupported {role} media type")
    signatures = {
        "image/jpeg": b"\xff\xd8\xff",
        "image/png": b"\x89PNG\r\n\x1a\n",
    }
    if not content.startswith(signatures[media_type]):
        raise AnticipatoryGcpError(f"Cloud Run {role} bytes do not match the media type")
    return _DecodedAsset(content, media_type, digest)


def _validate_identity(payload: Mapping[str, Any], *, expected_gpu: str) -> None:
    _require_equal(payload, "provider", EXPECTED_PROVIDER)
    _require_equal(payload, "gpu", expected_gpu)
    _require_equal(payload, "fast_model", FAST_MODEL)
    _require_equal(payload, "fast_model_revision", FAST_MODEL_REVISION)
    _require_equal(payload, "depth_model_revision", DEPTH_MODEL_REVISION)


def _require_equal(payload: Mapping[str, Any], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        raise AnticipatoryGcpError(f"Cloud Run response has unexpected {key}")


def _bounded_error(error: Exception) -> str:
    return (" ".join(str(error).split()) or error.__class__.__name__)[:300]
