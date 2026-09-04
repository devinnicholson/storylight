from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import struct
import time
import zlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bookforge.finite_modal_provider import (
    FastSceneRequest,
    FiniteModalProviderError,
    FiniteSceneBundle,
    MotionUpgradeRequest,
    SceneArtifact,
)
from bookforge.provider_router import SafeProviderFallbackError

PROVIDER_NAME = "gcp-vertex-gemini-image"
DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
MODEL_REVISION = "vertex-managed"
REQUEST_CONTRACT_REVISION = "story-scene-v3-int31-seed"
DEPTH_MODEL = "bookforge-projection-depth-bootstrap"
DEPTH_MODEL_REVISION = "vertical-gradient-v1"
DEFAULT_ESTIMATED_IMAGE_USD = 0.034
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,126}$")
_EXACT_COUNT_PHRASE = re.compile(
    r"\b(?P<count>[2-9]|10)\s+"
    r"(?P<label>(?:[A-Za-z][A-Za-z-]*\s+){0,2}"
    r"[A-Za-z][A-Za-z-]*(?:s|fish))\b",
    re.IGNORECASE,
)

AccessTokenSource = Callable[[], Awaitable[str]]


class VertexSceneProviderError(FiniteModalProviderError):
    pass


class VertexSceneAmbiguousError(VertexSceneProviderError):
    """The request may have reached a billable managed model."""


class VertexGeminiImageSceneProvider:
    """Managed Vertex image generation with a local zero-model depth bootstrap.

    Only the privacy-sanitized scene prompt is sent to Vertex. The generated
    master is paired immediately with a deterministic projection depth gradient;
    the Jetson can replace that sidecar with TensorRT depth asynchronously later.
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str = "global",
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 90.0,
        session_cost_cap_usd: float = 0.50,
        estimated_image_usd: float = DEFAULT_ESTIMATED_IMAGE_USD,
        token_source: AccessTokenSource | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
        api_origin: str = "https://aiplatform.googleapis.com",
    ) -> None:
        for value, label in (
            (project_id, "Vertex project ID"),
            (location, "Vertex location"),
            (model, "Vertex model"),
        ):
            if not _SAFE_COMPONENT.fullmatch(value.strip()):
                raise ValueError(f"{label} contains unsupported characters")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 300:
            raise ValueError("Vertex timeout must be between 1 and 300 seconds")
        if not math.isfinite(estimated_image_usd) or not 0 < estimated_image_usd <= 1:
            raise ValueError("Vertex estimated image cost must be between $0 and $1")
        if (
            not math.isfinite(session_cost_cap_usd)
            or not estimated_image_usd <= session_cost_cap_usd <= 10
        ):
            raise ValueError("Vertex session cost cap must cover one image and be at most $10")
        normalized_origin = api_origin.strip().rstrip("/")
        if not normalized_origin.startswith("https://"):
            raise ValueError("Vertex API origin must use HTTPS")
        self.project_id = project_id.strip()
        self.location = location.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.session_cost_cap_usd = session_cost_cap_usd
        self.estimated_image_usd = estimated_image_usd
        self.api_origin = normalized_origin
        self._token_source = token_source or google_access_token
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._token_lock = asyncio.Lock()
        self._cached_token = ""
        self._token_deadline = 0.0
        self._operation_lock = asyncio.Lock()
        self._reserved_usd = 0.0

    @property
    def endpoint(self) -> str:
        return (
            f"{self.api_origin}/v1/projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{self.model}:generateContent"
        )

    async def probe(self) -> tuple[bool, str]:
        try:
            token = await self._access_token()
        except Exception as error:
            return False, f"Vertex credentials are unavailable: {error}"
        if not token:
            return False, "Vertex credentials returned no access token"
        try:
            # generateContent is POST-only, so HEAD cannot create a prediction
            # or a billable image. It does establish the same DNS/TLS/HTTP/2
            # path that the first generation would otherwise pay for.
            client = await self._get_client()
            response = await client.head(
                self.endpoint,
                headers={"Authorization": f"Bearer {token}"},
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            return False, f"Vertex preconnect failed: {error}"
        if response.status_code in {401, 403}:
            return False, f"Vertex preconnect rejected credentials with HTTP {response.status_code}"
        if response.status_code >= 500:
            return False, f"Vertex preconnect returned HTTP {response.status_code}"
        return True, (
            f"Vertex managed image route is configured for {self.model} in {self.location}; "
            f"HTTP/2 path is warm"
        )

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        destination = output_dir.resolve()
        if destination.exists():
            raise VertexSceneProviderError(f"scene output already exists: {destination}")
        async with self._operation_lock:
            self._ensure_capacity()
            token = await self._access_token()
            started = time.perf_counter()
            try:
                client = await self._get_client()
                response = await client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                    json=_request_payload(request),
                )
            except (httpx.TimeoutException, httpx.TransportError) as error:
                self._reserve()
                raise VertexSceneAmbiguousError(
                    f"Vertex image request ended ambiguously: {error}"
                ) from error
            remote_seconds = time.perf_counter() - started
            if response.status_code >= 400:
                # Vertex pricing documents charge successful (200) predictions;
                # an explicit non-2xx response is therefore safe to route onward.
                raise SafeProviderFallbackError(
                    f"Vertex rejected image generation with HTTP {response.status_code}: "
                    f"{_safe_error_detail(response)}"
                )
            try:
                payload = response.json()
                master, media_type = _extract_image(payload)
                width, height = _image_dimensions(master, media_type)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self._reserve()
                raise VertexSceneAmbiguousError(
                    f"Vertex returned a billable response without a valid image: {error}"
                ) from error
            self._reserve()
            packaging_started = time.perf_counter()
            depth = _projection_depth_png(width, height)
            bundle = await asyncio.to_thread(
                _write_bundle,
                request,
                destination,
                master,
                media_type,
                width,
                height,
                depth,
                remote_seconds,
                self.estimated_image_usd,
                self.model,
                payload,
                packaging_started,
            )
            return bundle

    async def upgrade_motion(
        self,
        bundle: FiniteSceneBundle | Path,
        request: MotionUpgradeRequest,
    ) -> FiniteSceneBundle:
        del bundle, request
        raise VertexSceneProviderError(
            "Vertex managed image generation uses local depth motion instead of paid video"
        )

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = self._client_factory(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    follow_redirects=False,
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=1,
                        max_keepalive_connections=1,
                        keepalive_expiry=300,
                    ),
                )
            return self._client

    async def _access_token(self) -> str:
        now = time.monotonic()
        if self._cached_token and now < self._token_deadline:
            return self._cached_token
        async with self._token_lock:
            now = time.monotonic()
            if self._cached_token and now < self._token_deadline:
                return self._cached_token
            token = await self._token_source()
            if not token:
                raise VertexSceneProviderError("Vertex token source returned no access token")
            self._cached_token = token
            # Google access tokens normally live for an hour. A deliberately
            # conservative five-minute cache removes the probe/generate double
            # refresh while leaving ample expiry margin for custom sources.
            self._token_deadline = now + 300
            return token

    def _ensure_capacity(self) -> None:
        if self._reserved_usd + self.estimated_image_usd > self.session_cost_cap_usd + 1e-9:
            raise VertexSceneProviderError(
                f"Vertex session estimate would exceed ${self.session_cost_cap_usd:.2f} cap"
            )

    def _reserve(self) -> None:
        self._reserved_usd += self.estimated_image_usd


async def google_access_token() -> str:
    def fetch() -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as error:
            raise VertexSceneProviderError(
                "Google authentication is unavailable; install bookforge[gcp]"
            ) from error
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        token = credentials.token
        if not isinstance(token, str) or not token:
            raise VertexSceneProviderError("Google credentials returned no access token")
        return token

    return await asyncio.to_thread(fetch)


def _request_payload(request: FastSceneRequest) -> dict[str, Any]:
    exact_count_contract = _exact_count_contract(request.prompt)
    prompt = (
        f"STORY SCENE:\n{request.prompt}\n\n"
        "NON-NEGOTIABLE VISUAL CONTRACT:\n"
        "- Show every explicitly named subject, object, count, color, action, and spatial "
        "relationship.\n"
        "- Make each action unmistakable in body pose and physical contact; never replace an "
        "action with mere proximity.\n"
        "- Preserve exact counts and directions such as left, right, above, below, in front, "
        "and behind.\n"
        f"{exact_count_contract}"
        "- Create one coherent, full-bleed 16:9 storybook projection frame with an "
        "unambiguous primary subject, projection-bright midtones, clean silhouettes, and "
        "tactile foreground-to-background depth.\n"
        f"- Exclude: {request.negative_prompt}.\n"
        "Return no written words, caption, border, or interface inside the image."
    )
    return {
        "contents": [{"role": "USER", "parts": [{"text": prompt[:4_000]}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "candidateCount": 1,
            # Bookforge seeds are uint32; Vertex's GenerationConfig uses int32.
            "seed": request.seed & 0x7FFFFFFF,
            "imageConfig": {
                "aspectRatio": "16:9",
                "imageOutputOptions": {
                    "mimeType": "image/jpeg",
                    "compressionQuality": 95,
                },
                "personGeneration": "ALLOW_ALL",
            },
        },
    }


def _exact_count_contract(prompt: str) -> str:
    """Reinforce sanitized supporting-object counts without source-text access."""

    locks: list[str] = []
    seen: set[tuple[str, str]] = set()
    for match in _EXACT_COUNT_PHRASE.finditer(prompt):
        count = match.group("count")
        label = " ".join(match.group("label").split())
        key = (count, label.casefold())
        if key in seen:
            continue
        seen.add(key)
        locks.append(
            f"- COUNT LOCK: Show exactly {count} {label} total across the entire frame; "
            f"show no additional {label} and do not duplicate them.\n"
        )
    return "".join(locks)


def _extract_image(payload: object) -> tuple[bytes, str]:
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        reason = payload.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise ValueError(f"image generation returned {reason}")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise ValueError("candidate omitted content parts")
    for part in parts:
        inline = part.get("inlineData") if isinstance(part, dict) else None
        if not isinstance(inline, dict):
            continue
        media_type = inline.get("mimeType")
        encoded = inline.get("data")
        if media_type not in {"image/jpeg", "image/png"} or not isinstance(encoded, str):
            continue
        try:
            content_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("image data is not valid base64") from error
        if not 64 <= len(content_bytes) <= 20 * 1024 * 1024:
            raise ValueError("image byte length is outside the accepted range")
        return content_bytes, media_type
    raise ValueError("candidate contained no supported image part")


def _image_dimensions(content: bytes, media_type: str) -> tuple[int, int]:
    if media_type == "image/png":
        if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Vertex returned an invalid PNG")
        width = int.from_bytes(content[16:20], "big")
        height = int.from_bytes(content[20:24], "big")
        if width and height:
            return width, height
        raise ValueError("Vertex PNG has invalid dimensions")
    if len(content) < 4 or content[:2] != b"\xff\xd8":
        raise ValueError("Vertex returned an invalid JPEG")
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 3 < len(content):
        if content[offset] != 0xFF:
            raise ValueError("Vertex JPEG is malformed")
        while offset < len(content) and content[offset] == 0xFF:
            offset += 1
        if offset >= len(content):
            break
        marker = content[offset]
        offset += 1
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if offset + 2 > len(content):
            break
        segment_length = int.from_bytes(content[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(content):
            raise ValueError("Vertex JPEG is malformed")
        if marker in start_of_frame and segment_length >= 7:
            height = int.from_bytes(content[offset + 3 : offset + 5], "big")
            width = int.from_bytes(content[offset + 5 : offset + 7], "big")
            if width and height:
                return width, height
        if marker == 0xDA:
            break
        offset += segment_length
    raise ValueError("Vertex JPEG omitted dimensions")


def _projection_depth_png(width: int, height: int) -> bytes:
    if not 1 <= width <= 4_096 or not 1 <= height <= 4_096:
        raise ValueError("generated image dimensions are outside the projection limit")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = bytearray()
    denominator = max(1, height - 1)
    for y in range(height):
        value = 64 + round(128 * y / denominator)
        rows.extend(b"\x00")
        rows.extend(bytes([value]) * width)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=1))
        + chunk(b"IEND", b"")
    )


def _write_bundle(
    request: FastSceneRequest,
    destination: Path,
    master: bytes,
    master_media_type: str,
    width: int,
    height: int,
    depth: bytes,
    remote_seconds: float,
    estimated_image_usd: float,
    configured_model: str,
    response: Mapping[str, Any],
    packaging_started: float,
) -> FiniteSceneBundle:
    destination.mkdir(parents=True, exist_ok=False)
    master_suffix = ".jpg" if master_media_type == "image/jpeg" else ".png"
    master_path = destination / f"master{master_suffix}"
    depth_path = destination / "depth.png"
    manifest_path = destination / "scene.manifest.json"
    _atomic_write(master_path, master)
    _atomic_write(depth_path, depth)
    packaging_seconds = time.perf_counter() - packaging_started
    master_sha = hashlib.sha256(master).hexdigest()
    depth_sha = hashlib.sha256(depth).hexdigest()
    now = datetime.now(UTC).isoformat()
    response_id = response.get("responseId")
    manifest = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": request.scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "contract_revision": REQUEST_CONTRACT_REVISION,
            "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(
                request.negative_prompt.encode()
            ).hexdigest(),
            "seed": request.seed,
            "provider_seed": request.seed & 0x7FFFFFFF,
            "requested_width": request.width,
            "requested_height": request.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
        },
        "policy": {
            "source_text_allowed": False,
            "raw_media_allowed": False,
            "managed_service": True,
            "fallback_depth": True,
            "cost_scope": "managed-image-request-estimate",
        },
        "stages": {
            "fast": {
                "model": (
                    str(response["modelVersion"])
                    if response.get("modelVersion")
                    else configured_model
                ),
                "model_revision": MODEL_REVISION,
                "additional_models": [
                    {
                        "role": "depth",
                        "model": DEPTH_MODEL,
                        "model_revision": DEPTH_MODEL_REVISION,
                    }
                ],
                "gpu": "vertex-managed",
                "remote_seconds": remote_seconds,
                "inference_seconds": remote_seconds,
                "provider_overhead_seconds": 0.0,
                "image_seconds": remote_seconds,
                "depth_seconds": 0.0,
                "packaging_seconds": packaging_seconds,
                "warm_state": "managed",
                "estimated_gpu_usd": estimated_image_usd,
                "estimated_managed_service_usd": estimated_image_usd,
                "response_id": response_id if isinstance(response_id, str) else None,
                "output_width": width,
                "output_height": height,
            }
        },
        "artifacts": {
            "master": _artifact_payload(
                master_path,
                destination,
                master,
                master_media_type,
                width,
                height,
            ),
            "depth": _artifact_payload(
                depth_path,
                destination,
                depth,
                "image/png",
                width,
                height,
            ),
        },
    }
    _atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return FiniteSceneBundle(
        manifest_path=manifest_path,
        scene_id=request.scene_id,
        artifacts={
            "master": SceneArtifact(
                role="master",
                path=master_path,
                sha256=master_sha,
                mime_type=master_media_type,
                width=width,
                height=height,
            ),
            "depth": SceneArtifact(
                role="depth",
                path=depth_path,
                sha256=depth_sha,
                mime_type="image/png",
                width=width,
                height=height,
            ),
        },
        manifest=manifest,
    )


def _artifact_payload(
    path: Path,
    root: Path,
    content: bytes,
    mime_type: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "duration_ms": 0,
        "frames": 1,
        "fps": 0,
    }


def _safe_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return " ".join(response.text.split())[:300] or "request rejected"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return " ".join(message.split())[:300]
    return "request rejected"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
