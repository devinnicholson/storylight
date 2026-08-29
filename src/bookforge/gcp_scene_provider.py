from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bookforge.finite_modal_provider import (
    DEPTH_DTYPE,
    DEPTH_MODEL,
    DEPTH_MODEL_REVISION,
    FAST_MODEL,
    FAST_MODEL_REVISION,
    FastSceneRequest,
    FiniteModalProviderError,
    FiniteModalUnavailableError,
    FiniteSceneBundle,
    MotionUpgradeRequest,
    SceneArtifact,
    WarmPrewarmReport,
    WarmProviderStatus,
)

PROVIDER_NAME = "gcp-cloud-run"
GPU_USD_PER_SECOND = {
    "L4": 0.0001867,
    "RTX_PRO_6000": 0.00036522,
}
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 90
MAX_SCALEDOWN_WINDOW_SECONDS = 900

IdentityTokenSource = Callable[[str], Awaitable[str]]


class GcpSceneProviderError(FiniteModalProviderError):
    pass


class GcpSceneUnavailableError(FiniteModalUnavailableError, GcpSceneProviderError):
    pass


@dataclass(frozen=True, slots=True)
class _RemoteArtifact:
    content: bytes
    sha256: str
    media_type: str
    width: int
    height: int


class GoogleImpersonatedIdentityTokenSource:
    """Mint and cache short-lived Cloud Run ID tokens without a service-account key."""

    def __init__(
        self,
        target_principal: str,
        *,
        source_access_token: Callable[[], str] | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        normalized = target_principal.strip()
        if (
            not normalized
            or "@" not in normalized
            or not normalized.endswith(".iam.gserviceaccount.com")
        ):
            raise ValueError("GCP impersonation target must be a service-account email")
        self.target_principal = normalized
        self._lock = asyncio.Lock()
        self._token = ""
        self._audience = ""
        self._expires_at = 0.0
        self._source_access_token = source_access_token or _google_source_access_token
        self._client_factory = client_factory

    async def __call__(self, audience: str) -> str:
        async with self._lock:
            if self._token and audience == self._audience and time.time() < self._expires_at - 60:
                return self._token
            access_token = await asyncio.to_thread(self._source_access_token)
            try:
                async with self._client_factory(
                    timeout=httpx.Timeout(30),
                    follow_redirects=False,
                ) as client:
                    response = await client.post(
                        "https://iamcredentials.googleapis.com/v1/projects/-/"
                        f"serviceAccounts/{self.target_principal}:generateIdToken",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "application/json",
                        },
                        json={"audience": audience, "includeEmail": True},
                    )
                    response.raise_for_status()
                    token = response.json()["token"]
                expires_at = _jwt_expiration(token)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as error:
                raise GcpSceneUnavailableError(
                    f"Google service-account ID token mint failed: {error}"
                ) from error
            self._token = token
            self._audience = audience
            self._expires_at = expires_at
            return token


class GcpCloudRunSceneProvider:
    """Private, bounded Cloud Run GPU adapter for SANA master and depth generation."""

    def __init__(
        self,
        *,
        base_url: str,
        audience: str,
        impersonate_service_account: str = "",
        gpu: str = "L4",
        timeout_seconds: float = 180,
        session_gpu_cap_usd: float = 0.50,
        token_source: IdentityTokenSource | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        normalized_audience = audience.strip().rstrip("/")
        if not normalized_url.startswith("https://"):
            raise ValueError("Cloud Run scene URL must use HTTPS")
        if not normalized_audience.startswith("https://"):
            raise ValueError("Cloud Run audience must use HTTPS")
        if gpu not in GPU_USD_PER_SECOND:
            raise ValueError("Cloud Run scene GPU must be L4 or RTX_PRO_6000")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
            raise ValueError("Cloud Run timeout must be 1-600 seconds")
        if not math.isfinite(session_gpu_cap_usd) or not 0 < session_gpu_cap_usd <= 10:
            raise ValueError("GCP scene session cap must be between $0 and $10")
        if impersonate_service_account and token_source is not None:
            raise ValueError(
                "configure either GCP service-account impersonation or a token source, not both"
            )
        self.base_url = normalized_url
        self.audience = normalized_audience
        self.gpu = gpu
        self.gpu_usd_per_second = GPU_USD_PER_SECOND[gpu]
        self.timeout_seconds = timeout_seconds
        self.session_gpu_cap_usd = session_gpu_cap_usd
        self._token_source = token_source or (
            GoogleImpersonatedIdentityTokenSource(impersonate_service_account)
            if impersonate_service_account
            else google_identity_token
        )
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._estimated_gpu_usd = 0.0
        self._prewarm_id: str | None = None
        self._prewarm_deadline = 0.0
        self._remote_warm_deadline = 0.0
        self._scaledown_window_seconds = DEFAULT_SCALEDOWN_WINDOW_SECONDS
        self._operation_lock = asyncio.Lock()

    async def probe(self) -> tuple[bool, str]:
        try:
            # Cloud Run reserves some paths ending in "z", including common
            # healthz variants, before requests reach the container.
            payload, _ = await self._request("GET", "/health")
            _require_equal(payload, "provider", PROVIDER_NAME)
            _require_equal(payload, "fast_model_revision", FAST_MODEL_REVISION)
            _require_equal(payload, "depth_model_revision", DEPTH_MODEL_REVISION)
        except Exception as error:
            return False, f"private Cloud Run renderer is unreachable: {error}"
        return True, (f"Private Cloud Run {self.gpu} renderer is reachable with pinned models")

    async def prewarm(
        self,
        *,
        prewarm_id: str,
        include_motion: bool = False,
        scaledown_window_seconds: int = DEFAULT_SCALEDOWN_WINDOW_SECONDS,
    ) -> WarmPrewarmReport:
        _validate_identifier(prewarm_id)
        if include_motion:
            raise GcpSceneProviderError("the GCP master/depth service does not host motion")
        if not DEFAULT_SCALEDOWN_WINDOW_SECONDS <= scaledown_window_seconds <= 900:
            raise ValueError("scaledown window must be 90-900 seconds")
        async with self._operation_lock:
            if self._prewarm_id is not None and time.monotonic() < self._prewarm_deadline:
                raise GcpSceneProviderError("a Cloud Run renderer prewarm is already active")
            payload, remote_seconds = await self._paid_request(
                "POST",
                "/v1/prewarm",
                json_body={"prewarm_id": prewarm_id},
            )
            _validate_runtime_identity(payload, expected_gpu=self.gpu)
            self._prewarm_id = prewarm_id
            self._scaledown_window_seconds = scaledown_window_seconds
            self._prewarm_deadline = time.monotonic() + scaledown_window_seconds
            self._remote_warm_deadline = self._prewarm_deadline
            return WarmPrewarmReport(
                prewarm_id=prewarm_id,
                reservation_id=f"gcp-cloud-run:{prewarm_id}",
                include_motion=False,
                fast_remote_seconds=remote_seconds,
                motion_remote_seconds=0,
                full_session_ceiling_usd=self.session_gpu_cap_usd,
                fast_model_load_seconds=_non_negative_float(payload, "model_load_seconds"),
                motion_model_load_seconds=0,
                expires_in_seconds=scaledown_window_seconds,
                scaledown_window_seconds=scaledown_window_seconds,
                fast_inference_warmup_seconds=_non_negative_float(
                    payload, "inference_warmup_seconds"
                ),
            )

    async def warm_status(self) -> WarmProviderStatus:
        ready, detail = await self.probe()
        async with self._operation_lock:
            expires = max(0.0, self._prewarm_deadline - time.monotonic())
            if expires == 0:
                self._prewarm_id = None
            return WarmProviderStatus(
                ready=ready,
                detail=detail,
                state="prewarmed" if self._prewarm_id else "idle",
                prewarm_id=self._prewarm_id,
                include_motion=False,
                expires_in_seconds=expires,
                scaledown_window_seconds=self._scaledown_window_seconds,
            )

    async def is_prewarmed(self) -> bool:
        async with self._operation_lock:
            if time.monotonic() >= self._prewarm_deadline:
                self._prewarm_id = None
            return self._prewarm_id is not None

    async def is_renderer_likely_warm(self) -> bool:
        async with self._operation_lock:
            now = time.monotonic()
            if now >= self._prewarm_deadline:
                self._prewarm_id = None
            return self._prewarm_id is not None or now < self._remote_warm_deadline

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        if request.steps != 2:
            raise GcpSceneProviderError(
                "the GCP SANA Sprint renderer requires exactly 2 inference steps"
            )
        destination = output_dir.resolve()
        if destination.exists():
            raise GcpSceneProviderError(f"scene output already exists: {destination}")
        async with self._operation_lock:
            payload, remote_seconds = await self._paid_request(
                "POST",
                "/v1/generate",
                json_body={
                    "scene_id": request.scene_id,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "seed": request.seed,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                },
            )
            _validate_runtime_identity(payload, expected_gpu=self.gpu)
            _require_equal(payload, "scene_id", request.scene_id)
            master = _decode_artifact(payload, "master", request.width, request.height)
            depth = _decode_artifact(payload, "depth", request.width, request.height)
            if master.media_type != "image/jpeg" or depth.media_type != "image/jpeg":
                raise GcpSceneProviderError("Cloud Run returned unsupported scene media")
            estimated_gpu_usd = remote_seconds * self.gpu_usd_per_second
            bundle = await asyncio.to_thread(
                _write_bundle,
                request,
                destination,
                payload,
                master,
                depth,
                remote_seconds,
                estimated_gpu_usd,
            )
            self._remote_warm_deadline = (
                time.monotonic() + self._scaledown_window_seconds
            )
            return bundle

    async def upgrade_motion(
        self,
        bundle: FiniteSceneBundle | Path,
        request: MotionUpgradeRequest,
    ) -> FiniteSceneBundle:
        del bundle, request
        raise GcpSceneUnavailableError("motion is not enabled on the GCP fast-scene service")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        token = await self._token_source(self.audience)
        headers = {"Authorization": f"Bearer {token}"}
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                json=dict(json_body) if json_body is not None else None,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            raise GcpSceneUnavailableError(f"Cloud Run scene request failed: {error}") from error
        if not isinstance(payload, dict):
            raise GcpSceneProviderError("Cloud Run scene response must be a JSON object")
        return payload, time.perf_counter() - started

    async def _get_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is not None:
            return client
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

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()

    async def _paid_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float]:
        worst_case = self.timeout_seconds * self.gpu_usd_per_second
        self._ensure_estimate_capacity(worst_case)
        try:
            payload, remote_seconds = await self._request(
                method,
                path,
                json_body=json_body,
            )
        except BaseException:
            # The remote GPU can continue after the client loses its response.
            # Retain the full timeout reservation and require an operator restart
            # before another session can reuse that uncertain headroom.
            self._reserve_estimate(worst_case)
            raise
        self._reserve_estimate(remote_seconds * self.gpu_usd_per_second)
        return payload, remote_seconds

    def _ensure_estimate_capacity(self, amount: float) -> None:
        if self._estimated_gpu_usd + amount > self.session_gpu_cap_usd + 1e-9:
            raise GcpSceneProviderError(
                f"GCP request ceiling would exceed ${self.session_gpu_cap_usd:.2f} session cap"
            )

    def _reserve_estimate(self, amount: float) -> None:
        projected = self._estimated_gpu_usd + amount
        if projected > self.session_gpu_cap_usd + 1e-9:
            raise GcpSceneProviderError(
                f"GCP session GPU estimate would exceed ${self.session_gpu_cap_usd:.2f}"
            )
        self._estimated_gpu_usd = projected


def _google_source_access_token() -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as error:
        raise GcpSceneUnavailableError(
            "Google authentication is unavailable; install bookforge[gcp]"
        ) from error
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
    except Exception as error:
        raise GcpSceneUnavailableError(
            f"Google source credential refresh failed: {error}"
        ) from error
    token = credentials.token
    if not isinstance(token, str) or not token:
        raise GcpSceneUnavailableError("Google source credentials returned no access token")
    return token


def _jwt_expiration(token: str) -> float:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Google ID token is not a JWT")
    encoded = parts[1]
    encoded += "=" * (-len(encoded) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    expiration = payload.get("exp")
    if not isinstance(expiration, int) or expiration <= time.time():
        raise ValueError("Google ID token has no valid expiration")
    return float(expiration)


async def google_identity_token(audience: str) -> str:
    def fetch() -> str:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.id_token import fetch_id_token
        except ImportError as error:
            raise GcpSceneUnavailableError(
                "Google authentication is unavailable; install bookforge[gcp]"
            ) from error
        return fetch_id_token(Request(), audience)

    return await asyncio.to_thread(fetch)


def _validate_runtime_identity(payload: Mapping[str, Any], *, expected_gpu: str) -> None:
    _require_equal(payload, "provider", PROVIDER_NAME)
    _require_equal(payload, "gpu", expected_gpu)
    _require_equal(payload, "fast_model", FAST_MODEL)
    _require_equal(payload, "fast_model_revision", FAST_MODEL_REVISION)
    _require_equal(payload, "depth_model", DEPTH_MODEL)
    _require_equal(payload, "depth_model_revision", DEPTH_MODEL_REVISION)
    _require_equal(payload, "depth_dtype", DEPTH_DTYPE)


def _decode_artifact(
    payload: Mapping[str, Any],
    role: str,
    expected_width: int,
    expected_height: int,
) -> _RemoteArtifact:
    encoded = payload.get(f"{role}_b64")
    if not isinstance(encoded, str):
        raise GcpSceneProviderError(f"Cloud Run response omitted {role}_b64")
    try:
        content = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise GcpSceneProviderError(f"Cloud Run returned invalid {role} base64") from error
    digest = hashlib.sha256(content).hexdigest()
    _require_equal(payload, f"{role}_sha256", digest)
    width = _positive_int(payload, f"{role}_width")
    height = _positive_int(payload, f"{role}_height")
    if (width, height) != (expected_width, expected_height):
        raise GcpSceneProviderError(f"Cloud Run returned unexpected {role} dimensions")
    media_type = payload.get(f"{role}_media_type")
    if not isinstance(media_type, str):
        raise GcpSceneProviderError(f"Cloud Run response omitted {role}_media_type")
    return _RemoteArtifact(content, digest, media_type, width, height)


def _write_bundle(
    request: FastSceneRequest,
    destination: Path,
    remote: Mapping[str, Any],
    master: _RemoteArtifact,
    depth: _RemoteArtifact,
    remote_seconds: float,
    estimated_gpu_usd: float,
) -> FiniteSceneBundle:
    destination.mkdir(parents=True, exist_ok=False)
    master_path = destination / "master.jpg"
    depth_path = destination / "depth.jpg"
    manifest_path = destination / "scene.manifest.json"
    _atomic_write(master_path, master.content)
    _atomic_write(depth_path, depth.content)
    image_seconds = _non_negative_float(remote, "image_seconds")
    depth_seconds = _non_negative_float(remote, "depth_seconds")
    packaging_seconds = _non_negative_float(remote, "packaging_seconds")
    image_gpu_ms = _optional_non_negative_float(remote, "image_gpu_ms")
    depth_gpu_ms = _optional_non_negative_float(remote, "depth_gpu_ms")
    inference_seconds = image_seconds + depth_seconds
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": request.scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
        },
        "policy": {
            "private_iam_endpoint": True,
            "source_text_allowed": False,
            "max_instances": 1,
            "provider_mode": "gcp-cloud-run-gpu",
            "cost_scope": "request-wall-gpu-estimate",
        },
        "stages": {
            "fast": {
                "model": FAST_MODEL,
                "model_revision": FAST_MODEL_REVISION,
                "additional_models": [
                    {"model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION}
                ],
                "gpu": str(remote["gpu"]),
                "remote_seconds": remote_seconds,
                "inference_seconds": inference_seconds,
                "provider_overhead_seconds": max(0.0, remote_seconds - inference_seconds),
                "image_seconds": image_seconds,
                "depth_seconds": depth_seconds,
                "image_gpu_ms": image_gpu_ms,
                "depth_gpu_ms": depth_gpu_ms,
                "packaging_seconds": packaging_seconds,
                "model_load_seconds": _non_negative_float(remote, "model_load_seconds"),
                "container_age_seconds": _non_negative_float(remote, "container_age_seconds"),
                "warm_state": str(remote.get("warm_state", "unknown")),
                "estimated_gpu_usd": estimated_gpu_usd,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "master_jpeg_quality": 95,
                "depth_jpeg_quality": 85,
                "negative_prompt_supported": False,
            }
        },
        "artifacts": {
            "master": _artifact_manifest(master_path, master, destination),
            "depth": _artifact_manifest(depth_path, depth, destination),
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
                sha256=master.sha256,
                mime_type=master.media_type,
                width=master.width,
                height=master.height,
            ),
            "depth": SceneArtifact(
                role="depth",
                path=depth_path,
                sha256=depth.sha256,
                mime_type=depth.media_type,
                width=depth.width,
                height=depth.height,
            ),
        },
        manifest=manifest,
    )


def _artifact_manifest(path: Path, artifact: _RemoteArtifact, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": artifact.sha256,
        "bytes": len(artifact.content),
        "mime_type": artifact.media_type,
        "width": artifact.width,
        "height": artifact.height,
        "duration_ms": 0,
        "frames": 1,
        "fps": 0,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _require_equal(payload: Mapping[str, Any], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        raise GcpSceneProviderError(f"Cloud Run response has unexpected {key}")


def _non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as error:
        raise GcpSceneProviderError(f"Cloud Run response has invalid {key}") from error
    if not math.isfinite(value) or value < 0:
        raise GcpSceneProviderError(f"Cloud Run response has invalid {key}")
    return value


def _optional_non_negative_float(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    return _non_negative_float(payload, key)


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GcpSceneProviderError(f"Cloud Run response has invalid {key}")
    return value


def _validate_identifier(value: str) -> None:
    if (
        not value
        or len(value) > 96
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise ValueError("prewarm_id requires 1-96 letters, numbers, hyphens, or underscores")
