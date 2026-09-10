"""Inactive adapter for an externally owned, finite prepared Klein service.

The reviewed live-scene provider must validate original-source privacy before creating
FastSceneRequest. This boundary receives only its compiled prompt. Closing this HTTP
adapter does not delete the service; the external owner must enforce cloud teardown.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import re
import time
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

from storylight.finite_modal_provider import (
    FastSceneRequest,
    FiniteModalProviderError,
    FiniteSceneBundle,
    SceneArtifact,
    WarmPrewarmReport,
    WarmProviderStatus,
    _atomic_write,
    _atomic_write_json,
    _jpeg_dimensions,
    _validate_identifier,
)
from storylight.gcp_scene_provider import GoogleImpersonatedIdentityTokenSource
from storylight.privacy_policy import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    SENSITIVE_CONTENT_PATTERN,
    URL_PATTERN,
)

IDENTITY = {
    "buckets": [128, 256, 512], "capability": [12, 0], "compile_mode": "default",
    "contract": "klein-regional-default-v1", "cuda": "12.8",
    "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_revision": "b4769fd619394250528294b658587285526fab1c",
    "diffusers": "0.39.0", "dtype": "bfloat16", "fullgraph": True,
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "guidance": 1.0,
    "height": 576, "model": "black-forest-labs/FLUX.2-klein-4B",
    "model_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "runtime_sha256": "b3d1007f297bde37ac40a9f2dee9eeb89e985dd16fdb8bdd9478c0086f6ca92a",
    "steps": 4, "torch": "2.8.0+cu128", "transformers": "4.57.1",
    "triton": "3.4.0", "width": 1024,
}
LEASE_KEYS = {"schema_version", "state", "session_id", "instance_id", "service", "revision",
              "identity", "started_at", "expires_at", "warmups"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


IDENTITY_BYTES = encoded(IDENTITY)


def require(value):
    if not value:
        raise FiniteModalProviderError("prepared Klein evidence mismatch")


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
    return value


def validate_prepared_prompt(prompt):
    """Screen obvious sensitive data; source-aware privacy remains upstream."""
    require(isinstance(prompt, str) and bool(prompt.strip()) and len(prompt) <= 4000)
    require(not any(ord(c) < 32 and c not in "\n\t" for c in prompt))
    require(not any(p.search(prompt) for p in (
        EMAIL_PATTERN, PHONE_PATTERN, URL_PATTERN, SENSITIVE_CONTENT_PATTERN,
    )))


class PreparedKleinProvider:
    def __init__(self, *, base_url, service, revision, external_owner_id, claims_dir: Path,
                 impersonate_service_account="", token_source=None,
                 client_factory=httpx.AsyncClient, timeout_seconds=180,
                 session_gpu_cap_usd=3.50, clock=time.monotonic, wall_clock=time.time):
        url = httpx.URL(base_url)
        require(url.scheme == "https" and url.host.endswith(".run.app")
                and not url.userinfo and url.port in (None, 443)
                and url.path == "/" and not url.query and not url.fragment)
        _validate_identifier(service)
        _validate_identifier(revision)
        _validate_identifier(external_owner_id)
        require(1 <= number(timeout_seconds) <= 180)
        require(0 < number(session_gpu_cap_usd) <= 3.50)
        require(bool(token_source) != bool(impersonate_service_account))
        self.base_url = str(url).rstrip("/")
        self.service, self.revision = service, revision
        self.external_owner_id, self.claims_dir = external_owner_id, Path(claims_dir)
        self.token_source = token_source or GoogleImpersonatedIdentityTokenSource(
            impersonate_service_account)
        self.client_factory, self.timeout_seconds = client_factory, timeout_seconds
        self.session_gpu_cap_usd = session_gpu_cap_usd
        self.clock, self.wall_clock = clock, wall_clock
        self.session_id = uuid4().hex
        self.lock = asyncio.Lock()
        self.lease = self.report = None
        self.deadline = 0.0
        self.failed = self.closed = False
        self.requests = 0
        self._client = None
        self._close_task = None

    def _claim(self, label, request_id, body):
        for ancestor in (self.claims_dir, *self.claims_dir.parents):
            require(not ancestor.is_symlink())
            require(not ancestor.exists() or ancestor.is_dir())
        self.claims_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(encoded([self.base_url, self.service, label])).hexdigest()
        try:
            with (self.claims_dir / (key + ".json")).open("x") as stream:
                json.dump({"request_id": request_id, "session_id": self.session_id,
                           "body_sha256": hashlib.sha256(encoded(body)).hexdigest(),
                           "external_owner_id": self.external_owner_id}, stream)
        except FileExistsError:
            raise FiniteModalProviderError("prepared Klein operation already claimed") from None

    async def _call(self, path, body, seconds):
        try:
            async with asyncio.timeout(seconds):
                token = await self.token_source(self.base_url)
                require(isinstance(token, str) and bool(token.strip()))
                if self._client is None:
                    self._client = self.client_factory(
                        timeout=self.timeout_seconds, follow_redirects=False, trust_env=False)
                async with self._client.stream(
                    "POST", self.base_url + path, json=body, timeout=seconds,
                    headers={"Authorization": "Bearer " + token},
                ) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        require(size <= 12 * 1024**2)
                        chunks.append(chunk)
                return json.loads(b"".join(chunks), object_pairs_hook=_unique_pairs)
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception:
            self.failed = True
            raise FiniteModalProviderError(
                "prepared Klein request failed; no automatic retry or fallback is permitted"
            ) from None

    def _remaining(self):
        if self.failed or self.closed or self.lease is None:
            return 0.0
        return max(0.0, min(self.deadline - self.clock(),
                            self.lease["expires_at"] - self.wall_clock()))

    def _validate_lease(self, lease):
        _lease_shape(lease, self.service, self.revision)
        require(lease["session_id"] == self.session_id)
        require(lease["started_at"] <= self.wall_clock() < lease["expires_at"])
        if self.lease is not None:
            require(encoded(lease) == encoded(self.lease) and self._remaining() > 0)

    async def prewarm(self, *, prewarm_id, include_motion=False, scaledown_window_seconds=300):
        _validate_identifier(prewarm_id)
        require(not include_motion and scaledown_window_seconds == 300)
        async with self.lock:
            require(not self.failed and not self.closed and self._close_task is None)
            if self.report is not None:
                require(self.report.prewarm_id == prewarm_id and self._remaining() > 0)
                return replace(self.report, expires_in_seconds=self._remaining())
            body = {"session_id": self.session_id}
            self._claim("prepare", self.session_id, body)
            started = self.clock()
            try:
                result = await self._call("/v1/prewarm", body, self.timeout_seconds)
                require(set(result) == {"lease", "model_load_seconds", "warmup_seconds"})
                self._validate_lease(result["lease"])
                self.lease = result["lease"]
                self.deadline = min(started + 300, self.clock()
                                    + self.lease["expires_at"] - self.wall_clock())
                self.report = WarmPrewarmReport(
                    prewarm_id, self.external_owner_id, False, self.clock() - started, 0,
                    self.session_gpu_cap_usd, number(result["model_load_seconds"]), 0,
                    self._remaining(), 300, number(result["warmup_seconds"]))
                return self.report
            except asyncio.CancelledError:
                self.failed = True
                raise
            except Exception:
                self.failed = True
                raise FiniteModalProviderError(
                    "prepared Klein preparation failed; no retry") from None

    async def is_prewarmed(self):
        return self._remaining() > 0 and self.requests < 10 and self._close_task is None

    async def warm_status(self):
        remaining = self._remaining() if self.requests < 10 and self._close_task is None else 0.0
        return WarmProviderStatus(
            ready=remaining > 0,
            detail="Verified prepared Klein lease" if remaining else "No verified live Klein lease",
            state="prewarmed" if remaining else "idle",
            prewarm_id=self.report.prewarm_id if remaining else None,
            expires_in_seconds=remaining, scaledown_window_seconds=300)

    async def generate_fast(self, request: FastSceneRequest, *, output_dir: Path):
        require((request.width, request.height, request.steps, request.guidance_scale)
                == (1024, 576, 4, 1.0))
        require(not request.fidelity_label and not request.fidelity_object_label
                and not request.require_subject_object_overlap)
        require(type(request.seed) is int and 0 <= request.seed < 2**32)
        validate_prepared_prompt(request.prompt)
        async with self.lock:
            require(self._remaining() > 0 and self.requests < 10 and self._close_task is None)
            require(not output_dir.exists() and not output_dir.is_symlink())
            request_id = uuid4().hex
            body = {"session_id": self.session_id, "instance_id": self.lease["instance_id"],
                    "request_id": request_id, "scene_id": request.scene_id,
                    "prompt": request.prompt, "seed": request.seed}
            self._claim("generate:" + request.scene_id, request_id, body)
            self.requests += 1
            started = self.clock()
            try:
                payload = await self._call("/v1/generate", body,
                                           min(self.timeout_seconds, self._remaining()))
                require(set(payload) == {"lease", "request_id", "scene_id", "master_b64",
                                         "depth_b64", "metrics"})
                self._validate_lease(payload["lease"])
                require(payload["request_id"] == request_id
                        and payload["scene_id"] == request.scene_id)
                return _bundle(request, output_dir, payload, self.clock() - started,
                               self.external_owner_id)
            except asyncio.CancelledError:
                self.failed = True
                raise
            except Exception:
                self.failed = True
                raise FiniteModalProviderError(
                    "prepared Klein generation failed; no retry") from None

    async def _close(self):
        async with self.lock:
            self.closed = True
            if self._client is not None:
                await self._client.aclose()

    async def aclose(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            # Retain ownership through lock drain and pool close even if the caller leaves.
            while not self._close_task.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(self._close_task)
            self._close_task.result()
            raise


def _unique_pairs(items):
    value = {}
    for key, item in items:
        require(key not in value)
        value[key] = item
    return value


def _lease_shape(lease, service, revision):
    require(type(lease) is dict and set(lease) == LEASE_KEYS)
    require(type(lease["schema_version"]) is int and lease["schema_version"] == 1)
    require(lease["state"] == "READY")
    for key in ("session_id", "instance_id"):
        require(isinstance(lease[key], str) and re.fullmatch(r"[a-f0-9]{32}", lease[key]))
    require(lease["service"] == service and lease["revision"] == revision)
    require(encoded(lease["identity"]) == IDENTITY_BYTES)
    require(number(lease["expires_at"]) == number(lease["started_at"]) + 300)
    require(type(lease["warmups"]) is list and len(lease["warmups"]) == 2)
    for bucket, row in zip((128, 256), lease["warmups"], strict=True):
        require(type(row) is dict and set(row) == {
            "sequence_bucket", "seed", "master_sha256", "depth_sha256"})
        require(type(row["sequence_bucket"]) is int and row["sequence_bucket"] == bucket)
        require(type(row["seed"]) is int and 0 <= row["seed"] < 2**32)
        for role in ("master", "depth"):
            require(isinstance(row[role + "_sha256"], str)
                    and re.fullmatch(r"[a-f0-9]{64}", row[role + "_sha256"]))


def _artifacts(request, directory, payload):
    metrics = payload["metrics"]
    require(type(metrics) is dict and set(metrics) == {
        "seed", "token_count", "sequence_bucket", "image_seconds", "depth_seconds",
        "encoding_seconds", "total_seconds", "peak_allocated_gib", "peak_reserved_gib",
        "master_sha256", "depth_sha256"})
    require(type(metrics["seed"]) is int and metrics["seed"] == request.seed)
    require(type(metrics["token_count"]) is int and 1 <= metrics["token_count"] <= 256)
    require(type(metrics["sequence_bucket"]) is int and metrics["sequence_bucket"]
            == (128 if metrics["token_count"] <= 128 else 256))
    for name in ("image_seconds", "depth_seconds", "encoding_seconds", "total_seconds",
                 "peak_allocated_gib", "peak_reserved_gib"):
        number(metrics[name])
    artifacts, content = {}, {}
    for role in ("master", "depth"):
        data = base64.b64decode(payload[role + "_b64"], validate=True)
        digest = hashlib.sha256(data).hexdigest()
        require(0 < len(data) <= 4 * 1024**2 and digest == metrics[role + "_sha256"]
                and _jpeg_dimensions(data) == (1024, 576))
        with Image.open(io.BytesIO(data)) as image:
            require(image.format == "JPEG" and image.size == (1024, 576))
            image.load()
        content[role] = data
        artifacts[role] = SceneArtifact(role, directory / f"{role}.jpg", digest,
                                        "image/jpeg", 1024, 576)
    return artifacts, content


def _manifest(request, payload, artifacts, elapsed, owner):
    number(elapsed)
    require(isinstance(payload["request_id"], str)
            and re.fullmatch(r"[a-f0-9]{32}", payload["request_id"]))
    # Public us-central1 instance rates, https://cloud.google.com/run/pricing (2026-09-08).
    gpu, cpu, ram = elapsed * 0.00036522, elapsed * 20 * 0.000018, elapsed * 80 * 0.000002
    return {
        "schema_version": "1.0", "provider": "gcp-prepared-klein-candidate",
        "scene_id": request.scene_id, "request": asdict(request), "lease": payload["lease"],
        "request_id": payload["request_id"],
        "stages": {"fast": {**payload["metrics"], "remote_seconds": elapsed,
                             "model": IDENTITY["model"],
                             "model_revision": IDENTITY["model_revision"],
                             "additional_models": [{"role": "depth",
                                "model": IDENTITY["depth_model"],
                                "model_revision": IDENTITY["depth_revision"]}],
                             "gpu": IDENTITY["gpu"], "warm_state": "warm",
                             "steps": 4, "guidance_scale": 1.0,
                             # Existing bundle consumers use this legacy name for compute cost.
                             "estimated_gpu_usd": gpu + cpu + ram,
                             "estimated_gpu_only_usd": gpu, "estimated_cpu_usd": cpu,
                             "estimated_memory_usd": ram,
                             "cost_scope": "request GPU+20CPU+80GiB estimate; excludes lifecycle",
                             "negative_prompt_supported": False}},
        "policy": {"external_owner_id": owner, "automatic_retries": 0,
                   "visual_acceptance": "human-review-required",
                   "cloud_deletion_by_adapter": False},
        "artifacts": {role: {**asdict(a), "path": a.path.name} for role, a in artifacts.items()},
    }


def _bundle(request, directory, payload, elapsed, owner):
    artifacts, content = _artifacts(request, directory, payload)
    manifest = _manifest(request, payload, artifacts, elapsed, owner)
    directory.mkdir(parents=True, exist_ok=False)
    for role, artifact in artifacts.items():
        _atomic_write(artifact.path, content[role])
    path = directory / "scene.manifest.json"
    _atomic_write_json(path, manifest)
    return FiniteSceneBundle(path, request.scene_id, artifacts, manifest)


def load_prepared_klein_bundle(manifest_path: Path, *, request: FastSceneRequest,
                              service: str, revision: str, external_owner_id: str):
    """Recover only the exact requested verified bundle; never restores a warm lease."""
    try:
        path = Path(manifest_path)
        require(path.name == "scene.manifest.json")
        require(not any(p.is_symlink() for p in (path, *path.parents)))
        require(path.is_file() and path.stat().st_size <= 65536)
        stored = json.loads(path.read_bytes(), object_pairs_hook=_unique_pairs)
        require(stored["provider"] == "gcp-prepared-klein-candidate")
        require(encoded(stored["request"]) == encoded(asdict(request)))
        require((request.width, request.height, request.steps, request.guidance_scale)
                == (1024, 576, 4, 1.0))
        validate_prepared_prompt(request.prompt)
        _lease_shape(stored["lease"], service, revision)
        stage = stored["stages"]["fast"]
        keys = {"seed", "token_count", "sequence_bucket", "image_seconds", "depth_seconds",
                "encoding_seconds", "total_seconds", "peak_allocated_gib", "peak_reserved_gib",
                "master_sha256", "depth_sha256"}
        payload = {"lease": stored["lease"], "request_id": stored["request_id"],
                   "metrics": {key: stage[key] for key in keys}}
        for role in ("master", "depth"):
            asset = path.parent / f"{role}.jpg"
            require(asset.is_file() and not asset.is_symlink()
                    and asset.stat().st_size <= 4 * 1024**2)
            payload[role + "_b64"] = base64.b64encode(asset.read_bytes()).decode()
        artifacts, _ = _artifacts(request, path.parent, payload)
        expected = _manifest(request, payload, artifacts,
                             stage["remote_seconds"], external_owner_id)
        require(encoded(stored) == encoded(expected))
        return FiniteSceneBundle(path, request.scene_id, artifacts, stored)
    except Exception:
        raise FiniteModalProviderError("prepared Klein saved bundle failed verification") from None
