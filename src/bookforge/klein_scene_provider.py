"""Experimental Klein adapter using the normal live jobs, verified assets and projector."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import math
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from bookforge.finite_modal_provider import (
    DEPTH_MODEL,
    DEPTH_MODEL_REVISION,
    FastSceneRequest,
    FiniteModalBudgetError,
    FiniteModalProviderError,
    FiniteModalSceneProvider,
    FiniteSceneBundle,
    SceneArtifact,
    WarmPrewarmReport,
    WarmProviderStatus,
    _atomic_write,
    _atomic_write_json,
    _jpeg_dimensions,
)

MODEL = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
APP = "bookforge-klein-candidate"
CALL_CEILING_USD = 0.25  # 120s startup + 180s method + 90s idle, including CPU/RAM.


class KleinInvoker:
    def __init__(self):
        self.instance = None
        self.lock = asyncio.Lock()

    async def probe(self):
        async with self.lock:
            if self.instance is None:
                modal = importlib.import_module("modal")
                cls = modal.Cls.from_name(APP, "KleinSceneStudio")
                await cls.hydrate.aio()  # Metadata lookup does not allocate a GPU.
                self.instance = cls()
        return True, "Klein deployment metadata is reachable; lookup does not establish GPU warmth"

    async def invoke(self, method, **arguments):
        await self.probe()
        call = await getattr(self.instance, method).spawn.aio(**arguments)
        try:
            return await asyncio.wait_for(call.get.aio(), timeout=310)
        except BaseException:
            await asyncio.shield(call.cancel.aio(terminate_containers=True))
            raise


class KleinSceneProvider(FiniteModalSceneProvider):
    def __init__(self, *, invoker=None, **kwargs):
        super().__init__(**kwargs)
        self.invoker = invoker or KleinInvoker()
        self.operation_lock = asyncio.Lock()
        self.reserved_usd = 0.0
        self.warm_deadline = 0.0
        self.prewarm_id = None
        self.session_id = uuid4().hex
        self.reservation_id = None
        self.operations = set()

    async def probe(self):
        if not self.plan_file.is_file():
            return False, "Klein candidate requires an explicit Modal budget plan"
        try:
            return await self.invoker.probe()
        except Exception as error:
            return False, f"Klein candidate deployment unavailable: {type(error).__name__}"

    async def _call(self, operation, identity, **arguments):
        if self.reserved_usd + CALL_CEILING_USD > self.session_gpu_cap_usd + 1e-9:
            raise FiniteModalBudgetError("Klein candidate session compute allowance exhausted")
        key = (operation, identity)
        if key in self.operations:
            raise FiniteModalBudgetError("Klein operation already attempted; refusing duplicate")
        if self.reservation_id is None:
            self.reservation_id = await self._reserve_against_current_billing(
                experiment_id=f"klein:session:{self.session_id}",
                full_call_ceiling_usd=self.session_gpu_cap_usd,
            )
        # Retain the conservative complete-call reservation, even on cancellation/failure.
        # No billing-lag refunds and no implicit retry of a paid call.
        self.reserved_usd += CALL_CEILING_USD
        self.operations.add(key)
        return self.reservation_id, await self.invoker.invoke(operation, **arguments)

    async def prewarm(self, *, prewarm_id, include_motion=False, scaledown_window_seconds=90):
        if include_motion or scaledown_window_seconds != 90:
            raise FiniteModalProviderError(
                "Klein preview supports only a 90-second idle window and no video"
            )
        async with self.operation_lock:
            started = time.perf_counter()
            reservation, report = await self._call("prewarm", prewarm_id)
            self.warm_deadline = time.monotonic() + 90
            self.prewarm_id = prewarm_id
            return WarmPrewarmReport(
                prewarm_id=prewarm_id,
                reservation_id=reservation,
                include_motion=False,
                fast_remote_seconds=time.perf_counter() - started,
                motion_remote_seconds=0,
                full_session_ceiling_usd=self.session_gpu_cap_usd,
                fast_model_load_seconds=report["model_load_seconds"],
                motion_model_load_seconds=0,
                expires_in_seconds=90,
                fast_inference_warmup_seconds=report["warmup_seconds"],
            )

    async def warm_status(self):
        ready, detail = await self.probe()
        remaining = max(0.0, self.warm_deadline - time.monotonic())
        return WarmProviderStatus(
            ready=ready,
            detail=detail,
            state="prewarmed" if remaining and self.prewarm_id else "idle",
            prewarm_id=self.prewarm_id if remaining else None,
            expires_in_seconds=remaining,
        )

    async def generate_fast(self, request: FastSceneRequest, *, output_dir: Path):
        if (request.width, request.height, request.steps, request.guidance_scale) != (
            1024,
            576,
            4,
            1.0,
        ):
            raise ValueError(
                "Klein requires the qualified 1024x576, four-step, guidance-one profile"
            )
        if request.fidelity_label:
            raise ValueError("Klein preview has no automated visual acceptance gate; use deferred")
        if output_dir.exists():
            raise FiniteModalProviderError(
                "Klein output already exists; refusing a duplicate paid call"
            )
        async with self.operation_lock:
            started = time.perf_counter()
            reservation, payload = await self._call(
                "generate",
                request.scene_id,
                prompt=request.prompt,
                seed=request.seed,
            )
            elapsed = time.perf_counter() - started
            self.warm_deadline = time.monotonic() + 90
            return write_bundle(request, output_dir, payload, elapsed, reservation)


def write_bundle(request, output_dir, payload, elapsed, reservation):
    identity, metrics = payload["identity"], payload["metrics"]
    for key, expected in {
        "model": MODEL,
        "model_revision": REVISION,
        "depth_model": DEPTH_MODEL,
        "depth_revision": DEPTH_MODEL_REVISION,
        "width": 1024,
        "height": 576,
        "steps": 4,
        "guidance": 1.0,
        "gpu": "NVIDIA L4",
    }.items():
        if identity.get(key) != expected:
            raise FiniteModalProviderError(f"Klein provenance mismatch: {key}")
    if metrics["seed"] != request.seed or metrics["sequence_bucket"] not in (128, 256):
        raise FiniteModalProviderError("Klein changed seed or token profile")
    for field in ("image_seconds", "depth_seconds", "encoding_seconds", "total_seconds"):
        if not math.isfinite(metrics[field]) or metrics[field] < 0:
            raise FiniteModalProviderError("Klein returned invalid timing evidence")
    artifacts = {}
    for role in ("master", "depth"):
        content = payload[role]
        digest = hashlib.sha256(content).hexdigest()
        if digest != metrics[f"{role}_sha256"] or _jpeg_dimensions(content) != (1024, 576):
            raise FiniteModalProviderError(f"Klein {role} bytes failed verification")
        artifacts[role] = SceneArtifact(
            role, output_dir / f"{role}.jpg", digest, "image/jpeg", 1024, 576
        )
    stage = {
        "model": MODEL,
        "model_revision": REVISION,
        "additional_models": [
            {"role": "depth", "model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION}
        ],
        "gpu": "L4",
        "remote_seconds": elapsed,
        "image_seconds": metrics["image_seconds"],
        "depth_seconds": metrics["depth_seconds"],
        "packaging_seconds": metrics["encoding_seconds"],
        "model_load_seconds": payload["model_load_seconds"],
        "startup_seconds": payload["startup_seconds"],
        "cache_setup_seconds": payload["cache_setup_seconds"],
        "warm_state": payload["warm_state"],
        "sequence_bucket": metrics["sequence_bucket"],
        "token_count": metrics["token_count"],
        "steps": 4,
        "guidance_scale": 1.0,
        "estimated_gpu_usd": elapsed * 0.000222,
        "negative_prompt_supported": False,
    }
    manifest = {
        "schema_version": "1.0",
        "provider": "modal-klein-candidate",
        "scene_id": request.scene_id,
        "request": asdict(request),
        "stages": {"fast": stage},
        "identity": identity,
        "policy": {
            "reservation_id": reservation,
            "reserved_compute_usd": CALL_CEILING_USD,
            "visual_acceptance": "human-review-required",
            "automatic_retries": 0,
        },
        "artifacts": {role: {**asdict(a), "path": a.path.name} for role, a in artifacts.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for role, artifact in artifacts.items():
        _atomic_write(artifact.path, payload[role])
    path = output_dir / "scene.manifest.json"
    _atomic_write_json(path, manifest)
    return FiniteSceneBundle(path, request.scene_id, artifacts, manifest)
