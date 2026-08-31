from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from bookforge.finite_modal_provider import (
    FastSceneRequest,
    FiniteModalProviderError,
    FiniteModalUnavailableError,
    FiniteSceneBundle,
    MotionUpgradeRequest,
    SceneArtifact,
    WarmProviderStatus,
)


class SafeProviderFallbackError(FiniteModalUnavailableError):
    """A provider rejected work before it could create a billable result."""


class RoutedFastSceneProvider(Protocol):
    async def probe(self) -> tuple[bool, str]: ...

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle: ...


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    name: str
    provider: RoutedFastSceneProvider

    def __post_init__(self) -> None:
        if (
            not self.name
            or len(self.name) > 64
            or any(not (character.isalnum() or character in "-_") for character in self.name)
        ):
            raise ValueError("provider route names require 1-64 safe characters")


class ResilientFastSceneProvider:
    """Fail over before billing, never after an ambiguous paid request.

    Every route receives a bounded readiness probe. A failed probe or an explicit
    ``SafeProviderFallbackError`` opens a short circuit and advances to the next
    route. Any other generation error is intentionally terminal: the upstream
    request may still have produced a billable result, so starting another paid
    provider would risk duplicate spend and conflicting artwork.
    """

    def __init__(
        self,
        routes: Sequence[ProviderRoute],
        *,
        probe_timeout_seconds: float = 2.0,
        failure_cooldown_seconds: float = 300.0,
        healthy_probe_ttl_seconds: float = 30.0,
    ) -> None:
        if not routes:
            raise ValueError("resilient scene routing requires at least one provider")
        if len({route.name for route in routes}) != len(routes):
            raise ValueError("resilient scene route names must be unique")
        for value, label, maximum in (
            (probe_timeout_seconds, "probe timeout", 60),
            (failure_cooldown_seconds, "failure cooldown", 3_600),
            (healthy_probe_ttl_seconds, "healthy probe TTL", 300),
        ):
            if not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{label} must be finite and between 0 and {maximum}")
        self.routes = tuple(routes)
        self.probe_timeout_seconds = probe_timeout_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.healthy_probe_ttl_seconds = healthy_probe_ttl_seconds
        self._unavailable_until: dict[str, float] = {}
        self._healthy_until: dict[str, float] = {}
        self._last_detail: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def probe(self) -> tuple[bool, str]:
        attempts: list[dict[str, Any]] = []
        for route in self.routes:
            ready, detail, _ = await self._probe_route(route, attempts=attempts)
            if ready:
                return True, f"resilient route selected {route.name}: {detail}"
        return False, self._unavailable_message(attempts)

    async def warm_status(self) -> WarmProviderStatus:
        """Expose route readiness without prewarming or starting paid work."""

        ready, detail = await self.probe()
        return WarmProviderStatus(
            ready=ready,
            detail=detail,
            state="idle",
        )

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        recovered = await asyncio.to_thread(
            _recover_exact_bundle,
            request,
            output_dir,
        )
        if recovered is not None:
            return recovered
        attempts: list[dict[str, Any]] = []
        for route in self.routes:
            ready, _, probe_ms = await self._probe_route(route, attempts=attempts)
            if not ready:
                continue
            try:
                bundle = await route.provider.generate_fast(request, output_dir=output_dir)
            except SafeProviderFallbackError as error:
                detail = _bounded_detail(error)
                attempts.append(
                    {
                        "provider": route.name,
                        "outcome": "safe_generation_rejection",
                        "probe_ms": probe_ms,
                        "detail": detail,
                    }
                )
                await self._mark_unavailable(route.name, detail)
                continue
            except FiniteModalUnavailableError as error:
                # The readiness gate passed and generation started. Older
                # providers classify transport failures as "unavailable", but
                # they are billably ambiguous at this point. Convert them to a
                # terminal provider error so the live API does not advertise an
                # unsafe automatic retry.
                raise FiniteModalProviderError(
                    f"{route.name} failed after its paid boundary; automatic fallback "
                    "and retry are suppressed"
                ) from error
            except BaseException:
                # Cancellation and ambiguous paid failures must never cascade to
                # another provider. The original request may still finish remotely.
                raise
            await self._mark_healthy(route.name)
            attempts.append(
                {
                    "provider": route.name,
                    "outcome": "selected",
                    "probe_ms": probe_ms,
                }
            )
            return await asyncio.to_thread(
                _record_routing_evidence,
                bundle,
                route.name,
                attempts,
            )
        raise FiniteModalUnavailableError(self._unavailable_message(attempts))

    async def upgrade_motion(
        self,
        bundle: FiniteSceneBundle | Path,
        request: MotionUpgradeRequest,
    ) -> FiniteSceneBundle:
        del bundle, request
        raise FiniteModalProviderError(
            "resilient cloud routing uses local WebGL depth motion, not a second paid video call"
        )

    async def is_prewarmed(self) -> bool:
        return False

    async def is_renderer_likely_warm(self) -> bool:
        now = time.monotonic()
        async with self._lock:
            return any(deadline > now for deadline in self._healthy_until.values())

    async def aclose(self) -> None:
        for route in self.routes:
            close = getattr(route.provider, "aclose", None)
            if callable(close):
                await close()

    async def _probe_route(
        self,
        route: ProviderRoute,
        *,
        attempts: list[dict[str, Any]],
    ) -> tuple[bool, str, float]:
        now = time.monotonic()
        async with self._lock:
            unavailable_until = self._unavailable_until.get(route.name, 0.0)
            healthy_until = self._healthy_until.get(route.name, 0.0)
            last_detail = self._last_detail.get(route.name, "")
        if unavailable_until > now:
            attempts.append(
                {
                    "provider": route.name,
                    "outcome": "circuit_open",
                    "retry_in_seconds": round(unavailable_until - now, 3),
                    "detail": last_detail,
                }
            )
            return False, last_detail, 0.0
        if healthy_until > now:
            detail = last_detail or "recent readiness probe passed"
            return True, detail, 0.0

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.probe_timeout_seconds):
                ready, detail = await route.provider.probe()
        except TimeoutError:
            ready = False
            detail = f"readiness probe exceeded {self.probe_timeout_seconds:g} seconds"
        except Exception as error:
            ready = False
            detail = f"readiness probe failed: {_bounded_detail(error)}"
        probe_ms = (time.perf_counter() - started) * 1_000
        detail = _bounded_detail(detail)
        if ready:
            await self._mark_healthy(route.name, detail=detail)
            return True, detail, probe_ms
        await self._mark_unavailable(route.name, detail)
        attempts.append(
            {
                "provider": route.name,
                "outcome": "probe_unavailable",
                "probe_ms": probe_ms,
                "detail": detail,
            }
        )
        return False, detail, probe_ms

    async def _mark_healthy(self, name: str, *, detail: str = "") -> None:
        async with self._lock:
            self._unavailable_until.pop(name, None)
            self._healthy_until[name] = time.monotonic() + self.healthy_probe_ttl_seconds
            if detail:
                self._last_detail[name] = detail

    async def _mark_unavailable(self, name: str, detail: str) -> None:
        async with self._lock:
            self._healthy_until.pop(name, None)
            self._unavailable_until[name] = time.monotonic() + self.failure_cooldown_seconds
            self._last_detail[name] = detail

    @staticmethod
    def _unavailable_message(attempts: Sequence[Mapping[str, Any]]) -> str:
        details = [
            f"{attempt.get('provider', 'unknown')}: {attempt.get('detail', attempt['outcome'])}"
            for attempt in attempts
        ]
        return "no scene renderer passed its safe readiness gate" + (
            "; " + "; ".join(details) if details else ""
        )


def _record_routing_evidence(
    bundle: FiniteSceneBundle,
    selected_route: str,
    attempts: Sequence[Mapping[str, Any]],
) -> FiniteSceneBundle:
    payload = dict(bundle.manifest)
    payload["routing"] = {
        "policy": "preflight-fallback-ambiguous-paid-fail-closed-v1",
        "selected_route": selected_route,
        "selected_provider": str(bundle.manifest.get("provider", selected_route)),
        "attempts": [dict(attempt) for attempt in attempts],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = bundle.manifest_path.with_suffix(f"{bundle.manifest_path.suffix}.routing.tmp")
    temporary.write_bytes(serialized)
    temporary.replace(bundle.manifest_path)
    return replace(bundle, manifest=payload)


def _recover_exact_bundle(
    request: FastSceneRequest,
    output_dir: Path,
) -> FiniteSceneBundle | None:
    """Reuse a completed paid bundle after local packaging failed.

    The lookup uses only privacy-safe request hashes and render parameters. It
    never treats a partial directory or a checksum-invalid artifact as usable.
    A recovered copy records zero new provider time and zero incremental cost.
    """

    destination = output_dir.resolve()
    root = destination.parent
    candidates: list[Path] = []
    direct_manifest = destination / "scene.manifest.json"
    if direct_manifest.is_file():
        candidates.append(direct_manifest)
    if root.is_dir():
        discovered = sorted(
            (
                path
                for path in root.glob("*/scene.manifest.json")
                if path != direct_manifest
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(discovered[:128])

    for manifest_path in candidates:
        try:
            payload = json.loads(manifest_path.read_text())
            if not _request_identity_matches(payload, request):
                continue
            source = _load_routed_bundle(manifest_path, payload=payload)
        except (OSError, ValueError, json.JSONDecodeError, FiniteModalProviderError):
            continue
        if manifest_path == direct_manifest:
            return source
        if destination.exists():
            raise FiniteModalProviderError(
                f"scene output already exists without an exact recoverable bundle: {destination}"
            )
        return _copy_recovered_bundle(source, request=request, destination=destination)
    if destination.exists():
        raise FiniteModalProviderError(
            f"scene output already exists without an exact recoverable bundle: {destination}"
        )
    return None


def _request_identity_matches(
    payload: Mapping[str, Any],
    request: FastSceneRequest,
) -> bool:
    identity = payload.get("request")
    if not isinstance(identity, Mapping):
        return False
    expected = {
        "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
        "seed": request.seed,
        "requested_width": request.width,
        "requested_height": request.height,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        return False
    optional = {
        "negative_prompt_sha256": hashlib.sha256(request.negative_prompt.encode()).hexdigest(),
        "steps": request.steps,
        "guidance_scale": request.guidance_scale,
    }
    return all(key not in identity or identity.get(key) == value for key, value in optional.items())


def _copy_recovered_bundle(
    source: FiniteSceneBundle,
    *,
    request: FastSceneRequest,
    destination: Path,
) -> FiniteSceneBundle:
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    payload = json.loads(json.dumps(source.manifest))
    for role in ("master", "depth"):
        artifact = source.artifacts[role]
        target = destination / artifact.path.name
        shutil.copyfile(artifact.path, target)
        payload["artifacts"][role]["path"] = target.name
    payload["scene_id"] = request.scene_id
    stage = payload["stages"]["fast"]
    original_estimate = float(stage.get("estimated_gpu_usd", 0.0))
    stage.update(
        {
            "recovered_without_provider_call": True,
            "original_estimated_gpu_usd": original_estimate,
            "estimated_gpu_usd": 0.0,
            "estimated_managed_service_usd": 0.0,
            "remote_seconds": 0.0,
            "inference_seconds": 0.0,
            "provider_overhead_seconds": 0.0,
            "image_seconds": 0.0,
            "depth_seconds": 0.0,
            "packaging_seconds": time.perf_counter() - started,
            "warm_state": "cache",
        }
    )
    routing = payload.setdefault("routing", {})
    routing["recovery"] = {
        "source_scene_id": source.scene_id,
        "provider_call": False,
        "policy": "exact-hashed-request-and-checksum-v1",
    }
    manifest_path = destination / "scene.manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return _load_routed_bundle(manifest_path, payload=payload)


def _load_routed_bundle(
    manifest_path: Path,
    *,
    payload: Mapping[str, Any] | None = None,
) -> FiniteSceneBundle:
    path = manifest_path.resolve()
    if payload is None:
        payload = json.loads(path.read_text())
    if payload.get("schema_version") != "1.0":
        raise FiniteModalProviderError("unsupported routed scene manifest schema")
    provider = payload.get("provider")
    if not isinstance(provider, str) or not provider:
        raise FiniteModalProviderError("routed scene manifest has no provider")
    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        scene_id,
    ):
        raise FiniteModalProviderError("routed scene manifest has an invalid scene_id")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or not {"master", "depth"}.issubset(
        raw_artifacts
    ):
        raise FiniteModalProviderError("routed scene manifest requires master and depth")
    artifacts: dict[str, SceneArtifact] = {}
    for role in ("master", "depth"):
        raw = raw_artifacts[role]
        if not isinstance(raw, Mapping):
            raise FiniteModalProviderError(f"invalid routed {role} artifact")
        relative = raw.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise FiniteModalProviderError(f"routed {role} artifact has an invalid path")
        artifact_path = (path.parent / relative).resolve()
        if not artifact_path.is_relative_to(path.parent):
            raise FiniteModalProviderError(f"routed {role} artifact escapes its scene directory")
        content = artifact_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != raw.get("sha256"):
            raise FiniteModalProviderError(f"routed {role} checksum mismatch")
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if not 1 <= width <= 4_096 or not 1 <= height <= 4_096:
            raise FiniteModalProviderError(f"routed {role} dimensions are invalid")
        artifacts[role] = SceneArtifact(
            role=role,
            path=artifact_path,
            sha256=digest,
            mime_type=str(raw.get("mime_type", "")),
            width=width,
            height=height,
            duration_ms=int(raw.get("duration_ms", 0)),
            frames=int(raw.get("frames", 1)),
            fps=int(raw.get("fps", 0)),
        )
    stages = payload.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get("fast"), Mapping):
        raise FiniteModalProviderError("routed scene manifest requires fast-stage provenance")
    return FiniteSceneBundle(
        manifest_path=path,
        scene_id=scene_id,
        artifacts=artifacts,
        manifest=dict(payload),
    )


def _bounded_detail(value: object) -> str:
    text = " ".join(str(value).split())
    return text[:300] if text else "unavailable"
