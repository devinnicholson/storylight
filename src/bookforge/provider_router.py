from __future__ import annotations

import asyncio
import json
import math
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


def _bounded_detail(value: object) -> str:
    text = " ".join(str(value).split())
    return text[:300] if text else "unavailable"
