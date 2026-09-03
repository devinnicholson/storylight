"""Production assembly for the Bookforge GKE anticipatory service."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Literal, cast

from bookforge.anticipatory import AnticipatorySceneOrchestrator
from bookforge.anticipatory_gcp import (
    CloudRunAnticipatoryRenderer,
    MemorySceneAssetStore,
    StoredAssetNemotronCritic,
)
from bookforge.anticipatory_service import (
    AnticipatoryRuntime,
    SingleFlightPrewarm,
    create_anticipatory_service,
)
from bookforge.nemotron_critic import DEFAULT_NEMOTRON_VL_MODEL, NemotronVisionCritic


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _number(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be numeric") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def build_runtime() -> AnticipatoryRuntime:
    renderer_url = _required("BOOKFORGE_RENDERER_URL")
    audience = os.environ.get("BOOKFORGE_RENDERER_AUDIENCE", renderer_url).strip()
    expected_gpu = os.environ.get("BOOKFORGE_RENDERER_GPU", "RTX_PRO_6000").strip()
    if expected_gpu not in {"L4", "RTX_PRO_6000"}:
        raise RuntimeError("BOOKFORGE_RENDERER_GPU must be L4 or RTX_PRO_6000")

    def now() -> datetime:
        return datetime.now(UTC)

    store = MemorySceneAssetStore(
        max_bytes=_integer(
            "BOOKFORGE_ANTICIPATORY_ASSET_BYTES",
            64 * 1024 * 1024,
            minimum=16 * 1024 * 1024,
            maximum=1024 * 1024 * 1024,
        )
    )
    renderer = CloudRunAnticipatoryRenderer(
        base_url=renderer_url,
        audience=audience,
        asset_store=store,
        expected_gpu=cast(Literal["L4", "RTX_PRO_6000"], expected_gpu),
        timeout_seconds=_number(
            "BOOKFORGE_ANTICIPATORY_RENDER_TIMEOUT_SECONDS",
            30,
            minimum=1,
            maximum=120,
        ),
        now=now,
    )
    nim_base_url = os.environ.get(
        "BOOKFORGE_NEMOTRON_BASE_URL",
        "http://127.0.0.1:8000",
    ).strip()
    nim = NemotronVisionCritic(
        base_url=nim_base_url,
        model=os.environ.get("BOOKFORGE_NEMOTRON_MODEL", DEFAULT_NEMOTRON_VL_MODEL),
        timeout_seconds=_number(
            "BOOKFORGE_NEMOTRON_TIMEOUT_SECONDS",
            30,
            minimum=1,
            maximum=120,
        ),
        allow_loopback_http=True,
        allow_cluster_http=True,
    )
    critic = StoredAssetNemotronCritic(critic=nim, asset_store=store, now=now)

    async def prewarm() -> tuple[bool, str]:
        renderer_result, critic_result = await asyncio.gather(
            renderer.prewarm(),
            critic.prewarm(),
        )
        ready = renderer_result[0] and critic_result[0]
        return ready, f"{renderer_result[1]}; {critic_result[1]}"

    orchestrator = AnticipatorySceneOrchestrator(
        renderer=renderer,
        critic=critic,
        max_concurrent_renders=_integer(
            "BOOKFORGE_ANTICIPATORY_MAX_RENDERS",
            2,
            minimum=1,
            maximum=2,
        ),
        max_candidates=_integer(
            "BOOKFORGE_ANTICIPATORY_MAX_CANDIDATES",
            64,
            minimum=2,
            maximum=256,
        ),
        cache_entries=_integer(
            "BOOKFORGE_ANTICIPATORY_CACHE_ENTRIES",
            32,
            minimum=0,
            maximum=128,
        ),
        now=now,
    )
    return AnticipatoryRuntime(
        orchestrator=orchestrator,
        asset_store=store,
        renderer_probe=renderer.prewarm,
        critic_probe=critic.probe,
        runtime_prewarm=SingleFlightPrewarm(prewarm),
    )


app = create_anticipatory_service(build_runtime)
