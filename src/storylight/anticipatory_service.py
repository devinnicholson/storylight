"""HTTP surface for the bounded GKE anticipatory scene coordinator."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from storylight.anticipatory import (
    AnticipationCapacityError,
    AnticipationConflictError,
    AnticipationError,
    AnticipationNotFoundError,
    AnticipationStatus,
    AnticipatoryBatchRequest,
    AnticipatorySceneOrchestrator,
    BranchId,
    CommitRequest,
)
from storylight.anticipatory_gcp import AssetNotFoundError, MemorySceneAssetStore

AssetId = Annotated[str, StringConstraints(pattern=r"^asset_[a-f0-9]{64}$")]
Probe = Callable[[], Awaitable[tuple[bool, str]]]


class SingleFlightPrewarm:
    """Coalesce concurrent paid warmups and reuse a recent result."""

    def __init__(
        self,
        operation: Probe,
        *,
        cooldown_seconds: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(cooldown_seconds) or not 1 <= cooldown_seconds <= 900:
            raise ValueError("prewarm cooldown must be 1-900 seconds")
        self.operation = operation
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._lock = asyncio.Lock()
        self._completed_at = float("-inf")
        self._result: tuple[bool, str] | None = None

    async def __call__(self) -> tuple[bool, str]:
        async with self._lock:
            if (
                self._result is not None
                and self.clock() - self._completed_at < self.cooldown_seconds
            ):
                return self._result[0], f"{self._result[1]} (recent result reused)"
            result = await self.operation()
            self._completed_at = self.clock()
            self._result = result
            return result


@dataclass(frozen=True, slots=True)
class AnticipatoryRuntime:
    orchestrator: AnticipatorySceneOrchestrator
    asset_store: MemorySceneAssetStore
    renderer_probe: Probe
    critic_probe: Probe
    runtime_prewarm: Probe | None = None


RuntimeFactory = Callable[[], AnticipatoryRuntime]


class RendererProbeAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization: Annotated[
        str,
        StringConstraints(pattern=r"^I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU$"),
    ]


def create_anticipatory_service(runtime_factory: RuntimeFactory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory()
        app.state.anticipatory_runtime = runtime
        yield
        await runtime.orchestrator.close()

    app = FastAPI(
        title="Storylight Anticipatory Story Engine",
        version="1.0.0",
        description=(
            "Privacy-bounded speculative rendering and Nemotron promotion control. "
            "This service accepts scene direction, never passages or reader media."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "ready": True,
            "service": "storylight-anticipatory",
            "privacy_boundary": "sanitized_scene_spec_v1",
        }

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        critic = await runtime.critic_probe()
        is_ready = critic[0]
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "ready": is_ready,
                "critic": {"ready": critic[0], "detail": critic[1]},
                "renderer": {
                    "ready": None,
                    "detail": "not probed by readiness because a probe can wake a billable GPU",
                },
            },
        )

    @app.post("/v1/renderer:probe")
    async def probe_renderer(
        _payload: RendererProbeAuthorization,
        request: Request,
    ) -> dict[str, str | bool]:
        renderer = await _runtime(request).renderer_probe()
        return {"ready": renderer[0], "detail": renderer[1]}

    @app.post("/v1/runtime:prewarm")
    async def prewarm_runtime(
        _payload: RendererProbeAuthorization,
        request: Request,
    ) -> dict[str, str | bool]:
        prewarm = _runtime(request).runtime_prewarm
        if prewarm is None:
            raise HTTPException(status_code=501, detail="runtime prewarm is not configured")
        ready = await prewarm()
        return {"ready": ready[0], "detail": ready[1]}

    @app.post(
        "/v1/anticipations",
        response_model=AnticipationStatus,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit(
        payload: AnticipatoryBatchRequest,
        request: Request,
    ) -> AnticipationStatus:
        try:
            return await _runtime(request).orchestrator.submit(payload)
        except AnticipationConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AnticipationCapacityError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except AnticipationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get(
        "/v1/anticipations/{session_token}/{sequence}",
        response_model=AnticipationStatus,
    )
    async def get_status(
        request: Request,
        session_token: str = Path(pattern=r"^anticipate_[a-f0-9]{24}$"),
        sequence: int = Path(ge=0, le=2**31 - 1),
        wait_seconds: float = Query(default=0, ge=0, le=20, allow_inf_nan=False),
        branch_id: BranchId = "known_next",
    ) -> AnticipationStatus:
        try:
            orchestrator = _runtime(request).orchestrator
            if wait_seconds:
                with suppress(TimeoutError):
                    await orchestrator.wait_terminal(
                        session_token,
                        sequence,
                        branch_id,
                        timeout_seconds=wait_seconds,
                    )
            return await orchestrator.status(session_token, sequence)
        except AnticipationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/v1/anticipations:commit", response_model=AnticipationStatus)
    async def commit(payload: CommitRequest, request: Request) -> AnticipationStatus:
        try:
            return await _runtime(request).orchestrator.commit(payload)
        except AnticipationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AnticipationConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.delete(
        "/v1/anticipations/{session_token}/{sequence}",
        response_model=AnticipationStatus,
    )
    async def cancel(
        request: Request,
        session_token: str = Path(pattern=r"^anticipate_[a-f0-9]{24}$"),
        sequence: int = Path(ge=0, le=2**31 - 1),
    ) -> AnticipationStatus:
        try:
            return await _runtime(request).orchestrator.cancel(session_token, sequence)
        except AnticipationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/v1/assets/{asset_id}")
    async def get_asset(
        request: Request,
        asset_id: AssetId,
    ) -> Response:
        try:
            asset = await _runtime(request).asset_store.get(
                asset_id,
                now=datetime.now(UTC),
            )
        except AssetNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=asset.content,
            media_type=asset.media_type,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-SHA256": asset.sha256,
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app


def _runtime(request: Request) -> AnticipatoryRuntime:
    return request.app.state.anticipatory_runtime
