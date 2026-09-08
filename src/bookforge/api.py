import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bookforge import __version__
from bookforge.anticipatory import AnticipationStatus, BranchId, CommitRequest, SessionToken
from bookforge.anticipatory_edge import (
    AnticipatoryEdgeClient,
    AnticipatoryEdgeCoordinator,
    AnticipatoryEdgeError,
    LocalAnticipationPrepareRequest,
    LocalAnticipationPrepareResponse,
)
from bookforge.anticipatory_playback import (
    ActivateProjectionRequest,
    AnticipatoryPlayback,
    PreparedProjectionId,
    PreparedProjectionStatus,
    PrepareProjectionRequest,
    PrewarmProjectionRequest,
)
from bookforge.asr import TranscriptionError, build_asr_backend
from bookforge.asr_backend import AsrBackend, AsrBackendError, AsrBackendUnavailableError
from bookforge.asset_cache import AssetCache, AssetCacheError
from bookforge.asset_generator import (
    AssetGenerationError,
    AssetGeneratorUnavailableError,
    build_asset_generator,
    build_depth_estimator,
)
from bookforge.config import Settings, get_settings
from bookforge.domain import (
    CompileResponse,
    InterventionRequest,
    InterventionResponse,
    ModelProbe,
    SceneBuildResponse,
    StoryCompileRequest,
    StoryPack,
    TranscriptionResponse,
)
from bookforge.event_hub import (
    EventHubClosedError,
    PublishResult,
    ReaderEventHub,
    ReaderEventPublishRequest,
    SessionId,
)
from bookforge.gcp_scene_provider import google_identity_token
from bookforge.live_scene import (
    DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL,
    LiveSceneArtifactKind,
    LiveSceneCapacityError,
    LiveSceneConflictError,
    LiveSceneCreateRequest,
    LiveSceneJob,
    LiveSceneJobId,
    LiveSceneJobRegistry,
    LiveSceneNotFoundError,
    LiveScenePlannerPrepareRequest,
    LiveScenePlannerPrepareResponse,
    LiveScenePlannerWarmupResponse,
    LiveScenePrewarmRequest,
    LiveScenePrewarmResponse,
    LiveSceneRegistryClosedError,
    LiveSceneSessionEvent,
    LiveSceneSessionStatus,
    LiveSceneWarmProviderStatus,
    ProjectorTelemetry,
    build_live_scene_provider,
    live_scene_request_seed,
)
from bookforge.live_scene_planner import (
    CONCISE_RENDER_CONTRACT_REVISION,
    LIVE_SCENE_RENDER_CONTRACT_REVISION,
)
from bookforge.model_client import (
    ModelUnavailableError,
    StructuredModelClient,
    build_model_client,
)
from bookforge.nemotron_critic import (
    NemotronCriticEvidence,
    NemotronCriticRequest,
    NemotronCriticUnavailableError,
    NemotronVisionCritic,
)
from bookforge.reader_runtime import (
    ReaderSessionConfigureRequest,
    ReaderSessionGenerationMismatchError,
    ReaderSessionNotConfiguredError,
    ReaderSessionPageMismatchError,
    ReaderSessionRegistry,
    ReaderSessionStatus,
    TranscriptUpdateRequest,
    TranscriptUpdateResult,
)
from bookforge.runtime_status import RuntimeComponent, RuntimeStatus
from bookforge.scene_foundry import SceneFoundry, SceneFoundryError
from bookforge.service import BookforgeService
from bookforge.story_store import StoryPackCorruptError, StoryPackNotFoundError, StoryPackStore
from bookforge.tensorrt_slot_client import TensorRTSlotModelClient


def _completed_pack_matches_planner_mode(
    pack: StoryPack,
    *,
    planner_mode: str,
    render_revision: str = LIVE_SCENE_RENDER_CONTRACT_REVISION,
) -> bool:
    """Prevent an old generic fallback from shadowing a real model-planned retry."""

    compiler = pack.compiler_model.casefold()
    if planner_mode == "model":
        return (
            pack.compiler_contract_revision == render_revision
            and not any(marker in compiler for marker in ("deterministic", "fallback", "fixture"))
        )
    if compiler.startswith("deterministic-live-scene-planner-"):
        return compiler == DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL
    return True


async def _warm_live_scene_planner_at_startup(registry: LiveSceneJobRegistry) -> None:
    """Load the private planner before the first reader request reaches it.

    The task deliberately runs in the background so model loading does not hold
    API readiness hostage. ``StructuredLiveScenePlanner`` coalesces this call
    with browser warmups and with an early generation request, so at most one
    local model-load request can occupy Ollama's single inference slot.
    """

    planner = getattr(registry.provider, "planner", None)
    warmup = getattr(planner, "warmup", None)
    if not callable(warmup):
        return
    try:
        await warmup()
    except Exception:
        # Generation still fails closed before a paid render if the local
        # planner is unavailable. A best-effort latency optimization must not
        # prevent the appliance from exposing diagnostics and repair controls.
        return


def _build_live_scene_planner_client(
    settings: Settings,
    *,
    fallback: StructuredModelClient,
) -> StructuredModelClient:
    if settings.live_scene_planner_backend == "configured":
        return fallback
    if settings.live_scene_planner_compact_wire:
        raise ValueError("TensorRT slot planning requires the standard wire contract")
    return TensorRTSlotModelClient(
        base_url=settings.live_scene_planner_base_url,
        model=settings.live_scene_planner_model_name,
        timeout_seconds=settings.live_scene_planner_timeout_seconds,
        max_output_tokens=settings.live_scene_planner_max_output_tokens,
        fallback=fallback,
        fallback_ready_seconds=settings.live_scene_planner_fallback_ready_seconds,
        defer_fallback_sanitization=True,
        protocol=(
            "hybrid"
            if settings.live_scene_planner_backend in {"tensorrt_hybrid", "tensorrt_graph"}
            else "slots"
        ),
        scene_facts_enabled=settings.live_scene_planner_backend
        in {"tensorrt_graph", "tensorrt_accepted_graph"},
        planning_scope=settings.live_scene_planner_scope,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = build_model_client(settings)
    planner_client = _build_live_scene_planner_client(settings, fallback=client)
    app.state.settings = settings
    app.state.service = BookforgeService(settings, client)
    app.state.transcriber = build_asr_backend(settings)
    app.state.projector_telemetry = {}
    app.state.reader_events = ReaderEventHub()
    app.state.reader_sessions = ReaderSessionRegistry()
    app.state.reader_pipeline_lock = asyncio.Lock()
    app.state.story_store = StoryPackStore(settings.data_dir / "story-packs")
    await app.state.story_store.initialize()
    settings.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.cache_dir.chmod(0o700)
    app.state.asset_cache = AssetCache(settings.cache_dir / "assets")
    await app.state.asset_cache.initialize()
    app.state.live_scene_critic = None
    if settings.live_scene_critic_backend == "nemotron":
        if not settings.live_scene_critic_url:
            raise ValueError("BOOKFORGE_LIVE_SCENE_CRITIC_URL is required for Nemotron")
        token_source = None
        if settings.live_scene_critic_audience:
            audience = settings.live_scene_critic_audience

            async def critic_token_source() -> str:
                return await google_identity_token(audience)

            token_source = critic_token_source
        app.state.live_scene_critic = NemotronVisionCritic(
            base_url=settings.live_scene_critic_url,
            model=settings.live_scene_critic_model,
            timeout_seconds=settings.live_scene_critic_timeout_seconds,
            api_key=settings.live_scene_critic_api_key,
            token_source=token_source,
        )

    async def find_completed_live_scene(payload: LiveSceneCreateRequest) -> StoryPack | None:
        from bookforge.bounded_description import REVISION as reviewed_revision

        pack = await app.state.story_store.find_live_scene(
            text=payload.text,
            visual_style=payload.visual_style,
            seed=live_scene_request_seed(payload),
            session_id=payload.session_id,
            planning_scope=settings.live_scene_planner_scope,
        )
        if pack is not None and (
            pack.compiler_model.startswith("bounded-description-") != payload.reviewed_description
            or (payload.reviewed_description and pack.compiler_model != reviewed_revision)
        ):
            return None
        if pack is not None and not _completed_pack_matches_planner_mode(
            pack,
            planner_mode=settings.live_scene_planner,
            render_revision=(
                CONCISE_RENDER_CONTRACT_REVISION
                if settings.live_scene_backend == "modal_klein"
                else LIVE_SCENE_RENDER_CONTRACT_REVISION
            ),
        ):
            return None
        return pack

    async def validate_completed_live_scene(pack: StoryPack) -> StoryPack:
        return await app.state.asset_cache.install_pack(pack, Path("."))

    app.state.scene_foundry = SceneFoundry(
        generator=build_asset_generator(settings),
        depth_estimator=build_depth_estimator(settings),
        cache=app.state.asset_cache,
        width=settings.asset_width,
        height=settings.asset_height,
    )
    live_scene_output_dir = _jetson_writable_runtime_path(
        settings.environment,
        settings.data_dir,
        settings.live_scene_output_dir,
    )
    modal_ledger_path = _jetson_writable_runtime_path(
        settings.environment,
        settings.data_dir,
        settings.live_scene_modal_ledger_path,
    )
    app.state.live_scenes = LiveSceneJobRegistry(
        build_live_scene_provider(
            settings.live_scene_backend,
            asset_backend=settings.asset_backend,
            cache=app.state.asset_cache,
            output_root=live_scene_output_dir,
            enable_motion=settings.live_scene_enable_motion,
            enable_preview=settings.live_scene_enable_preview,
            modal_session_gpu_cap_usd=settings.live_scene_modal_session_gpu_cap_usd,
            modal_executable=settings.asset_modal_command,
            modal_plan_file=settings.live_scene_modal_plan_file,
            modal_ledger_path=modal_ledger_path,
            gcp_url=settings.live_scene_gcp_url,
            gcp_audience=settings.live_scene_gcp_audience,
            gcp_impersonate_service_account=(settings.live_scene_gcp_impersonate_service_account),
            gcp_gpu=settings.live_scene_gcp_gpu,
            gcp_timeout_seconds=settings.live_scene_gcp_timeout_seconds,
            gcp_session_gpu_cap_usd=settings.live_scene_gcp_session_gpu_cap_usd,
            vertex_project_id=settings.live_scene_vertex_project_id,
            vertex_location=settings.live_scene_vertex_location,
            vertex_model=settings.live_scene_vertex_model,
            vertex_timeout_seconds=settings.live_scene_vertex_timeout_seconds,
            vertex_session_cost_cap_usd=settings.live_scene_vertex_session_cost_cap_usd,
            vertex_estimated_image_usd=settings.live_scene_vertex_estimated_image_usd,
            routing_probe_timeout_seconds=(settings.live_scene_routing_probe_timeout_seconds),
            routing_failure_cooldown_seconds=(settings.live_scene_routing_failure_cooldown_seconds),
            planner_mode=settings.live_scene_planner,
            model_client=planner_client,
            planner_timeout_seconds=settings.live_scene_planner_timeout_seconds,
            planner_model_revision=settings.live_scene_planner_model_revision,
            planner_cache_entries=settings.live_scene_planner_cache_entries,
            planner_cache_dir=settings.cache_dir / "live-scene-plans",
            planner_compact_wire=settings.live_scene_planner_compact_wire,
            master_width=settings.live_scene_master_width,
            master_height=settings.live_scene_master_height,
            master_steps=settings.live_scene_master_steps,
            master_guidance_scale=settings.live_scene_master_guidance_scale,
            fidelity_mode=settings.live_scene_fidelity_mode,
            auto_prewarm_on_submit=settings.live_scene_auto_prewarm_on_submit,
        ),
        max_active_jobs=settings.live_scene_max_active_jobs,
        max_retained_jobs=settings.live_scene_max_retained_jobs,
        event_queue_size=settings.live_scene_event_queue_size,
        completed_pack_sink=app.state.story_store.save,
        completed_pack_source=find_completed_live_scene,
        completed_pack_validator=validate_completed_live_scene,
    )
    app.state.live_scene_planner_warmup_task = None
    if settings.live_scene_planner_auto_warmup:
        app.state.live_scene_planner_warmup_task = asyncio.create_task(
            _warm_live_scene_planner_at_startup(app.state.live_scenes),
            name="bookforge-live-planner-startup-warmup",
        )
    app.state.anticipatory = None
    app.state.anticipatory_playback = None
    if settings.anticipatory_backend == "gke":
        if not settings.anticipatory_url:
            raise ValueError("BOOKFORGE_ANTICIPATORY_URL is required for the GKE backend")
        planner = getattr(app.state.live_scenes.provider, "planner", None)
        if planner is None or not callable(getattr(planner, "plan", None)):
            raise ValueError(
                "GKE anticipation requires BOOKFORGE_LIVE_SCENE_PLANNER=model"
            )
        token_source = None
        if settings.anticipatory_audience:
            audience = settings.anticipatory_audience

            async def anticipation_token_source() -> str:
                return await google_identity_token(audience)

            token_source = anticipation_token_source
        app.state.anticipatory = AnticipatoryEdgeCoordinator(
            planner=planner,
            client=AnticipatoryEdgeClient(
                base_url=settings.anticipatory_url,
                timeout_seconds=settings.anticipatory_timeout_seconds,
                token_source=token_source,
                allow_loopback_http=settings.anticipatory_allow_loopback_http,
            ),
            edge_gate_revision=settings.anticipatory_edge_gate_revision,
        )
        app.state.anticipatory_playback = AnticipatoryPlayback(
            edge=app.state.anticipatory,
            cache=app.state.asset_cache,
            registry=app.state.live_scenes,
        )
    yield
    planner_warmup_task = app.state.live_scene_planner_warmup_task
    if planner_warmup_task is not None:
        if not planner_warmup_task.done():
            planner_warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await planner_warmup_task
    await app.state.live_scenes.close()
    if app.state.anticipatory is not None:
        await app.state.anticipatory.aclose()
    await app.state.reader_events.close()
    closed_http_clients: set[int] = set()
    for model_client in (planner_client, client):
        http_client = getattr(model_client, "client", None)
        if http_client is not None and id(http_client) not in closed_http_clients:
            await http_client.aclose()
            closed_http_clients.add(id(http_client))


def _jetson_writable_runtime_path(
    environment: str,
    data_dir: Path,
    configured_path: Path,
) -> Path:
    if environment == "jetson" and not configured_path.is_absolute():
        return data_dir / configured_path
    return configured_path


app = FastAPI(
    title="Bookforge API",
    version=__version__,
    description="Shared API for private edge intervention and cloud Story Pack compilation.",
    lifespan=lifespan,
)

settings = get_settings()
static_directory = Path(__file__).parent / "static"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/workbench-assets", StaticFiles(directory=static_directory), name="workbench-assets")


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(_: Request, error: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(TranscriptionError)
async def transcription_error_handler(_: Request, error: TranscriptionError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(AsrBackendUnavailableError)
async def asr_unavailable_handler(_: Request, error: AsrBackendUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(AsrBackendError)
async def asr_error_handler(_: Request, error: AsrBackendError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(SceneFoundryError)
async def scene_foundry_error_handler(_: Request, error: SceneFoundryError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(AssetGeneratorUnavailableError)
async def asset_generator_unavailable_handler(_: Request, error: AssetGeneratorUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(AssetGenerationError)
async def asset_generation_error_handler(_: Request, error: AssetGenerationError):
    return JSONResponse(status_code=502, content={"detail": str(error)})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    store: StoryPackStore = request.app.state.story_store
    code = 200 if store.ready else 503
    return JSONResponse(status_code=code, content={"ready": store.ready})


@app.get("/v1/runtime:status", response_model=RuntimeStatus)
async def runtime_status(request: Request) -> RuntimeStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Runtime status is local-only")
    service: BookforgeService = request.app.state.service
    transcriber: AsrBackend = request.app.state.transcriber
    store: StoryPackStore = request.app.state.story_store
    model_ready, model_detail = await service.model_client.probe()
    return RuntimeStatus(
        ready=store.ready
        and model_ready
        and (transcriber.available or service.settings.asr_backend == "disabled"),
        environment=service.settings.environment,
        version=__version__,
        loopback_only=True,
        model=RuntimeComponent(
            ready=model_ready,
            name=f"{service.settings.model_backend}:{service.settings.model_name}",
            detail=model_detail,
        ),
        asr=RuntimeComponent(
            ready=transcriber.available,
            name=transcriber.name,
            detail="available" if transcriber.available else "disabled or unavailable",
        ),
        storage=RuntimeComponent(
            ready=store.ready,
            name="story-pack-store",
            detail=str(store.root),
        ),
        data_dir=str(service.settings.data_dir),
        cache_dir=str(service.settings.cache_dir),
    )


@app.get("/workbench", include_in_schema=False)
async def workbench() -> FileResponse:
    return FileResponse(static_directory / "workbench.html")


@app.get("/projector", include_in_schema=False)
async def projector() -> FileResponse:
    return FileResponse(
        static_directory / "projector.html",
        headers={
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data:; media-src 'self'; "
                "connect-src 'self' ws: wss:; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'"
            )
        },
    )


@app.get("/v1/models:probe", response_model=ModelProbe)
async def probe_model(request: Request) -> ModelProbe:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Model status is local-only")
    service: BookforgeService = request.app.state.service
    ready, detail = await service.model_client.probe()
    return ModelProbe(
        ready=ready,
        backend=service.settings.model_backend,
        model=service.settings.model_name,
        detail=detail,
    )


@app.post("/v1/audio:transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: Request) -> TranscriptionResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Audio transcription is local-only")
    transcriber: AsrBackend = request.app.state.transcriber
    content_type = request.headers.get("content-type", "audio/webm")
    if not content_type.lower().startswith("audio/"):
        raise HTTPException(status_code=415, detail="Expected an audio content type")
    maximum_bytes = request.app.state.settings.asr_max_audio_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
            if declared_length < 0:
                raise ValueError
            if declared_length > maximum_bytes:
                raise HTTPException(status_code=413, detail="Recording exceeds the audio limit")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from error
    audio = bytearray()
    async for chunk in request.stream():
        if len(audio) + len(chunk) > maximum_bytes:
            raise HTTPException(status_code=413, detail="Recording exceeds the audio limit")
        audio.extend(chunk)
    return await transcriber.transcribe(bytes(audio), content_type)


@app.post("/v1/interventions:select", response_model=InterventionResponse)
async def select_intervention(
    payload: InterventionRequest,
    request: Request,
) -> InterventionResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Interventions are local-only")
    service: BookforgeService = request.app.state.service
    return await service.select_intervention(payload)


@app.post("/v1/story-packs:compile", response_model=CompileResponse)
async def compile_story(payload: StoryCompileRequest, request: Request) -> CompileResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Story compilation is local-only")
    service: BookforgeService = request.app.state.service
    store: StoryPackStore = request.app.state.story_store
    try:
        response = await service.compile_story(payload)
        await store.save(response.story_pack)
        return response
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/story-packs:build", response_model=SceneBuildResponse)
async def build_story_pack(payload: StoryCompileRequest, request: Request) -> SceneBuildResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Story generation is local-only")
    service: BookforgeService = request.app.state.service
    foundry: SceneFoundry = request.app.state.scene_foundry
    store: StoryPackStore = request.app.state.story_store
    try:
        compiled = await service.compile_story(payload)
        story_pack, generation_metrics = await foundry.build(compiled.story_pack)
        await store.save(story_pack)
        return SceneBuildResponse(
            story_pack=story_pack,
            compile_metrics=compiled.metrics,
            generation_metrics=generation_metrics,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/story-packs/latest", response_model=StoryPack)
async def latest_story_pack(request: Request) -> StoryPack:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Stored Story Packs are local-only")
    store: StoryPackStore = request.app.state.story_store
    try:
        return await store.latest()
    except StoryPackNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except StoryPackCorruptError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/v1/assets/{checksum}/{filename}", response_class=FileResponse)
async def cached_asset(checksum: str, filename: str, request: Request) -> FileResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Cached assets are local-only")
    cache: AssetCache = request.app.state.asset_cache
    try:
        return FileResponse(
            cache.resolve(checksum, filename),
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{checksum}"',
            },
        )
    except AssetCacheError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/v1/projector-telemetry", response_model=ProjectorTelemetry)
async def record_projector_telemetry(
    payload: ProjectorTelemetry,
    request: Request,
) -> ProjectorTelemetry:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Projector telemetry is local-only")
    snapshot = payload.model_copy(update={"captured_at": datetime.now(UTC)})
    request.app.state.projector_telemetry[payload.session_id] = snapshot
    return snapshot


@app.get("/v1/projector-telemetry/{session_id}", response_model=ProjectorTelemetry)
async def latest_projector_telemetry(
    session_id: SessionId,
    request: Request,
) -> ProjectorTelemetry:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Projector telemetry is local-only")
    snapshot = request.app.state.projector_telemetry.get(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No projector telemetry is available")
    return snapshot


@app.post(
    "/v1/live-scenes",
    response_model=LiveSceneJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_live_scene(
    payload: LiveSceneCreateRequest,
    request: Request,
    response: Response,
) -> LiveSceneJob:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene generation is local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    try:
        job = await registry.submit(payload)
        if payload.session_id is not None:
            pointer = await registry.get_session(payload.session_id)
            response.headers["X-Bookforge-Server-Instance-Id"] = pointer.server_instance_id
            response.headers["X-Bookforge-Session-Revision"] = str(pointer.session_revision)
        return job
    except LiveSceneCapacityError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except LiveSceneRegistryClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _configured_warm_status_provider(registry: LiveSceneJobRegistry):
    adapter = registry.provider
    provider = getattr(adapter, "provider", None)
    if provider is None or not callable(getattr(provider, "warm_status", None)):
        raise HTTPException(
            status_code=409,
            detail="The configured live-scene provider does not expose readiness status",
        )
    return adapter, provider


def _configured_prewarm_scene_provider(registry: LiveSceneJobRegistry):
    adapter, provider = _configured_warm_status_provider(registry)
    if not callable(getattr(provider, "prewarm", None)):
        raise HTTPException(
            status_code=409,
            detail="The configured live-scene provider does not support explicit prewarming",
        )
    return adapter, provider


@app.get(
    "/v1/live-scene-provider/warm-status",
    response_model=LiveSceneWarmProviderStatus,
)
async def live_scene_warm_status(request: Request) -> LiveSceneWarmProviderStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene provider status is local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    _, provider = _configured_warm_status_provider(registry)
    try:
        report = await provider.warm_status()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return LiveSceneWarmProviderStatus.model_validate(asdict(report))


@app.post(
    "/v1/live-scene-provider/prewarm",
    response_model=LiveScenePrewarmResponse,
)
async def prewarm_live_scene_provider(
    payload: LiveScenePrewarmRequest,
    request: Request,
) -> LiveScenePrewarmResponse:
    """Explicitly prewarm bounded Modal classes; disabled by default in the UI."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene prewarm is local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    adapter, provider = _configured_prewarm_scene_provider(registry)
    if payload.include_motion and not adapter.enable_motion:
        raise HTTPException(
            status_code=409,
            detail="Enable BOOKFORGE_LIVE_SCENE_ENABLE_MOTION before prewarming motion",
        )
    try:
        report = await provider.prewarm(
            prewarm_id=payload.prewarm_id,
            include_motion=payload.include_motion,
            scaledown_window_seconds=payload.scaledown_window_seconds,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return LiveScenePrewarmResponse.model_validate(asdict(report))


@app.post(
    "/v1/live-scene-planner/prepare",
    response_model=LiveScenePlannerPrepareResponse,
)
async def prepare_live_scene_planner(
    payload: LiveScenePlannerPrepareRequest,
    request: Request,
) -> LiveScenePlannerPrepareResponse:
    """Prime the private local edge-plan cache without starting a paid render."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene planning is local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    planner = getattr(registry.provider, "planner", None)
    if not payload.reviewed_description and (
        planner is None or not callable(getattr(planner, "plan", None))
    ):
        raise HTTPException(
            status_code=409,
            detail="BOOKFORGE_LIVE_SCENE_PLANNER is not configured as model",
        )
    if payload.session_id is not None:
        await registry.set_session_planner_active(payload.session_id, True)
    try:
        if payload.reviewed_description:
            from bookforge.bounded_description import plan_bounded_description

            result = plan_bounded_description(payload.text, payload.visual_style, payload.seed)
        else:
            result = await planner.plan(
                text=payload.text,
                visual_style=payload.visual_style,
                seed=payload.seed,
            )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        if payload.session_id is not None:
            await registry.set_session_planner_active(payload.session_id, False)
    return LiveScenePlannerPrepareResponse(
        planning_ms=result.wall_ms,
        cache_hit=result.cache_hit,
        model=result.metrics.model,
        revision=result.model_revision,
        input_tokens=result.metrics.input_tokens,
        output_tokens=result.metrics.output_tokens,
    )


@app.post(
    "/v1/live-scene-planner/warmup",
    response_model=LiveScenePlannerWarmupResponse,
)
async def warmup_live_scene_planner(request: Request) -> LiveScenePlannerWarmupResponse:
    """Hide local Ollama model load using fixed synthetic input and no story text."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene planning is local-only")
    settings = request.app.state.settings
    if not settings.live_scene_planner_auto_warmup:
        return LiveScenePlannerWarmupResponse(ready=False)
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    planner = getattr(registry.provider, "planner", None)
    warmup = getattr(planner, "warmup", None)
    if not callable(warmup):
        return LiveScenePlannerWarmupResponse(ready=False)
    try:
        result = await warmup()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return LiveScenePlannerWarmupResponse(
        ready=True,
        warmup_ms=result.wall_ms,
        model=result.metrics.model,
        input_tokens=result.metrics.input_tokens,
        output_tokens=result.metrics.output_tokens,
    )


def _prepared_playback(request: Request) -> AnticipatoryPlayback:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Prepared projection controls are local-only")
    playback = request.app.state.anticipatory_playback
    if playback is None:
        raise HTTPException(status_code=409, detail="GKE next-page preparation is not configured")
    return playback


@app.get("/v1/prepared-projections/runtime")
async def prepared_projection_runtime(request: Request) -> dict:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Prepared projection controls are local-only")
    return {
        "enabled": request.app.state.anticipatory_playback is not None,
        "server_instance_id": request.app.state.live_scenes.server_instance_id,
        "detail": "No cloud request was made. Start the bounded GKE runtime before warming it.",
    }


@app.post("/v1/prepared-projections:prewarm")
async def prewarm_prepared_projection(payload: PrewarmProjectionRequest, request: Request) -> dict:
    playback = _prepared_playback(request)
    try:
        return await playback.edge.client.prewarm_runtime()
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/v1/prepared-projections", response_model=PreparedProjectionStatus, status_code=202)
async def prepare_projection(payload: PrepareProjectionRequest, request: Request):
    playback = _prepared_playback(request)
    try:
        return await playback.prepare(payload)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/v1/prepared-projections/{prepared_id}", response_model=PreparedProjectionStatus)
async def prepared_projection_status(
    prepared_id: PreparedProjectionId,
    request: Request,
    wait_seconds: float = Query(default=0, ge=0, le=20, allow_inf_nan=False),
):
    playback = _prepared_playback(request)
    try:
        return await playback.status(prepared_id, wait_seconds=wait_seconds)
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/prepared-projections/{prepared_id}/stage", response_model=PreparedProjectionStatus)
async def stage_prepared_projection(prepared_id: PreparedProjectionId, request: Request):
    playback = _prepared_playback(request)
    try:
        return await playback.stage(prepared_id)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/prepared-projections:activate", response_model=LiveSceneSessionStatus)
async def activate_prepared_projection(payload: ActivateProjectionRequest, request: Request):
    playback = _prepared_playback(request)
    try:
        return await playback.activate(payload)
    except (AnticipatoryEdgeError, LiveSceneConflictError, LiveSceneCapacityError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.delete("/v1/prepared-projections/{prepared_id}", status_code=204)
async def discard_prepared_projection(prepared_id: PreparedProjectionId, request: Request):
    playback = _prepared_playback(request)
    try:
        await playback.discard(prepared_id)
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _anticipatory_coordinator(request: Request) -> AnticipatoryEdgeCoordinator:
    coordinator = request.app.state.anticipatory
    if coordinator is None:
        raise HTTPException(
            status_code=409,
            detail="BOOKFORGE_ANTICIPATORY_BACKEND is disabled",
        )
    return coordinator


@app.post(
    "/v1/anticipations:prepare",
    response_model=LocalAnticipationPrepareResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def prepare_anticipation(
    payload: LocalAnticipationPrepareRequest,
    request: Request,
) -> LocalAnticipationPrepareResponse:
    """Plan private text on the edge and send only the sanitized scene contract."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Anticipatory planning is local-only")
    try:
        return await _anticipatory_coordinator(request).prepare(payload)
    except (AnticipatoryEdgeError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get(
    "/v1/anticipations/{session_token}/{sequence}",
    response_model=AnticipationStatus,
)
async def anticipation_status(
    request: Request,
    session_token: SessionToken,
    sequence: int,
) -> AnticipationStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Anticipatory status is local-only")
    try:
        return await _anticipatory_coordinator(request).status(session_token, sequence)
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post(
    "/v1/anticipations:commit",
    response_model=AnticipationStatus,
)
async def commit_anticipation(
    payload: CommitRequest,
    request: Request,
) -> AnticipationStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Anticipatory commit is local-only")
    try:
        return await _anticipatory_coordinator(request).commit(payload)
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.delete(
    "/v1/anticipations/{session_token}/{sequence}",
    response_model=AnticipationStatus,
)
async def cancel_anticipation(
    request: Request,
    session_token: SessionToken,
    sequence: int,
) -> AnticipationStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Anticipatory cancellation is local-only")
    try:
        return await _anticipatory_coordinator(request).cancel(session_token, sequence)
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/v1/anticipations/{session_token}/{sequence}/{branch_id}/{kind}")
async def anticipation_asset(
    request: Request,
    session_token: SessionToken,
    sequence: int,
    branch_id: BranchId,
    kind: Literal["master", "depth"],
) -> Response:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Anticipatory assets are local-only")
    try:
        content, media_type, digest = await _anticipatory_coordinator(request).asset(
            session_token=session_token,
            sequence=sequence,
            branch_id=branch_id,
            kind=kind,
        )
    except AnticipatoryEdgeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-SHA256": digest,
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/v1/live-scenes/{job_id}", response_model=LiveSceneJob)
async def live_scene_status(
    job_id: LiveSceneJobId,
    request: Request,
) -> LiveSceneJob:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene status is local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    try:
        return await registry.get(job_id)
    except LiveSceneNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/v1/live-scenes/{job_id}:critique",
    response_model=NemotronCriticEvidence,
)
async def critique_live_scene(
    job_id: LiveSceneJobId,
    request: Request,
) -> NemotronCriticEvidence:
    """Evaluate an already-visible synthetic plate without delaying first-image delivery."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene criticism is local-only")
    critic: NemotronVisionCritic | None = request.app.state.live_scene_critic
    if critic is None:
        raise HTTPException(
            status_code=409,
            detail="BOOKFORGE_LIVE_SCENE_CRITIC_BACKEND is disabled",
        )
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    cache: AssetCache = request.app.state.asset_cache
    try:
        job = await registry.get(job_id)
    except LiveSceneNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if not job.complete or job.story_pack is None:
        raise HTTPException(status_code=409, detail="Scene master is not complete")
    if job.metrics.planning_status.value != "model":
        raise HTTPException(
            status_code=409,
            detail="Nemotron requires a locally privacy-gated model scene plan",
        )
    master = next(
        (artifact for artifact in job.artifacts if artifact.kind is LiveSceneArtifactKind.MASTER),
        None,
    )
    if master is None:
        raise HTTPException(status_code=409, detail="Scene has no master artifact")
    parts = master.uri.split("/")
    if len(parts) != 5 or parts[1:3] != ["v1", "assets"]:
        raise HTTPException(status_code=500, detail="Scene master has an invalid cache URI")
    try:
        master_path = cache.resolve(parts[3], parts[4])
        image_bytes = await asyncio.to_thread(master_path.read_bytes)
    except (AssetCacheError, OSError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    page = job.story_pack.pages[0]
    if page.scene_spec is None:
        raise HTTPException(status_code=500, detail="Scene master has no visual specification")
    normalized_source = " ".join(job.request.text.casefold().split())
    normalized_brief = " ".join(page.scene_spec.master_prompt.casefold().split())
    if normalized_source and normalized_source in normalized_brief:
        raise HTTPException(
            status_code=409,
            detail="Nemotron visual brief failed the outbound privacy boundary",
        )
    expected_subjects = [layer.prompt[:300] for layer in page.layers if layer.kind != "background"][
        :8
    ]
    critic_request = NemotronCriticRequest(
        visual_brief=page.scene_spec.master_prompt,
        expected_subjects=expected_subjects,
        forbidden_content=["readable text", "duplicate principal subject", "interface chrome"],
    )
    media_type = "image/png" if master_path.suffix.lower() == ".png" else "image/jpeg"
    try:
        return await critic.evaluate(
            critic_request,
            image_bytes=image_bytes,
            media_type=media_type,
        )
    except (NemotronCriticUnavailableError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get(
    "/v1/live-scene-sessions/{session_id}",
    response_model=LiveSceneSessionStatus,
)
async def live_scene_session_status(
    session_id: SessionId,
    request: Request,
) -> LiveSceneSessionStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene sessions are local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    try:
        return await registry.get_session(session_id)
    except LiveSceneNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/v1/live-scene-sessions/{session_id}/events",
    response_class=StreamingResponse,
)
async def live_scene_session_events(
    session_id: SessionId,
    request: Request,
) -> StreamingResponse:
    """Stream the server epoch before any job, then every current-job revision."""

    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene session events are local-only")
    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    try:
        subscription = await registry.subscribe_session(session_id)
    except LiveSceneRegistryClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def stream():
        try:
            async with subscription:
                while True:
                    event: LiveSceneSessionEvent = await subscription.receive()
                    job_revision = event.job.revision if event.job is not None else 0
                    event_id = f"{event.server_instance_id}:{event.session_revision}:{job_revision}"
                    yield (
                        f"id: {event_id}\nevent: scene.session\ndata: {event.model_dump_json()}\n\n"
                    )
        except LiveSceneRegistryClosedError:
            return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/live-scenes/{job_id}/events", response_class=StreamingResponse)
async def live_scene_events(
    job_id: LiveSceneJobId,
    request: Request,
    after_revision: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Live-scene events are local-only")

    last_event_id = request.headers.get("last-event-id")
    if last_event_id and after_revision == 0:
        try:
            after_revision = int(last_event_id)
            if after_revision < 0:
                raise ValueError
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Last-Event-ID header") from error

    registry: LiveSceneJobRegistry = request.app.state.live_scenes
    try:
        subscription = await registry.subscribe(job_id, after_revision=after_revision)
    except LiveSceneNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LiveSceneRegistryClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def stream():
        try:
            async with subscription:
                while True:
                    snapshot = await subscription.receive()
                    yield (
                        f"id: {snapshot.revision}\n"
                        "event: scene.job\n"
                        f"data: {snapshot.model_dump_json()}\n\n"
                    )
                    if snapshot.terminal:
                        return
        except LiveSceneRegistryClosedError:
            return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


def _is_local_connection(request: Request | WebSocket) -> bool:
    if any(header in request.headers for header in ("forwarded", "x-forwarded-for", "x-real-ip")):
        return False
    client = request.client
    if client is None:
        return False
    if client.host == "testclient":
        return True
    try:
        return ip_address(client.host).is_loopback
    except ValueError:
        return client.host == "localhost"


@app.put(
    "/v1/reader-sessions/{session_id}",
    response_model=ReaderSessionStatus,
)
async def configure_reader_session(
    session_id: SessionId,
    payload: ReaderSessionConfigureRequest,
    request: Request,
) -> ReaderSessionStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Reader sessions are local-only")
    registry: ReaderSessionRegistry = request.app.state.reader_sessions
    async with request.app.state.reader_pipeline_lock:
        return await registry.configure(session_id, payload)


@app.get(
    "/v1/reader-sessions/{session_id}",
    response_model=ReaderSessionStatus,
)
async def reader_session_status(
    session_id: SessionId,
    request: Request,
) -> ReaderSessionStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Reader sessions are local-only")
    registry: ReaderSessionRegistry = request.app.state.reader_sessions
    try:
        return await registry.status(session_id)
    except ReaderSessionNotConfiguredError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/v1/reader-sessions/{session_id}:reset",
    response_model=ReaderSessionStatus,
)
async def reset_reader_session(
    session_id: SessionId,
    request: Request,
) -> ReaderSessionStatus:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Reader sessions are local-only")
    registry: ReaderSessionRegistry = request.app.state.reader_sessions
    hub: ReaderEventHub = request.app.state.reader_events
    async with request.app.state.reader_pipeline_lock:
        try:
            session = await registry.reset(session_id)
        except ReaderSessionNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await hub.publish(
            session_id,
            "session.reset",
            {
                "page_id": session.page_id,
                "page_text": session.page_text,
                "generation": session.generation,
            },
        )
    return session


@app.post(
    "/v1/reader-sessions/{session_id}/transcripts:simulate",
    response_model=TranscriptUpdateResult,
)
async def ingest_reader_transcript(
    session_id: SessionId,
    payload: TranscriptUpdateRequest,
    request: Request,
) -> TranscriptUpdateResult:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Reader sessions are local-only")
    registry: ReaderSessionRegistry = request.app.state.reader_sessions
    hub: ReaderEventHub = request.app.state.reader_events
    async with request.app.state.reader_pipeline_lock:
        try:
            result = await registry.ingest(session_id, payload)
        except ReaderSessionNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ReaderSessionPageMismatchError, ReaderSessionGenerationMismatchError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        await hub.publish(
            session_id,
            "transcript.partial",
            {
                "transcript": payload.text,
                "page_id": result.status.page_id,
                "generation": result.status.generation,
                "source": payload.source,
                "language": payload.language,
                "is_final": payload.is_final,
            },
        )
        for event in result.word_events:
            await hub.publish_word_reached(
                session_id,
                page_id=event.page_id,
                index=event.index,
                word=event.word,
                generation=result.status.generation,
            )
    return result


@app.post(
    "/v1/reader-sessions/{session_id}/events:publish",
    response_model=PublishResult,
)
async def publish_reader_event(
    session_id: SessionId,
    payload: ReaderEventPublishRequest,
    request: Request,
) -> PublishResult:
    if not _is_local_connection(request):
        raise HTTPException(status_code=403, detail="Reader session events are local-only")
    if request.app.state.settings.environment.lower() == "jetson":
        raise HTTPException(status_code=403, detail="Direct event publishing is disabled on Jetson")
    hub: ReaderEventHub = request.app.state.reader_events
    try:
        return await hub.publish_request(session_id, payload)
    except EventHubClosedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.websocket("/v1/reader-sessions/{session_id}/events")
async def reader_session_events(websocket: WebSocket, session_id: SessionId) -> None:
    if not _is_local_connection(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    hub: ReaderEventHub = websocket.app.state.reader_events
    try:
        subscription = await hub.subscribe(session_id)
    except EventHubClosedError:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    await websocket.accept()

    async def forward_events() -> None:
        try:
            while True:
                event = await subscription.receive()
                await websocket.send_json(event.model_dump(mode="json"))
        except (EventHubClosedError, WebSocketDisconnect, RuntimeError):
            return

    sender_task = asyncio.create_task(forward_events())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        await subscription.close()
        await asyncio.gather(sender_task, return_exceptions=True)
