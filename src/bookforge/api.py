import asyncio
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bookforge import __version__
from bookforge.asr import TranscriptionError, build_asr_backend
from bookforge.asr_backend import AsrBackend, AsrBackendError, AsrBackendUnavailableError
from bookforge.asset_cache import AssetCache, AssetCacheError
from bookforge.config import get_settings
from bookforge.domain import (
    CompileResponse,
    InterventionRequest,
    InterventionResponse,
    ModelProbe,
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
from bookforge.model_client import ModelUnavailableError, build_model_client
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
from bookforge.service import BookforgeService
from bookforge.story_store import StoryPackCorruptError, StoryPackNotFoundError, StoryPackStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = build_model_client(settings)
    app.state.settings = settings
    app.state.service = BookforgeService(settings, client)
    app.state.transcriber = build_asr_backend(settings)
    app.state.reader_events = ReaderEventHub()
    app.state.reader_sessions = ReaderSessionRegistry()
    app.state.reader_pipeline_lock = asyncio.Lock()
    app.state.story_store = StoryPackStore(settings.data_dir / "story-packs")
    await app.state.story_store.initialize()
    settings.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.cache_dir.chmod(0o700)
    app.state.asset_cache = AssetCache(settings.cache_dir / "assets")
    await app.state.asset_cache.initialize()
    yield
    await app.state.reader_events.close()
    http_client = getattr(client, "client", None)
    if http_client is not None:
        await http_client.aclose()


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
    allow_methods=["GET", "POST", "PUT"],
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
    return FileResponse(static_directory / "projector.html")


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
        return FileResponse(cache.resolve(checksum, filename))
    except AssetCacheError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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
