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
from bookforge.config import get_settings
from bookforge.domain import (
    CompileResponse,
    InterventionRequest,
    InterventionResponse,
    ModelProbe,
    StoryCompileRequest,
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
    ReaderSessionNotConfiguredError,
    ReaderSessionPageMismatchError,
    ReaderSessionRegistry,
    ReaderSessionStatus,
    TranscriptUpdateRequest,
    TranscriptUpdateResult,
)
from bookforge.service import BookforgeService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = build_model_client(settings)
    app.state.settings = settings
    app.state.service = BookforgeService(settings, client)
    app.state.transcriber = build_asr_backend(settings)
    app.state.reader_events = ReaderEventHub()
    app.state.reader_sessions = ReaderSessionRegistry()
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


@app.get("/workbench", include_in_schema=False)
async def workbench() -> FileResponse:
    return FileResponse(static_directory / "workbench.html")


@app.get("/projector", include_in_schema=False)
async def projector() -> FileResponse:
    return FileResponse(static_directory / "projector.html")


@app.get("/v1/models:probe", response_model=ModelProbe)
async def probe_model(request: Request) -> ModelProbe:
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
    transcriber: AsrBackend = request.app.state.transcriber
    content_type = request.headers.get("content-type", "audio/webm")
    if not content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Expected an audio content type")
    return await transcriber.transcribe(await request.body(), content_type)


@app.post("/v1/interventions:select", response_model=InterventionResponse)
async def select_intervention(
    payload: InterventionRequest,
    request: Request,
) -> InterventionResponse:
    service: BookforgeService = request.app.state.service
    return await service.select_intervention(payload)


@app.post("/v1/story-packs:compile", response_model=CompileResponse)
async def compile_story(payload: StoryCompileRequest, request: Request) -> CompileResponse:
    service: BookforgeService = request.app.state.service
    try:
        return await service.compile_story(payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
    return await registry.configure(session_id, payload)


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
    try:
        result = await registry.ingest(session_id, payload)
    except ReaderSessionNotConfiguredError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ReaderSessionPageMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    await hub.publish(
        session_id,
        "transcript.partial",
        {
            "transcript": payload.text,
            "page_id": result.status.page_id,
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
