from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bookforge import __version__
from bookforge.asr import LocalTranscriber, TranscriptionError
from bookforge.config import get_settings
from bookforge.domain import (
    CompileResponse,
    InterventionRequest,
    InterventionResponse,
    ModelProbe,
    StoryCompileRequest,
    TranscriptionResponse,
)
from bookforge.model_client import ModelUnavailableError, build_model_client
from bookforge.service import BookforgeService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = build_model_client(settings)
    app.state.settings = settings
    app.state.service = BookforgeService(settings, client)
    app.state.transcriber = LocalTranscriber(settings)
    yield
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
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/workbench-assets", StaticFiles(directory=static_directory), name="workbench-assets")


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(_: Request, error: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(TranscriptionError)
async def transcription_error_handler(_: Request, error: TranscriptionError):
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
    transcriber: LocalTranscriber = request.app.state.transcriber
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
