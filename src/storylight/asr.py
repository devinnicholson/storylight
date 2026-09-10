import asyncio
import tempfile
from contextlib import suppress
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

from storylight.asr_backend import AsrBackend, DisabledAsrBackend, TestAsrBackend
from storylight.config import Settings
from storylight.domain import TranscriptionResponse


class TranscriptionError(RuntimeError):
    """Raised when local speech recognition cannot process an audio clip."""


class LocalTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._transcribe_lock = asyncio.Lock()
        self.preparation_ms: float | None = None

    async def prepare(self) -> None:
        if not self.settings.asr_startup_audio or self.preparation_ms is not None:
            return
        path = Path(self.settings.asr_startup_audio)
        if not path.is_file():
            raise TranscriptionError("Startup audio must be a regular local file")
        if path.suffix.lower() not in {".webm", ".wav", ".ogg", ".mp4", ".m4a"}:
            raise TranscriptionError("Unsupported startup audio format")
        maximum = self.settings.asr_max_audio_mb * 1024 * 1024
        with path.open("rb") as source:
            audio = source.read(maximum + 1)
        result = await self.transcribe(audio, f"audio/{path.suffix[1:].lower()}")
        self.preparation_ms = result.total_ms

    @property
    def name(self) -> str:
        return "mlx_whisper"

    @property
    def available(self) -> bool:
        return (
            self.settings.asr_backend == "mlx_whisper"
            and find_spec("mlx_whisper") is not None
            and Path(self.settings.asr_model).exists()
        )

    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        if self.settings.asr_backend == "disabled":
            raise TranscriptionError("Local speech recognition is disabled")

        maximum = self.settings.asr_max_audio_mb * 1024 * 1024
        if not audio:
            raise TranscriptionError("The recording was empty")
        if len(audio) > maximum:
            raise TranscriptionError(
                f"Recording exceeds the {self.settings.asr_max_audio_mb} MB limit"
            )

        suffix = _audio_suffix(content_type)
        started = perf_counter()
        async with self._transcribe_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._transcribe_sync, audio, suffix))
            try:
                text, language = await asyncio.shield(worker)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await worker
                raise
        elapsed_ms = (perf_counter() - started) * 1000
        if not text:
            raise TranscriptionError("No speech was detected in the recording")

        return TranscriptionResponse(
            text=text,
            language=language,
            model=self.settings.asr_model,
            total_ms=round(elapsed_ms, 2),
            audio_bytes=len(audio),
        )

    def _transcribe_sync(self, audio: bytes, suffix: str) -> tuple[str, str]:
        try:
            import mlx_whisper
        except ImportError as error:
            raise TranscriptionError(
                "MLX Whisper is not installed; run `uv sync --extra mac-asr`"
            ) from error

        model = Path(self.settings.asr_model)
        if not model.exists():
            raise TranscriptionError(
                f"Local Whisper model not found at {model}; run `make asr-model-pull`"
            )

        with tempfile.TemporaryDirectory(prefix="storylight-asr-") as directory:
            audio_path = Path(directory) / f"recording{suffix}"
            audio_path.write_bytes(audio)
            try:
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=str(model),
                    language=(None if self.settings.asr_language == "auto"
                              else self.settings.asr_language),
                    verbose=None,
                    condition_on_previous_text=False,
                    # Do not let a confident decoded token override the model's
                    # no-speech decision on a silent recording window.
                    logprob_threshold=None,
                )
            except Exception as error:
                raise TranscriptionError(f"Local transcription failed: {error}") from error

        return str(result.get("text", "")).strip(), str(result.get("language", "en"))


def build_asr_backend(settings: Settings) -> AsrBackend:
    """Build the configured local ASR implementation behind one portable contract."""

    if settings.asr_backend == "mlx_whisper":
        return LocalTranscriber(settings)
    if settings.asr_backend == "whisper_trt":
        from storylight.asr_jetson import WhisperTrtBackend

        return WhisperTrtBackend(
            model_name=settings.asr_model,
            engine_path=settings.asr_engine_path,
            maximum_audio_mb=settings.asr_max_audio_mb,
        )
    if settings.asr_backend == "test":
        return TestAsrBackend()
    return DisabledAsrBackend()


def _audio_suffix(content_type: str) -> str:
    normalized = content_type.lower()
    if "mp4" in normalized or "m4a" in normalized:
        return ".m4a"
    if "ogg" in normalized:
        return ".ogg"
    if "wav" in normalized:
        return ".wav"
    return ".webm"
