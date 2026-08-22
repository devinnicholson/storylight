import asyncio
import tempfile
from pathlib import Path
from time import perf_counter

from bookforge.config import Settings
from bookforge.domain import TranscriptionResponse


class TranscriptionError(RuntimeError):
    """Raised when local speech recognition cannot process an audio clip."""


class LocalTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings

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
        text, language = await asyncio.to_thread(self._transcribe_sync, audio, suffix)
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

        with tempfile.TemporaryDirectory(prefix="bookforge-asr-") as directory:
            audio_path = Path(directory) / f"recording{suffix}"
            audio_path.write_bytes(audio)
            try:
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=str(model),
                    language="en",
                    verbose=None,
                    condition_on_previous_text=False,
                )
            except Exception as error:
                raise TranscriptionError(f"Local transcription failed: {error}") from error

        return str(result.get("text", "")).strip(), str(result.get("language", "en"))


def _audio_suffix(content_type: str) -> str:
    normalized = content_type.lower()
    if "mp4" in normalized or "m4a" in normalized:
        return ".m4a"
    if "ogg" in normalized:
        return ".ogg"
    if "wav" in normalized:
        return ".wav"
    return ".webm"
