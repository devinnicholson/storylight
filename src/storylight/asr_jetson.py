from __future__ import annotations

import asyncio
import importlib
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from storylight.asr_backend import AsrBackendError, AsrBackendUnavailableError
from storylight.domain import TranscriptionResponse


class WhisperTrtModel(Protocol):
    def transcribe(self, audio: str) -> dict[str, Any]: ...


ModelLoader = Callable[..., WhisperTrtModel]


class WhisperTrtBackend:
    """Lazy NVIDIA WhisperTRT adapter for Jetson Orin devices."""

    name = "whisper_trt"

    def __init__(
        self,
        *,
        model_name: str = "base.en",
        engine_path: str = "",
        maximum_audio_mb: int = 20,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.model_name = model_name
        self.engine_path = engine_path
        self.maximum_audio_bytes = maximum_audio_mb * 1024 * 1024
        self._model_loader = model_loader
        self._model: WhisperTrtModel | None = None
        self._lock = asyncio.Lock()
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._model_loader is not None:
            return True
        if self._available is None:
            try:
                module = importlib.import_module("whisper_trt")
                self._available = callable(getattr(module, "load_trt_model", None))
            except Exception:
                self._available = False
        return self._available

    def _load_model(self) -> WhisperTrtModel:
        loader = self._model_loader
        if loader is None:
            try:
                from whisper_trt import load_trt_model
            except ImportError as error:
                raise AsrBackendUnavailableError(
                    "whisper_trt is not installed for this Jetson runtime"
                ) from error
            loader = load_trt_model

        if self.engine_path:
            engine = Path(self.engine_path)
            engine.parent.mkdir(parents=True, exist_ok=True)
            return loader(self.model_name, path=str(engine))
        return loader(self.model_name)

    def _transcribe_sync(self, audio_path: str) -> dict[str, Any]:
        if self._model is None:
            self._model = self._load_model()
        return self._model.transcribe(audio_path)

    async def warmup(self) -> None:
        if not self.available:
            raise AsrBackendUnavailableError(
                "whisper_trt is unavailable; install and validate it on the Jetson first"
            )
        async with self._lock:
            if self._model is not None:
                return
            worker = asyncio.create_task(asyncio.to_thread(self._load_model))
            try:
                self._model = await asyncio.shield(worker)
            except asyncio.CancelledError:
                completed = await _finish_worker_after_cancellation(worker)
                if completed is not None:
                    self._model = completed
                raise
            except Exception as error:
                raise AsrBackendError(f"WhisperTRT engine build failed: {error}") from error

    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        if not self.available:
            raise AsrBackendUnavailableError(
                "whisper_trt is unavailable; install and validate it on the Jetson first"
            )
        if not audio:
            raise AsrBackendError("The recording was empty")
        if not content_type.strip().lower().startswith("audio/"):
            raise AsrBackendError("The content type must be audio/*")
        if len(audio) > self.maximum_audio_bytes:
            maximum_mb = self.maximum_audio_bytes // (1024 * 1024)
            raise AsrBackendError(f"Recording exceeds the {maximum_mb} MB limit")

        suffix = _audio_suffix(content_type)
        started = perf_counter()
        async with self._lock:
            with tempfile.TemporaryDirectory(prefix="storylight-whisper-trt-") as directory:
                audio_path = Path(directory) / f"recording{suffix}"
                audio_path.write_bytes(audio)
                worker = asyncio.create_task(
                    asyncio.to_thread(self._transcribe_sync, str(audio_path))
                )
                try:
                    result = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    await _finish_worker_after_cancellation(worker)
                    raise
                except Exception as error:
                    raise AsrBackendError(f"WhisperTRT transcription failed: {error}") from error

        text = str(result.get("text", "")).strip()
        if not text:
            raise AsrBackendError("No speech was detected in the recording")
        return TranscriptionResponse(
            text=text,
            language=str(result.get("language", "en")),
            model=f"whisper_trt:{self.model_name}",
            total_ms=round((perf_counter() - started) * 1000, 2),
            audio_bytes=len(audio),
        )


async def _finish_worker_after_cancellation(worker: asyncio.Task[Any]) -> Any | None:
    """Keep ownership of a thread worker until it stops, despite repeated cancellation."""

    while not worker.done():
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            return None
    with suppress(Exception, asyncio.CancelledError):
        return worker.result()
    return None


def _audio_suffix(content_type: str) -> str:
    normalized = content_type.lower()
    if "mp4" in normalized or "m4a" in normalized:
        return ".m4a"
    if "ogg" in normalized:
        return ".ogg"
    if "wav" in normalized:
        return ".wav"
    return ".webm"
