"""Portable speech-recognition backend contracts.

This module intentionally imports no accelerator or platform-specific packages.  A
Jetson implementation can live in a separate module and satisfy :class:`AsrBackend`
without making development and contract tests depend on JetPack.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from storylight.domain import TranscriptionResponse


class AsrBackendError(RuntimeError):
    """Base error raised by an ASR backend."""


class AsrBackendUnavailableError(AsrBackendError):
    """Raised when transcription was intentionally disabled or is unavailable."""


@runtime_checkable
class AsrBackend(Protocol):
    """Cross-platform contract implemented by speech-recognition backends."""

    @property
    def name(self) -> str:
        """Stable backend identifier used in diagnostics."""
        ...

    @property
    def available(self) -> bool:
        """Whether this configured backend can currently accept audio."""
        ...

    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        """Transcribe one encoded audio clip or raise :class:`AsrBackendError`."""
        ...


@dataclass(frozen=True, slots=True)
class DisabledAsrBackend:
    """Explicitly unavailable backend for privacy modes and incomplete installs."""

    reason: str = "Speech recognition is disabled"
    name: str = "disabled"
    available: bool = False

    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        del audio, content_type
        raise AsrBackendUnavailableError(self.reason)


@dataclass(frozen=True, slots=True)
class TestAsrBackend:
    """Deterministic backend for API and platform-independent contract tests."""

    transcript: str = "The moon gate opened."
    language: str = "en"
    name: str = "test"
    available: bool = True

    async def transcribe(self, audio: bytes, content_type: str) -> TranscriptionResponse:
        if not audio:
            raise AsrBackendError("The recording was empty")
        if not content_type.strip().lower().startswith("audio/"):
            raise AsrBackendError("The content type must be audio/*")

        started = perf_counter()
        return TranscriptionResponse(
            text=self.transcript,
            language=self.language,
            model=self.name,
            total_ms=round((perf_counter() - started) * 1000, 2),
            audio_bytes=len(audio),
        )


def create_portable_asr_backend(name: str) -> AsrBackend:
    """Create a backend that has no Jetson-only import or runtime dependency."""

    normalized = name.strip().lower()
    if normalized == "disabled":
        return DisabledAsrBackend()
    if normalized == "test":
        return TestAsrBackend()
    raise ValueError(f"Unknown portable ASR backend: {name!r}")
