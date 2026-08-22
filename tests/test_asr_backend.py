import asyncio

import pytest

from bookforge.asr import LocalTranscriber, build_asr_backend
from bookforge.asr_backend import (
    AsrBackend,
    AsrBackendError,
    AsrBackendUnavailableError,
    DisabledAsrBackend,
    create_portable_asr_backend,
)
from bookforge.asr_backend import TestAsrBackend as DeterministicAsrBackend
from bookforge.config import Settings


def test_disabled_backend_satisfies_contract_and_fails_closed() -> None:
    backend = DisabledAsrBackend(reason="No microphone consent")

    assert isinstance(backend, AsrBackend)
    assert backend.name == "disabled"
    assert backend.available is False
    with pytest.raises(AsrBackendUnavailableError, match="No microphone consent"):
        asyncio.run(backend.transcribe(b"audio", "audio/webm"))


def test_test_backend_returns_deterministic_response() -> None:
    backend = DeterministicAsrBackend(transcript="A red fox ran.", language="en-US")

    response = asyncio.run(backend.transcribe(b"recording", "audio/wav"))

    assert isinstance(backend, AsrBackend)
    assert backend.available is True
    assert response.text == "A red fox ran."
    assert response.language == "en-US"
    assert response.model == "test"
    assert response.audio_bytes == len(b"recording")
    assert response.total_ms >= 0


@pytest.mark.parametrize(
    ("audio", "content_type", "message"),
    [
        (b"", "audio/wav", "empty"),
        (b"recording", "application/octet-stream", "audio/"),
    ],
)
def test_test_backend_rejects_invalid_input(audio: bytes, content_type: str, message: str) -> None:
    backend = DeterministicAsrBackend()

    with pytest.raises(AsrBackendError, match=message):
        asyncio.run(backend.transcribe(audio, content_type))


def test_portable_factory_has_an_explicit_allowlist() -> None:
    assert isinstance(create_portable_asr_backend(" disabled "), DisabledAsrBackend)
    assert isinstance(create_portable_asr_backend("TEST"), DeterministicAsrBackend)

    with pytest.raises(ValueError, match="Unknown portable ASR backend"):
        create_portable_asr_backend("jetson-maybe")


def test_application_backend_factory_uses_portable_contract() -> None:
    disabled = build_asr_backend(Settings(asr_backend="disabled"))
    testing = build_asr_backend(Settings(asr_backend="test"))
    local = build_asr_backend(Settings(asr_backend="mlx_whisper"))

    assert isinstance(disabled, AsrBackend)
    assert isinstance(testing, AsrBackend)
    assert isinstance(local, LocalTranscriber)
    assert local.name == "mlx_whisper"
    assert isinstance(local.available, bool)
