import asyncio

import pytest

from storylight.asr import LocalTranscriber, build_asr_backend
from storylight.asr_backend import (
    AsrBackend,
    AsrBackendUnavailableError,
    DisabledAsrBackend,
)
from storylight.config import Settings


def test_disabled_backend_satisfies_contract_and_fails_closed() -> None:
    backend = DisabledAsrBackend(reason="No microphone consent")

    assert isinstance(backend, AsrBackend)
    assert backend.name == "disabled"
    assert backend.available is False
    with pytest.raises(AsrBackendUnavailableError, match="No microphone consent"):
        asyncio.run(backend.transcribe(b"audio", "audio/webm"))


def test_application_backend_factory_uses_portable_contract() -> None:
    disabled = build_asr_backend(Settings(asr_backend="disabled"))
    testing = build_asr_backend(Settings(asr_backend="test"))
    local = build_asr_backend(Settings(asr_backend="mlx_whisper"))

    assert isinstance(disabled, AsrBackend)
    assert isinstance(testing, AsrBackend)
    assert isinstance(local, LocalTranscriber)
    assert local.name == "mlx_whisper"
    assert isinstance(local.available, bool)
