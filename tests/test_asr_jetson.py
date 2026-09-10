import asyncio
import threading
from pathlib import Path

import pytest

from storylight.asr_backend import AsrBackendError
from storylight.asr_jetson import WhisperTrtBackend


class FakeWhisperTrtModel:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe(self, audio: str) -> dict[str, str]:
        path = Path(audio)
        assert path.read_bytes() == b"encoded-audio"
        self.paths.append(path)
        return {"text": "The small moth.", "language": "en"}


def test_whisper_trt_backend_loads_once_and_transcribes_serially(tmp_path: Path) -> None:
    model = FakeWhisperTrtModel()
    loads: list[tuple[str, str]] = []

    def load_model(name: str, *, path: str) -> FakeWhisperTrtModel:
        loads.append((name, path))
        return model

    backend = WhisperTrtBackend(
        model_name="base.en",
        engine_path=str(tmp_path / "engines" / "base-en.pth"),
        model_loader=load_model,
    )

    async def transcribe_twice():
        return await asyncio.gather(
            backend.transcribe(b"encoded-audio", "audio/webm"),
            backend.transcribe(b"encoded-audio", "audio/webm"),
        )

    first, second = asyncio.run(transcribe_twice())

    assert loads == [("base.en", str(tmp_path / "engines" / "base-en.pth"))]
    assert first.text == second.text == "The small moth."
    assert first.model == "whisper_trt:base.en"
    assert len(model.paths) == 2


def test_whisper_trt_backend_wraps_engine_build_failure() -> None:
    def fail_loader(_: str) -> FakeWhisperTrtModel:
        raise RuntimeError("engine build failed")

    backend = WhisperTrtBackend(model_loader=fail_loader)

    with pytest.raises(AsrBackendError, match="engine build failed"):
        asyncio.run(backend.transcribe(b"encoded-audio", "audio/webm"))


def test_cancelled_engine_build_holds_serialization_lock() -> None:
    started = threading.Event()
    release = threading.Event()
    load_count = 0

    def blocking_loader(_: str) -> FakeWhisperTrtModel:
        nonlocal load_count
        load_count += 1
        started.set()
        release.wait(timeout=2)
        return FakeWhisperTrtModel()

    backend = WhisperTrtBackend(model_loader=blocking_loader)

    async def exercise() -> None:
        first = asyncio.create_task(backend.transcribe(b"encoded-audio", "audio/webm"))
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        second = asyncio.create_task(backend.transcribe(b"encoded-audio", "audio/webm"))
        await asyncio.sleep(0.02)
        assert load_count == 1
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1].text == "The small moth."
        assert load_count == 1

    asyncio.run(exercise())


def test_repeated_cancellation_keeps_warmup_serialized() -> None:
    started = threading.Event()
    release = threading.Event()
    load_count = 0

    def blocking_loader(_: str) -> FakeWhisperTrtModel:
        nonlocal load_count
        load_count += 1
        started.set()
        release.wait(timeout=2)
        return FakeWhisperTrtModel()

    backend = WhisperTrtBackend(model_loader=blocking_loader)

    async def exercise() -> None:
        first = asyncio.create_task(backend.warmup())
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        await asyncio.sleep(0.01)
        first.cancel()
        second = asyncio.create_task(backend.warmup())
        await asyncio.sleep(0.02)
        assert load_count == 1
        assert backend._lock.locked()
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1] is None
        assert load_count == 1

    asyncio.run(exercise())
