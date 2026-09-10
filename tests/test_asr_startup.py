import asyncio
import threading

import pytest

from storylight.asr import LocalTranscriber, TranscriptionError
from storylight.config import Settings


def test_startup_runs_once_and_preserves_real_transcription(tmp_path, monkeypatch):
    async def scenario():
        primer = tmp_path / "primer.webm"
        primer.write_bytes(b"primer")
        backend = LocalTranscriber(Settings(asr_startup_audio=str(primer)))
        calls = []

        def decode(audio, suffix):
            calls.append((audio, suffix))
            return ("Synthetic primer." if audio == b"primer" else "A green cat.", "en")

        monkeypatch.setattr(backend, "_transcribe_sync", decode)
        await backend.prepare()
        await backend.prepare()
        assert backend.preparation_ms is not None
        assert (await backend.transcribe(b"real", "audio/webm")).text == "A green cat."
        assert calls == [(b"primer", ".webm"), (b"real", ".webm")]

    asyncio.run(scenario())


def test_failed_or_disabled_preparation_never_claims_success(tmp_path, monkeypatch):
    async def scenario():
        backend = LocalTranscriber(Settings())
        await backend.prepare()
        assert backend.preparation_ms is None
        primer = tmp_path / "primer.wav"
        primer.write_bytes(b"")
        backend = LocalTranscriber(Settings(asr_startup_audio=str(primer)))
        with pytest.raises(TranscriptionError, match="empty"):
            await backend.prepare()
        assert backend.preparation_ms is None
        primer.write_bytes(b"silence")
        monkeypatch.setattr(backend, "_transcribe_sync", lambda *_: ("", "en"))
        with pytest.raises(TranscriptionError, match="No speech"):
            await backend.prepare()
        assert backend.preparation_ms is None

    asyncio.run(scenario())


def test_cancelled_startup_joins_worker_before_releasing_lock(tmp_path, monkeypatch):
    async def scenario():
        primer = tmp_path / "primer.webm"
        primer.write_bytes(b"primer")
        backend = LocalTranscriber(Settings(asr_startup_audio=str(primer)))
        entered, release = threading.Event(), threading.Event()

        def decode(*_):
            entered.set()
            assert release.wait(5)
            return "Synthetic primer.", "en"

        monkeypatch.setattr(backend, "_transcribe_sync", decode)
        task = asyncio.create_task(backend.prepare())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert backend._transcribe_lock.locked()
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not backend._transcribe_lock.locked()
        assert backend.preparation_ms is None

    asyncio.run(scenario())


def test_api_prepares_before_creating_clients_and_rejects_wrong_backend(tmp_path, monkeypatch):
    from fastapi import FastAPI

    import storylight.api as api

    settings = Settings(asr_startup_audio=str(tmp_path / "primer.webm"))
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    events = []

    async def prepare(self):
        events.append("prepared")

    def client(_):
        assert events == ["prepared"]
        raise RuntimeError("end of startup probe")

    monkeypatch.setattr(LocalTranscriber, "prepare", prepare)
    monkeypatch.setattr(api, "build_model_client", client)

    async def scenario():
        with pytest.raises(RuntimeError, match="end of startup probe"):
            async with api.lifespan(FastAPI()):
                pytest.fail("Startup exposed an unfinished application")
        settings.asr_backend = "test"
        with pytest.raises(ValueError, match="requires mlx_whisper"):
            async with api.lifespan(FastAPI()):
                pytest.fail("Startup accepted incompatible backend")

    asyncio.run(scenario())


def test_fixture_bounds_fail_before_decoding(tmp_path, monkeypatch):
    def decode(*_):
        pytest.fail("Invalid startup fixture reached decoder")

    async def scenario():
        for path, content, message in (
            (tmp_path, None, "regular local file"),
            (tmp_path / "bad.txt", b"audio", "Unsupported"),
            (tmp_path / "large.wav", b"x" * (1024 * 1024 + 1), "exceeds"),
        ):
            if content is not None:
                path.write_bytes(content)
            backend = LocalTranscriber(Settings(asr_startup_audio=str(path), asr_max_audio_mb=1))
            monkeypatch.setattr(backend, "_transcribe_sync", decode)
            with pytest.raises(TranscriptionError, match=message):
                await backend.prepare()
            assert backend.preparation_ms is None

    asyncio.run(scenario())
