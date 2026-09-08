import sys
from pathlib import Path
from types import SimpleNamespace

from bookforge.asr import LocalTranscriber
from bookforge.config import Settings


def test_local_asr_uses_native_no_speech_gate_without_rewriting_text(tmp_path, monkeypatch):
    calls = []

    def transcribe(path, **options):
        assert Path(path).read_bytes() == b"synthetic-audio"
        calls.append((Path(path), options))
        return {"text": " A cat chasing a mouse. You are beside it. ", "language": "en"}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    model = tmp_path / "model"
    model.mkdir()
    backend = LocalTranscriber(Settings(asr_backend="mlx_whisper", asr_model=str(model)))
    text, language = backend._transcribe_sync(b"synthetic-audio", ".webm")
    assert text == "A cat chasing a mouse. You are beside it."
    assert language == "en"
    assert len(calls) == 1
    path, options = calls[0]
    assert not path.exists()
    assert options == {
        "path_or_hf_repo": str(model),
        "language": "en",
        "verbose": None,
        "condition_on_previous_text": False,
        "logprob_threshold": None,
    }
