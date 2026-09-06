import hashlib
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from klein_reference_runtime import MAX_REFERENCE_BYTES, KleinReferenceRuntime  # noqa: E402
from klein_scene_runtime import KleinSceneRuntime  # noqa: E402
from test_klein_latency_runtime import fake_runtime as fake_runtime  # noqa: E402


def jpeg(*, size=(1024, 576), mode="RGB", format="JPEG"):
    buffer = io.BytesIO()
    Image.new(mode, size, 80).save(buffer, format=format)
    content = buffer.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def test_text_delegation_preserves_bytes_arguments_identity_and_compiler(fake_runtime, capsys):
    original = KleinSceneRuntime(Path("unused"))
    candidate = KleinReferenceRuntime(Path("unused"))
    old, old_master, old_depth = original.render("A synthetic boat beside a dock.", 47)
    new, master, depth = candidate.render("A synthetic boat beside a dock.", 47)
    assert (master, depth) == (old_master, old_depth)
    assert candidate.pipe.calls == original.pipe.calls
    assert candidate.identity == original.identity
    assert KleinReferenceRuntime.compile is KleinSceneRuntime.compile
    assert new["master_sha256"] == old["master_sha256"]
    assert new["reference_sha256"] is None and new["reference_conditioned"] is False
    assert new["reference_width"] is new["reference_height"] is None
    assert new["reference_runtime_sha256"] == hashlib.sha256(
        Path("deploy/klein_reference_runtime.py").read_bytes()
    ).hexdigest()
    assert capsys.readouterr().out == ""


def test_reference_uses_one_decoded_image_with_unchanged_sampling_and_source_free_metrics(
    fake_runtime, monkeypatch, capsys
):
    candidate = KleinReferenceRuntime(Path("unused"))
    original = KleinSceneRuntime(Path("unused"))
    inference = []

    @contextmanager
    def inference_mode():
        inference.append("enter")
        yield
        inference.append("exit")

    monkeypatch.setattr(sys.modules["torch"], "inference_mode", inference_mode)
    content, proof = jpeg()
    prompt = "A synthetic boat beside a dock."
    metrics, master, depth = candidate.render(
        prompt, 47, reference_jpeg=content, reference_sha256=proof
    )
    old, old_master, old_depth = original.render(prompt, 47)
    # Fake pipeline output depends on sampling arguments; the real model's pixels may change.
    assert (master, depth) == (old_master, old_depth)
    assert inference == ["enter", "exit", "enter", "exit"]
    call = candidate.pipe.calls[0].copy()
    reference = call.pop("image")
    with Image.open(io.BytesIO(content)) as expected:
        assert reference.tobytes() == expected.tobytes()
    assert reference.size == (1024, 576) and reference.mode == "RGB" and reference.info == {}
    assert call == original.pipe.calls[0]
    assert metrics["reference_conditioned"] is True and metrics["reference_sha256"] == proof
    assert (metrics["reference_width"], metrics["reference_height"]) == (1024, 576)
    assert metrics["sequence_bucket"] == old["sequence_bucket"] == 128
    assert hashlib.sha256(master).hexdigest() == metrics["master_sha256"]
    assert hashlib.sha256(depth).hexdigest() == metrics["depth_sha256"]
    assert prompt not in repr(metrics) and content not in metrics.values()
    assert capsys.readouterr().out == ""


def test_reference_validation_and_token_limit_refuse_before_gpu_or_pipeline(fake_runtime):
    _, cuda = fake_runtime
    candidate = KleinReferenceRuntime(Path("unused"))
    gpu_calls = []
    cuda.reset_peak_memory_stats = lambda: gpu_calls.append("reset")
    content, proof = jpeg()
    invalid = [
        (None, proof),
        ("/private/reference.jpg", proof),
        (content, "0" * 64),
        (b"x" * (MAX_REFERENCE_BYTES + 1), proof),
        jpeg(size=(512, 288)),
        jpeg(mode="L"),
        jpeg(format="PNG"),
        (b"not JPEG", hashlib.sha256(b"not JPEG").hexdigest()),
    ]
    for image, checksum in invalid:
        with pytest.raises(ValueError):
            candidate.render("synthetic scene", 1, reference_jpeg=image, reference_sha256=checksum)
    for prompt, seed in [("", 1), ("synthetic scene", -1), ("synthetic scene", 2**32)]:
        with pytest.raises(ValueError):
            candidate.render(prompt, seed, reference_jpeg=content, reference_sha256=proof)
    candidate.pipe.tokenizer.forced_count = 513
    with pytest.raises(ValueError, match="not truncating"):
        candidate.render("synthetic scene", 1, reference_jpeg=content, reference_sha256=proof)
    assert not gpu_calls and not candidate.pipe.calls
