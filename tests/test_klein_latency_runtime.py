from __future__ import annotations

import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from klein_latency_runtime import LatencySceneRuntime  # noqa: E402
from klein_scene_runtime import KleinSceneRuntime  # noqa: E402


@pytest.fixture
def fake_runtime(monkeypatch):
    class Tokenizer:
        forced_count = None

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return messages[0]["content"]

        def __call__(self, text):
            count = self.forced_count or (200 if len(text) > 100 else 40)
            return {"input_ids": list(range(count))}

    class Pipeline:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.calls = []
            self.fail_next = False

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("synthetic pipeline failure")
            color = (kwargs["generator"] % 256, 80, kwargs["max_sequence_length"] % 256)
            return SimpleNamespace(
                images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), color)]
            )

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    cuda = SimpleNamespace(
        Event=Event,
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        max_memory_allocated=lambda: 1024,
        max_memory_reserved=lambda: 2048,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            inference_mode=nullcontext,
            Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed),
        ),
    )

    def initialize(self, model_root):
        self.identity = {"runtime_sha256": "historical-runtime", "model": "pinned-model"}
        self.pipe = Pipeline()
        self.depth_pipe = lambda master: {"depth": master.convert("L")}

    monkeypatch.setattr(KleinSceneRuntime, "__init__", initialize)
    return LatencySceneRuntime(Path("unused")), cuda


def test_instrumentation_preserves_pipeline_arguments_bytes_and_cache_identity(
    fake_runtime, capsys
):
    instrumented, _ = fake_runtime
    original = KleinSceneRuntime(Path("unused"))
    prompt = "A synthetic boat beside a dock."
    old, old_master, old_depth = original.render(prompt, 47)
    new, new_master, new_depth = instrumented.render(prompt, 47)
    assert (new_master, new_depth) == (old_master, old_depth)
    assert instrumented.pipe.calls == original.pipe.calls
    assert instrumented.identity == original.identity
    assert LatencySceneRuntime.compile is KleinSceneRuntime.compile
    assert new["master_sha256"] == old["master_sha256"]
    assert new["depth_sha256"] == old["depth_sha256"]
    assert new["cuda_image_seconds"] == 0.002
    assert new["tokenization_seconds"] >= 0 and new["pipeline_seconds"] >= 0
    assert new["image_seconds"] >= new["tokenization_seconds"] + new["pipeline_seconds"]
    assert new["total_seconds"] >= sum(
        new[key] for key in ("image_seconds", "depth_seconds", "encoding_seconds")
    )
    assert (
        new["instrumentation_sha256"]
        == hashlib.sha256(Path("deploy/klein_latency_runtime.py").read_bytes()).hexdigest()
    )
    assert prompt not in repr(new) and capsys.readouterr().out == ""


def test_bucket_warmth_requires_success_for_that_bucket_and_cuda_events_are_optional(fake_runtime):
    runtime, cuda = fake_runtime
    del cuda.Event
    first, _, _ = runtime.render("short synthetic scene", 1)
    assert first["sequence_bucket"] == 128 and first["bucket_was_warm"] is False
    assert first["cuda_image_seconds"] is None
    runtime.pipe.fail_next = True
    with pytest.raises(RuntimeError):
        runtime.render("long synthetic scene " * 20, 1)
    second, _, _ = runtime.render("long synthetic scene " * 20, 1)
    assert second["sequence_bucket"] == 256 and second["bucket_was_warm"] is False
    assert runtime.render("another short scene", 2)[0]["bucket_was_warm"] is True
    assert runtime.render("another long scene " * 20, 2)[0]["bucket_was_warm"] is True
    calls_before = len(runtime.pipe.calls)
    runtime.pipe.tokenizer.forced_count = 257
    with pytest.raises(ValueError, match="not been qualified"):
        runtime.render("unqualified synthetic scene", 3)
    assert len(runtime.pipe.calls) == calls_before
    assert 512 not in runtime._completed_buckets


def test_warmup_measures_both_fixed_buckets_without_exporting_images(fake_runtime):
    runtime, _ = fake_runtime
    report = runtime.warmup()
    assert [row["sequence_bucket"] for row in report["renders"]] == [128, 256]
    assert [row["bucket_was_warm"] for row in report["renders"]] == [False, False]
    assert [row["seed"] for row in report["renders"]] == [0, 0]
    assert report["warmup_seconds"] >= sum(row["total_seconds"] for row in report["renders"])
    assert len(runtime.pipe.calls) == 2 and "images" not in report and "prompt" not in report
    runtime.pipe.tokenizer.forced_count = 40
    with pytest.raises(ValueError, match="qualified token bucket"):
        runtime.warmup()
