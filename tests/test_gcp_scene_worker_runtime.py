from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "storylight_worker_test", Path("deploy/gcp_live_scene_worker/app.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_partial_model_initialization_is_not_ready(worker):
    runtime = worker.SceneRuntime()
    runtime.image_pipe = object()
    calls = []

    def finish_load():
        calls.append(True)
        runtime.depth_pipe = object()

    runtime._load = finish_load
    asyncio.run(runtime.ensure_loaded())
    assert calls == [True]
    worker.runtime = runtime
    assert asyncio.run(worker.health())["loaded"] is True
    runtime.depth_pipe = None
    assert asyncio.run(worker.health())["loaded"] is False


@pytest.mark.parametrize("strategy", ["cpu_then_cuda", "direct_cuda"])
def test_loader_publishes_both_models_atomically(worker, monkeypatch, strategy):
    options = []
    transfers = []
    image_pipe = SimpleNamespace(set_progress_bar_config=lambda **kw: None)
    image_pipe.to = lambda device: transfers.append(device) or image_pipe
    depth_pipe = object()
    fail_depth = True

    def load_image(*args, **kwargs):
        options.append(kwargs)
        return image_pipe

    def load_depth(**kwargs):
        if fail_depth:
            raise RuntimeError("depth load failed")
        return depth_pipe

    monkeypatch.setattr(worker, "MODEL_LOAD_STRATEGY", strategy)
    monkeypatch.setattr(worker, "EXPECTED_GPU", "L4")
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            SanaSprintPipeline=SimpleNamespace(from_pretrained=load_image),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_name=lambda _: "NVIDIA L4",
                get_device_capability=lambda _: (8, 9),
                get_arch_list=lambda: ["sm_89"],
            ),
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            bfloat16="bf16",
            float16="fp16",
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=load_depth))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: "/models/depth",
        ),
    )
    runtime = worker.SceneRuntime()
    with pytest.raises(RuntimeError, match="depth load failed"):
        runtime._load()
    assert runtime.image_pipe is None
    assert runtime.depth_pipe is None
    fail_depth = False
    runtime._load()
    assert runtime.image_pipe is image_pipe
    assert runtime.depth_pipe is depth_pipe
    assert options[-1]["torch_dtype"] == "bf16"
    assert options[-1]["local_files_only"] is True
    assert options[-1].get("device_map") == ("cuda" if strategy == "direct_cuda" else None)
    assert transfers == (["cuda", "cuda"] if strategy == "cpu_then_cuda" else [])
    assert sum(runtime.load_stages.values()) == pytest.approx(runtime.model_load_seconds)


@pytest.mark.parametrize("cancel_twice", [False, True])
def test_cancelled_load_keeps_lock_until_thread_finishes(worker, cancel_twice):
    runtime = worker.SceneRuntime()
    started = threading.Event()
    release = threading.Event()
    loads = []

    def load():
        loads.append(True)
        started.set()
        assert release.wait(timeout=3)
        runtime.image_pipe = runtime.depth_pipe = object()

    runtime._load = load

    async def scenario():
        first = asyncio.create_task(runtime.ensure_loaded())
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        await asyncio.sleep(0)
        if cancel_twice:
            first.cancel()
            await asyncio.sleep(0)
        second = asyncio.create_task(runtime.ensure_loaded())
        await asyncio.sleep(0.01)
        try:
            assert runtime._load_lock.locked()
            assert not second.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert loads == [True]

    asyncio.run(scenario())


def test_cancelled_inference_cannot_overlap_next_request(worker):
    started = threading.Event()
    release = threading.Event()
    lock = asyncio.Lock()
    sequence = []

    def work():
        started.set()
        assert release.wait(timeout=3)
        sequence.append("finished")

    async def scenario():
        async def run():
            async with lock:
                await worker._finish_thread_before_unlock(work)

        async def next_request():
            async with lock:
                sequence.append("next")

        first = asyncio.create_task(run())
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        second = asyncio.create_task(next_request())
        await asyncio.sleep(0.01)
        try:
            assert not second.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert sequence == ["finished", "next"]

    asyncio.run(scenario())
