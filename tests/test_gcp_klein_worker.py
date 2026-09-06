from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def setup_worker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("deploy").resolve()))
    spec = importlib.util.spec_from_file_location(
        "gcp_klein_test", "deploy/gcp_klein_worker/app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    from klein_scene_runtime import PROFILE

    sources = {}
    for key, source in {
        "worker": Path("deploy/gcp_klein_worker/app.py"),
        "runtime": Path("deploy/klein_scene_runtime.py"),
        "weights": Path("deploy/gcp_klein_worker/klein_weights.py"),
    }.items():
        data = source.read_bytes()
        (tmp_path / source.name).write_bytes(data)
        sources[key] = hashlib.sha256(data).hexdigest()
    identity = (
        PROFILE
        | {key: module.PACKAGES[key] for key in ("torch", "diffusers", "transformers", "triton")}
        | {
            "cuda": "12.8",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell",
            "capability": [12, 0],
            "runtime_sha256": sources["runtime"],
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "authorized",
        "experiment_id": "synthetic-test",
        "expires_at": int(time.time()) + 600,
        "max_requests": 2,
        "expected_identity": identity,
        "cases": [{"case_id": "boat", "seed": 9, "prompt": "A watercolor boat."}],
        "sources": sources,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    calls = []

    class Runtime:
        def __init__(self, expected):
            calls.append("load")
            self.identity = expected

        def compile(self, cache):
            assert cache is None
            calls.append("compile")
            return 0.1

        def render(self, prompt, seed):
            assert (prompt, seed) == ("A watercolor boat.", 9)
            calls.append("render")
            return (
                {
                    "seed": seed,
                    "sequence_bucket": 128,
                    "master_sha256": hashlib.sha256(b"master").hexdigest(),
                    "depth_sha256": hashlib.sha256(b"depth").hexdigest(),
                },
                b"master",
                b"depth",
            )

    return module, path, manifest, Runtime, calls


def test_fixed_admission_identity_and_cold_warm_receipts(setup_worker, monkeypatch):
    module, path, manifest, runtime, calls = setup_worker
    monkeypatch.setenv("K_SERVICE", "klein-test")
    monkeypatch.setenv("K_REVISION", "klein-test-00001")
    app = module.create_app(path, runtime)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/health")).json()["loaded"] is False
            for raw in (
                '{"case_id":"boat","seed":true}',
                '{"case_id":"boat","seed":9,"prompt":"private"}',
                '{"case_id":"boat","seed":9,"seed":9}',
                '{"case_id":"boat","seed":8}',
                '"' + "x" * 2048 + '"',
            ):
                reply = await client.post("/generate", content=raw)
                assert reply.status_code == 400 and reply.json() == {"error": "rejected"}
            assert calls == []
            for ordinal in (1, 2):
                reply = (await client.post("/generate", json={"case_id": "boat", "seed": 9})).json()
                assert reply["cold"] is (ordinal == 1)
                assert reply["bucket_was_warm"] is (ordinal == 2)
                assert reply["request_ordinal"] == ordinal
                assert reply["identity"] == manifest["expected_identity"]
                assert reply["service"] == "klein-test" and reply["revision"] == "klein-test-00001"
                assert reply["compile_seconds"] == 0.1
            assert (
                await client.post("/generate", json={"case_id": "boat", "seed": 9})
            ).status_code == 409
            assert calls == ["load", "compile", "render", "render"]
            manifest["expires_at"] = int(time.time()) - 1
            path.write_text(json.dumps(manifest))
            assert (
                await client.post("/generate", json={"case_id": "boat", "seed": 9})
            ).status_code == 400

    asyncio.run(exercise())
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: module.PACKAGES[name])
    fake = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "NVIDIA L4"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(ValueError, match="qualification rejected"):
        module.load_runtime(manifest["expected_identity"])
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "wrong")
    with pytest.raises(ValueError, match="qualification rejected"):
        module.load_runtime(manifest["expected_identity"])


def test_cancellation_retains_inference_lock_and_failed_attempt(setup_worker):
    module, path, manifest, runtime, calls = setup_worker
    entered, release = threading.Event(), threading.Event()

    class Blocking(runtime):
        def render(self, prompt, seed):
            entered.set()
            assert release.wait(3)
            return super().render(prompt, seed)

    app = module.create_app(path, Blocking)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.post("/generate", json={"case_id": "boat", "seed": 9})
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert (
                    await client.post("/generate", json={"case_id": "boat", "seed": 9})
                ).status_code == 429
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (
                await client.post("/generate", json={"case_id": "boat", "seed": 9})
            ).status_code == 409
            assert app.state.worker.requests == 1 and app.state.worker.failed
            assert calls == ["load", "compile", "render"]

    asyncio.run(exercise())
