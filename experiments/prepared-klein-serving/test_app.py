"""Exercise the real session lifecycle with a fake, joinable render thread."""

import asyncio
import base64
import hashlib
import importlib.util
import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent


@pytest.fixture
def module():
    with patch.dict(sys.modules):
        for name in ("runtime_worker", "session_state", "session_worker"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("prepared_serving_test", HERE / "app.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        yield result


class FakeWorker:
    def __init__(self, path, factory):
        self.instance_id = "b" * 32
        self.lock = asyncio.Lock()
        self.runtime, self.failed, self.requests = None, False, 0
        self.calls = []
        self.started = self.release = None

    def identity(self):
        return {"instance_id": self.instance_id, "service": "serving-test", "revision": "trial"}

    def render(self, manifest, case):
        self.calls.append(case)
        self.runtime = object()
        if self.started:
            self.started.set()
            assert self.release.wait(5)
        metrics = {
            "sequence_bucket": 256 if case["prompt"] == "warm-256" else 128,
            "seed": case["seed"],
            "total_seconds": 0.1,
            "master_sha256": "c" * 64,
            "depth_sha256": "d" * 64,
        }
        return {
            **self.identity(),
            "identity": manifest["expected_identity"],
            "request_ordinal": self.requests,
            "case_id": case["case_id"],
            "metrics": metrics,
            "load_seconds": 1.5,
            "bucket_was_warm": self.requests > 2,
            "master_b64": "bWFzdGVy",
            "depth_b64": "ZGVwdGg=",
        }


def fixture(module, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "serving-test")
    clock = [10.0]
    manifest = {
        "experiment_id": "serving-test",
        "expires_at": 1000,
        "expected_identity": {"runtime": "pinned"},
        "cases": [
            {"case_id": str(i), "prompt": "warm-256" if i == 1 else "warm-128", "seed": i}
            for i in range(8)
        ],
    }
    app = module.create_app(
        worker_type=FakeWorker,
        manifest_reader=lambda _: (manifest, "manifest"),
        session_reader=lambda _: ({"lifetime_seconds": 300}, "session"),
        bucket_for=lambda _, prompt: 256 if prompt == "warm-256" else 128,
        clock=lambda: clock[0],
        wall_clock=lambda: clock[0],
    )
    payload = {
        "session_id": "a" * 32,
        "instance_id": "b" * 32,
        "request_id": "1" * 32,
        "scene_id": "scene-1",
        "prompt": "A watercolor fox.",
        "seed": 42,
    }
    return app, app.state.serving, clock, payload


def test_two_warmups_exact_replay_and_finite_capacity(module, monkeypatch):
    async def run():
        app, serving, _, payload = fixture(module, monkeypatch)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as c:
                ready = await c.post("/v1/prewarm", json={"session_id": "a" * 32})
                assert ready.status_code == 200
                lease = ready.json()["lease"]
                assert [r["sequence_bucket"] for r in lease["warmups"]] == [128, 256]
                assert lease["expires_at"] - lease["started_at"] == 300
                assert ready.json()["model_load_seconds"] == 1.5
                assert ready.json()["warmup_seconds"] == 0.2
                assert (
                    await c.post("/v1/prewarm", json={"session_id": "a" * 32})
                ).json() == ready.json()
                assert (
                    await c.post("/v1/prewarm", json={"session_id": "e" * 32})
                ).status_code == 409
                first = await c.post("/v1/generate", json=payload)
                assert first.status_code == 200 and first.json()["lease"] == lease
                assert (await c.post("/v1/generate", json=payload)).json() == first.json()
                assert len(serving.session.worker.calls) == 3
                for changed in ({"prompt": "A watercolor bird."}, {"instance_id": "e" * 32}):
                    assert (await c.post("/v1/generate", json=payload | changed)).status_code == 409
                for i in range(2, 11):
                    assert (
                        await c.post("/v1/generate", json=payload | {"request_id": f"{i:032x}"})
                    ).status_code == 200
                assert (
                    await c.post("/v1/generate", json=payload | {"request_id": "f" * 32})
                ).status_code == 409
                assert len(serving.claims) == 10 and len(serving.session.worker.calls) == 12
                assert (await c.post("/v1/generate", json=payload)).json() == first.json()
                assert (await c.post("/generate", json=payload)).status_code == 404
        finally:
            await serving.session.close()

    asyncio.run(run())


@pytest.mark.parametrize("disconnect", [False, True])
def test_cancel_joins_render_and_claim_cannot_rerender(module, monkeypatch, disconnect):
    async def run():
        _, serving, _, payload = fixture(module, monkeypatch)
        await serving.prewarm({"session_id": "a" * 32})
        worker = serving.session.worker
        worker.started, worker.release = threading.Event(), threading.Event()
        disconnected = asyncio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        operation = (
            module.owned_request(
                SimpleNamespace(receive=receive), lambda: serving.generate(payload)
            )
            if disconnect
            else serving.generate(payload)
        )
        pending = asyncio.create_task(operation)
        try:
            assert await asyncio.to_thread(worker.started.wait, 2)
            with pytest.raises(module.SessionUnavailable):
                await serving.generate(payload)
            if disconnect:
                disconnected.set()
            else:
                pending.cancel()
            await asyncio.sleep(0.01)
            assert not pending.done() and worker.lock.locked()
        finally:
            worker.release.set()
        with pytest.raises(module.SessionUnavailable if disconnect else asyncio.CancelledError):
            await pending
        with pytest.raises(module.SessionUnavailable):
            await serving.generate(payload)
        assert len(worker.calls) == 3 and worker.failed
        await serving.session.close()

    asyncio.run(run())


def test_expiry_and_invalid_prompt_do_not_render(module, monkeypatch):
    async def run():
        _, serving, clock, payload = fixture(module, monkeypatch)
        await serving.prewarm({"session_id": "a" * 32})
        try:
            for changed in (
                {"seed": True},
                {"prompt": "Contact a@example.com"},
                {"prompt": "A fox.\x00"},
                {"text": "raw voice"},
            ):
                with pytest.raises(ValueError):
                    await serving.generate(payload | changed)
            assert not serving.claims and len(serving.session.worker.calls) == 2
            clock[0] += 300
            with pytest.raises(module.SessionUnavailable):
                await serving.generate(payload)
            with pytest.raises(module.SessionUnavailable):
                await serving.prewarm({"session_id": "a" * 32})
            assert len(serving.session.worker.calls) == 2
        finally:
            await serving.session.close()

    asyncio.run(run())


def test_pinned_composition_and_pack_constant(module, tmp_path):
    original = HERE.parents[1] / "benchmarks/prepared-gcp-2026-09-08/t/context-t/rootfs/app"
    for name, digest in module.SOURCES.items():
        data = (HERE / (name + ".py")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert (
            data
            == (original / ("app.py" if name == "session_worker" else name + ".py")).read_bytes()
        )
        (tmp_path / (name + ".py")).write_bytes(data)
    module.verify_sources(tmp_path)
    assert module.PACK_PROOF_SHA256 == module.runtime_worker.PACK_PROOF_SHA256
    (tmp_path / "session_worker.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        module.verify_sources(tmp_path)


def test_adapter_packages_actual_worker_wire_without_network(module, monkeypatch, tmp_path):
    from storylight.finite_modal_provider import FastSceneRequest
    from storylight.prepared_klein_provider import IDENTITY, PreparedKleinProvider

    stream = io.BytesIO()
    Image.new("RGB", (1024, 576), "blue").save(stream, format="JPEG")
    jpeg = stream.getvalue()
    digest = hashlib.sha256(jpeg).hexdigest()
    original = FakeWorker.render

    def render(self, manifest, case):
        result = original(self, manifest, case)
        result["metrics"].update(
            token_count=result["metrics"]["sequence_bucket"],
            image_seconds=0.06,
            depth_seconds=0.03,
            encoding_seconds=0.01,
            peak_allocated_gib=1,
            peak_reserved_gib=1,
            master_sha256=digest,
            depth_sha256=digest,
        )
        result.update(
            master_b64=base64.b64encode(jpeg).decode(), depth_b64=base64.b64encode(jpeg).decode()
        )
        return result

    monkeypatch.setattr(FakeWorker, "render", render)

    async def run():
        app, serving, clock, _ = fixture(module, monkeypatch)
        original_reader = serving.session.manifest_reader
        serving.session.manifest_reader = lambda path: (
            original_reader(path)[0] | {"expected_identity": IDENTITY},
            "manifest",
        )

        async def token(audience):
            assert audience == "https://serving-test.run.app"
            return "test-token"

        def client_factory(**kwargs):
            return httpx.AsyncClient(transport=httpx.ASGITransport(app), **kwargs)

        provider = PreparedKleinProvider(
            base_url="https://serving-test.run.app",
            service="serving-test",
            revision="trial",
            external_owner_id="test-owner",
            claims_dir=tmp_path / "claims",
            token_source=token,
            client_factory=client_factory,
            clock=lambda: clock[0],
            wall_clock=lambda: clock[0],
        )
        try:
            report = await provider.prewarm(prewarm_id="test")
            assert report.expires_in_seconds == 300
            bundle = await provider.generate_fast(
                FastSceneRequest("scene-1", "A watercolor fox.", steps=4, guidance_scale=1),
                output_dir=tmp_path / "scene",
            )
            assert bundle.master.path.read_bytes() == bundle.depth.path.read_bytes() == jpeg
            assert len(serving.session.worker.calls) == 3
            assert bundle.manifest["lease"]["instance_id"] == serving.session.worker.instance_id
        finally:
            await provider.aclose()
            await serving.session.close()

    asyncio.run(run())
