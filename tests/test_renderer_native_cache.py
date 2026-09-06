from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def cache_worker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("deploy").resolve()))
    spec = importlib.util.spec_from_file_location(
        "native_cache_test", "experiments/renderer-native-cache/app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("K_SERVICE", "native-cache")
    monkeypatch.setenv("K_REVISION", "native-cache-00001")
    from klein_scene_runtime import PROFILE

    sources = {}
    for key, source in {
        "worker": Path(spec.origin),
        "runtime": Path("deploy/klein_scene_runtime.py"),
        "weights": Path("deploy/gcp_klein_worker/klein_weights.py"),
    }.items():
        data = source.read_bytes()
        (tmp_path / ("app.py" if key == "worker" else source.name)).write_bytes(data)
        sources[key] = hashlib.sha256(data).hexdigest()
    identity = (
        PROFILE
        | {k: module.PACKAGES[k] for k in ("torch", "diffusers", "transformers", "triton")}
        | {
            "cuda": "12.8",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "capability": [12, 0],
            "runtime_sha256": sources["runtime"],
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "authorized",
        "expires_at": int(time.time()) + 600,
        "experiment_id": "cache-test",
        "max_requests": 10,
        "expected_identity": identity,
        "sources": sources,
        "cases": [{"case_id": "boat", "seed": 1, "prompt": "A watercolor boat."}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    calls = []

    class Runtime:
        def __init__(self, expected):
            self.identity = expected
            calls.append("load")

        def compile(self, cache):
            assert cache is None
            calls.append("compile")
            return 0.1

        def render(self, prompt, seed):
            calls.append("render")
            return (
                {
                    "seed": seed,
                    "sequence_bucket": 128 if calls.count("render") % 2 else 256,
                    "master_sha256": hashlib.sha256(b"master").hexdigest(),
                    "depth_sha256": hashlib.sha256(b"depth").hexdigest(),
                },
                b"master",
                b"depth",
            )

        def save_cache(self, target):
            calls.append("export")
            target.mkdir()
            (target / "artifacts.bin").write_bytes(b"compiled-artifact")
            saved = {
                "identity": self.identity,
                "sha256": hashlib.sha256(b"compiled-artifact").hexdigest(),
                "cache_info": "must-not-be-exported",
            }
            (target / "manifest.json").write_text(json.dumps(saved))
            return saved

    return module, path, manifest, Runtime, calls


def test_export_admission_receipts_and_single_serialization(cache_worker, monkeypatch):
    module, path, manifest, runtime, calls = cache_worker
    app = module.create_app(path, runtime)
    worker = app.state.worker
    query = {
        "instance_id": worker.instance_id,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            assert calls == []
            original_manifest = path.read_bytes()
            path.write_text(json.dumps(manifest | {"expires_at": 1}))
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            path.write_bytes(original_manifest)
            assert calls == []
            for _ in range(10):
                result = await client.post("/generate", json={"case_id": "boat", "seed": 1})
                assert result.status_code == 200
            before = list(calls)
            for invalid in (
                dict(query, instance_id="0" * 32),
                dict(query, manifest_sha256="0" * 64),
                dict(query, extra="x"),
            ):
                assert (await client.get("/compiler-cache", params=invalid)).status_code == 409
            worker.failed = True
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            worker.failed = False
            last = worker.receipts.pop()
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            worker.receipts.append(last)
            worker.completed_buckets.remove(256)
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            worker.completed_buckets.add(256)
            await worker.lock.acquire()
            assert (await client.get("/compiler-cache", params=query)).status_code == 429
            worker.lock.release()
            assert calls == before
            reply = await client.get("/compiler-cache", params=query)
            assert reply.status_code == 200 and reply.content.startswith(module.CACHE_MAGIC)
            size = int.from_bytes(reply.content[6:10], "big")
            metadata = json.loads(reply.content[10 : 10 + size])
            artifact = reply.content[10 + size :]
            assert artifact == b"compiled-artifact"
            assert metadata["sha256"] == hashlib.sha256(artifact).hexdigest()
            assert metadata["producer"]["instance_id"] == worker.instance_id
            assert metadata["manifest_sha256"] == query["manifest_sha256"]
            assert metadata["successful_results"] == 10 and metadata["buckets"] == [128, 256]
            assert (
                metadata["receipts_sha256"]
                == hashlib.sha256(module.encoded(worker.receipts)).hexdigest()
            )
            assert b"must-not-be-exported" not in reply.content
            assert (await client.get("/compiler-cache", params=query)).status_code == 409
            assert calls == before + ["export"]
            worker.export_attempted = False
            monkeypatch.setattr(module, "MAX_CACHE_BYTES", 2)
            assert (await client.get("/compiler-cache", params=query)).json() == {"error": "failed"}
            assert worker.export_attempted

    asyncio.run(exercise())


def test_consumer_pins_precede_load_and_cache_result_precedes_compile(
    cache_worker, monkeypatch, capsys
):
    module, path, manifest, runtime, calls = cache_worker
    module.CACHE_ROOT.mkdir()
    case = manifest["cases"][0]
    with pytest.raises(ValueError):
        module.Worker(path, runtime).render(manifest, case)
    assert calls == []
    artifact = b"compiled-artifact"
    raw = module.encoded(
        {
            "identity": manifest["expected_identity"],
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "size_bytes": len(artifact),
        }
    )
    (module.CACHE_ROOT / "metadata.json").write_bytes(raw)
    (module.CACHE_ROOT / "artifacts.bin").write_bytes(artifact)
    monkeypatch.setattr(module, "CACHE_METADATA_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(module, "CACHE_ARTIFACT_SHA256", hashlib.sha256(artifact).hexdigest())
    with pytest.raises(ValueError):
        module.cache_input(manifest["expected_identity"] | {"gpu": "NVIDIA L4"})
    assert calls == []
    (module.CACHE_ROOT / "artifacts.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        module.Worker(path, runtime).render(manifest, case)
    assert calls == []
    (module.CACHE_ROOT / "artifacts.bin").write_bytes(artifact)

    def restore(data):
        assert data == artifact
        calls.append("restore")
        return None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(compiler=SimpleNamespace(load_cache_artifacts=restore)),
    )
    with pytest.raises(ValueError):
        module.Worker(path, runtime).render(manifest, case)
    assert calls == ["load", "restore"]

    def valid_restore(data):
        restore(data)
        return object()

    sys.modules["torch"].compiler.load_cache_artifacts = valid_restore
    worker = module.Worker(path, runtime)
    result = worker.render(manifest, case)
    assert calls[-4:] == ["load", "restore", "compile", "render"]
    assert result["bucket_was_warm"] is False
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["returned_info"] for e in events] == [False, True]
    assert all(e["event"] == "qualification.cache_restore" and e["seconds"] >= 0 for e in events)
    app = module.create_app(path, runtime)

    async def refused_export():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            assert (await client.get("/compiler-cache")).status_code == 409

    asyncio.run(refused_export())
