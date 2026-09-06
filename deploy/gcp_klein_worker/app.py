"""IAM-private, finite synthetic Klein qualification; no live provider routing."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    "torch": "2.8.0+cu128",
    "diffusers": "0.39.0",
    "transformers": "4.57.1",
    "triton": "3.4.0",
    "accelerate": "1.10.1",
    "Pillow": "11.1.0",
}
GPUS = {"NVIDIA L4": [8, 9], "NVIDIA RTX PRO 6000 Blackwell": [12, 0]}


def require(condition):
    if not condition:
        raise ValueError("qualification rejected")


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs, parse_constant=lambda _: require(False))


def read_manifest(path):
    from klein_scene_runtime import PROFILE

    raw = path.read_bytes()
    require(len(raw) <= 32_768)
    manifest = decode(raw)
    require(
        set(manifest)
        == {
            "schema_version",
            "status",
            "experiment_id",
            "expires_at",
            "max_requests",
            "expected_identity",
            "cases",
            "sources",
        }
    )
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1)
    require(manifest["status"] == "authorized")
    require(
        type(manifest["expires_at"]) is int
        and time.time() < manifest["expires_at"] <= time.time() + 7200
    )
    require(type(manifest["max_requests"]) is int and 1 <= manifest["max_requests"] <= 16)
    require(
        isinstance(manifest["experiment_id"], str) and 1 <= len(manifest["experiment_id"]) <= 96
    )
    sources = manifest["sources"]
    require(set(sources) == {"worker", "runtime", "weights"})
    for key, filename in (
        ("worker", "app.py"),
        ("runtime", "klein_scene_runtime.py"),
        ("weights", "klein_weights.py"),
    ):
        require(sources[key] == hashlib.sha256((ROOT / filename).read_bytes()).hexdigest())
    identity = manifest["expected_identity"]
    require(identity.get("gpu") in GPUS and identity.get("capability") == GPUS[identity["gpu"]])
    expected = (
        PROFILE
        | {key: PACKAGES[key] for key in ("torch", "diffusers", "transformers", "triton")}
        | {
            "cuda": "12.8",
            "gpu": identity["gpu"],
            "capability": GPUS[identity["gpu"]],
            "runtime_sha256": sources["runtime"],
        }
    )
    require(json.dumps(identity, sort_keys=True) == json.dumps(expected, sort_keys=True))
    cases = manifest["cases"]
    require(isinstance(cases, list) and 1 <= len(cases) <= 12)
    seen = set()
    for case in cases:
        require(set(case) == {"case_id", "prompt", "seed"})
        key = case["case_id"]
        require(
            isinstance(key, str)
            and 1 <= len(key) <= 96
            and key.isascii()
            and all(c.isalnum() or c in "-_" for c in key)
            and key not in seen
        )
        seen.add(key)
        require(isinstance(case["prompt"], str) and 1 <= len(case["prompt"].strip()) <= 4000)
        require(type(case["seed"]) is int and 0 <= case["seed"] <= 2**32 - 1)
    return manifest, hashlib.sha256(raw).hexdigest()


def load_runtime(identity):
    # Package/device checks precede all model loading and compilation.
    for package, version in PACKAGES.items():
        require(importlib.metadata.version(package) == version)
    import torch
    from klein_scene_runtime import KleinSceneRuntime

    require(torch.cuda.is_available() and torch.version.cuda == identity["cuda"])
    require(torch.cuda.get_device_name(0) == identity["gpu"])
    require(list(torch.cuda.get_device_capability(0)) == identity["capability"])
    require(
        f"sm_{identity['capability'][0]}{identity['capability'][1]}" in torch.cuda.get_arch_list()
    )
    return KleinSceneRuntime(Path("/models"))


class Worker:
    def __init__(self, manifest_path, factory=load_runtime):
        self.path, self.factory = manifest_path, factory
        self.instance_id = uuid.uuid4().hex
        self.runtime = None
        self.lock = asyncio.Lock()
        self.failed = False
        self.requests = 0
        self.completed_buckets = set()
        self.load_seconds = self.compile_seconds = 0.0
        self.stage = "idle"

    def identity(self):
        return {
            "instance_id": self.instance_id,
            "revision": os.environ.get("K_REVISION", ""),
            "service": os.environ.get("K_SERVICE", ""),
        }

    def render(self, manifest, case):
        started = time.perf_counter()
        cold = self.runtime is None
        if cold:
            self.stage = "load"
            load_started = time.perf_counter()
            runtime = self.factory(manifest["expected_identity"])
            require(runtime.identity == manifest["expected_identity"])
            self.load_seconds = time.perf_counter() - load_started
            require(time.time() < manifest["expires_at"])
            self.stage = "compile"
            self.compile_seconds = runtime.compile(None)
            self.runtime = runtime
        require(time.time() < manifest["expires_at"])
        self.stage = "render"
        metrics, master, depth = self.runtime.render(case["prompt"], case["seed"])
        self.stage = "verify"
        require(metrics["seed"] == case["seed"])
        require(metrics["master_sha256"] == hashlib.sha256(master).hexdigest())
        require(metrics["depth_sha256"] == hashlib.sha256(depth).hexdigest())
        bucket = metrics["sequence_bucket"]
        warm = bucket in self.completed_buckets
        self.completed_buckets.add(bucket)
        self.stage = "complete"
        return {
            "schema_version": 1,
            "case_id": case["case_id"],
            **self.identity(),
            "identity": self.runtime.identity,
            "metrics": metrics,
            "master_b64": base64.b64encode(master).decode("ascii"),
            "depth_b64": base64.b64encode(depth).decode("ascii"),
            "cold": cold,
            "bucket_was_warm": warm,
            "load_seconds": self.load_seconds,
            "compile_seconds": self.compile_seconds,
            "worker_seconds": time.perf_counter() - started,
            "request_ordinal": self.requests,
        }


def create_app(manifest_path, factory=load_runtime):
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    worker = Worker(manifest_path, factory)
    application.state.worker = worker

    @application.get("/health")
    async def health():
        try:
            _, digest = read_manifest(worker.path)
            return {
                **worker.identity(),
                "manifest_sha256": digest,
                "loaded": worker.runtime is not None,
                "failed": worker.failed,
            }
        except Exception:
            return JSONResponse({"error": "unavailable"}, status_code=503)

    @application.post("/generate")
    async def generate(request: Request):
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                require(len(raw) <= 2048)
            payload = decode(raw)
            require(isinstance(payload, dict) and set(payload) == {"case_id", "seed"})
            require(type(payload["seed"]) is int)
            manifest, digest = read_manifest(worker.path)
            case = next(
                case
                for case in manifest["cases"]
                if case["case_id"] == payload["case_id"] and case["seed"] == payload["seed"]
            )
        except Exception:
            return JSONResponse({"error": "rejected"}, status_code=400)
        if worker.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=429)
        async with worker.lock:
            if worker.failed or worker.requests >= manifest["max_requests"]:
                return JSONResponse({"error": "exhausted"}, status_code=409)
            worker.requests += 1  # Retain failed/ambiguous attempts; never retry initialization.
            task = asyncio.create_task(asyncio.to_thread(worker.render, manifest, case))
            try:
                result = await asyncio.shield(task)
                return result | {"manifest_sha256": digest}
            except asyncio.CancelledError:
                worker.failed = True
                print(
                    json.dumps(
                        {
                            "event": "qualification.failed",
                            "stage": worker.stage,
                            "exception_type": "CancelledError",
                        }
                    ),
                    flush=True,
                )
                # HTTP cancellation cannot stop CUDA. Keep exclusive ownership until completion.
                while not task.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(task)
                if not task.cancelled():
                    task.exception()
                raise
            except Exception as error:
                worker.failed = True
                print(
                    json.dumps(
                        {
                            "event": "qualification.failed",
                            "stage": worker.stage,
                            "exception_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                return JSONResponse({"error": "failed"}, status_code=503)

    return application


app = create_app(ROOT / "manifest.json")
