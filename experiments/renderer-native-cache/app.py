"""IAM-private, finite synthetic Klein qualification; no live provider routing."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path("/compiler-cache")
CACHE_ARTIFACT_SHA256 = None
CACHE_METADATA_SHA256 = None
MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8192
CACHE_MAGIC = b"BFKC1\n"
PACKAGES = {
    "torch": "2.8.0+cu128",
    "diffusers": "0.39.0",
    "transformers": "4.57.1",
    "triton": "3.4.0",
    "accelerate": "1.10.1",
    "Pillow": "11.1.0",
}
GPUS = {
    "NVIDIA L4": [8, 9],
    "NVIDIA RTX PRO 6000 Blackwell": [12, 0],
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": [12, 0],
}


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


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def bounded_file(path, limit):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit)
    return data


def cache_input(identity):
    pins = (CACHE_ARTIFACT_SHA256, CACHE_METADATA_SHA256)
    if pins == (None, None):
        require(not CACHE_ROOT.exists() and not CACHE_ROOT.is_symlink())
        return None
    require(all(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin) for pin in pins))
    require(CACHE_ROOT.is_dir() and not CACHE_ROOT.is_symlink())
    metadata_raw = bounded_file(CACHE_ROOT / "metadata.json", MAX_METADATA_BYTES)
    require(hashlib.sha256(metadata_raw).hexdigest() == CACHE_METADATA_SHA256)
    metadata = decode(metadata_raw)
    require(encoded(metadata["identity"]) == encoded(identity))
    artifact = bounded_file(CACHE_ROOT / "artifacts.bin", MAX_CACHE_BYTES)
    require(hashlib.sha256(artifact).hexdigest() == CACHE_ARTIFACT_SHA256 == metadata["sha256"])
    require(type(metadata["size_bytes"]) is int and len(artifact) == metadata["size_bytes"] > 0)
    return artifact


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
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    version_pattern = r"[0-9]+(?:\.[0-9]+)*(?:\.?(?:a|b|rc|post|dev)[0-9]+)?(?:\+(?:cu[0-9]+|cpu))?"

    def safe_version(value):
        return (
            value
            if isinstance(value, str) and len(value) <= 64 and re.fullmatch(version_pattern, value)
            else None
        )

    def check(ok, stage):
        if not ok:
            print(
                json.dumps({"event": "qualification.preflight_rejected", "stage": stage}),
                flush=True,
            )
        require(ok)

    print(
        json.dumps(
            {
                "event": "qualification.preflight",
                "stage": "packages",
                "versions": {key: safe_version(value) for key, value in versions.items()},
            }
        ),
        flush=True,
    )
    check(versions == PACKAGES, "packages")
    import torch
    from klein_scene_runtime import KleinSceneRuntime

    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else None
    capability = list(torch.cuda.get_device_capability(0)) if available else None
    arches = torch.cuda.get_arch_list()
    print(
        json.dumps(
            {
                "event": "qualification.preflight",
                "stage": "device",
                "cuda_available": bool(available),
                "cuda_version": safe_version(torch.version.cuda),
                "gpu": name
                if isinstance(name, str)
                and re.fullmatch(r"(?:NVIDIA|Tesla) [A-Za-z0-9 -]{1,96}", name)
                else None,
                "capability": capability
                if isinstance(capability, list)
                and len(capability) == 2
                and all(type(v) is int and 0 <= v <= 99 for v in capability)
                else None,
                "compiled_arches": [
                    arch
                    for arch in arches[:32]
                    if isinstance(arch, str)
                    and re.fullmatch(r"(?:sm|compute)_[0-9]{2,3}[af]?", arch)
                ],
            }
        ),
        flush=True,
    )
    check(available and torch.version.cuda == identity["cuda"], "cuda")
    check(name == identity["gpu"], "gpu")
    check(capability == identity["capability"], "capability")
    check(f"sm_{identity['capability'][0]}{identity['capability'][1]}" in arches, "compiled_arches")
    print(json.dumps({"event": "qualification.load", "stage": "model_load"}), flush=True)
    return KleinSceneRuntime(Path("/models"))


class Worker:
    def __init__(self, manifest_path, factory=load_runtime):
        self.path, self.factory = manifest_path, factory
        self.instance_id = uuid.uuid4().hex
        self.runtime = None
        self.lock = asyncio.Lock()
        self.failed = False
        self.requests = 0
        self.receipts = []
        self.export_attempted = False
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
            artifact = cache_input(manifest["expected_identity"])
            self.stage = "load"
            load_started = time.perf_counter()
            runtime = self.factory(manifest["expected_identity"])
            require(runtime.identity == manifest["expected_identity"])
            self.load_seconds = time.perf_counter() - load_started
            require(time.time() < manifest["expires_at"])
            if artifact is not None:
                import torch

                self.stage = "cache_restore"
                cache_started = time.perf_counter()
                loaded = torch.compiler.load_cache_artifacts(artifact) is not None
                print(
                    json.dumps(
                        {
                            "event": "qualification.cache_restore",
                            "seconds": time.perf_counter() - cache_started,
                            "returned_info": loaded,
                        }
                    ),
                    flush=True,
                )
                require(loaded)  # Cache population is not proof of graph hits.
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
        require(type(bucket) is int and bucket in (128, 256, 512))
        warm = bucket in self.completed_buckets
        self.completed_buckets.add(bucket)
        self.receipts.append(
            {
                "ordinal": self.requests,
                "case_id": case["case_id"],
                "seed": case["seed"],
                "bucket": bucket,
                "master_sha256": metrics["master_sha256"],
                "depth_sha256": metrics["depth_sha256"],
            }
        )
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

    def export_cache(self, digest):
        with tempfile.TemporaryDirectory(prefix="klein-cache-") as directory:
            target = Path(directory) / "export"
            saved = self.runtime.save_cache(target)
            artifact = bounded_file(target / "artifacts.bin", MAX_CACHE_BYTES)
            require(len(artifact) > 0)
            require(encoded(saved["identity"]) == encoded(self.runtime.identity))
            require(saved["sha256"] == hashlib.sha256(artifact).hexdigest())
            producer = self.identity()
            require(
                all(
                    isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)
                    for value in producer.values()
                )
            )
            header = encoded(
                {
                    "schema_version": 1,
                    "identity": self.runtime.identity,
                    "sha256": saved["sha256"],
                    "size_bytes": len(artifact),
                    "producer": producer,
                    "manifest_sha256": digest,
                    "buckets": sorted(self.completed_buckets),
                    "successful_results": len(self.receipts),
                    "receipts_sha256": hashlib.sha256(encoded(self.receipts)).hexdigest(),
                }
            )
            require(len(header) <= MAX_METADATA_BYTES)
            return CACHE_MAGIC + len(header).to_bytes(4, "big") + header + artifact


def create_app(manifest_path, factory=load_runtime):
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    worker = Worker(manifest_path, factory)
    application.state.worker = worker

    @application.get("/compiler-cache")
    async def compiler_cache(request: Request):
        try:
            manifest, digest = read_manifest(worker.path)
            require(CACHE_ARTIFACT_SHA256 is None and CACHE_METADATA_SHA256 is None)
            require(not CACHE_ROOT.exists() and not CACHE_ROOT.is_symlink())
            require(len(request.query_params.multi_items()) == 2)
            require(
                dict(request.query_params)
                == {"instance_id": worker.instance_id, "manifest_sha256": digest}
            )
            require(manifest["max_requests"] == 10 and worker.runtime is not None)
            require(not worker.failed and len(worker.receipts) == worker.requests == 10)
            require({128, 256} <= worker.completed_buckets and not worker.export_attempted)
        except Exception:
            return JSONResponse({"error": "rejected"}, status_code=409)
        if worker.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=429)
        async with worker.lock:
            worker.export_attempted = True
            task = asyncio.create_task(asyncio.to_thread(worker.export_cache, digest))
            try:
                payload = await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(task)
                if not task.cancelled():
                    task.exception()
                raise
            except Exception:
                return JSONResponse({"error": "failed"}, status_code=503)
            return StreamingResponse(
                (payload[i : i + 65536] for i in range(0, len(payload), 65536)),
                media_type="application/octet-stream",
            )

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
