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
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

ROOT = Path(__file__).resolve().parent
PACK_PROOF_SHA256 = "6cb1afce2751161aee7d401a89bdc3ee4fd4860715b64deeaf7b512c15c756e7"
HELPER_SHA256 = "e83534508132f5be6717ee19582b354d3c1b16a65fd66e252ddedb8390bf19ee"
ORACLE_SHA256 = "76ff344f96125fda87f53f3a041233c23edecf70f26a1cdd087baa7cc62d85a4"
BASELINE_RUNTIME = "f87ab40c8b6a1ed457a10f1bce6bf92e07075eed40c0fdb224ba3ed18c5c3441"
CANDIDATE_RUNTIME = "b3d1007f297bde37ac40a9f2dee9eeb89e985dd16fdb8bdd9478c0086f6ca92a"
PACK_DIRECTORY = Path("/transformer-pack")
ORACLE_SECONDS = 60
AUXILIARY_SECONDS = 30
CACHE_ROOT = Path("/compiler-cache")
CACHE_ARTIFACT_SHA256 = "65b3b2415865e7a7121dc06e3c53b1b4b4eb0ca051fad538f56931239704d05f"
CACHE_METADATA_SHA256 = "2ba821d82acb20344f9e67fe9a8e9cd08f849dbf4338dcef04df0da164f505ac"
MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8192
CACHE_MAGIC = b"BFKC1\n"
NOISE_HELPER_SHA256 = "88f993d8bb6e4ae16da8c8918d7f8c2cd5e70b6378d6daefd76def18603381b4"
DIAGNOSTICS_HELPER_SHA256 = "6b0fa5f64673cfcaae00db10ad6b061057272735a82b68edef4ac7505bd1886b"

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
    require(type(manifest["max_requests"]) is int and manifest["max_requests"] == 12)
    require(
        isinstance(manifest["experiment_id"], str) and 1 <= len(manifest["experiment_id"]) <= 96
    )
    for name, expected in (
        ("klein_flashpack.py", HELPER_SHA256),
        ("transformer_oracle.py", ORACLE_SHA256),
        ("noise.py", NOISE_HELPER_SHA256),
        ("diagnostics.py", DIAGNOSTICS_HELPER_SHA256),
    ):
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected)
    sources = manifest["sources"]
    require(set(sources) == {"worker", "runtime", "weights"})
    for key, filename in (
        ("worker", "app.py"),
        ("runtime", "klein_scene_runtime.py"),
        ("weights", "klein_weights.py"),
    ):
        require(sources[key] == hashlib.sha256((ROOT / filename).read_bytes()).hexdigest())
    require(sources["runtime"] == CANDIDATE_RUNTIME)
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
    require(isinstance(cases, list) and len(cases) == 8)
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
    if identity["runtime_sha256"] == CANDIDATE_RUNTIME:
        # All candidate proof/helper work stays inside the original whole-factory timer.
        from klein_flashpack import inspect_pack

        inspect_pack(
            PACK_DIRECTORY, Path("/models/klein/transformer/config.json"), PACK_PROOF_SHA256
        )
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
        self.completed_buckets = set()
        self.load_seconds = self.compile_seconds = 0.0
        self.stage = "idle"
        self.manifest_sha256 = None
        self.receipts = []
        self.oracle_attempted = False
        self.oracle_succeeded = False
        self.noise_attempted = self.noise_succeeded = self.export_attempted = False
        self.noise_receipts = []

    def identity(self):
        return {
            "instance_id": self.instance_id,
            "revision": os.environ.get("K_REVISION", ""),
            "service": os.environ.get("K_SERVICE", ""),
        }

    def render(self, manifest, case):
        started = time.perf_counter()
        from diagnostics import emit, snapshot

        cold = self.runtime is None
        if cold:
            self.stage = "load"
            load_started = time.perf_counter()
            runtime = self.factory(manifest["expected_identity"])
            require(runtime.identity == manifest["expected_identity"])
            require(time.time() < manifest["expires_at"])
            artifact = cache_input(runtime.identity)
            if artifact is not None:
                import torch

                require(time.time() < manifest["expires_at"])
                self.stage = "cache_restore"
                require(torch.compiler.load_cache_artifacts(artifact) is not None)
            self.load_seconds = time.perf_counter() - load_started
            require(time.time() < manifest["expires_at"])
            self.stage = "compile"
            _compiler_before = snapshot()
            self.compile_seconds = runtime.compile(None)
            emit("compile", _compiler_before, snapshot(), self.instance_id)
            self.runtime = runtime
        require(time.time() < manifest["expires_at"])
        self.stage = "render"
        import torch
        from noise import noise_receipt, prepared_noise

        _compiler_before = snapshot() if self.requests in (1, 2) else None
        with prepared_noise(self.runtime.pipe, case, torch) as observed:
            metrics, master, depth = self.runtime.render(case["prompt"], case["seed"])
        noise_row = noise_receipt(observed, self.requests, case)
        if self.requests in (1, 2):
            emit(
                "first_128" if self.requests == 1 else "first_256",
                _compiler_before,
                snapshot(),
                self.instance_id,
            )
        self.stage = "verify"
        require(metrics["seed"] == case["seed"])
        require(metrics["master_sha256"] == hashlib.sha256(master).hexdigest())
        require(metrics["depth_sha256"] == hashlib.sha256(depth).hexdigest())
        bucket = metrics["sequence_bucket"]
        warm = bucket in self.completed_buckets
        self.completed_buckets.add(bucket)
        self.stage = "complete"
        self.noise_receipts.append(noise_row)
        self.receipts.append(
            {
                "ordinal": self.requests,
                "case_id": case["case_id"],
                "seed": case["seed"],
                "sequence_bucket": bucket,
                "master_sha256": metrics["master_sha256"],
                "depth_sha256": metrics["depth_sha256"],
            }
        )
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

    def receipts_sha256(self):
        return hashlib.sha256(
            json.dumps(self.receipts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def compare_tensors(self, deadline, cancelled):
        started = time.perf_counter()
        self.stage = "transformer_oracle"
        path = PACK_DIRECTORY / "proof.json"
        require(path.is_file() and 0 < path.stat().st_size <= 262144)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == PACK_PROOF_SHA256)
        proof = decode(raw)
        from transformer_oracle import run_oracle

        result = run_oracle(self.runtime.pipe.transformer, proof, deadline, cancelled)
        self.stage = "complete"
        return {
            "schema_version": 1,
            "kind": "transformer-tensor-oracle",
            **self.identity(),
            "manifest_sha256": self.manifest_sha256,
            "receipts_sha256": self.receipts_sha256(),
            "proof_sha256": PACK_PROOF_SHA256,
            "oracle_sha256": ORACLE_SHA256,
            "identity": self.runtime.identity,
            **result,
            "seconds": time.perf_counter() - started,
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

    def noise_oracle(self):
        return {
            "schema_version": 1,
            "kind": "klein-initial-noise-oracle",
            **self.identity(),
            "manifest_sha256": self.manifest_sha256,
            "identity": self.runtime.identity,
            "receipts_sha256": self.receipts_sha256(),
            "noise_sha256": hashlib.sha256(encoded(self.noise_receipts)).hexdigest(),
            "noise_helper_sha256": NOISE_HELPER_SHA256,
            "records": self.noise_receipts,
        }


def create_app(manifest_path, factory=load_runtime):
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    worker = Worker(manifest_path, factory)
    application.state.worker = worker

    async def auxiliary(request, kind):
        if worker.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=429)
        async with worker.lock:
            try:
                manifest, digest = read_manifest(worker.path)
                require(len(request.query_params.multi_items()) == 2)
                require(
                    dict(request.query_params)
                    == {
                        "instance_id": worker.instance_id,
                        "manifest_sha256": digest,
                    }
                )
                require(digest == worker.manifest_sha256 and worker.runtime is not None)
                require(not worker.failed and worker.oracle_succeeded)
                require(worker.requests == len(worker.receipts) == len(worker.noise_receipts) == 10)
                if kind == "noise":
                    require(not worker.noise_attempted)
                    worker.noise_attempted = True
                    operation = worker.noise_oracle
                else:
                    require(CACHE_ARTIFACT_SHA256 is CACHE_METADATA_SHA256 is None)
                    require(not CACHE_ROOT.exists() and not CACHE_ROOT.is_symlink())
                    require(worker.noise_succeeded and not worker.export_attempted)
                    require({128, 256} <= worker.completed_buckets)
                    worker.export_attempted = True

                    def operation():
                        return worker.export_cache(digest)
            except Exception:
                return JSONResponse({"error": "rejected"}, status_code=409)
            budget = min(AUXILIARY_SECONDS, manifest["expires_at"] - time.time())
            if budget <= 0:
                worker.failed = True
                return JSONResponse({"error": "failed"}, status_code=503)
            task = asyncio.create_task(asyncio.to_thread(operation))

            async def disconnected():
                while (await request.receive())["type"] != "http.disconnect":
                    pass

            watcher = asyncio.create_task(disconnected())
            try:
                done, _ = await asyncio.wait(
                    {task, watcher}, timeout=budget, return_when=asyncio.FIRST_COMPLETED
                )
                require(task in done and watcher not in done)
                payload = task.result()
                require(time.time() < manifest["expires_at"])
                if kind == "noise":
                    require(len(encoded(payload)) <= MAX_METADATA_BYTES)
                    worker.noise_succeeded = True
                    return JSONResponse(payload)
                require(len(payload) <= len(CACHE_MAGIC) + 4 + MAX_METADATA_BYTES + MAX_CACHE_BYTES)
                return Response(payload, media_type="application/octet-stream")
            except (asyncio.CancelledError, Exception) as error:
                worker.failed = True
                # Keep ownership until the non-cancellable CPU/CUDA work has stopped.
                while not task.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(task)
                if not task.cancelled():
                    task.exception()
                if isinstance(error, asyncio.CancelledError):
                    raise
                return JSONResponse({"error": "failed"}, status_code=503)
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher

    @application.get("/noise-oracle")
    async def noise_oracle(request: Request):
        return await auxiliary(request, "noise")

    @application.get("/compiler-cache")
    async def compiler_cache(request: Request):
        return await auxiliary(request, "cache")

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
            schedule = manifest["cases"] + manifest["cases"][:2]
            if worker.manifest_sha256 not in (None, digest) or case != schedule[worker.requests]:
                return JSONResponse({"error": "sequence"}, status_code=409)
            worker.manifest_sha256 = digest
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

    @application.post("/transformer-oracle")
    async def transformer_oracle(request: Request):
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                require(len(raw) <= 1024)
            payload = decode(raw)
            require(
                isinstance(payload, dict)
                and set(payload)
                == {"instance_id", "manifest_sha256", "receipts_sha256", "proof_sha256"}
            )
            manifest, digest = read_manifest(worker.path)
            require(payload["instance_id"] == worker.instance_id)
            require(payload["manifest_sha256"] == digest == worker.manifest_sha256)
            require(payload["receipts_sha256"] == worker.receipts_sha256())
            require(payload["proof_sha256"] == PACK_PROOF_SHA256)
        except Exception:
            return JSONResponse({"error": "rejected"}, status_code=400)
        if worker.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=429)
        async with worker.lock:
            if (
                worker.failed
                or worker.oracle_attempted
                or worker.runtime is None
                or worker.requests != 10
                or len(worker.receipts) != 10
            ):
                return JSONResponse({"error": "unavailable"}, status_code=409)
            worker.oracle_attempted = True
            cancelled = threading.Event()
            budget = min(ORACLE_SECONDS, manifest["expires_at"] - time.time())
            deadline = time.monotonic() + budget
            task = asyncio.create_task(
                asyncio.to_thread(worker.compare_tensors, deadline, cancelled)
            )

            async def disconnected():
                while (await request.receive())["type"] != "http.disconnect":
                    pass

            watcher = asyncio.create_task(disconnected())
            try:
                done, _ = await asyncio.wait(
                    {task, watcher}, timeout=budget, return_when=asyncio.FIRST_COMPLETED
                )
                if watcher in done:
                    raise ConnectionError("oracle disconnected")
                if task not in done:
                    raise TimeoutError("oracle deadline")
                result = task.result()
                worker.oracle_succeeded = (
                    result["all_tensor_hashes_match"] is True and result["tensor_count"] == 169
                )
                return result
            except (asyncio.CancelledError, Exception) as error:
                worker.failed = True
                cancelled.set()
                print(
                    json.dumps(
                        {
                            "event": "qualification.failed",
                            "stage": "transformer_oracle",
                            "exception_type": type(error).__name__
                            if type(error).__module__ == "builtins"
                            else "Exception",
                        }
                    ),
                    flush=True,
                )
                # Coroutine cancellation cannot interrupt an in-flight CUDA transfer.
                while not task.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(task)
                if not task.cancelled():
                    task.exception()
                if isinstance(error, asyncio.CancelledError):
                    raise
                return JSONResponse({"error": "failed"}, status_code=503)
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher

    return application


app = create_app(ROOT / "manifest.json")
