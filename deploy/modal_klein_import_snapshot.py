"""Bounded GPU-assisted framework import snapshots; all model work follows restore."""

import hashlib
import json
import os
import re
import resource
import sys
import time
import uuid
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
EXPERIMENT = "klein-import-snapshot-20260905-a"
APP_NAME = "bookforge-klein-import-snapshot"
RUNTIME_SHA256 = "f87ab40c8b6a1ed457a10f1bce6bf92e07075eed40c0fdb224ba3ed18c5c3441"
CASES_SHA256 = "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
MANIFEST = Path("/root/import-snapshot-manifest.json")
VERSIONS = {
    "torch": "2.8.0+cu128",
    "diffusers": "0.39.0",
    "transformers": "4.57.1",
    "triton": "3.4.0",
}

if modal.is_local():
    image = (
        modal.Image.from_id(IMAGE_ID)
        .add_local_file(DEPLOY / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
        .add_local_file(
            DEPLOY.parent / "benchmarks/renderer-import-snapshot-2026-09-05/manifest.json",
            str(MANIFEST),
        )
    )
else:
    image = None

app = modal.App(APP_NAME)
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _expiry(manifest):
    expires = manifest.get("expires_at")
    now = time.time()
    if type(expires) is not int or not now < expires <= now + 7200:
        raise ValueError("import snapshot authorization expired")


def _manifest():
    with MANIFEST.open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("import snapshot manifest exceeds bound")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("import snapshot authorization differs")
    _expiry(manifest)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("status") != "authorized"
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("image_id") != IMAGE_ID
        or manifest.get("cache_id") != CACHE_ID
        or type(manifest.get("max_operations")) is not int
        or manifest["max_operations"] != 5
        or type(manifest.get("max_captures")) is not int
        or manifest["max_captures"] != 3
        or manifest.get("runtime_sha256") != RUNTIME_SHA256
        or _sha((DEPLOY / "klein_scene_runtime.py").read_bytes()) != RUNTIME_SHA256
        or _sha(Path(__file__).read_bytes()) != manifest.get("deployment_sha256")
        or _sha(json.dumps(manifest.get("cases"), sort_keys=True, separators=(",", ":")).encode())
        != CASES_SHA256
    ):
        raise ValueError("import snapshot authorization differs")
    operations = manifest.get("operations")
    if (
        not isinstance(operations, list)
        or len(operations) != 5
        or any(
            not isinstance(row, dict)
            or set(row) != {"ordinal", "request_id", "variant"}
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or row["variant"] != "snapshot"
            or not isinstance(row["request_id"], str)
            or not re.fullmatch(r"[a-f0-9]{32}", row["request_id"])
            for ordinal, row in enumerate(operations)
        )
        or len({row["request_id"] for row in operations}) != 5
    ):
        raise ValueError("import snapshot operations differ")
    if (
        os.environ.get("MODAL_CLOUD_PROVIDER") != "CLOUD_PROVIDER_AWS"
        or os.environ.get("MODAL_REGION") != "us-east-1"
        or not os.environ.get("MODAL_TASK_ID")
    ):
        raise ValueError("import snapshot placement differs")
    return manifest, _sha(raw)


def _event(phase, state, **fields):
    print(
        json.dumps(
            {"import_snapshot_stage": {"phase": phase, "state": state, **fields}}, sort_keys=True
        ),
        flush=True,
    )


@app.cls(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    startup_timeout=120,
    timeout=180,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    single_use_containers=True,
    cloud="aws",
    region="us-east",
    routing_region="us-east",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/compiled": cache},
    include_source=True,
)
class ImportSnapshot:
    @modal.enter(snap=True)
    def capture(self):
        _, self.manifest_sha256 = _manifest()
        capture_id = uuid.uuid4().hex
        for ordinal in range(3):
            if claims.put(f"capture:{ordinal}", capture_id, skip_if_exists=True):
                break
        else:
            raise ValueError("import snapshot capture allowance exhausted")
        self.capture_id = capture_id
        self.capture_container_sha256 = _sha(os.environ["MODAL_TASK_ID"].encode())
        _event(
            "capture",
            "start",
            capture_id=capture_id,
            ordinal=ordinal,
            container_sha256=self.capture_container_sha256,
        )
        started = time.perf_counter()
        try:
            import diffusers
            import torch
            import transformers
            import triton
            from transformers import (  # noqa: F401
                AutoImageProcessor,
                AutoModelForDepthEstimation,
                pipeline,
            )

            self.versions = {
                name: str(module.__version__)
                for name, module in (
                    ("diffusers", diffusers),
                    ("torch", torch),
                    ("transformers", transformers),
                    ("triton", triton),
                )
            }
            self.cuda_available = bool(torch.cuda.is_available())
            if self.versions != VERSIONS or not self.cuda_available:
                raise ValueError("framework identity differs")
        except Exception:
            _event("capture", "failed", capture_id=capture_id)
            raise RuntimeError("import snapshot capture failed") from None
        self.imports_seconds = time.perf_counter() - started
        _event("capture", "end", capture_id=capture_id, imports_seconds=self.imports_seconds)

    @modal.enter(snap=False)
    def activate(self):
        _, manifest_sha256 = _manifest()
        if manifest_sha256 != self.manifest_sha256:
            raise ValueError("import snapshot manifest changed")
        self.activation_id = uuid.uuid4().hex
        self.container_sha256 = _sha(os.environ["MODAL_TASK_ID"].encode())
        self.started = time.perf_counter()
        _event(
            "activate",
            "end",
            capture_id=self.capture_id,
            activation_id=self.activation_id,
            container_sha256=self.container_sha256,
        )

    @modal.method()
    def cycle(self, request_id: str):
        manifest, manifest_sha256 = _manifest()
        if manifest_sha256 != self.manifest_sha256 or not hasattr(self, "activation_id"):
            raise ValueError("import snapshot activation differs")
        if not any(row["request_id"] == request_id for row in manifest["operations"]):
            raise ValueError("import snapshot request is not authorized")
        if not claims.put(f"request:{request_id}", True, skip_if_exists=True):
            raise ValueError("import snapshot request already claimed")
        request_sha256 = _sha(request_id.encode())

        def event(phase, state, **fields):
            _event(
                phase,
                state,
                request_sha256=request_sha256,
                activation_id=self.activation_id,
                elapsed_seconds=time.perf_counter() - self.started,
                **fields,
            )

        try:
            from klein_scene_runtime import KleinSceneRuntime

            event("runtime_load", "start")
            runtime = KleinSceneRuntime(Path("/models"))
            event("runtime_load", "end")
            if runtime.identity != manifest["expected_identity"]:
                raise ValueError("runtime identity differs")
            _expiry(manifest)
            event("cache_restore", "start")
            cache_seconds = runtime.compile(Path("/compiled") / CACHE_ID)
            event("cache_restore", "end")
            samples = []
            for repeat in range(2):
                for case_index, case in enumerate(manifest["cases"]):
                    _expiry(manifest)
                    event("render", "start", repeat=repeat, case_index=case_index)
                    metrics, master, depth = runtime.render(case["prompt"], case["seed"])
                    if metrics["sequence_bucket"] != case["expected_bucket"]:
                        raise ValueError("token bucket differs")
                    samples.append(
                        {
                            "repeat": repeat,
                            "case_index": case_index,
                            "metrics": metrics,
                            "master": master,
                            "depth": depth,
                        }
                    )
                    event("render", "end", repeat=repeat, case_index=case_index)
        except Exception:
            event("cycle", "failed")
            raise RuntimeError("import snapshot cycle failed") from None
        event("complete", "end")
        return {
            "request_id": request_id,
            "variant": "snapshot",
            "identity": runtime.identity,
            "location": {
                "cloud": os.environ["MODAL_CLOUD_PROVIDER"],
                "compute_region": os.environ["MODAL_REGION"],
                "container_sha256": self.container_sha256,
            },
            "snapshot": {
                "capture_id": self.capture_id,
                "capture_container_sha256": self.capture_container_sha256,
                "activation_id": self.activation_id,
                "imports_seconds": self.imports_seconds,
                "versions": self.versions,
                "cuda_available": self.cuda_available,
            },
            "stages": {
                "imports_seconds": self.imports_seconds,
                "model_load_seconds": runtime.load_seconds,
                "cache_setup_seconds": cache_seconds,
                "worker_seconds": time.perf_counter() - self.started,
                "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            },
            "samples": samples,
        }
