"""One verified warmed-runtime capture and three single-use snapshot activations."""

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
EXPERIMENT = "klein-warmed-snapshot-20260905-a"
APP_NAME = "bookforge-klein-warmed-snapshot"
RUNTIME_SHA256 = "f87ab40c8b6a1ed457a10f1bce6bf92e07075eed40c0fdb224ba3ed18c5c3441"
CASES_SHA256 = "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
MANIFEST = Path("/root/warmed-snapshot-manifest.json")
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
            DEPLOY.parent / "benchmarks/renderer-warmed-snapshot-2026-09-05/manifest.json",
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
        raise ValueError("warmed snapshot authorization expired")


def _manifest():
    with MANIFEST.open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("warmed snapshot manifest exceeds bound")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("warmed snapshot authorization differs")
    _expiry(manifest)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("status") != "authorized"
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("image_id") != IMAGE_ID
        or manifest.get("cache_id") != CACHE_ID
        or type(manifest.get("max_operations")) is not int
        or manifest["max_operations"] != 3
        or type(manifest.get("max_captures")) is not int
        or manifest["max_captures"] != 1
        or manifest.get("runtime_sha256") != RUNTIME_SHA256
        or _sha((DEPLOY / "klein_scene_runtime.py").read_bytes()) != RUNTIME_SHA256
        or _sha(Path(__file__).read_bytes()) != manifest.get("deployment_sha256")
        or _sha(json.dumps(manifest.get("cases"), sort_keys=True, separators=(",", ":")).encode())
        != CASES_SHA256
    ):
        raise ValueError("warmed snapshot authorization differs")
    operations = manifest.get("operations")
    if (
        not isinstance(operations, list)
        or len(operations) != 3
        or any(
            not isinstance(row, dict)
            or set(row) != {"ordinal", "request_id", "variant"}
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or row["variant"] != "warmed_snapshot"
            or not isinstance(row["request_id"], str)
            or not re.fullmatch(r"[a-f0-9]{32}", row["request_id"])
            for ordinal, row in enumerate(operations)
        )
        or len({row["request_id"] for row in operations}) != 3
    ):
        raise ValueError("warmed snapshot operations differ")
    if (
        os.environ.get("MODAL_CLOUD_PROVIDER")
        not in {"CLOUD_PROVIDER_AWS", "CLOUD_PROVIDER_GCP", "CLOUD_PROVIDER_OCI"}
        or not re.fullmatch(r"us-[a-z0-9-]{1,48}", os.environ.get("MODAL_REGION", ""))
        or not os.environ.get("MODAL_TASK_ID")
    ):
        raise ValueError("warmed snapshot placement differs")
    return manifest, _sha(raw)


def _event(phase, state, **fields):
    print(
        json.dumps(
            {"warmed_snapshot_stage": {"phase": phase, "state": state, **fields}}, sort_keys=True
        ),
        flush=True,
    )


@app.cls(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    startup_timeout=120,
    timeout=60,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    single_use_containers=True,
    region="us",
    routing_region="us-east",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/compiled": cache},
    include_source=True,
)
class WarmedSnapshot:
    @modal.enter(snap=True)
    def capture(self):
        manifest, self.manifest_sha256 = _manifest()
        capture_id = uuid.uuid4().hex
        if not claims.put("capture:0", capture_id, skip_if_exists=True):
            raise ValueError("warmed snapshot capture allowance exhausted")
        self.capture_id = capture_id
        self.capture_container_sha256 = _sha(os.environ["MODAL_TASK_ID"].encode())
        _event(
            "capture",
            "start",
            capture_id=capture_id,
            ordinal=0,
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
            self.imports_seconds = time.perf_counter() - started
            from klein_scene_runtime import KleinSceneRuntime

            _expiry(manifest)
            _event("runtime_load", "start", capture_id=capture_id)
            self.runtime = KleinSceneRuntime(Path("/models"))
            if self.runtime.identity != manifest["expected_identity"]:
                raise ValueError("runtime identity differs")
            _event("runtime_load", "end", capture_id=capture_id)
            _expiry(manifest)
            _event("cache_restore", "start", capture_id=capture_id)
            cache_seconds = self.runtime.compile(Path("/compiled") / CACHE_ID)
            _event("cache_restore", "end", capture_id=capture_id)
            warmup_started = time.perf_counter()
            # Only bounded hash/timing rows survive capture; generated image bytes are discarded.
            self.warmups = self._renders(manifest, "warmup", retain_images=False)
            self.initialization = {
                "model_load_seconds": self.runtime.load_seconds,
                "cache_setup_seconds": cache_seconds,
                "warmup_seconds": time.perf_counter() - warmup_started,
            }
            _expiry(manifest)
        except Exception:
            _event("capture", "failed", capture_id=capture_id)
            raise RuntimeError("warmed snapshot capture failed") from None
        _event("capture", "end", capture_id=capture_id, imports_seconds=self.imports_seconds)

    @modal.enter(snap=False)
    def activate(self):
        _, manifest_sha256 = _manifest()
        if manifest_sha256 != self.manifest_sha256:
            raise ValueError("warmed snapshot manifest changed")
        import torch

        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_name(0) != self.runtime.identity["gpu"]
            or list(torch.cuda.get_device_capability(0)) != self.runtime.identity["capability"]
            or torch.version.cuda != self.runtime.identity["cuda"]
        ):
            raise ValueError("warmed snapshot restored GPU differs")
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
            raise ValueError("warmed snapshot activation differs")
        if not any(row["request_id"] == request_id for row in manifest["operations"]):
            raise ValueError("warmed snapshot request is not authorized")
        if not claims.put(f"request:{request_id}", True, skip_if_exists=True):
            raise ValueError("warmed snapshot request already claimed")
        try:
            samples = self._renders(
                manifest, "render", retain_images=True, request_sha256=_sha(request_id.encode())
            )
        except Exception:
            _event("cycle", "failed", request_sha256=_sha(request_id.encode()))
            raise RuntimeError("warmed snapshot cycle failed") from None
        _event("complete", "end", request_sha256=_sha(request_id.encode()))
        return {
            "request_id": request_id,
            "variant": "warmed_snapshot",
            "identity": self.runtime.identity,
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
                "initialization": self.initialization,
                "warmups": self.warmups,
            },
            "stages": {
                "imports_seconds": self.imports_seconds,
                "model_load_seconds": self.initialization["model_load_seconds"],
                "cache_setup_seconds": self.initialization["cache_setup_seconds"],
                "worker_seconds": time.perf_counter() - self.started,
                "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            },
            "samples": samples,
        }

    def _renders(self, manifest, phase, *, retain_images, request_sha256=None):
        rows = []
        for repeat in range(2):
            for case_index, case in enumerate(manifest["cases"]):
                _expiry(manifest)
                fields = {"capture_id": self.capture_id, "repeat": repeat, "case_index": case_index}
                if request_sha256 is not None:
                    fields.update(request_sha256=request_sha256, activation_id=self.activation_id)
                _event(phase, "start", **fields)
                metrics, master, depth = self.runtime.render(case["prompt"], case["seed"])
                if (
                    metrics["sequence_bucket"] != case["expected_bucket"]
                    or _sha(master) != case["master_sha256"]
                    or _sha(depth) != case["depth_sha256"]
                    or metrics["master_sha256"] != case["master_sha256"]
                    or metrics["depth_sha256"] != case["depth_sha256"]
                ):
                    raise ValueError("warmed snapshot render differs from reference")
                if retain_images:
                    rows.append(
                        {
                            "repeat": repeat,
                            "case_index": case_index,
                            "metrics": metrics,
                            "master": master,
                            "depth": depth,
                        }
                    )
                else:
                    rows.append(
                        {
                            "repeat": repeat,
                            "case_index": case_index,
                            "sequence_bucket": metrics["sequence_bucket"],
                            "total_seconds": metrics["total_seconds"],
                            "master_sha256": case["master_sha256"],
                            "depth_sha256": case["depth_sha256"],
                            "hashes_match": True,
                        }
                    )
                _event(phase, "end", **fields)
        return rows
