"""Isolated, authenticated SDK and HTTP transports for one frozen comparison."""

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
from klein_latency_protocol import digest, pack_response, validate_request  # noqa: E402
from klein_latency_runtime import LatencySceneRuntime  # noqa: E402

if modal.is_local():
    sys.path.insert(0, str(DEPLOY.parent / "experiments" / "renderer-fidelity"))
    from klein_restart import WEIGHTS
    from klein_weights import bake_weights
    from modal_compare import image as base_image

    # Match the qualified bake layer, then install the HTTP server before runtime mounts.
    image = (
        base_image.add_local_file(
            DEPLOY.parent / "experiments/renderer-fidelity/klein_weights.py",
            "/root/klein_weights.py",
            copy=True,
        )
        .run_function(bake_weights, args=(WEIGHTS,), cpu=4, memory=16384, timeout=300)
        .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        .uv_pip_install("uvicorn==0.35.0")
        .add_local_file(DEPLOY / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
    )
    for name in ("klein_latency_protocol.py", "klein_latency_runtime.py", "klein_latency_http.py"):
        image = image.add_local_file(DEPLOY / name, f"/root/{name}")
    image = image.add_local_file(
        DEPLOY.parent / "benchmarks/renderer-latency-2026-09-05/manifest.json",
        "/root/latency-manifest.json",
    )
else:
    image = None

sdk_app = modal.App("bookforge-klein-latency-sdk")
http_app = modal.App("bookforge-klein-latency-http")
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")
claims = modal.Dict.from_name("bookforge-klein-latency-20260905-claims", create_if_missing=True)
RESOURCES = dict(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    startup_timeout=120,
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
    volumes={"/compiled": cache},
)


class Runtime:
    def __init__(self, transport):
        self.manifest = json.loads(Path("/root/latency-manifest.json").read_text())
        if time.time() >= self.manifest["expires_at"]:
            raise RuntimeError("experiment expired")
        for field, name in (
            ("deployment_sha256", "modal_klein_latency.py"),
            ("protocol_sha256", "klein_latency_protocol.py"),
            ("http_server_sha256", "klein_latency_http.py"),
            ("runtime_sha256", "klein_scene_runtime.py"),
            ("instrumentation_sha256", "klein_latency_runtime.py"),
        ):
            if hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest() != self.manifest[field]:
                raise RuntimeError("deployed source identity differs")
        self.allowed = {
            row["request"]["request_id"]: digest(row["request"])
            for row in self.manifest["operations"]
            if row["transport"] == transport
        }
        self.transport = transport
        started = time.perf_counter()
        self.runtime = LatencySceneRuntime(Path("/models"))
        self.cache_seconds = self.runtime.compile(
            Path("/compiled/f305950a0fbb4ecf89acfb80a3990351")
        )
        self.startup_seconds = time.perf_counter() - started
        if self.runtime.identity != self.manifest["expected_identity"]:
            raise RuntimeError("renderer identity differs")
        if self.runtime.instrumentation_sha256 != self.manifest["instrumentation_sha256"]:
            raise RuntimeError("instrumentation identity differs")

    def claim(self, request_id):
        return claims.put(request_id, True, skip_if_exists=True)

    def invoke(self, request):
        started = time.perf_counter()
        validate_request(request)
        if time.time() >= self.manifest["expires_at"] or self.allowed.get(
            request["request_id"]
        ) != digest(request):
            raise ValueError("request not authorized")
        payload = {
            "request_id": request["request_id"],
            "identity": self.runtime.identity,
            "instrumentation_sha256": self.runtime.instrumentation_sha256,
            "deployment_sha256": self.manifest["deployment_sha256"],
            "model_load_seconds": self.runtime.load_seconds,
            "startup_seconds": self.startup_seconds,
            "cache_setup_seconds": self.cache_seconds,
            "location": {
                "compute_region": bounded_environment("MODAL_REGION"),
                "cloud": bounded_environment("MODAL_CLOUD_PROVIDER"),
                "container_sha256": hashlib.sha256(
                    os.environ.get("MODAL_TASK_ID", "unavailable").encode()
                ).hexdigest(),
                "routing_region": "us-east",
            },
        }
        if request["operation"] == "prewarm":
            payload.update(self.runtime.warmup(), master=b"", depth=b"")
        else:
            metrics, master, depth = self.runtime.render(request["prompt"], request["seed"])
            if metrics["sequence_bucket"] not in (128, 256):
                raise ValueError("unqualified token bucket")
            payload.update(
                metrics=metrics,
                master=master,
                depth=depth,
                warm_state="warm" if metrics["bucket_was_warm"] else "cold",
            )
        payload["server_seconds"] = time.perf_counter() - started
        return payload


def bounded_environment(name):
    value = os.environ.get(name, "")
    return value if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) else None


@sdk_app.cls(**RESOURCES, timeout=180, retries=0)
@modal.concurrent(max_inputs=1)
class LatencyStudio:
    @modal.enter()
    def load(self):
        self.worker = Runtime("sdk")

    @modal.method()
    def invoke(self, request: dict):
        validate_request(request)
        if self.worker.allowed.get(request["request_id"]) != digest(request):
            raise ValueError("request not authorized")
        if not self.worker.claim(request["request_id"]):
            raise ValueError("request already claimed")
        return pack_response(self.worker.invoke(request))


@http_app.server(
    **RESOURCES, port=8000, unauthenticated=False, exit_grace_period=0, routing_region="us-east"
)
class LatencyServer:
    @modal.enter()
    def load(self):
        import uvicorn
        from klein_latency_http import build_app

        worker = Runtime("http")
        application = build_app(
            handler=worker.invoke,
            claim=worker.claim,
            terminate=lambda: os._exit(70),
            allowed_requests=worker.allowed,
            expires_at=worker.manifest["expires_at"],
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                application,
                host="0.0.0.0",
                port=8000,
                access_log=False,
                log_level="critical",
                lifespan="off",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
