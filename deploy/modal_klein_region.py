"""Region-controlled successor to the frozen Klein transport comparison."""

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
from klein_latency_protocol import digest, pack_response, validate_request  # noqa: E402
from klein_latency_runtime import LatencySceneRuntime  # noqa: E402
from modal_klein_latency import RESOURCES, Runtime  # noqa: E402
from modal_klein_latency import image as original_image  # noqa: E402

if modal.is_local():
    image = original_image.add_local_file(
        DEPLOY.parent / "benchmarks/renderer-region-2026-09-05/manifest.json",
        "/root/region-manifest.json",
    ).add_local_file(
        DEPLOY.parent / "src/bookforge/klein_region_client.py", "/root/klein_region_client.py"
    )
else:
    image = None

sdk_app = modal.App("bookforge-klein-region-sdk")
http_app = modal.App("bookforge-klein-region-http")
claims = modal.Dict.from_name("bookforge-klein-region-20260905-claims", create_if_missing=True)
resources = {**RESOURCES, "image": image, "cloud": "aws"}


def require_placement():
    if (
        os.environ.get("MODAL_CLOUD_PROVIDER") != "CLOUD_PROVIDER_AWS"
        or os.environ.get("MODAL_REGION") != "us-west-2"
    ):
        raise RuntimeError("compute placement does not match the comparison")


class RegionRuntime(Runtime):
    def __init__(self, transport):
        require_placement()
        self.manifest = json.loads(Path("/root/region-manifest.json").read_text())
        expires = self.manifest.get("expires_at")
        if (
            self.manifest.get("status") != "authorized"
            or type(expires) is not int
            or not time.time() < expires <= time.time() + 7200
        ):
            raise RuntimeError("region experiment is not authorized or has expired")
        if self.manifest.get("placement") != {
            "cloud": "aws",
            "compute_region": "us-west",
            "routing_region": "us-east",
            "expected_cloud": "CLOUD_PROVIDER_AWS",
            "expected_compute_region": "us-west-2",
        }:
            raise RuntimeError("region experiment placement differs")
        for field, name in (
            ("deployment_sha256", "modal_klein_latency.py"),
            ("protocol_sha256", "klein_latency_protocol.py"),
            ("http_server_sha256", "klein_latency_http.py"),
            ("runtime_sha256", "klein_scene_runtime.py"),
            ("instrumentation_sha256", "klein_latency_runtime.py"),
            ("region_deployment_sha256", "modal_klein_region.py"),
            ("region_client_sha256", "klein_region_client.py"),
        ):
            if hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest() != self.manifest[field]:
                raise RuntimeError("region experiment source identity differs")
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
        if (
            self.runtime.identity != self.manifest["expected_identity"]
            or self.runtime.instrumentation_sha256 != self.manifest["instrumentation_sha256"]
        ):
            raise RuntimeError("region renderer identity differs")

    def claim(self, request_id):
        return claims.put(request_id, True, skip_if_exists=True)

    def invoke(self, request):
        payload = super().invoke(request)
        payload["deployment_sha256"] = self.manifest["region_deployment_sha256"]
        return payload


@sdk_app.cls(**resources, region="us-west", routing_region="us-east", timeout=180, retries=0)
@modal.concurrent(max_inputs=1)
class RegionStudio:
    @modal.enter()
    def load(self):
        self.worker = RegionRuntime("sdk")

    @modal.method()
    def invoke(self, request: dict):
        validate_request(request)
        if self.worker.allowed.get(request["request_id"]) != digest(request):
            raise ValueError("request not authorized")
        if not self.worker.claim(request["request_id"]):
            raise ValueError("request already claimed")
        return pack_response(self.worker.invoke(request))


@http_app.server(
    **resources,
    compute_region="us-west",
    routing_region="us-east",
    port=8000,
    unauthenticated=False,
    exit_grace_period=0,
)
class RegionServer:
    @modal.enter()
    def load(self):
        import uvicorn
        from klein_latency_http import build_app

        worker = RegionRuntime("http")
        application = build_app(
            worker.invoke,
            worker.claim,
            lambda: os._exit(70),
            worker.allowed,
            worker.manifest["expires_at"],
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
