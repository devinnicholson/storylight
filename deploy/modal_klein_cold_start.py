"""Six single-use cold starts; only the guaranteed host-memory request differs."""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
EXPERIMENT = "klein-cold-start-20260905-a"
CASES_SHA256 = "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
APP_NAME = "bookforge-klein-cold-start"
MANIFEST = Path("/root/cold-start-manifest.json")

if modal.is_local():
    image = (
        modal.Image.from_id(IMAGE_ID)
        .add_local_file(DEPLOY / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
        .add_local_file(
            DEPLOY.parent / "benchmarks/renderer-cold-start-2026-09-05/manifest.json",
            str(MANIFEST),
        )
    )
else:
    image = None

app = modal.App(APP_NAME)
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")
resources = {
    "image": image,
    "gpu": "L4",
    "cpu": (8, 8),
    "timeout": 180,
    "startup_timeout": 120,
    "retries": 0,
    "min_containers": 0,
    "max_containers": 1,
    "scaledown_window": 2,
    "single_use_containers": True,
    "cloud": "aws",
    "region": "us-east",
    "routing_region": "us-east",
    "volumes": {"/compiled": cache},
}


def run_cycle(request_id, variant):
    started = time.perf_counter()
    manifest = json.loads(MANIFEST.read_text())
    expires = manifest.get("expires_at")
    if (
        manifest.get("status") != "authorized"
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("image_id") != IMAGE_ID
        or manifest.get("cache_id") != CACHE_ID
        or type(expires) is not int
        or not time.time() < expires <= time.time() + 7200
    ):
        raise ValueError("cold-start experiment authorization differs")
    for key, path in (
        ("runtime_sha256", DEPLOY / "klein_scene_runtime.py"),
        ("deployment_sha256", Path(__file__)),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("cold-start source identity differs")
    allowed = [
        row
        for row in manifest["operations"]
        if row["request_id"] == request_id and row["variant"] == variant
    ]
    if len(allowed) != 1 or len(manifest["operations"]) != 6:
        raise ValueError("cold-start request is not authorized")
    if (
        hashlib.sha256(
            json.dumps(manifest.get("cases"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != CASES_SHA256
    ):
        raise ValueError("cold-start cases differ from the two frozen references")
    if (
        os.environ.get("MODAL_CLOUD_PROVIDER") != "CLOUD_PROVIDER_AWS"
        or os.environ.get("MODAL_REGION") != "us-east-1"
    ):
        raise ValueError("cold-start placement differs")
    if not claims.put(request_id, True, skip_if_exists=True):
        raise ValueError("cold-start request already claimed")

    def peak_rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    def event(phase, state, **fields):
        print(
            json.dumps(
                {
                    "cold_start_stage": {
                        "request_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
                        "variant": variant,
                        "phase": phase,
                        "state": state,
                        "elapsed_seconds": time.perf_counter() - started,
                        "process_peak_rss_gib": peak_rss(),
                        **fields,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    event("framework_imports", "start")
    imports_started = time.perf_counter()
    import diffusers  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import triton  # noqa: F401
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline  # noqa: F401

    imports_seconds = time.perf_counter() - imports_started
    event("framework_imports", "end")
    from klein_scene_runtime import KleinSceneRuntime

    event("runtime_load", "start")
    runtime = KleinSceneRuntime(Path("/models"))
    event("runtime_load", "end")
    if runtime.identity != manifest["expected_identity"]:
        raise ValueError("cold-start runtime identity differs")
    event("cache_restore", "start")
    cache_seconds = runtime.compile(Path("/compiled") / CACHE_ID)
    event("cache_restore", "end")
    samples = []
    for repeat in range(2):
        for case_index, case in enumerate(manifest["cases"]):
            event("render", "start", repeat=repeat, case_index=case_index)
            metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            if metrics["sequence_bucket"] != case["expected_bucket"]:
                raise ValueError("cold-start token bucket differs")
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
    event("complete", "end")
    return {
        "request_id": request_id,
        "variant": variant,
        "identity": runtime.identity,
        "location": {
            "cloud": os.environ["MODAL_CLOUD_PROVIDER"],
            "compute_region": os.environ["MODAL_REGION"],
            "container_sha256": hashlib.sha256(os.environ["MODAL_TASK_ID"].encode()).hexdigest(),
        },
        "stages": {
            "imports_seconds": imports_seconds,
            "model_load_seconds": runtime.load_seconds,
            "cache_setup_seconds": cache_seconds,
            "worker_seconds": time.perf_counter() - started,
            "process_peak_rss_gib": peak_rss(),
        },
        "samples": samples,
    }


@app.function(**resources, memory=(65536, 65536))
def baseline_cycle(request_id: str):
    return run_cycle(request_id, "baseline")


@app.function(**resources, memory=(16384, 65536))
def candidate_cycle(request_id: str):
    return run_cycle(request_id, "candidate")
