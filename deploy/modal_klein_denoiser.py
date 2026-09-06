"""One finite, isolated comparison of Klein transformer graph replay."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
EXPERIMENT = "klein-denoiser-20260906-a"
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
MANIFEST = Path("/root/denoiser-manifest.json")
SOURCES = (
    "klein_scene_runtime.py",
    "klein_denoiser_graph.py",
    "klein_denoiser_probe.py",
    "modal_klein_denoiser.py",
)

if modal.is_local():
    image = modal.Image.from_id(IMAGE_ID)
    for name in SOURCES[:-1]:
        image = image.add_local_file(DEPLOY / name, f"/root/{name}")
    image = image.add_local_file(
        DEPLOY.parent / "benchmarks/renderer-denoiser-2026-09-06/manifest.json", str(MANIFEST)
    )
else:
    image = None

app = modal.App("bookforge-klein-denoiser")
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")


def require(condition):
    if not condition:
        raise ValueError("denoiser comparison differs from its authorization")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def configuration():
    require(MANIFEST.stat().st_size <= 32768)
    value = json.loads(MANIFEST.read_bytes())
    require(value["schema_version"] == 1 and value["status"] == "authorized")
    require(value["experiment_id"] == EXPERIMENT and value["image_id"] == IMAGE_ID)
    require(value["cache_id"] == CACHE_ID and value["maximum_calls"] == 1)
    require(
        type(value["expires_at"]) is int and time.time() < value["expires_at"] <= time.time() + 7200
    )
    require(
        digest(value["cases"]) == "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
    )
    require(
        digest(value["expected_identity"])
        == "91f975f485ed30aba6b85f252be21f183bab402aac441804641649bcf3f6a3f3"
    )
    for name in SOURCES:
        require(hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest() == value["sources"][name])
    require(os.environ.get("MODAL_CLOUD_PROVIDER") == "CLOUD_PROVIDER_AWS")
    require(os.environ.get("MODAL_REGION") == "us-east-1" and os.environ.get("MODAL_TASK_ID"))
    return value


@app.function(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    timeout=120,
    startup_timeout=30,
    retries=0,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
    cloud="aws",
    region="us-east",
    routing_region="us-east",
    volumes={"/compiled": cache},
    include_source=True,
)
@modal.concurrent(max_inputs=1)
def compare():
    value = configuration()
    require(claims.put("comparison", True, skip_if_exists=True))
    started = time.perf_counter()
    print(json.dumps({"denoiser_initialization": "start"}), flush=True)
    from klein_denoiser_probe import run_comparison
    from klein_scene_runtime import KleinSceneRuntime

    runtime = KleinSceneRuntime(Path("/models"))
    require(runtime.identity == value["expected_identity"])
    cache_seconds = runtime.compile(Path("/compiled") / CACHE_ID)
    print(
        json.dumps(
            {
                "denoiser_initialization": "complete",
                "model_load_seconds": runtime.load_seconds,
                "startup_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    require(time.time() < value["expires_at"])
    comparison = run_comparison(runtime, value["cases"])
    return {
        "identity": runtime.identity,
        **comparison,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "model_load_seconds": runtime.load_seconds,
        "cache_setup_seconds": cache_seconds,
        "worker_seconds": time.perf_counter() - started,
        "location": {
            "cloud": os.environ["MODAL_CLOUD_PROVIDER"],
            "region": os.environ["MODAL_REGION"],
            "container_sha256": hashlib.sha256(os.environ["MODAL_TASK_ID"].encode()).hexdigest(),
        },
    }
