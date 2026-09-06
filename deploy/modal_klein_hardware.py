"""One finite L40S comparison with unchanged Klein weights and fresh regional compilation."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
EXPERIMENT = "klein-hardware-20260906-c"
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = None
MANIFEST = Path("/root/hardware-manifest.json")
SOURCES = (
    "klein_scene_runtime.py",
    "klein_hardware_probe.py",
    "modal_klein_hardware.py",
)

if modal.is_local():
    image = modal.Image.from_id(IMAGE_ID)
    for name in SOURCES[:-1]:
        image = image.add_local_file(DEPLOY / name, f"/root/{name}")
    image = image.add_local_file(
        DEPLOY.parent / "benchmarks/renderer-hardware-2026-09-06/retry-c/manifest.json",
        str(MANIFEST),
    )
else:
    image = None

app = modal.App("bookforge-klein-hardware")
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)


def require(condition):
    if not condition:
        raise ValueError("hardware comparison differs from its authorization")


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
        == "893db078c92f3299b646d3ccafe1e498922ec8cf9b0d7d8f82ef455681472ca3"
    )
    for name in SOURCES:
        require(hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest() == value["sources"][name])
    require(os.environ.get("MODAL_CLOUD_PROVIDER", "").startswith("CLOUD_PROVIDER_"))
    require(os.environ.get("MODAL_REGION", "").startswith("us-"))
    require(os.environ.get("MODAL_TASK_ID"))
    return value


@app.function(
    image=image,
    gpu="L40S",
    cpu=(8, 8),
    memory=(32768, 65536),
    timeout=120,
    startup_timeout=30,
    retries=0,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
    region="us",
    routing_region="us-east",
    include_source=True,
)
@modal.concurrent(max_inputs=1)
def compare():
    value = configuration()
    require(claims.put("comparison", True, skip_if_exists=True))
    started = time.perf_counter()
    print(json.dumps({"hardware_initialization": "start"}), flush=True)
    from klein_hardware_probe import run_comparison
    from klein_scene_runtime import KleinSceneRuntime

    runtime = KleinSceneRuntime(Path("/models"))
    require(runtime.identity == value["expected_identity"])
    cache_seconds = runtime.compile()
    print(
        json.dumps(
            {
                "hardware_initialization": "complete",
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
