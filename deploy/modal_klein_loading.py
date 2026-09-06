"""Four fresh-process loading comparisons; inference and compiler artifacts stay fixed."""

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
EXPERIMENT = "klein-loading-20260905-a"
CASES_SHA256 = "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
IDENTITY_SHA256 = "91f975f485ed30aba6b85f252be21f183bab402aac441804641649bcf3f6a3f3"
MANIFEST = Path("/root/loading-manifest.json")
MODELS = Path("/models")
COMPILED = Path("/compiled") / CACHE_ID
SCHEDULE = ("baseline", "candidate", "candidate", "baseline")

if modal.is_local():
    image = (
        modal.Image.from_id(IMAGE_ID)
        .add_local_file(DEPLOY / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
        .add_local_file(
            DEPLOY.parent / "benchmarks/renderer-loading-2026-09-05/manifest.json", str(MANIFEST)
        )
    )
else:
    image = None

app = modal.App("bookforge-klein-loading")
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition):
    if not condition:
        raise ValueError("loading experiment validation failed")


def shard_inventory():
    inventory = {}
    for component in ("klein/transformer", "klein/text_encoder", "klein/vae", "depth"):
        directory = MODELS / component
        indexes = sorted(directory.glob("*.safetensors.index.json"))
        require(len(indexes) <= 1)
        shards = set()
        for index in indexes:
            require(index.stat().st_size <= 2_000_000)
            weights = json.loads(index.read_bytes())["weight_map"]
            require(isinstance(weights, dict) and bool(weights))
            shards.update(weights.values())
        require(all(isinstance(name, str) and Path(name).name == name for name in shards))
        require(all((directory / name).is_file() for name in shards))
        inventory[component] = {
            "index_count": len(indexes),
            "indexed_shards": len(shards),
            "safetensors_files": len(list(directory.glob("*.safetensors"))),
            "referenced_bytes": sum((directory / name).stat().st_size for name in shards),
        }
    require(any(row["indexed_shards"] > 1 for row in inventory.values()))
    return inventory


def configuration(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 4)
    require(MANIFEST.stat().st_size <= 32_768)
    manifest = json.loads(MANIFEST.read_bytes())
    require(
        set(manifest)
        == {
            "schema_version",
            "status",
            "experiment_id",
            "expires_at",
            "image_id",
            "cache_id",
            "runtime_sha256",
            "deployment_sha256",
            "client_sha256",
            "expected_identity",
            "cases",
            "operations",
        }
    )
    expires = manifest["expires_at"]
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1)
    require(manifest["status"] == "authorized" and manifest["experiment_id"] == EXPERIMENT)
    require(manifest["image_id"] == IMAGE_ID and manifest["cache_id"] == CACHE_ID)
    require(type(expires) is int and time.time() < expires <= time.time() + 7200)
    require(digest(manifest["cases"]) == CASES_SHA256)
    require(digest(manifest["expected_identity"]) == IDENTITY_SHA256)
    for key, path in (
        ("runtime_sha256", DEPLOY / "klein_scene_runtime.py"),
        ("deployment_sha256", Path(__file__)),
    ):
        require(file_hash(path) == manifest[key])
    operations = [
        {
            "ordinal": index,
            "request_id": hashlib.sha256(f"{EXPERIMENT}:{index}".encode()).hexdigest()[:32],
            "variant": variant,
        }
        for index, variant in enumerate(SCHEDULE)
    ]
    require(manifest["operations"] == operations)
    require(os.environ.get("MODAL_CLOUD_PROVIDER") == "CLOUD_PROVIDER_AWS")
    require(os.environ.get("MODAL_REGION") == "us-east-1" and bool(os.environ.get("MODAL_TASK_ID")))
    require(
        not any(name in sys.modules for name in ("torch", "diffusers", "transformers", "triton"))
    )
    cache_manifest = json.loads((COMPILED / "manifest.json").read_bytes())
    require(cache_manifest["identity"] == manifest["expected_identity"])
    require(file_hash(COMPILED / "artifacts.bin") == cache_manifest["sha256"])
    return manifest


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
def loading_cycle(ordinal: int):
    started = time.perf_counter()
    manifest = configuration(ordinal)
    operation = manifest["operations"][ordinal]
    request_id, variant = operation["request_id"], operation["variant"]
    require(ordinal == 0 or claims.get(f"completed:{ordinal - 1}", False) is True)
    require(claims.put(request_id, True, skip_if_exists=True))
    environment = {
        "HF_ENABLE_PARALLEL_LOADING": "true" if variant == "candidate" else "false",
        "HF_PARALLEL_LOADING_WORKERS": "4",
    }
    os.environ.update(environment)
    inventory = shard_inventory()

    def peak_rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    def event(phase, state):
        print(
            json.dumps(
                {
                    "loading_stage": {
                        "ordinal": ordinal,
                        "variant": variant,
                        "phase": phase,
                        "state": state,
                        "elapsed_seconds": time.perf_counter() - started,
                        "process_peak_rss_gib": peak_rss(),
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

    require(time.time() < manifest["expires_at"])
    event("runtime_load", "start")
    runtime = KleinSceneRuntime(MODELS)
    event("runtime_load", "end")
    require(runtime.identity == manifest["expected_identity"])
    event("cache_restore", "start")
    cache_seconds = runtime.compile(COMPILED)
    event("cache_restore", "end")
    samples = []
    first_artifact_seconds = None
    for repeat in range(2):
        for case_index, case in enumerate(manifest["cases"]):
            require(time.time() < manifest["expires_at"])
            event("render", "start")
            metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            require(metrics["sequence_bucket"] == case["expected_bucket"])
            require(
                hashlib.sha256(master).hexdigest()
                == metrics["master_sha256"]
                == case["master_sha256"]
            )
            require(
                hashlib.sha256(depth).hexdigest() == metrics["depth_sha256"] == case["depth_sha256"]
            )
            if first_artifact_seconds is None:
                first_artifact_seconds = time.perf_counter() - started
            samples.append(
                {
                    "repeat": repeat,
                    "case_index": case_index,
                    "metrics": metrics,
                    "master": master,
                    "depth": depth,
                }
            )
            event("render", "end")
    require(claims.put(f"completed:{ordinal}", True, skip_if_exists=True))
    event("complete", "end")
    return {
        "request_id": request_id,
        "ordinal": ordinal,
        "variant": variant,
        "identity": runtime.identity,
        "loading_environment": environment,
        "shard_inventory": inventory,
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
            "first_artifact_seconds": first_artifact_seconds,
            "process_peak_rss_gib": peak_rss(),
        },
        "samples": samples,
    }
