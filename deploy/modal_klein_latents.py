"""Finite capture/replay of identical Klein starting noise across two GPUs."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
MANIFEST = Path("/root/latent-manifest.json")
PHASE = (
    os.environ["BOOKFORGE_LATENT_PHASE"]
    if modal.is_local()
    else json.loads(MANIFEST.read_bytes())["phase"]
)
if PHASE not in ("capture", "replay"):
    raise ValueError("invalid latent experiment phase")
EXPERIMENT = f"klein-latents-{PHASE}-20260906-a"
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
SOURCES = ("klein_scene_runtime.py", "klein_latent_probe.py", "modal_klein_latents.py")
EVIDENCE = DEPLOY.parent / "benchmarks/renderer-latents-2026-09-06"

if modal.is_local():
    image = modal.Image.from_id(IMAGE_ID)
    for name in SOURCES[:-1]:
        image = image.add_local_file(DEPLOY / name, f"/root/{name}")
    image = image.add_local_file(EVIDENCE / PHASE / "manifest.json", str(MANIFEST))
    if PHASE == "replay":
        for index in range(2):
            image = image.add_local_file(
                EVIDENCE / "capture" / f"latent-{index}.bin", f"/root/latent-{index}.bin"
            )
else:
    image = None

app = modal.App(f"bookforge-klein-latents-{PHASE}")
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
volumes = (
    {"/compiled": modal.Volume.from_name("bookforge-klein-compile-cache-v1")}
    if PHASE == "capture"
    else {}
)


def require(condition):
    if not condition:
        raise ValueError("latent comparison differs from its authorization")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@app.function(
    image=image,
    gpu="L4" if PHASE == "capture" else "L40S",
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
    volumes=volumes,
    include_source=True,
)
@modal.concurrent(max_inputs=1)
def compare():
    require(MANIFEST.stat().st_size <= 32768)
    value = json.loads(MANIFEST.read_bytes())
    require(value["schema_version"] == 1 and value["status"] == "authorized")
    require(value["phase"] == PHASE and value["experiment_id"] == EXPERIMENT)
    require(value["image_id"] == IMAGE_ID and value["maximum_calls"] == 1)
    require(
        type(value["expires_at"]) is int and time.time() < value["expires_at"] <= time.time() + 7200
    )
    require(value["cache_id"] == (CACHE_ID if PHASE == "capture" else None))
    require(
        digest(value["cases"]) == "0b2b218049531ffee74450515164bfbbd9c727526e9e7990631b599e47c6fbe4"
    )
    require(
        digest(value["expected_identity"])
        == {
            "capture": "91f975f485ed30aba6b85f252be21f183bab402aac441804641649bcf3f6a3f3",
            "replay": "893db078c92f3299b646d3ccafe1e498922ec8cf9b0d7d8f82ef455681472ca3",
        }[PHASE]
    )
    for name in SOURCES:
        require(hashlib.sha256((DEPLOY / name).read_bytes()).hexdigest() == value["sources"][name])
    require(os.environ.get("MODAL_REGION", "").startswith("us-"))
    require(os.environ.get("MODAL_CLOUD_PROVIDER", "").startswith("CLOUD_PROVIDER_"))
    require(os.environ.get("MODAL_TASK_ID"))
    artifacts = []
    require(type(value["artifacts"]) is list)
    if PHASE == "capture":
        require(value["artifacts"] == [])
    else:
        require(len(value["artifacts"]) == 2)
        from klein_latent_probe import validate_artifact

        for index, metadata in enumerate(value["artifacts"]):
            path = Path(f"/root/latent-{index}.bin")
            require(path.stat().st_size == 589824)
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == metadata["sha256"])
            artifacts.append(
                validate_artifact({**metadata, "data": data}, value["cases"][index], index)
            )
    require(claims.put("comparison", True, skip_if_exists=True))
    started = time.perf_counter()
    from klein_latent_probe import run_capture, run_replay
    from klein_scene_runtime import KleinSceneRuntime

    runtime = KleinSceneRuntime(Path("/models"))
    require(runtime.identity == value["expected_identity"])
    cache_seconds = runtime.compile(Path("/compiled") / CACHE_ID if PHASE == "capture" else None)
    if PHASE == "capture":
        comparison = run_capture(runtime, value["cases"])
    else:
        comparison = run_replay(runtime, value["cases"], artifacts)
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
