"""Three fresh L4 containers: CPU-baked weights and portable compiler-cache reuse.

No public endpoint, GPU snapshots, automatic retries, or changes to live routing.
Each GPU call is bounded to 300 seconds and runs in a single-use container.
"""

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import modal
from klein_weights import bake_weights
from modal_compare import PROMPTS, STYLE, SUFFIX
from modal_compare import image as base_image

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "deploy" if modal.is_local() else Path("/root")
sys.path.insert(0, str(RUNTIME_DIR))
from klein_scene_runtime import DEPTH_MODEL, DEPTH_REVISION, MODEL, MODEL_REVISION  # noqa: E402

WEIGHTS = {"klein": [MODEL, MODEL_REVISION], "depth": [DEPTH_MODEL, DEPTH_REVISION]}
app = modal.App("storylight-klein-restart-comparison")
cache = modal.Volume.from_name("storylight-klein-compile-cache-v1", create_if_missing=True)


image = (
    base_image.add_local_file(
        Path(__file__).with_name("klein_weights.py"), "/root/klein_weights.py", copy=True
    )
    .run_function(bake_weights, args=(WEIGHTS,), cpu=4, memory=16384, timeout=300)
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_file(RUNTIME_DIR / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
    .add_local_file(Path(__file__).with_name("modal_compare.py"), "/root/modal_compare.py")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=300,
    startup_timeout=120,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
    volumes={"/compiled": cache},
)
def run_cycle(run_id: str, phase: str, cases: list[dict]):
    from klein_scene_runtime import KleinSceneRuntime

    if phase not in {"build", "restore"} or uuid.UUID(run_id).hex != run_id:
        raise ValueError("invalid experiment identity")
    started = time.perf_counter()
    report = {
        "phase": phase,
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "process_uuid": uuid.uuid4().hex,
        "network_weights_allowed": False,
        "samples": [],
    }
    assets = {}
    try:
        if json.loads(Path("/models/identities.json").read_text()) != WEIGHTS:
            raise ValueError("baked weights identity mismatch")
        runtime = KleinSceneRuntime(Path("/models"))
        report.update(identity=runtime.identity, model_load_seconds=runtime.load_seconds)
        print(json.dumps({"model_load_seconds": runtime.load_seconds, "phase": phase}), flush=True)
        cache_directory = Path("/compiled") / run_id
        report["cache_load_and_compile_setup_seconds"] = runtime.compile(
            cache_directory if phase == "restore" else None
        )
        for repeat in range(2):
            for case in cases:
                row, master, depth = runtime.render(case["prompt"], case["seed"])
                stem = f"{case['id']}-r{repeat}"
                row.update(
                    case=case["id"],
                    repeat=repeat,
                    master_file=f"{stem}.jpg",
                    depth_file=f"{stem}-depth.jpg",
                )
                report["samples"].append(row)
                assets[row["master_file"]] = master
                assets[row["depth_file"]] = depth
                print(json.dumps(row), flush=True)
        if phase == "build":
            report["cache"] = runtime.save_cache(cache_directory)
            cache.commit()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
    report["worker_seconds"] = time.perf_counter() - started
    return report, assets


def benchmark_cases() -> list[dict]:
    from storylight.live_scene_planner import (
        LiveSceneWireFocus,
        LiveSceneWireMagic,
        LiveSceneWirePlan,
    )

    cases = [
        {"id": f"short-{i}", "prompt": STYLE + prompt + SUFFIX, "seed": 20260903 + i}
        for i, prompt in enumerate(PROMPTS)
    ]
    contracts = [
        ("calm indigo pond", "one golden paper boat", "floating on the pond", "crescent moon"),
        ("tall cedar trees", "one silver fox", "standing left of a golden lantern", "soft mist"),
        ("calm sea", "one owl", "perching on a branch", "lighthouse far to the left of the owl"),
        ("snowy forest", "one red fox", "carrying a golden lantern in its mouth", "falling snow"),
        ("calm blue pond", "exactly two red paper boats", "floating side by side", "soft mist"),
        (
            "wooden bridge",
            "one child",
            "holding an open green book in both hands",
            "no other people",
        ),
    ]
    for index, (background, subject, action, accent) in enumerate(contracts):
        wire = LiveSceneWirePlan(
            background_prompt=background,
            focus=LiveSceneWireFocus(kind="character", subject=subject, action=action),
            magic=LiveSceneWireMagic(kind="effect", prompt=accent),
        )
        plan = wire.privacy_sanitized(source_text=PROMPTS[index]).to_live_scene_plan(
            context_text=PROMPTS[index]
        )
        page = plan.to_page(
            source_text=PROMPTS[index], visual_style=STYLE.strip(), seed=20260903 + index
        )
        cases.append(
            {
                "id": f"contract-{index}",
                "prompt": page.scene_spec.master_prompt,
                "seed": 20260903 + index,
            }
        )
    return cases


@app.local_entrypoint()
def main(output_dir: str):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    cases = benchmark_cases()
    report = {
        "run_id": uuid.uuid4().hex,
        "automatic_retries": 0,
        "single_use_containers": True,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "cycles": [],
    }
    try:
        for index, phase in enumerate(("build", "restore", "restore")):
            started = time.perf_counter()
            call = run_cycle.spawn(report["run_id"], phase, cases)
            print(f"Recoverable function call: {call.object_id}", flush=True)
            try:
                cycle, assets = call.get(timeout=420)
            except Exception:
                call.cancel(terminate_containers=True)
                raise
            cycle.update(call_id=call.object_id, whole_call_seconds=time.perf_counter() - started)
            report["cycles"].append(cycle)
            directory = destination / f"cycle-{index}"
            directory.mkdir()
            for filename, content in assets.items():
                (directory / filename).write_bytes(content)
            (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            if "failure" in cycle:
                raise RuntimeError(f"cycle failed: {cycle['failure']}")
        identities = {cycle["container_id"] for cycle in report["cycles"]}
        if None in identities or len(identities) != 3:
            raise RuntimeError("fresh containers were not established")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
        raise
    finally:
        (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
