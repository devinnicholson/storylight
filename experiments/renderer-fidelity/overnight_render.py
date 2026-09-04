"""Finite comparison from captured Gemma contracts; original text stays local."""

import hashlib
import json
import time
from pathlib import Path

from klein_restart import app, run_cycle


@app.local_entrypoint(name="overnight_comparison")
def main(capture: str, output_dir: str):
    source = Path(capture)
    captured = json.loads(source.read_text())
    if captured["split"] not in {"development", "validation"}:
        raise ValueError("privacy/adversarial inputs must not be rendered")
    cases = []
    for row in captured["cases"]:
        if "failure" in row:
            continue
        pair = [
            {"id": f"{row['id']}-{kind}", "prompt": row[f"{kind}_prompt"], "seed": row["seed"]}
            for kind in ("full", "concise")
        ]
        cases.extend(pair if len(cases) % 4 == 0 else reversed(pair))
    if not 1 <= len(cases) <= 48:
        raise ValueError("expected a bounded original synthetic batch")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "capture_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
        "comparison": "same Klein model; accepted full contract vs opt-in concise contract",
        "planner": "actual captured Jetson Gemma",
        "automatic_retries": 0,
    }
    call = run_cycle.spawn("f305950a0fbb4ecf89acfb80a3990351", "restore", cases)
    report["call_id"] = call.object_id
    print(f"Recoverable call: {call.object_id}", flush=True)
    try:
        cycle, assets = call.get(timeout=420)
        report["cycle"] = cycle
        for name, data in assets.items():
            (destination / name).write_bytes(data)
        if "failure" in cycle:
            raise RuntimeError(str(cycle["failure"]))
        for row in cycle["samples"]:
            for role in ("master", "depth"):
                if (
                    hashlib.sha256((destination / row[f"{role}_file"]).read_bytes()).hexdigest()
                    != row[f"{role}_sha256"]
                ):
                    raise ValueError("artifact checksum mismatch")
        report["all_artifacts_verified"] = True
    except BaseException:
        call.cancel(terminate_containers=True)
        raise
    finally:
        report["whole_call_seconds"] = time.perf_counter() - started
        (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
