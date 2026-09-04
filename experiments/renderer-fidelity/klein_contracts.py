"""Synthetic-only full-versus-concise contracts through the actual TensorRT parser.

No model-planner accuracy claim: the four-slot answers are authored fixtures.
Uses the private cache from the fresh-container test in one bounded L4 call.
"""

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from klein_restart import app, run_cycle
from modal_compare import PROMPTS, STYLE


def contract_cases():
    from bookforge.live_scene_planner import validate_live_scene_plan_privacy
    from bookforge.tensorrt_slot_client import _semantic_privacy_separator, tensor_slot_wire_plan

    fixtures = [
        ("calm indigo pond", "one golden paper boat", "floating on the pond", "crescent moon"),
        ("tall cedar trees", "one silver fox", "standing left of a golden lantern", "soft mist"),
        ("calm sea", "one owl", "perching on a branch", "lighthouse far to the left of the owl"),
        ("snowy forest", "one red fox", "carrying a golden lantern in its mouth", "falling snow"),
        ("calm blue pond", "exactly two red paper boats", "floating side by side", "soft mist"),
        (
            "wooden bridge",
            "one child",
            "standing on bridge holding an open green book in both hands",
            "no other people",
        ),
    ]
    cases = []
    for index, values in enumerate(fixtures):
        slots = "\n".join(
            f"{label}: {value}"
            for label, value in zip(("SETTING", "ACTOR", "ACTION", "MAGIC"), values, strict=True)
        )
        wire = tensor_slot_wire_plan(slots, source_text=PROMPTS[index])
        plan = wire.to_live_scene_plan(context_text=PROMPTS[index])
        page = plan.to_page(
            source_text=PROMPTS[index], visual_style=STYLE.strip(), seed=20260903 + index
        )
        subject = _semantic_privacy_separator(
            f"{wire.focus.subject}, {wire.focus.action}", source_text=PROMPTS[index]
        )
        detail_label = (
            "Constraint" if re.match(r"(?i)^(no|without)\b", wire.magic.prompt) else "Detail"
        )
        concise = (
            f"{STYLE}Setting: {wire.background_prompt}. Subject: {subject}. "
            f"{detail_label}: {wire.magic.prompt}. One continuous scene, full-bleed, no text."
        )
        # Check the complete assembled prompt too, including clause boundaries.
        validate_live_scene_plan_privacy(
            plan.model_copy(update={"art_direction": concise}), source_text=PROMPTS[index]
        )
        pair = [
            {
                "id": f"full-{index}",
                "prompt": page.scene_spec.master_prompt,
                "seed": 20260903 + index,
            },
            {"id": f"concise-{index}", "prompt": concise, "seed": 20260903 + index},
        ]
        cases.extend(pair if index % 2 == 0 else reversed(pair))
    return cases


@app.local_entrypoint(name="contract_comparison")
def main(output_dir: str, cache_run_id: str):
    if uuid.UUID(cache_run_id).hex != cache_run_id:
        raise ValueError("cache run ID must be a UUID hex value")
    cases = contract_cases()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_run_id": cache_run_id,
        "cases": cases,
        "planner": "authored slots through production TensorRT parser; no model inference",
        "automatic_retries": 0,
    }
    started = time.perf_counter()
    call = run_cycle.spawn(cache_run_id, "restore", cases)
    report["call_id"] = call.object_id
    print(f"Recoverable function call: {call.object_id}", flush=True)
    try:
        cycle, assets = call.get(timeout=420)
        report["cycle"] = cycle
        for filename, content in assets.items():
            (destination / filename).write_bytes(content)
        if "failure" in cycle:
            raise RuntimeError(f"contract comparison failed: {cycle['failure']}")
    except Exception as error:
        call.cancel(terminate_containers=True)
        report["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
        raise
    finally:
        report["whole_call_seconds"] = time.perf_counter() - started
        (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
