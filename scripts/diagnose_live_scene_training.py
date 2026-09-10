#!/usr/bin/env python3
"""Trace the first 32 public training controls through the live scene pipeline.

Inputs are exact target slots, not model responses. Retain only hashes, fixed
stage labels, and scores so the diagnostic cannot be mistaken for learned accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path

import storylight
from storylight import fidelity_evaluation
from storylight.fidelity_dataset import generate_split
from storylight.fidelity_evaluation import evaluate_surface
from storylight.fidelity_schema import DatasetSplit
from storylight.live_scene_facts import adapt_live_scene_facts
from storylight.live_scene_planner import (
    LiveScenePlannerPrivacyError,
    validate_live_scene_plan_privacy,
)
from storylight.tensorrt_slot_client import (
    parse_tensor_graph_slots,
    tensor_accepted_graph_wire_plan,
    tensor_slot_wire_plan,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_report() -> dict:
    records = tuple(islice(generate_split(DatasetSplit.TRAIN), 32))
    totals: Counter[str] = Counter()
    cases = []
    for record in records:
        raw = record.target.as_wire()
        result = adapt_live_scene_facts(parse_tensor_graph_slots(raw), source_text=record.passage)
        case = {"record_sha256": record.sha256(), "adapter": "valid"}
        totals["cases"] += 1
        if result.facts is None:
            case["adapter"] = result.refusal.value
        else:
            score = evaluate_surface(record, result.facts, surface="postprocessed")
            totals["graph_valid"] += 1
            totals["graph_exact"] += int(score.exact_example_pass)
            totals["graph_required_atoms"] += score.required_atoms
            totals["graph_passed_atoms"] += score.passed_atoms
            prompt = result.facts.to_renderer_prompt(
                source_text=record.passage, visual_style="watercolor"
            )
            rendered = evaluate_surface(
                record,
                {
                    "master_prompt": prompt,
                    "scene_facts": result.facts.model_dump(),
                    "visual_style": "watercolor",
                },
                surface="renderer",
            )
            totals["verified_renderer_schema_valid"] += int(rendered.schema_valid)
            totals["verified_renderer_exact"] += int(rendered.exact_example_pass)
        totals["adapter_" + case["adapter"]] += 1
        stage = "accepted_wire"
        try:
            wire = tensor_slot_wire_plan(raw, source_text=record.passage)
            totals[stage + "_valid"] += 1
            stage = "accepted_plan"
            plan = wire.to_live_scene_plan(context_text=record.passage)
            totals[stage + "_valid"] += 1
            stage = "accepted_privacy"
            validate_live_scene_plan_privacy(plan, source_text=record.passage)
            totals[stage + "_valid"] += 1
            stage = "accepted_renderer"
            plan.to_page(source_text=record.passage, visual_style="watercolor", seed=0)
            totals[stage + "_valid"] += 1
            case["accepted"] = "valid"
        except (ValueError, LiveScenePlannerPrivacyError):
            case["accepted"] = stage + "_refused"
            totals[case["accepted"]] += 1
        try:
            candidate = tensor_accepted_graph_wire_plan(raw, source_text=record.passage)
            case["integrated"] = "graph" if candidate.scene_facts is not None else "fallback"
        except (ValueError, LiveScenePlannerPrivacyError):
            case["integrated"] = "refused"
        totals["integrated_" + case["integrated"]] += 1
        cases.append(case)
    source_dir = Path(storylight.__file__).resolve().parent
    return {
        "schema_version": 1,
        "diagnostic": "live-scene-first-32-training-controls",
        "evaluator_revision": getattr(fidelity_evaluation, "FIDELITY_EVALUATOR_REVISION", "legacy"),
        "selection": "first 32 generator records in training order, fixed before edits",
        "split": "train",
        "dataset_sha256": digest([record.sha256() for record in records]),
        "implementation_sha256": digest(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source_dir.glob("*.py"))
            }
        ),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_requests": 0,
        "hidden_evaluated": False,
        "development_evaluated": False,
        "learned_accuracy_measured": False,
        "graph_score_scope": "adapter-valid cases only; refusals are excluded from atom totals",
        "totals": dict(sorted(totals.items())),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
