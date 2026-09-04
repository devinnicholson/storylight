#!/usr/bin/env python3
"""Reproduce public-development scoring controls without invoking a model or network.

These controls measure representation and adapter limits; they are not learned accuracy.
Only aggregate scores and provenance hashes are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bookforge.fidelity_dataset import DATASET_ID, generate_split
from bookforge.fidelity_evaluation import SurfaceEvaluation, evaluate_surface
from bookforge.fidelity_graph_targets import derive_fidelity_graph_target
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from bookforge.live_scene_facts import adapt_live_scene_facts
from bookforge.scene_facts import SceneFactsGroundingError, SceneFactsPrivacyError, SceneFactsV2
from bookforge.tensorrt_slot_client import parse_tensor_graph_slots


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Control:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter(
            evaluated=0,
            schema_valid=0,
            exact=0,
            required_atoms=0,
            passed_atoms=0,
            privacy_failures=0,
            forbidden_hits=0,
            unsupported_concepts=0,
        )
        self.missed_slots: Counter[str] = Counter()

    def add(self, evaluation: SurfaceEvaluation) -> None:
        self.counts.update(
            evaluated=1,
            schema_valid=int(evaluation.schema_valid),
            exact=int(evaluation.exact_example_pass),
            required_atoms=evaluation.required_atoms,
            passed_atoms=evaluation.passed_atoms,
            privacy_failures=int(not evaluation.privacy.passed),
            forbidden_hits=len(evaluation.forbidden_hits),
            unsupported_concepts=len(evaluation.unsupported_concepts),
        )
        for atom in evaluation.expectation_results:
            if atom.required and not atom.passed:
                self.missed_slots[
                    atom.slot if atom.slot in {"SETTING", "ACTOR", "ACTION", "MAGIC"} else "other"
                ] += 1

    def report(self) -> dict[str, Any]:
        required = self.counts["required_atoms"]
        return {
            **self.counts,
            "semantic_atom_recall": self.counts["passed_atoms"] / required if required else None,
            "missed_required_slots": dict(sorted(self.missed_slots.items())),
        }


def compile_control(
    record: FidelityRecord, facts: SceneFactsV2, control: Control, refusals: Counter[str]
) -> None:
    try:
        prompt = facts.to_renderer_prompt(source_text=record.passage)
    except SceneFactsGroundingError:
        refusals["grounding"] += 1
    except SceneFactsPrivacyError:
        refusals["privacy"] += 1
    except ValueError:
        refusals["validation"] += 1
    else:
        control.add(evaluate_surface(record, {"master_prompt": prompt}, surface="renderer"))


def build_report() -> dict[str, Any]:
    records = tuple(generate_split(DatasetSplit.DEVELOPMENT))
    controls = {
        name: Control()
        for name in (
            "exact_target_raw",
            "literal_target_renderer",
            "typed_target",
            "native_compiled_target_renderer",
            "live_adapter_from_exact_target",
            "live_adapter_compiled_renderer",
        )
    }
    target_refusals: Counter[str] = Counter()
    native_compile_refusals: Counter[str] = Counter()
    adapter_refusals: Counter[str] = Counter()
    adapter_compile_refusals: Counter[str] = Counter()
    for record in records:
        raw = record.target.as_wire()
        controls["exact_target_raw"].add(evaluate_surface(record, raw, surface="raw"))
        controls["literal_target_renderer"].add(
            evaluate_surface(record, {"master_prompt": raw}, surface="renderer")
        )
        target = derive_fidelity_graph_target(record, token_budget=64)
        if target.facts is None:
            target_refusals[target.refusal.value] += 1
        else:
            controls["typed_target"].add(
                evaluate_surface(record, target.facts, surface="postprocessed")
            )
            compile_control(
                record,
                target.facts,
                controls["native_compiled_target_renderer"],
                native_compile_refusals,
            )
        adapted = adapt_live_scene_facts(parse_tensor_graph_slots(raw), source_text=record.passage)
        if adapted.facts is None:
            adapter_refusals[adapted.refusal.value] += 1
        else:
            controls["live_adapter_from_exact_target"].add(
                evaluate_surface(record, adapted.facts, surface="postprocessed")
            )
            compile_control(
                record,
                adapted.facts,
                controls["live_adapter_compiled_renderer"],
                adapter_compile_refusals,
            )
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/bookforge").glob("*.py"))]
    return {
        "schema_version": 1,
        "diagnostic": "live-scene-facts-scoring-positive-controls",
        "dataset_id": DATASET_ID,
        "split": "development",
        "total": len(records),
        "token_budget": 64,
        "hidden_evaluated": False,
        "network_calls": 0,
        "model_requests": 0,
        "learned_accuracy_measured": False,
        "dataset_sha256": digest([record.model_dump(mode="json") for record in records]),
        "implementation_sha256": digest(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
        ),
        "renderer_style_sha256": digest("luminous storybook illustration"),
        "controls": {name: control.report() for name, control in controls.items()},
        "target_refusals": dict(sorted(target_refusals.items())),
        "native_compile_refusals": dict(sorted(native_compile_refusals.items())),
        "adapter_refusals": dict(sorted(adapter_refusals.items())),
        "adapter_compile_refusals": dict(sorted(adapter_compile_refusals.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rendered = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
        return 0
    except Exception:
        print("scoring diagnostic failed; no private error detail retained")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
