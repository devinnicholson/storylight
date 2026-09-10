"""Offline V5 paired scoring with unchanged strict atoms and pinned current grounding."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORER = HERE / "support/base-evaluate.py"
SCORER_SHA256 = "85a4e6085bead918e1fc86480b30516de2834c152416f3f3c603630ffff0ea2d"
ARMS = ("old", "new")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scorer():
    require(digest(SCORER) == SCORER_SHA256, "strict scorer changed")
    spec = importlib.util.spec_from_file_location("v3_frozen_strict_scorer", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_records(gold, records):
    require(len(gold) == len({r["id"] for r in gold}) == 128, "gold case count")
    expected = {(row["id"], arm, rep) for row in gold for arm in ARMS for rep in range(2)}
    require(isinstance(records, list) and len(records) == 512, "incomplete paired experiment")
    for row in records:
        require(type(row["repetition"]) is int and row["repetition"] in (0, 1), "repetition")
        require(
            row["arm"] in ARMS and isinstance(row["output"], str) and len(row["output"]) <= 16384,
            "prediction fields",
        )
        elapsed = row["latency_ms"]
        require(
            type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
            "latency must be finite and positive",
        )
    keys = [(r["id"], r["arm"], r["repetition"]) for r in records]
    require(len(set(keys)) == 512 and set(keys) == expected, "duplicate or mismatched observations")
    by_key = dict(zip(keys, records, strict=True))
    strict = scorer()
    results = []
    for row in gold:
        for arm in ARMS:
            for rep in range(2):
                raw = by_key[row["id"], arm, rep]
                result = strict.score(row, raw["output"])
                results.append(
                    {**result, "arm": arm, "repetition": rep, "latency_ms": raw["latency_ms"]}
                )
    summaries = {}
    for arm in ARMS:
        observations = [r for r in results if r["arm"] == arm]
        case_results = []
        for row in gold:
            pair = [r for r in observations if r["id"] == row["id"]]
            case_results.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "expected": row["expected"],
                    "both_exact": all(r["exact"] for r in pair),
                    "both_refused": all(r["refused"] for r in pair),
                }
            )
        positive = [r for r in case_results if r["expected"] == "V2"]
        refusals = [r for r in case_results if r["expected"] == "REFUSE"]
        require(len(positive) == 96 and len(refusals) == 32, "gold class counts")
        summaries[arm] = {
            "positive_case_exact": sum(r["both_exact"] for r in positive),
            "correct_refusal_cases": sum(r["both_refused"] for r in refusals),
            "schema_valid_observations": sum(r["schema_valid"] for r in observations),
            "grounded_private_observations": sum(r["grounded_private"] for r in observations),
            "unexpected_admission_observations": sum(
                r["schema_valid"] and r["expected"] == "REFUSE" for r in observations
            ),
            "grounded_unexpected_admission_observations": sum(
                r["grounded_private"] and r["expected"] == "REFUSE" for r in observations
            ),
            "per_repetition": {
                str(rep): {
                    "positive_exact": sum(
                        r["exact"] and r["expected"] == "V2"
                        for r in observations
                        if r["repetition"] == rep
                    ),
                    "correct_refusals": sum(
                        r["refused"] and r["expected"] == "REFUSE"
                        for r in observations
                        if r["repetition"] == rep
                    ),
                }
                for rep in range(2)
            },
            "latency_median_ms": median(r["latency_ms"] for r in observations),
            "latency_max_ms": max(r["latency_ms"] for r in observations),
            "cases": case_results,
        }
    constrained = summaries["new"]
    gate = (
        constrained["positive_case_exact"] > summaries["old"]["positive_case_exact"]
        and constrained["correct_refusal_cases"] == 32
        and constrained["unexpected_admission_observations"] == 0
    )
    paired_reduction = [
        1
        - by_key[row["id"], "new", rep]["latency_ms"]
        / by_key[row["id"], "old", rep]["latency_ms"]
        for row in gold
        for rep in range(2)
    ]
    return {
        "arms": summaries,
        "results": results,
        "median_paired_fractional_latency_reduction": median(paired_reduction),
        "adapter_comparison_gate_passed": gate,
        "demo_preservation_review_required": True,
        "production_promotion": False,
    }


def evaluate(receipt):
    manifest = json.loads((HERE / "screen-manifest.json").read_text())
    for name in ("screen-inputs.jsonl", "screen-gold.jsonl"):
        require(digest(HERE / name) == manifest["files"][name]["sha256"], "screen changed")
    for name, sha in manifest["sources"].items():
        require(digest(ROOT / name) == sha, "validator source changed")
    require(digest(HERE / "protocol.json") == manifest["protocol_sha256"], "protocol changed")
    require(digest(Path(__file__)) == manifest["evaluator_sha256"], "evaluator changed")
    require(
        type(receipt["schema_version"]) is int and receipt["schema_version"] == 1, "receipt schema"
    )
    require(
        receipt["input_sha256"] == manifest["files"]["screen-inputs.jsonl"]["sha256"],
        "inference inputs changed",
    )
    require(
        isinstance(receipt["provenance"], dict)
        and isinstance(receipt["provenance"].get("implementation"), str)
        and bool(receipt["provenance"]["implementation"].strip()),
        "missing provenance",
    )
    gold = [json.loads(line) for line in (HERE / "screen-gold.jsonl").read_text().splitlines()]
    return {
        "schema_version": 1,
        "screen_manifest_sha256": digest(HERE / "screen-manifest.json"),
        "protocol_sha256": digest(HERE / "protocol.json"),
        "evaluator_sha256": digest(Path(__file__)),
        "strict_scorer_sha256": SCORER_SHA256,
        "provenance": receipt["provenance"],
        **score_records(gold, receipt["records"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(json.loads(args.predictions.read_text()))
    report["predictions_sha256"] = digest(args.predictions)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
