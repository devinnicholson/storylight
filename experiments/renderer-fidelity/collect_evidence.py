"""Collect this supervised experiment's outputs and compute descriptive statistics.

Does not make network requests or run models. Raw prompts here are synthetic test cases.
"""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path


def collect_model_run(source: Path, destination: Path) -> None:
    report = json.loads((source / "results.json").read_text())
    groups: dict[str, list[float]] = {}
    for row in report["samples"]:
        filename = row["filename"]
        if Path(filename).name != filename:
            raise ValueError("invalid model evidence filename")
        content = (source / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"model evidence checksum mismatch: {filename}")
        (destination / filename).write_bytes(content)
        if not row["first_inference"]:
            key = f"{row['model']}-seq{row.get('max_sequence_length', 'native')}"
            groups.setdefault(key, []).append(row["inference_seconds"])
    shutil.copyfile(source / "results.json", destination / "results.json")
    summary = {
        "boundary": "warm image inference only, excluding depth, network, and model loading",
        "groups": {
            name: {
                "samples": len(values),
                "median_seconds": statistics.median(values),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
            }
            for name, values in groups.items()
        },
        "general_accuracy_claim": False,
        "production_promoted": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model-run", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    if args.model_run:
        collect_model_run(args.source, args.destination)
        return
    manifest = {}
    names = [
        "blind-fox-v1.json",
        "blind-fox-v2.json",
        "blind-fox-v3.json",
        "blind-holdout-v3.json",
        "question-v1.json",
        "holdout-cases.json",
        "renderer-baseline.json",
        "renderer-direct.json",
        "renderer-direct-v2.json",
        "scene-a.jpg",
        "scene-b.jpg",
        "scene-c.jpg",
    ]
    for name in names:
        source = args.source / name
        shutil.copyfile(source, args.destination / name)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    renderers = {}
    for name in ["renderer-baseline", "renderer-direct-v2"]:
        payload = json.loads((args.source / f"{name}.json").read_text())
        rows = payload["samples"]
        renderers[name] = {
            "cold_request_ms": rows[0]["wall_ms"],
            "cold_samples": 1,
            "warm_samples": len(rows) - 1,
            "warm_median_ms": statistics.median(row["wall_ms"] for row in rows[1:]),
            "warm_maximum_ms": max(row["wall_ms"] for row in rows[1:]),
            "load_stages": rows[0]["load_stages"],
        }
    baseline = json.loads((args.source / "renderer-baseline.json").read_text())["samples"]
    direct = json.loads((args.source / "renderer-direct-v2.json").read_text())["samples"]
    renderers["all_six_master_depth_pairs_byte_identical"] = all(
        before[role + "_sha256"] == after[role + "_sha256"]
        for before, after in zip(baseline, direct, strict=True)
        for role in ["master", "depth"]
    )
    reviews = {}
    for name in ["blind-fox-v2", "blind-fox-v3", "blind-holdout-v3"]:
        rows = json.loads((args.source / f"{name}.json").read_text())["results"]
        policies = {}
        for policy in ["baseline", "candidate"]:
            predictions = [
                row["baseline"]["verdict"]["decision"] == "accept"
                if policy == "baseline"
                else row["candidate"]["m"]
                for row in rows
            ]
            policies[policy] = {
                "cases": len(rows),
                "correct": sum(
                    p == row["expected_accept"] for p, row in zip(predictions, rows, strict=True)
                ),
                "false_accepts": sum(
                    p and not row["expected_accept"]
                    for p, row in zip(predictions, rows, strict=True)
                ),
                "false_rejects": sum(
                    not p and row["expected_accept"]
                    for p, row in zip(predictions, rows, strict=True)
                ),
                "median_ms": statistics.median(
                    row["baseline"]["latency_ms"] if policy == "baseline" else row["latency_ms"]
                    for row in rows
                ),
            }
        reviews[name] = policies
    result = {
        "schema_version": "1.0",
        "renderer": renderers,
        "reviews": reviews,
        "file_sha256": manifest,
        "limitations": [
            "Only one cold startup per loader; no production p95 claim.",
            "Review experiments are small diagnostic sets, not general accuracy estimates.",
            "Direct v1 failed before inference due to the occupied one-GPU quota.",
            "None of the alternative review prompts was promoted.",
        ],
    }
    (args.destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
