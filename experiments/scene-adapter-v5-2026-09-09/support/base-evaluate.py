"""Offline, strict scoring of frozen synthetic adapter predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from storylight.scene_facts import SceneFactsV2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_screen():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        if digest((HERE / name).read_bytes()) != entry["sha256"]:
            raise ValueError("dataset file changed: " + name)
    for path, pin in manifest["sources"].items():
        if digest((ROOT / path).read_bytes()) != pin:
            raise ValueError("validator source changed: " + path)
    for name, field in (("generate.py", "generator_sha256"), ("prompt.txt", "prompt_sha256")):
        if digest((HERE / name).read_bytes()) != manifest[field]:
            raise ValueError("generation contract changed: " + name)
    rows = (HERE / "screen-gold.jsonl").read_text().splitlines()
    return manifest, [json.loads(line) for line in rows]


def atoms(facts):
    """Reference/order neutral; literal phrases remain strict, without guessed synonyms."""
    value = facts.model_dump(mode="json")
    refs = {e["ref"]: (kind, e["label"], e["color"])
            for kind in ("subjects", "objects") for e in value[kind]}
    events = {e["ref"]: (refs[e["source"]], e["action"], refs.get(e["object"]))
              for e in value["events"]}
    result = []
    result.append(("setting", value["setting"]["label"]))
    result += [("setting_attribute", a) for a in value["setting"]["attributes"]]
    for kind in ("subjects", "objects"):
        for entity in value[kind]:
            ref = refs[entity["ref"]]
            result.append(("entity", ref))
            result.append(("count", ref, entity["count"]))
            for field in ("attributes", "actions", "states"):
                result += [(field, ref, item) for item in entity.get(field, [])]
    for field in ("relationships", "motions", "salience", "negatives", "events", "temporal_order"):
        for item in value[field]:
            rewritten = {}
            for key, data in item.items():
                if field == "events" and key == "ref":
                    continue
                if key in {"source", "target", "secondary_target", "destination", "object"}:
                    data = refs.get(data)
                if key in {"before", "after"}:
                    data = events[data]
                rewritten[key] = data
            result.append((field, rewritten))
    if value["transformation"] is not None:
        transformed = dict(value["transformation"])
        transformed["source"] = refs[transformed["source"]]
        result.append(("transformation", transformed))
    return {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in result}


def score(row, output):
    result = {"id": row["id"], "family": row["family"], "expected": row["expected"],
              "refused": output.strip() == "REFUSE", "schema_valid": False,
              "grounded_private": False, "exact": False, "matched_atoms": 0,
              "missing_atoms": 0, "extra_atoms": 0}
    expected = set() if row["facts"] is None else atoms(SceneFactsV2.model_validate(row["facts"]))
    result["required_atoms"] = len(expected)
    result["missing_atoms"] = len(expected)
    if result["refused"]:
        result["exact"] = row["expected"] == "REFUSE"
        return result
    try:
        facts = SceneFactsV2.from_wire(output.strip())
    except ValueError:
        return result
    result["schema_valid"] = True
    observed = atoms(facts)
    result.update(matched_atoms=len(expected & observed), missing_atoms=len(expected - observed),
                  extra_atoms=len(observed - expected))
    try:
        facts.to_renderer_prompt(source_text=row["source"], visual_style="rich watercolor")
        result["grounded_private"] = True
    except ValueError:
        pass
    result["exact"] = bool(expected) and observed == expected and result["grounded_private"]
    return result


def evaluate(receipt):
    manifest, gold = load_screen()
    if set(receipt) != {"schema_version", "system", "input_sha256", "provenance", "records"}:
        raise ValueError("prediction receipt fields")
    if (receipt["schema_version"] != 1
            or receipt["system"] not in {"base", "current_parser", "tuned"}):
        raise ValueError("prediction system")
    if receipt["input_sha256"] != manifest["files"]["screen-inputs.jsonl"]["sha256"]:
        raise ValueError("wrong inference inputs")
    provenance = receipt["provenance"]
    if not isinstance(provenance, dict) or not provenance.get("implementation"):
        raise ValueError("implementation provenance required")
    predictions = receipt["records"]
    if len(predictions) != len(gold) or {r["id"] for r in predictions} != {r["id"] for r in gold}:
        raise ValueError("missing, duplicate, or extra predictions")
    by_id = {r["id"]: r for r in predictions}
    results = []
    for row in gold:
        prediction = by_id[row["id"]]
        if set(prediction) != {"id", "output", "latency_ms"}:
            raise ValueError("prediction fields")
        elapsed = prediction["latency_ms"]
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid latency")
        output = prediction["output"]
        if not isinstance(output, str) or len(output) > 16384:
            raise ValueError("output bounds")
        result = score(row, output)
        result["latency_ms"] = elapsed
        results.append(result)
    return {
        "schema_version": 1, "system": receipt["system"], "provenance": provenance,
        "scope": "32 synthetic groups; literal graph contract, not general language accuracy",
        "manifest_sha256": digest((HERE / "manifest.json").read_bytes()),
        "protocol_sha256": digest((HERE / "protocol.json").read_bytes()),
        "evaluator_sha256": digest(Path(__file__).read_bytes()),
        "summary": {
            "cases": len(results), "exact": sum(r["exact"] for r in results),
            "positive_cases": sum(r["expected"] == "V2" for r in results),
            "positive_exact": sum(r["exact"] and r["expected"] == "V2" for r in results),
            "refusal_cases": sum(r["expected"] == "REFUSE" for r in results),
            "correct_refusals": sum(r["exact"] and r["expected"] == "REFUSE" for r in results),
            "unexpected_admissions": sum(
                r["schema_valid"] and r["expected"] == "REFUSE" for r in results),
            "latency_median_ms": median(r["latency_ms"] for r in results),
            "latency_max_ms": max(r["latency_ms"] for r in results),
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.predictions.read_text()))
    report["predictions_sha256"] = digest(args.predictions.read_bytes())
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
