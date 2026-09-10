"""Local strict diagnostics after independent runtime verification; no inference."""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

from storylight.scene_facts import SceneFactsV2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / "experiments/scene-adapter-v5-2026-09-09"
PROTOCOL_SHA = "1a2579008b49f64d2a2d45f00a2c4ed5c63c0621d1ca6accbf0637ff4f79bd69"
MANIFEST_SHA = "c6bf8fbc28b139b3ed46dceac7835a426004b2fe27cf1e95945dbfd22a783112"
GOLD_SHA = "ca8b08ef36fdb63c25b2d127977c4dac08cf6ca03b177682d360d929254a4c70"
SCORER_SHA = "85a4e6085bead918e1fc86480b30516de2834c152416f3f3c603630ffff0ea2d"
MODES = ("dynamic", "bridge-compile")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def pinned(path, sha):
    require(digest(path) == sha, "artifact pin")
    return json.loads(path.read_text())


def summarize(scores):
    positive = [r for r in scores if r["expected"] != "REFUSE"]
    refusal = [r for r in scores if r["expected"] == "REFUSE"]
    require(len(positive) == 96 and len(refusal) == 32, "diagnostic scope")
    return dict(
        positive_cases=96,
        positive_exact=sum(r["exact"] for r in positive),
        positive_valid_grounded=sum(r["grounded_private"] for r in positive),
        refusal_cases=32,
        literal_refusals=sum(r["refused"] for r in refusal),
        refusal_schema_admissions=sum(r["schema_valid"] for r in refusal),
        refusal_grounded_admissions=sum(r["grounded_private"] for r in refusal),
    )


def analyze(args):
    proof = pinned(args.verification, args.verification_sha256)
    require(
        proof["verified"] is True
        and proof["observations"] == 260
        and proof["measured_observations"] == 256
        and proof["measured_pairs"] == 128
        and proof["tokenizer_and_grammar_replayed"] is True
        and proof["held_out"] is False
        and proof["gold_read"] is False
        and proof["review_script_sha256"] == args.verifier_sha256,
        "independent verification required",
    )
    complete = pinned(args.directory / "completed.json", proof["completed_sha256"])
    require(not (args.directory / "failure.json").exists(), "failed execution")
    raw_path = args.directory / "raw.jsonl"
    require(
        digest(raw_path) == complete["files"]["raw.jsonl"] == proof["raw_sha256"],
        "verified raw bytes",
    )
    protocol = pinned(HERE / "protocol.json", PROTOCOL_SHA)
    inputs_path = HERE / "inputs.jsonl"
    require(digest(inputs_path) == protocol["input_sha256"], "input pin")
    inputs = list(map(json.loads, inputs_path.read_text().splitlines()))
    raw = list(map(json.loads, raw_path.read_text().splitlines()))
    schedule = [item for mode in MODES for item in protocol["schedule"][mode]]
    require(
        len(raw) == len(schedule) == 260
        and all(
            all(row[k] == v for k, v in dispatch.items())
            for row, dispatch in zip(raw, schedule, strict=True)
        ),
        "verified schedule",
    )
    require(
        all(
            row["terminated_with_eos"] is True
            and row["grammar_accepts_complete_tokens"] is True
            and len(row["prediction"].encode()) <= 4096
            for row in raw
        ),
        "complete outputs",
    )
    manifest = pinned(V5 / "screen-manifest.json", MANIFEST_SHA)
    for path, pin in manifest["sources"].items():
        require(digest(ROOT / path) == pin, "validator source changed")
    require(
        digest(Path(inspect.getsourcefile(SceneFactsV2)))
        == manifest["sources"]["src/storylight/scene_facts.py"],
        "imported validator",
    )
    scorer_path = ROOT / "experiments/scene-adapter-2026-09-08/evaluate.py"
    require(digest(scorer_path) == SCORER_SHA, "strict scorer changed")
    spec = importlib.util.spec_from_file_location("strict_cache_score", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    # Gold is local and joined only after the independent execution proof is checked.
    require(digest(args.gold) == GOLD_SHA, "gold pin")
    gold = {r["id"]: r for r in map(json.loads, args.gold.read_text().splitlines())}
    expected_ids = [r["id"] for r in inputs[2:]]
    require(
        len(expected_ids) == len(set(expected_ids)) == 128 and set(expected_ids) == set(gold),
        "gold scope",
    )
    by_mode = {
        mode: {r["id"]: r for r in raw if r["mode"] == mode and not r["warmup"]} for mode in MODES
    }
    scores = {
        mode: [scorer.score(gold[id_], by_mode[mode][id_]["prediction"]) for id_ in expected_ids]
        for mode in MODES
    }
    changes = []
    matched = 0
    for index, id_ in enumerate(expected_ids):
        a, b = (by_mode[mode][id_] for mode in MODES)
        identical = a["token_ids"] == b["token_ids"]
        matched += identical
        before, after = (scores[mode][index] for mode in MODES)
        if not identical or before != after:
            changes.append(
                dict(id=id_, identical_tokens=identical, dynamic=before, bridge_compile=after)
            )
    return dict(
        source_sha256=digest(Path(__file__)),
        independent_verification_sha256=args.verification_sha256,
        independent_verifier_sha256=args.verifier_sha256,
        completed_sha256=proof["completed_sha256"],
        raw_sha256=digest(raw_path),
        protocol_sha256=PROTOCOL_SHA,
        input_sha256=protocol["input_sha256"],
        gold_sha256=GOLD_SHA,
        scorer_sha256=SCORER_SHA,
        validator_sources=manifest["sources"],
        measured_pairs=128,
        identical_token_pairs=matched,
        modes={m: summarize(scores[m]) for m in MODES},
        changed_cases=changes,
        scores=scores,
        held_out=False,
        primary_gate_unchanged=True,
        production_promotion=False,
        scope="Fresh scores of this merged runtime; previously exposed synthetic cases",
        original_unmerged_scores_inherited=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "verification", "gold", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("verification-sha256", "verifier-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output exists")
    value = analyze(args)
    with args.output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value["modes"]))


if __name__ == "__main__":
    main()
