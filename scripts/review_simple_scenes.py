#!/usr/bin/env python3
"""Build and score a blind review of the fixed simple-scene comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_simple_scenes as benchmark  # noqa: E402
from scripts.build_fidelity_review import HTML  # noqa: E402

require = benchmark.require


def checks(control):
    return [
        f"{label}: {check['text']}"
        for label, group in (
            ("State 1", "before_checks"),
            ("State 2", "after_checks"),
            ("Across both states", "continuity_checks"),
        )
        for check in control[group]
    ]


def verified(args):
    manifest = benchmark.read_manifest(args.manifest)
    authorization = benchmark.read_authorization(
        args.authorization, args.authorization_sha256, args.manifest
    )
    summary = benchmark.aggregate(args, manifest, benchmark.header(args, manifest, authorization))
    require(summary["complete"] and summary["artifacts_verified"] == 24)
    return manifest, summary


def build(args):
    manifest, summary = verified(args)
    gallery, key = args.gallery.absolute(), args.key.absolute()
    require(gallery.resolve() == gallery and key.resolve() == key)
    require(not gallery.exists() and not key.exists() and gallery not in key.parents)
    require(gallery != key and key.parent.is_dir())
    require(key.parent.stat().st_uid == os.getuid())
    require(stat.S_IMODE(key.parent.stat().st_mode) == 0o700)
    journal = (args.output / "journal.jsonl").read_bytes()
    require(hashlib.sha256(journal).hexdigest() == summary["journal_sha256"])
    hashes = {
        row["ordinal"]: row["payload"]["metrics"]["master_sha256"]
        for row in (json.loads(line) for line in journal.splitlines())
        if row.get("kind") == "result"
    }
    public = {"schema_version": 1, "review_id": secrets.token_hex(16), "pages": []}
    mapping = {
        "review_id": public["review_id"],
        "manifest_sha256": benchmark.legacy.file_hash(args.manifest),
        "journal_sha256": summary["journal_sha256"],
        "pages": [],
    }
    assets = {}
    for index, control in enumerate(benchmark.fixture()["controls"]):
        group = manifest["operations"][index * 3 : index * 3 + 3]
        base, *variants = group
        secrets.SystemRandom().shuffle(variants)
        options, private_options = [], []
        for label, variant in zip("AB", variants, strict=True):
            frames = []
            for operation in (base, variant):
                content = benchmark.read_artifact(
                    benchmark.artifact(args.output, operation["ordinal"], "master"),
                    hashes[operation["ordinal"]],
                )
                name = f"frame-{hashlib.sha256(content).hexdigest()}.jpg"
                assets[name] = content
                frames.append(name)
            options.append({"label": label, "frames": frames})
            private_options.append({"label": label, "variant": variant["variant"]})
        public["pages"].append(
            {
                "page": index + 1,
                "states": 2,
                "required": checks(control),
                "forbidden": [],
                "options": options,
            }
        )
        mapping["pages"].append({"page": index + 1, "options": private_options})
    mapping["review_data_sha256"] = benchmark.legacy.protocol.digest(public)
    benchmark.legacy.write_exclusive(key, benchmark.preparation.encoded(mapping))
    gallery.mkdir(mode=0o700)
    for name, content in assets.items():
        benchmark.legacy.write_exclusive(gallery / name, content)
    benchmark.legacy.write_exclusive(
        gallery / "review-data.json", benchmark.preparation.encoded(public)
    )
    benchmark.legacy.write_exclusive(gallery / "index.html", HTML.encode())


def read_json(path):
    return benchmark.legacy.protocol.decode_json(benchmark.preparation.read_code(path))


def score(args):
    manifest, summary = verified(args)
    public = read_json(args.gallery / "review-data.json")
    mapping, review = read_json(args.key), read_json(args.review)
    require(mapping["manifest_sha256"] == benchmark.legacy.file_hash(args.manifest))
    require(mapping["journal_sha256"] == summary["journal_sha256"])
    require(mapping["review_data_sha256"] == benchmark.legacy.protocol.digest(public))
    require(set(review) == {"schema_version", "review_id", "pages"})
    require(type(review["schema_version"]) is int and review["schema_version"] == 1)
    require(review["review_id"] == public["review_id"] == mapping["review_id"])
    require(len(review["pages"]) == len(public["pages"]) == len(mapping["pages"]) == 4)
    counts = {
        variant: {"correct": 0, "incorrect": 0, "unrated": 0}
        for variant in ("text_next", "reference_next")
    }
    annotations_complete = True
    outcomes = []
    for index, (ratings, page, private) in enumerate(
        zip(review["pages"], public["pages"], mapping["pages"], strict=True)
    ):
        control = benchmark.fixture()["controls"][index]
        outcome = {"control_id": control["id"], "variants": {}}
        require(page["required"] == checks(control) and page["forbidden"] == [])
        require(set(ratings) == {"page", "options"})
        require(
            type(ratings["page"]) is int
            and ratings["page"] == page["page"] == private["page"] == index + 1
        )
        require(len(ratings["options"]) == len(page["options"]) == len(private["options"]) == 2)
        require({o["variant"] for o in private["options"]} == set(counts))
        base_ratings = [
            value
            for option in ratings["options"]
            for value in option["required"][: len(control["before_checks"])]
        ]
        base_wrong = "missing" in base_ratings
        base_clear = all(value == "visible" for value in base_ratings)
        for label, rating, option, hidden in zip(
            "AB", ratings["options"], page["options"], private["options"], strict=True
        ):
            require(
                set(rating)
                == {"label", "correct", "legible", "continuity", "rating", "required", "forbidden"}
            )
            require(rating["label"] == option["label"] == hidden["label"] == label)
            require(rating["correct"] in {"", "correct", "incorrect", "unsure"})
            require(rating["legible"] in {"", "clear", "unclear"})
            require(rating["continuity"] in {"", "consistent", "inconsistent", "unsure"})
            require(rating["rating"] in {"", "1", "2", "3", "4", "5"})
            require(rating["forbidden"] == [] and isinstance(rating["required"], list))
            require(len(rating["required"]) == len(page["required"]))
            require(all(v in {"", "visible", "missing", "unsure"} for v in rating["required"]))
            annotations_complete = annotations_complete and (
                rating["correct"] in {"correct", "incorrect"}
                and rating["legible"] in {"clear", "unclear"}
                and rating["continuity"] in {"consistent", "inconsistent"}
                and all(v in {"visible", "missing"} for v in rating["required"])
            )
            group = manifest["operations"][index * 3 : index * 3 + 3]
            variant = next(op for op in group if op["variant"] == hidden["variant"])
            expected_frames = []
            for operation in (group[0], variant):
                asset = benchmark.artifact(args.output, operation["ordinal"], "master")
                checksum = benchmark.legacy.file_hash(asset)
                name = f"frame-{checksum}.jpg"
                require(benchmark.legacy.file_hash(args.gallery / name) == checksum)
                expected_frames.append(name)
            require(option["frames"] == expected_frames)
            wrong = (
                base_wrong
                or rating["correct"] == "incorrect"
                or rating["legible"] == "unclear"
                or rating["continuity"] == "inconsistent"
                or "missing" in rating["required"]
            )
            passed = (
                base_clear
                and rating["correct"] == "correct"
                and rating["legible"] == "clear"
                and rating["continuity"] == "consistent"
                and all(v == "visible" for v in rating["required"])
            )
            verdict = "incorrect" if wrong else "correct" if passed else "unrated"
            counts[hidden["variant"]][verdict] += 1
            outcome["variants"][hidden["variant"]] = verdict
        outcomes.append(outcome)
    return {
        "schema_version": 1,
        "review_sha256": benchmark.legacy.file_hash(args.review),
        "journal_sha256": summary["journal_sha256"],
        "variants": counts,
        "controls": outcomes,
        "latency_pass": summary["latency_pass"],
        "human_review_complete": annotations_complete,
        "verdicts_complete": all(v["unrated"] == 0 for v in counts.values()),
        "physical_display_measured": False,
        "production_promotion_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "authorization", "output", "gallery", "key"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.review:
            print(json.dumps(score(args), indent=2))
        else:
            build(args)
        return 0
    except (ValueError, KeyError, TypeError, OSError, StopIteration):
        print(
            "simple-scene review refused; check verified artifacts and review bindings",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
