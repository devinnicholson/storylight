#!/usr/bin/env python3
"""Validate and summarize a frozen story gallery review without inferring blank ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import install_fidelity_display as installer  # noqa: E402

assembly = installer.assembly


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Rating(StrictModel):
    label: Literal["A", "B"]
    correct: Literal["", "correct", "incorrect", "unsure"]
    legible: Literal["", "clear", "unclear"]
    continuity: Literal["", "consistent", "inconsistent", "unsure"]
    rating: Literal["", "1", "2", "3", "4", "5"]
    required: Annotated[list[Literal["", "visible", "missing", "unsure"]], Field(max_length=32)]
    forbidden: Annotated[list[Literal["", "absent", "present", "unsure"]], Field(max_length=32)]


class ReviewedPage(StrictModel):
    page: Annotated[int, Field(ge=1, le=6)]
    options: Annotated[list[Rating], Field(min_length=1, max_length=2)]


class Review(StrictModel):
    schema_version: Literal[1]
    review_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    pages: Annotated[list[ReviewedPage], Field(min_length=6, max_length=6)]


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[dict, str]:
    content = installer.read_bounded(path, 262144)
    value = json.loads(content, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("review input must be an object")
    return value, hashlib.sha256(content).hexdigest()


def bind_gallery(args):
    review_data, review_sha = read_json(args.review)
    review = Review.model_validate(review_data)
    gallery, gallery_sha = read_json(args.gallery_data)
    key, key_sha = read_json(args.key_file)
    if (
        type(review_data["schema_version"]) is not int
        or type(gallery.get("schema_version")) is not int
        or args.key_file.stat().st_uid != os.getuid()
        or args.key_file.parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(args.key_file.stat().st_mode) != 0o600
        or stat.S_IMODE(args.key_file.parent.stat().st_mode) != 0o700
        or set(key)
        != {"review_id", "batch_sha256", "journal_sha256", "pages", "review_data_sha256"}
        or key["review_id"] != review.review_id
        or any(
            not isinstance(key[name], str) or not re.fullmatch(r"[a-f0-9]{64}", key[name])
            for name in ("batch_sha256", "journal_sha256", "review_data_sha256")
        )
        or assembly.digest(gallery) != key["review_data_sha256"]
    ):
        raise ValueError("review identity or gallery proof differs")
    batch = installer.render.read_batch(args.batch, key["batch_sha256"])
    _, journal_sha = installer.completed_bundles(batch, key["batch_sha256"], args.render_dir)
    if journal_sha != key["journal_sha256"]:
        raise ValueError("review journal proof differs")
    frames = {
        row.id: "frame-"
        + assembly.file_hash(args.render_dir / f"image-{ordinal:02}/master.jpg")
        + ".jpg"
        for ordinal, row in enumerate(installer.render.ordered_requests(batch))
    }
    manifest, _ = assembly.smoke.load_manifest(assembly.probe.STORY)
    expected = {"schema_version": 1, "review_id": review.review_id, "pages": []}
    if not isinstance(key["pages"], list) or len(key["pages"]) != 6:
        raise ValueError("review mapping pages differ")
    bindings = []
    for index, (page, mapping, reviewed) in enumerate(
        zip(batch.pages, key["pages"], review.pages, strict=True), 1
    ):
        groups = {
            variant: [
                row.id
                for row in batch.requests
                if row.source_page_index == index and row.variant == variant
            ]
            for variant in ("candidate", "accepted")
        }
        groups = {variant: requests for variant, requests in groups.items() if requests}
        labels = list("AB"[: len(groups)])
        if (
            not isinstance(mapping, dict)
            or set(mapping) != {"page", "options"}
            or type(mapping["page"]) is not int
            or mapping["page"] != index
            or reviewed.page != index
            or page.source_page_index != index
            or not isinstance(mapping["options"], list)
            or len(mapping["options"]) != len(groups)
            or [option.label for option in reviewed.options] != labels
        ):
            raise ValueError("review pages or options differ")
        options, variants = [], {}
        source = manifest["pages"][index - 1]
        for label, option, rating in zip(labels, mapping["options"], reviewed.options, strict=True):
            if (
                not isinstance(option, dict)
                or set(option) != {"label", "variant", "requests"}
                or option["label"] != label
                or option["variant"] not in groups
                or option["variant"] in variants
                or option["requests"] != groups[option["variant"]]
                or len(rating.required) != len(source["required_facts"])
                or len(rating.forbidden) != len(source["forbidden_mistakes"])
            ):
                raise ValueError("review mapping or checklist length differs")
            selected = [frames[request] for request in option["requests"]]
            if len(selected) == 1:
                selected *= len(page.display_step_ids)
            options.append({"label": label, "frames": selected})
            variants[option["variant"]] = rating
        expected["pages"].append(
            {
                "page": index,
                "states": len(page.display_step_ids),
                "options": options,
                "required": source["required_facts"],
                "forbidden": source["forbidden_mistakes"],
            }
        )
        bindings.append(variants)
    if gallery != expected:
        raise ValueError("gallery differs from frozen checklists and verified frames")
    return (
        review,
        bindings,
        {
            "review_sha256": review_sha,
            "gallery_file_sha256": gallery_sha,
            "gallery_canonical_sha256": key["review_data_sha256"],
            "mapping_sha256": key_sha,
            "batch_sha256": key["batch_sha256"],
            "render_journal_sha256": journal_sha,
            "story_manifest_sha256": assembly.smoke.MANIFEST_SHA256,
        },
    )


def summarize(review: Review, bindings: list[dict], provenance: dict) -> dict:
    pages, rejected = [], []
    paired = {"candidate_wins": 0, "accepted_wins": 0, "ties": 0, "unrated": 0}
    correctness_pairs = {"both_incorrect": 0, "both_correct": 0, "different": 0, "unresolved": 0}
    counts = {
        variant: {
            "available": 0,
            "correct": 0,
            "incorrect": 0,
            "unsure": 0,
            "unrated": 0,
            "clear": 0,
            "unclear": 0,
            "legibility_unrated": 0,
        }
        for variant in ("candidate", "accepted")
    }
    facts = {"total": 0, "rated": 0, "unrated": 0, "unsure": 0}
    candidate_complete = True
    for index, variants in enumerate(bindings, 1):
        page = {"page": index, "accepted_available": "accepted" in variants, "options": {}}
        for variant, rating in variants.items():
            counts[variant]["available"] += 1
            counts[variant][rating.correct or "unrated"] += 1
            counts[variant][rating.legible or "legibility_unrated"] += 1
            values = [*rating.required, *rating.forbidden]
            facts["total"] += len(values)
            facts["rated"] += sum(bool(value) for value in values)
            facts["unrated"] += values.count("")
            facts["unsure"] += values.count("unsure")
            page["options"][variant] = {
                name: [value or None for value in getattr(rating, name)]
                if name in {"required", "forbidden"}
                else getattr(rating, name) or None
                for name in ("correct", "legible", "continuity", "rating", "required", "forbidden")
            }
        candidate = variants["candidate"]
        if (
            candidate.correct == "incorrect"
            or "missing" in candidate.required
            or "present" in candidate.forbidden
        ):
            rejected.append(index)
        candidate_complete &= (
            candidate.correct == "correct"
            and candidate.legible == "clear"
            and candidate.continuity == "consistent"
            and bool(candidate.rating)
            and all(value == "visible" for value in candidate.required)
            and all(value == "absent" for value in candidate.forbidden)
        )
        if "accepted" in variants:
            accepted = variants["accepted"]
            if candidate.rating and accepted.rating:
                relation = (
                    "candidate_wins"
                    if int(candidate.rating) > int(accepted.rating)
                    else "accepted_wins"
                    if int(candidate.rating) < int(accepted.rating)
                    else "ties"
                )
            else:
                relation = "unrated"
            paired[relation] += 1
            page["paired_overall_rating"] = relation
            correctness = (
                "both_incorrect"
                if candidate.correct == accepted.correct == "incorrect"
                else "both_correct"
                if candidate.correct == accepted.correct == "correct"
                else "different"
                if {candidate.correct, accepted.correct} == {"correct", "incorrect"}
                else "unresolved"
            )
            correctness_pairs[correctness] += 1
            page["paired_correctness"] = correctness
        pages.append(page)
    fact_annotations_complete = facts["unrated"] == facts["unsure"] == 0
    return {
        "schema_version": 1,
        "kind": "fidelity-story-human-review-summary",
        "review_id": review.review_id,
        "provenance": provenance,
        "harness_sha256": assembly.file_hash(Path(__file__)),
        "pages": pages,
        "counts": counts,
        "paired_overall_ratings": paired,
        "paired_correctness": correctness_pairs,
        "fact_annotations": facts,
        "fact_annotations_complete": fact_annotations_complete,
        "candidate_rejected_pages": rejected,
        "decision": "reject"
        if rejected
        else "review_complete"
        if candidate_complete and fact_annotations_complete
        else "incomplete",
        "ratings_are_page_level": True,
        "paired_rating_wins_are_fidelity_wins": False,
        "general_accuracy_measured": False,
        "production_promotion_authorized": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "gallery-data", "key-file", "batch", "render-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("summary requires a fresh output")
        review, bindings, provenance = bind_gallery(args)
        result = summarize(review, bindings, provenance)
        with os.fdopen(
            os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w"
        ) as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception:
        print("review summary refused; verify frozen local inputs and rating completeness")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
