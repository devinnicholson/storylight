from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PAGE_PATTERN = re.compile(r"(?:^|-)page-(\d+)(?:-|$)")


class VisualSelectionError(RuntimeError):
    pass


def merge_technical_payloads(payloads: list[dict]) -> dict:
    evaluations = []
    for payload in payloads:
        batch = payload.get("evaluations")
        if not isinstance(batch, list):
            raise VisualSelectionError("technical evaluations are required")
        evaluations.extend(batch)
    return {"schema_version": "1.0", "evaluations": evaluations}


def merge_semantic_payloads(payloads: list[dict]) -> dict:
    if not payloads:
        raise VisualSelectionError("at least one semantic payload is required")
    identity = (payloads[0].get("model"), payloads[0].get("model_revision"))
    scores = []
    for payload in payloads:
        if (payload.get("model"), payload.get("model_revision")) != identity:
            raise VisualSelectionError("semantic payload model identities do not match")
        batch = payload.get("scores")
        if not isinstance(batch, list):
            raise VisualSelectionError("semantic scores are required")
        scores.extend(batch)
    return {
        "schema_version": "1.0",
        "model": identity[0],
        "model_revision": identity[1],
        "scores": scores,
    }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    page_number: int
    sha256: str
    story_fidelity: float
    character_consistency: float
    child_safety: float
    no_text: float
    projection_legibility: float
    human_approved: bool
    review_notes: str
    overall: float


def _unit(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise VisualSelectionError(f"{name} must be a finite score from 0 to 1")
    return number


def score_candidate(
    *,
    candidate_id: str,
    sha256: str,
    story_fidelity: float,
    character_consistency: float,
    child_safety: float,
    no_text: float,
    projection_legibility: float,
    human_approved: bool = True,
    review_notes: str = "",
) -> CandidateScore:
    match = PAGE_PATTERN.search(candidate_id)
    if not match:
        raise VisualSelectionError(f"candidate ID has no page number: {candidate_id}")
    values = {
        "story_fidelity": _unit(story_fidelity, "story_fidelity"),
        "character_consistency": _unit(character_consistency, "character_consistency"),
        "child_safety": _unit(child_safety, "child_safety"),
        "no_text": _unit(no_text, "no_text"),
        "projection_legibility": _unit(projection_legibility, "projection_legibility"),
    }
    # Fidelity and cross-page consistency dominate. Safety and accidental text
    # are explicit gates as well as weighted evidence, so a pretty unsafe frame
    # cannot win by compensating elsewhere.
    overall = (
        0.32 * values["story_fidelity"]
        + 0.26 * values["character_consistency"]
        + 0.18 * values["projection_legibility"]
        + 0.14 * values["child_safety"]
        + 0.10 * values["no_text"]
    )
    if values["child_safety"] < 0.55 or values["no_text"] < 0.55:
        overall = min(overall, 0.49)
    if not human_approved:
        overall = min(overall, 0.49)
    return CandidateScore(
        candidate_id=candidate_id,
        page_number=int(match.group(1)),
        sha256=sha256,
        human_approved=human_approved,
        review_notes=review_notes,
        overall=round(overall, 6),
        **{name: round(value, 6) for name, value in values.items()},
    )


def build_selection_report(
    technical: dict,
    semantic: dict,
    *,
    expected_pages: int,
    human_review: dict | None = None,
) -> dict:
    evaluations = technical.get("evaluations")
    semantic_scores = semantic.get("scores")
    if not isinstance(evaluations, list) or not isinstance(semantic_scores, list):
        raise VisualSelectionError("technical evaluations and semantic scores are required")
    technical_by_id = {}
    for evaluation in evaluations:
        candidate_id = Path(evaluation["path"]).stem.split("__", 1)[0]
        if candidate_id in technical_by_id:
            raise VisualSelectionError(f"duplicate technical candidate: {candidate_id}")
        technical_by_id[candidate_id] = evaluation
    semantic_by_id = {score["id"]: score for score in semantic_scores}
    if set(technical_by_id) != set(semantic_by_id):
        raise VisualSelectionError("technical and semantic candidate sets do not match")
    review_by_id: dict[str, dict] = {}
    if human_review is not None:
        reviews = human_review.get("reviews")
        if not isinstance(reviews, list):
            raise VisualSelectionError("human review requires a reviews list")
        review_by_id = {str(review.get("id")): review for review in reviews}
        if set(review_by_id) != set(technical_by_id):
            raise VisualSelectionError("human review and candidate sets do not match")
        if any(not isinstance(review.get("approved"), bool) for review in reviews):
            raise VisualSelectionError("every human review requires boolean approved")

    candidates: list[CandidateScore] = []
    for candidate_id, evaluation in technical_by_id.items():
        score = semantic_by_id[candidate_id]
        review = review_by_id.get(candidate_id, {"approved": True, "notes": ""})
        if evaluation["sha256"] != score["sha256"]:
            raise VisualSelectionError(f"checksum mismatch for {candidate_id}")
        candidates.append(
            score_candidate(
                candidate_id=candidate_id,
                sha256=evaluation["sha256"],
                story_fidelity=score["story_fidelity"],
                character_consistency=score["character_consistency"],
                child_safety=score["child_safety"],
                no_text=score["no_text"],
                projection_legibility=evaluation["projection"]["projection_legibility"],
                human_approved=review["approved"],
                review_notes=str(review.get("notes", "")),
            )
        )

    winners = []
    for page_number in range(1, expected_pages + 1):
        page_candidates = [item for item in candidates if item.page_number == page_number]
        if not page_candidates:
            raise VisualSelectionError(f"page {page_number:02d} has no candidates")
        winner = max(page_candidates, key=lambda item: (item.overall, item.candidate_id))
        if winner.overall < 0.5:
            raise VisualSelectionError(f"page {page_number:02d} has no candidate passing gates")
        winners.append(winner)
    return {
        "schema_version": "1.0",
        "expected_pages": expected_pages,
        "model": semantic.get("model", "unknown"),
        "model_revision": semantic.get("model_revision", "unknown"),
        "human_review": (
            {
                "reviewer": human_review.get("reviewer", "unspecified"),
                "reviewed_at": human_review.get("reviewed_at", "unspecified"),
            }
            if human_review is not None
            else None
        ),
        "candidates": [
            asdict(item) for item in sorted(candidates, key=lambda item: item.candidate_id)
        ],
        "winners": [asdict(item) for item in winners],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select one projection master per story page")
    parser.add_argument("technical", type=Path)
    parser.add_argument("semantic", type=Path)
    parser.add_argument("--additional-technical", action="append", default=[], type=Path)
    parser.add_argument("--additional-semantic", action="append", default=[], type=Path)
    parser.add_argument("--expected-pages", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--human-review", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.expected_pages <= 12:
        raise VisualSelectionError("expected pages must be between 1 and 12")
    technical = merge_technical_payloads(
        [
            json.loads(path.read_text())
            for path in [arguments.technical, *arguments.additional_technical]
        ]
    )
    semantic = merge_semantic_payloads(
        [
            json.loads(path.read_text())
            for path in [arguments.semantic, *arguments.additional_semantic]
        ]
    )
    report = build_selection_report(
        technical,
        semantic,
        expected_pages=arguments.expected_pages,
        human_review=(
            json.loads(arguments.human_review.read_text()) if arguments.human_review else None
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
