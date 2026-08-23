from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PAGE_PATTERN = re.compile(r"(?:^|-)page-(\d+)(?:-|$)")


class MotionSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MotionCandidateScore:
    candidate_id: str
    page_number: int
    sha256: str
    projection_legibility: float
    endpoint_ssim: float
    temporal_change: float
    motion_stability: float
    human_approved: bool
    review_notes: str
    overall: float


def build_motion_selection_report(
    technical: dict,
    *,
    expected_pages: int,
    human_review: dict | None = None,
) -> dict:
    evaluations = technical.get("evaluations")
    if not isinstance(evaluations, list):
        raise MotionSelectionError("technical motion evaluations are required")
    review_by_id: dict[str, dict] = {}
    if human_review is not None:
        reviews = human_review.get("reviews")
        if not isinstance(reviews, list):
            raise MotionSelectionError("human review requires a reviews list")
        review_by_id = {str(review.get("id")): review for review in reviews}
        if len(review_by_id) != len(reviews):
            raise MotionSelectionError("human review contains duplicate candidate IDs")
        if any(not isinstance(review.get("approved"), bool) for review in reviews):
            raise MotionSelectionError("every human review requires boolean approved")
    candidates: list[MotionCandidateScore] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        candidate_id = Path(evaluation["path"]).stem.split("__", 1)[0]
        if candidate_id in seen:
            raise MotionSelectionError(f"duplicate motion candidate: {candidate_id}")
        seen.add(candidate_id)
        match = PAGE_PATTERN.search(candidate_id)
        motion = evaluation.get("motion")
        if not match or not isinstance(motion, dict):
            raise MotionSelectionError(f"invalid motion candidate: {candidate_id}")
        values = {
            "projection_legibility": float(evaluation["projection"]["projection_legibility"]),
            "endpoint_ssim": float(motion["endpoint_ssim"]),
            "temporal_change": float(motion["temporal_change"]),
            "motion_stability": float(motion["motion_stability"]),
        }
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise MotionSelectionError(f"invalid motion score: {candidate_id}")
        duration = float(motion["duration_seconds"])
        fps = float(motion["fps"])
        overall = (
            0.6 * values["motion_stability"]
            + 0.25 * values["projection_legibility"]
            + 0.15 * values["endpoint_ssim"]
        )
        if not 3 <= duration <= 8 or fps < 20 or values["endpoint_ssim"] < 0.94:
            overall = min(overall, 0.49)
        review = review_by_id.get(candidate_id, {"approved": True, "notes": ""})
        if not review["approved"]:
            overall = min(overall, 0.49)
        candidates.append(
            MotionCandidateScore(
                candidate_id=candidate_id,
                page_number=int(match.group(1)),
                sha256=str(evaluation["sha256"]),
                human_approved=review["approved"],
                review_notes=str(review.get("notes", "")),
                overall=round(overall, 6),
                **{name: round(value, 6) for name, value in values.items()},
            )
        )
    if human_review is not None and set(review_by_id) != seen:
        raise MotionSelectionError("human review and candidate sets do not match")
    winners = []
    for page_number in range(1, expected_pages + 1):
        page_candidates = [item for item in candidates if item.page_number == page_number]
        if len(page_candidates) < 2:
            raise MotionSelectionError(
                f"page {page_number:02d} requires at least two motion candidates"
            )
        winner = max(page_candidates, key=lambda item: (item.overall, item.candidate_id))
        if winner.overall < 0.5:
            raise MotionSelectionError(
                f"page {page_number:02d} has no motion candidate passing gates"
            )
        winners.append(winner)
    return {
        "schema_version": "1.0",
        "expected_pages": expected_pages,
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
    parser = argparse.ArgumentParser(description="Select one stable motion loop per story page")
    parser.add_argument("technical", type=Path)
    parser.add_argument("--expected-pages", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--human-review", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.expected_pages <= 12:
        raise MotionSelectionError("expected pages must be between 1 and 12")
    report = build_motion_selection_report(
        json.loads(arguments.technical.read_text()),
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
