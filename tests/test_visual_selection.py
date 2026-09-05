from copy import deepcopy

import pytest

from bookforge.visual_selection import (
    VisualSelectionError,
    build_selection_report,
    score_candidate,
)


def evaluation(candidate_id: str, checksum: str, legibility: float) -> dict:
    return {
        "path": f"/tmp/{candidate_id}__42.png",
        "sha256": checksum,
        "projection": {"projection_legibility": legibility},
    }


def semantic(candidate_id: str, checksum: str, fidelity: float, consistency: float) -> dict:
    return {
        "id": candidate_id,
        "sha256": checksum,
        "story_fidelity": fidelity,
        "character_consistency": consistency,
        "child_safety": 0.95,
        "no_text": 0.91,
    }


def payloads() -> tuple[dict, dict]:
    candidates = [
        ("lost-words-page-01-a", "a" * 64, 0.82, 0.80, 0.91),
        ("lost-words-page-01-b", "b" * 64, 0.75, 0.90, 0.87),
        ("lost-words-page-02-a", "c" * 64, 0.78, 0.89, 0.88),
        ("lost-words-page-02-b", "d" * 64, 0.86, 0.92, 0.90),
    ]
    technical = {"evaluations": [evaluation(item[0], item[1], item[4]) for item in candidates]}
    scores = {
        "model": "siglip",
        "model_revision": "revision",
        "scores": [semantic(item[0], item[1], item[2], item[3]) for item in candidates],
    }
    return technical, scores


def test_safety_and_text_are_non_compensating_gates() -> None:
    score = score_candidate(
        candidate_id="lost-words-page-01-a",
        sha256="a" * 64,
        story_fidelity=1,
        character_consistency=1,
        child_safety=0.2,
        no_text=1,
        projection_legibility=1,
    )

    assert score.overall == 0.49


def test_rejects_mismatched_sets_or_checksums() -> None:
    technical, scores = payloads()
    scores["scores"].pop()
    with pytest.raises(VisualSelectionError, match="sets do not match"):
        build_selection_report(technical, scores, expected_pages=2)

    technical, scores = payloads()
    broken = deepcopy(scores)
    broken["scores"][0]["sha256"] = "f" * 64
    with pytest.raises(VisualSelectionError, match="checksum mismatch"):
        build_selection_report(technical, broken, expected_pages=2)


def test_human_review_can_veto_a_high_scoring_candidate() -> None:
    technical, scores = payloads()
    reviews = {
        "reviewer": "visual acceptance",
        "reviewed_at": "2026-08-23T00:00:00Z",
        "reviews": [
            {
                "id": item["id"],
                "approved": item["id"] != "lost-words-page-02-b",
                "notes": "missing required child" if item["id"].endswith("02-b") else "passes",
            }
            for item in scores["scores"]
        ],
    }

    report = build_selection_report(
        technical,
        scores,
        expected_pages=2,
        human_review=reviews,
    )

    assert report["winners"][1]["candidate_id"] == "lost-words-page-02-a"
    vetoed = next(item for item in report["candidates"] if item["candidate_id"].endswith("02-b"))
    assert vetoed["human_approved"] is False
    assert vetoed["overall"] == 0.49


def test_human_review_must_cover_exact_candidate_set() -> None:
    technical, scores = payloads()

    with pytest.raises(VisualSelectionError, match="sets do not match"):
        build_selection_report(
            technical,
            scores,
            expected_pages=2,
            human_review={"reviews": [{"id": "lost-words-page-01-a", "approved": True}]},
        )
