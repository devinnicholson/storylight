import pytest

from storylight.motion_selection import MotionSelectionError, build_motion_selection_report


def candidate(page: int, variant: str, *, stability: float, ssim: float = 0.98) -> dict:
    candidate_id = f"lost-words-page-{page:02d}-a-motion-{variant}"
    return {
        "path": f"artifacts/{candidate_id}__42.mp4",
        "sha256": (variant[0] if variant[0] in "abcdef" else "a") * 64,
        "projection": {"projection_legibility": 0.8},
        "motion": {
            "duration_seconds": 4,
            "fps": 24,
            "endpoint_ssim": ssim,
            "temporal_change": 0.01,
            "motion_stability": stability,
        },
    }


def test_rejects_loops_that_jump_at_endpoint() -> None:
    with pytest.raises(MotionSelectionError, match="no motion candidate passing"):
        build_motion_selection_report(
            {
                "evaluations": [
                    candidate(1, "ambient", stability=0.9, ssim=0.8),
                    candidate(1, "parallax", stability=0.9, ssim=0.7),
                ]
            },
            expected_pages=1,
        )


def test_human_review_vetoes_a_smooth_but_visually_malformed_loop() -> None:
    rows = [
        candidate(1, "ambient", stability=0.98),
        candidate(1, "parallax", stability=0.85),
    ]
    review = {
        "reviewer": "motion acceptance",
        "reviewed_at": "2026-08-23T00:00:00Z",
        "reviews": [
            {
                "id": "lost-words-page-01-a-motion-ambient",
                "approved": False,
                "notes": "face morph",
            },
            {
                "id": "lost-words-page-01-a-motion-parallax",
                "approved": True,
                "notes": "clean",
            },
        ],
    }

    report = build_motion_selection_report(
        {"evaluations": rows},
        expected_pages=1,
        human_review=review,
    )

    assert report["winners"][0]["candidate_id"].endswith("parallax")
    vetoed = next(item for item in report["candidates"] if item["candidate_id"].endswith("ambient"))
    assert vetoed["overall"] == 0.49


def test_motion_human_review_must_cover_the_exact_candidate_set() -> None:
    rows = [
        candidate(1, "ambient", stability=0.9),
        candidate(1, "parallax", stability=0.9),
    ]

    with pytest.raises(MotionSelectionError, match="sets do not match"):
        build_motion_selection_report(
            {"evaluations": rows},
            expected_pages=1,
            human_review={
                "reviews": [{"id": "lost-words-page-01-a-motion-ambient", "approved": True}]
            },
        )

    with pytest.raises(MotionSelectionError, match="duplicate candidate IDs"):
        build_motion_selection_report(
            {"evaluations": rows},
            expected_pages=1,
            human_review={
                "reviews": [
                    {"id": "lost-words-page-01-a-motion-ambient", "approved": True},
                    {"id": "lost-words-page-01-a-motion-ambient", "approved": False},
                ]
            },
        )
