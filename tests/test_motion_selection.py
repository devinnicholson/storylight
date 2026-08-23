import pytest

from bookforge.motion_selection import MotionSelectionError, build_motion_selection_report


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


def test_selects_stable_winner_for_every_page() -> None:
    report = build_motion_selection_report(
        {
            "evaluations": [
                candidate(1, "ambient", stability=0.92),
                candidate(1, "parallax", stability=0.86),
                candidate(2, "ambient", stability=0.81),
                candidate(2, "parallax", stability=0.94),
            ]
        },
        expected_pages=2,
    )

    assert [winner["candidate_id"] for winner in report["winners"]] == [
        "lost-words-page-01-a-motion-ambient",
        "lost-words-page-02-a-motion-parallax",
    ]


def test_requires_two_candidates_per_page() -> None:
    with pytest.raises(MotionSelectionError, match="at least two"):
        build_motion_selection_report(
            {"evaluations": [candidate(1, "ambient", stability=0.9)]},
            expected_pages=1,
        )


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
