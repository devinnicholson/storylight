import pytest

from bookforge.planner_benchmark import (
    CONTEST_CASES,
    BenchmarkCase,
    SemanticExpectation,
    _contains_semantic_alternative,
    _contract_order_for_case,
    _parser,
    _require_loopback,
    _select_cases,
    _semantic_evidence,
    _semantic_summary,
    _summarize,
)


def test_planner_benchmark_requires_loopback_model_endpoint() -> None:
    assert _require_loopback("http://127.0.0.1:11435/") == "http://127.0.0.1:11435"
    assert _require_loopback("http://localhost:11434") == "http://localhost:11434"
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback("https://models.example.com")


def test_planner_benchmark_summary_preserves_latency_and_token_maxima() -> None:
    summary = _summarize(
        [
            {"planning_ms": 4000.125, "output_tokens": 108},
            {"planning_ms": 5000.375, "output_tokens": 116},
        ]
    )

    assert summary == {
        "cases": 2,
        "mean_planning_ms": 4500.25,
        "median_planning_ms": 4500.25,
        "maximum_planning_ms": 5000.375,
        "mean_output_tokens": 112.0,
        "maximum_output_tokens": 116,
    }


def test_planner_benchmark_counterbalances_contract_order() -> None:
    contracts = ("standard", "compact")

    assert _contract_order_for_case(0, contracts) == ("standard", "compact")
    assert _contract_order_for_case(1, contracts) == ("compact", "standard")
    assert _contract_order_for_case(2, contracts) == ("standard", "compact")
    assert _contract_order_for_case(3, ("compact",)) == ("compact",)


def test_planner_benchmark_can_target_bundled_openai_compatible_server() -> None:
    args = _parser().parse_args(
        [
            "--backend",
            "openai",
            "--base-url",
            "http://127.0.0.1:11436",
            "--suite",
            "contest",
        ]
    )

    assert args.backend == "openai"
    assert args.base_url == "http://127.0.0.1:11436"
    assert args.suite == "contest"


def test_contest_suite_has_twenty_unique_synthetic_cases() -> None:
    assert len(CONTEST_CASES) == 20
    assert len({case.case_id for case in CONTEST_CASES}) == 20
    assert all(case.expectations for case in CONTEST_CASES)


def test_planner_benchmark_can_select_validator_failed_subset() -> None:
    selected = _select_cases("contest", ["teacup_boat", "bottle_city"])

    assert [case.case_id for case in selected] == ["teacup_boat", "bottle_city"]

    with pytest.raises(ValueError, match="unknown_case"):
        _select_cases("contest", ["unknown_case"])


def test_semantic_alternative_matching_normalizes_punctuation_and_inflection() -> None:
    assert _contains_semantic_alternative(
        "A luminous, folded-paper bird rises.",
        "paper bird",
    )
    assert _contains_semantic_alternative("Origami birds cross the sky", "origami bird")
    assert _contains_semantic_alternative("One glowing firefly appears", "fireflies")
    assert _contains_semantic_alternative("Wildflowers cover the desert", "flowers")
    assert _contains_semantic_alternative("Flowers blooming across sand", "bloom")
    assert not _contains_semantic_alternative("A clock tower", "octopus")


def test_semantic_evidence_requires_expected_ideas_and_rejects_forbidden_ones() -> None:
    case = BenchmarkCase(
        case_id="negation",
        text="synthetic",
        visual_style="paper",
        seed=1,
        expectations=(
            SemanticExpectation(label="subject", alternatives=("blue moth", "moth")),
            SemanticExpectation(label="shadow", alternatives=("cathedral",)),
        ),
        forbidden_terms=("dragon",),
    )

    passed = _semantic_evidence(
        case,
        generated_text="A blue moth casts a cathedral-shaped shadow.",
    )
    failed = _semantic_evidence(
        case,
        generated_text="A dragon casts a cathedral-shaped shadow.",
    )

    assert passed["automatic_semantic_pass"] is True
    assert failed["automatic_semantic_pass"] is False
    assert failed["forbidden_checks"] == [{"term": "dragon", "pass": False}]


def test_semantic_summary_counts_passes_and_failures() -> None:
    assert _semantic_summary(
        [
            {"automatic_semantic_pass": True},
            {"automatic_semantic_pass": False},
            {"automatic_semantic_pass": True},
        ]
    ) == {"cases": 3, "passed": 2, "failed": 1, "all_passed": False}
