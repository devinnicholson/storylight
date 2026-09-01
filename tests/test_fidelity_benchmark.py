from __future__ import annotations

from dataclasses import replace

import pytest

from bookforge.fidelity_benchmark import (
    FidelitySummary,
    PopulationContract,
    RuntimeEvidence,
    benchmark_predictions,
    decide_promotion,
    summarize_evaluations,
)
from bookforge.fidelity_dataset import CATEGORIES
from bookforge.fidelity_evaluation import evaluate_surface


def _record(record_id: str, variant: str) -> dict[str, object]:
    subject = "red fox" if variant == "a" else "blue owl"
    return {
        "record_id": record_id,
        "split": "development",
        "pair_id": "pair-1",
        "pair_variant": variant,
        "categories": ["attribute"],
        "passage": f"A {subject} waits in a cave.",
        "expectations": [
            {
                "kind": "slot",
                "label": "actor",
                "slot": "ACTOR",
                "alternatives": [subject],
            }
        ],
        "allowed_concepts": [subject, "cave"],
        "forbidden_terms": [],
        "privacy_terms": [],
    }


def _raw(subject: str) -> str:
    return f"SETTING: cave\nACTOR: {subject}\nACTION: waits\nMAGIC: quiet shadows"


def test_summary_aggregates_semantics_categories_privacy_and_counterfactuals() -> None:
    evaluations = [
        evaluate_surface(_record("one", "a"), _raw("red fox"), surface="raw"),
        evaluate_surface(_record("two", "b"), _raw("blue owl"), surface="raw"),
    ]

    summary = summarize_evaluations(evaluations)

    assert summary.records == 2
    assert summary.schema_valid_rate == 1
    assert summary.privacy_pass_rate == 1
    assert summary.semantic_atom_recall == 1
    assert summary.exact_example_pass_rate == 1
    assert summary.category_pass_rates == {"attribute": 1}
    assert summary.counterfactual_pairs == 1
    assert summary.counterfactual_sensitivity == 1


def test_summary_rejects_mixed_surfaces_and_empty_input() -> None:
    raw = evaluate_surface(_record("one", "a"), _raw("red fox"), surface="raw")
    repaired = evaluate_surface(
        _record("two", "b"),
        {"master_prompt": "blue owl waits in cave"},
        surface="renderer",
    )

    with pytest.raises(ValueError, match="at least one"):
        summarize_evaluations([])
    with pytest.raises(ValueError, match="cannot combine"):
        summarize_evaluations([raw, repaired])


def _summary(*, exact: float, category: float = 1.0) -> FidelitySummary:
    category_rates = {name: 1.0 for name in CATEGORIES}
    category_rates["attributes"] = category
    return FidelitySummary(
        surface="raw",
        split="hidden",
        records=512,
        record_ids_sha256="6" * 64,
        category_record_counts={name: 32 for name in CATEGORIES},
        schema_valid_rate=1,
        privacy_pass_rate=1,
        semantic_atom_recall=0.99,
        exact_example_pass_rate=exact,
        category_pass_rates=category_rates,
        counterfactual_pairs=256,
        counterfactual_sensitivity=0.99,
        unsupported_concept_rate=0,
        pii_leaks=0,
        privacy_term_leaks=0,
        source_echoes=0,
        injection_leaks=0,
        forbidden_hits=0,
    )


def _population() -> PopulationContract:
    return PopulationContract(
        split="hidden",
        records=512,
        pairs=256,
        record_ids_sha256="6" * 64,
        category_record_counts={name: 32 for name in CATEGORIES},
    )


def _development_summary(*, exact: float, category: float = 1.0) -> FidelitySummary:
    return replace(
        _summary(exact=exact, category=category),
        split="development",
        record_ids_sha256="7" * 64,
    )


def _development_population() -> PopulationContract:
    return replace(_population(), split="development", record_ids_sha256="7" * 64)


def _runtime(**updates: object) -> RuntimeEvidence:
    values: dict[str, object] = {
        "maximum_output_tokens": 64,
        "p50_seconds": 1.5,
        "p95_seconds": 1.8,
        "maximum_seconds": 2.1,
        "p95_regression_fraction": 0.03,
        "unified_memory_peak_gb": 3.9,
        "available_memory_mib": 900,
        "planner_ready_seconds": 75,
        "projector_flow_passed": True,
        "restoration_demonstrated": True,
    }
    values.update(updates)
    return RuntimeEvidence(**values)  # type: ignore[arg-type]


def test_promotion_passes_only_with_semantic_runtime_and_human_evidence() -> None:
    decision = decide_promotion(
        _summary(exact=0.97),
        baseline=_summary(exact=0.90),
        development_candidate=_development_summary(exact=0.97),
        development_baseline=_development_summary(exact=0.90),
        runtime=_runtime(),
        contest_suite_passed=True,
        human_review_passed=True,
        population=_population(),
        development_population=_development_population(),
    )

    assert decision.passed is True
    assert decision.reasons == ()
    assert all(decision.checks.values())


def test_promotion_fails_closed_on_latency_review_and_category_regression() -> None:
    decision = decide_promotion(
        _summary(exact=0.97, category=0.89),
        baseline=_summary(exact=0.90, category=1),
        development_candidate=_development_summary(exact=0.97, category=0.89),
        development_baseline=_development_summary(exact=0.90, category=1),
        runtime=_runtime(p95_seconds=2.01),
        contest_suite_passed=True,
        human_review_passed=False,
        population=_population(),
        development_population=_development_population(),
    )

    assert decision.passed is False
    assert "human_review" in decision.reasons
    assert "p95_latency" in decision.reasons
    assert "category:attributes" in decision.reasons
    assert "no_category_regression" in decision.reasons


def test_prediction_join_fails_when_a_record_is_missing() -> None:
    records = [_record("one", "a"), _record("two", "b")]

    with pytest.raises(ValueError, match="missing prediction"):
        benchmark_predictions(records, [{"record_id": "one", "raw": _raw("red fox")}])


def test_promotion_rejects_any_manifest_population_mismatch() -> None:
    population = _population()
    candidate = _summary(exact=0.97)
    mismatched = replace(candidate, record_ids_sha256="5" * 64)
    decision = decide_promotion(
        mismatched,
        baseline=_summary(exact=0.90),
        development_candidate=_development_summary(exact=0.97),
        development_baseline=_development_summary(exact=0.90),
        runtime=_runtime(),
        contest_suite_passed=True,
        human_review_passed=True,
        population=population,
        development_population=_development_population(),
    )
    assert decision.passed is False
    assert "population_record_ids" in decision.reasons


def test_promotion_uses_only_development_summaries_for_improvement_gates() -> None:
    decision = decide_promotion(
        _summary(exact=0.97),
        baseline=_summary(exact=0.96),
        development_candidate=_development_summary(exact=0.97),
        development_baseline=_development_summary(exact=0.90),
        runtime=_runtime(),
        contest_suite_passed=True,
        human_review_passed=True,
        population=_population(),
        development_population=_development_population(),
    )

    assert decision.passed is True
    assert decision.checks["development_improvement"] is True
