"""Lock one merged candidate only after it improves the full development set."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bookforge.fidelity_benchmark import (
    CRITICAL_CATEGORIES,
    DEFAULT_PROMOTION_THRESHOLDS,
    FidelitySummary,
    population_contract_from_manifest,
)
from bookforge.fidelity_schema import DatasetSplit

from .configuration import load_config
from .integrity import canonical_json_bytes, canonical_sha256, sha256_file
from .prediction_evidence import (
    PredictionEvidenceError,
    validate_prediction_completion,
)

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ELIGIBILITY_FIELDS = {
    "schema_version",
    "candidate_id",
    "stage",
    "eligibility_decision",
    "checks",
    "reasons",
    "summary",
    "baseline_summary_sha256",
    "candidate_summary_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "development_records_sha256",
    "predictions_sha256",
    "prediction_completion_sha256",
    "evaluation_completion_sha256",
    "evaluation_input_sha256",
    "candidate_manifest_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_content_sha256",
    "training_run_id",
}
_EVALUATION_COMPLETION_FIELDS = {
    "schema_version",
    "status",
    "stage",
    "surface",
    "candidate_id",
    "config_sha256",
    "dataset_manifest_sha256",
    "development_records_sha256",
    "predictions_sha256",
    "prediction_completion_sha256",
    "evaluation_input_sha256",
    "candidate_manifest_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_content_sha256",
    "report_sha256",
    "predictions",
    "hidden_evaluated",
}


class DevelopmentEligibilityError(ValueError):
    """Development evidence is malformed or bound to another population."""


def _approved_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None or sha256_file(path) != expected_sha256:
        raise DevelopmentEligibilityError(f"{label} differs from its approved SHA-256")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentEligibilityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise DevelopmentEligibilityError(f"{label} must contain one JSON object")
    return value


def _summary(document: dict[str, Any], label: str) -> FidelitySummary:
    value = document.get("summary")
    if not isinstance(value, dict):
        raise DevelopmentEligibilityError(f"{label} has no summary")
    try:
        return FidelitySummary(**value)
    except TypeError as error:
        raise DevelopmentEligibilityError(f"{label} summary is malformed") from error


def _population_matches(summary: FidelitySummary, population: object) -> bool:
    return (
        summary.surface == "raw"
        and summary.split == "development"
        and summary.records == population.records
        and summary.record_ids_sha256 == population.record_ids_sha256
        and summary.counterfactual_pairs == population.pairs
        and dict(summary.category_record_counts) == dict(population.category_record_counts)
    )


def decide_development_eligibility(
    candidate: FidelitySummary,
    baseline: FidelitySummary,
) -> dict[str, bool]:
    thresholds = DEFAULT_PROMOTION_THRESHOLDS
    baseline_failures = baseline.records * (1 - baseline.exact_example_pass_rate)
    candidate_failures = candidate.records * (1 - candidate.exact_example_pass_rate)
    improvement = candidate.exact_example_pass_rate - baseline.exact_example_pass_rate
    categories = set(candidate.category_pass_rates) | set(baseline.category_pass_rates)
    checks = {
        "schema_valid": candidate.schema_valid_rate >= thresholds.schema_valid_rate,
        "privacy": candidate.privacy_pass_rate >= thresholds.privacy_pass_rate,
        "zero_leaks": (
            candidate.pii_leaks
            + candidate.privacy_term_leaks
            + candidate.source_echoes
            + candidate.injection_leaks
            == 0
        ),
        "zero_forbidden_hits": candidate.forbidden_hits == 0,
        "semantic_atom_recall": (
            candidate.semantic_atom_recall >= thresholds.semantic_atom_recall
        ),
        "exact_example_pass": (
            candidate.exact_example_pass_rate >= thresholds.exact_example_pass_rate
        ),
        "counterfactual_sensitivity": (
            candidate.counterfactual_sensitivity >= thresholds.counterfactual_sensitivity
        ),
        "unsupported_concepts": (
            candidate.unsupported_concept_rate <= thresholds.unsupported_concept_rate
        ),
        "development_improvement": (
            improvement >= thresholds.minimum_development_improvement
            or (baseline_failures > 0 and candidate_failures <= baseline_failures / 2)
        ),
        "no_category_regression": all(
            candidate.category_pass_rates.get(category, 0)
            >= baseline.category_pass_rates.get(category, 0)
            - thresholds.maximum_category_regression
            for category in categories
        ),
    }
    for category, rate in candidate.category_pass_rates.items():
        minimum = (
            thresholds.critical_category_pass_rate
            if category in CRITICAL_CATEGORIES
            else thresholds.category_pass_rate
        )
        checks[f"category:{category}"] = rate >= minimum
    return checks


def validate_development_eligibility(
    document: Mapping[str, Any],
    *,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    training_run_id: str,
) -> FidelitySummary:
    """Validate the complete producer contract for a passing development decision."""

    checks = document.get("checks")
    reasons = document.get("reasons")
    if (
        set(document) != _ELIGIBILITY_FIELDS
        or document.get("schema_version") != "1.0"
        or document.get("candidate_id") != candidate_id
        or document.get("stage") != "development"
        or document.get("config_sha256") != config_sha256
        or document.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("training_run_id") != training_run_id
        or document.get("eligibility_decision")
        != {"passed": True, "hidden_evaluated": False}
        or not isinstance(checks, dict)
        or not checks
        or any(type(result) is not bool or not result for result in checks.values())
        or reasons != []
        or _SHA256.fullmatch(str(document.get("baseline_summary_sha256"))) is None
        or _SHA256.fullmatch(str(document.get("candidate_summary_sha256"))) is None
        or document.get("evaluation_input_sha256")
        != canonical_sha256(
            {
                "development_records_sha256": document.get(
                    "development_records_sha256"
                ),
                "predictions_sha256": document.get("predictions_sha256"),
                "prediction_completion_sha256": document.get(
                    "prediction_completion_sha256"
                ),
            }
        )
        or any(
            _SHA256.fullmatch(str(document.get(name))) is None
            for name in (
                "development_records_sha256",
                "predictions_sha256",
                "prediction_completion_sha256",
                "evaluation_completion_sha256",
                "candidate_manifest_sha256",
                "checkpoint_manifest_sha256",
                "checkpoint_content_sha256",
            )
        )
    ):
        raise DevelopmentEligibilityError(
            "candidate evaluation is not a complete passing development-only decision"
        )
    summary = _summary(dict(document), "candidate development evaluation")
    if summary.surface != "raw" or summary.split != "development" or summary.records < 1:
        raise DevelopmentEligibilityError(
            "candidate development evaluation has the wrong summary population"
        )
    return summary


def _evaluation_completion(
    path: Path,
    *,
    expected_sha256: str,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    candidate_report_sha256: str,
    prediction_completion_sha256: str,
    predictions_sha256: str,
) -> dict[str, Any]:
    document = _approved_json(path, expected_sha256, "development evaluation completion")
    if (
        set(document) != _EVALUATION_COMPLETION_FIELDS
        or document.get("schema_version") != "1.0"
        or document.get("status") != "succeeded"
        or document.get("stage") != "development-evaluation"
        or document.get("surface") != "raw"
        or document.get("candidate_id") != candidate_id
        or document.get("config_sha256") != config_sha256
        or document.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("report_sha256") != candidate_report_sha256
        or document.get("prediction_completion_sha256")
        != prediction_completion_sha256
        or document.get("predictions_sha256") != predictions_sha256
        or document.get("evaluation_input_sha256")
        != canonical_sha256(
            {
                "development_records_sha256": document.get(
                    "development_records_sha256"
                ),
                "predictions_sha256": predictions_sha256,
                "prediction_completion_sha256": prediction_completion_sha256,
            }
        )
        or document.get("predictions") != 512
        or document.get("hidden_evaluated") is not False
        or any(
            _SHA256.fullmatch(str(document.get(name))) is None
            for name in (
                "development_records_sha256",
                "candidate_manifest_sha256",
                "checkpoint_manifest_sha256",
                "checkpoint_content_sha256",
            )
        )
    ):
        raise DevelopmentEligibilityError(
            "development evaluation completion has another prediction or candidate lineage"
        )
    return document


def validate_development_evidence_chain(
    *,
    predictions_path: Path,
    predictions_sha256: str,
    prediction_completion_path: Path,
    prediction_completion_sha256: str,
    evaluation_completion_path: Path,
    evaluation_completion_sha256: str,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    candidate_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that prediction, evaluation, and candidate bytes form one lineage."""

    try:
        prediction = validate_prediction_completion(
            prediction_completion_path,
            expected_sha256=prediction_completion_sha256,
            predictions_path=predictions_path,
            expected_predictions_sha256=predictions_sha256,
            expected_candidate_id=candidate_id,
            expected_config_sha256=config_sha256,
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
        )
    except PredictionEvidenceError as error:
        raise DevelopmentEligibilityError(str(error)) from error
    evaluation = _evaluation_completion(
        evaluation_completion_path,
        expected_sha256=evaluation_completion_sha256,
        candidate_id=candidate_id,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        candidate_report_sha256=candidate_report_sha256,
        prediction_completion_sha256=prediction_completion_sha256,
        predictions_sha256=predictions_sha256,
    )
    for name in (
        "development_records_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
    ):
        if evaluation[name] != prediction[name]:
            raise DevelopmentEligibilityError(
                f"development evaluation completion changed {name}"
            )
    return prediction, evaluation


def _write_once(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    from .release import candidate_id_for_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--merged-hf-checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--prediction-completion", type=Path, required=True)
    parser.add_argument("--prediction-completion-sha256", required=True)
    parser.add_argument("--evaluation-completion", type=Path, required=True)
    parser.add_argument("--evaluation-completion-sha256", required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    population = population_contract_from_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
        split=DatasetSplit.DEVELOPMENT,
    )
    candidate_document = _approved_json(
        args.candidate_report,
        args.candidate_report_sha256,
        "candidate development report",
    )
    baseline_document = _approved_json(
        args.baseline_report,
        args.baseline_report_sha256,
        "baseline development report",
    )
    candidate = _summary(candidate_document, "candidate development report")
    baseline = _summary(baseline_document, "baseline development report")
    if not _population_matches(candidate, population) or not _population_matches(
        baseline, population
    ):
        raise DevelopmentEligibilityError("development reports use another population")

    candidate_id = candidate_id_for_checkpoint(
        config_path=args.config,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        training_run_id=args.training_run_id,
        merged_hf_checkpoint=args.merged_hf_checkpoint,
    )
    config_sha256 = load_config(args.config).sha256
    prediction, evaluation_completion = validate_development_evidence_chain(
        predictions_path=args.predictions,
        predictions_sha256=args.predictions_sha256,
        prediction_completion_path=args.prediction_completion,
        prediction_completion_sha256=args.prediction_completion_sha256,
        evaluation_completion_path=args.evaluation_completion,
        evaluation_completion_sha256=args.evaluation_completion_sha256,
        candidate_id=candidate_id,
        config_sha256=config_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        candidate_report_sha256=args.candidate_report_sha256,
    )
    checks = decide_development_eligibility(candidate, baseline)
    passed = all(checks.values())
    document = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "stage": "development",
        "eligibility_decision": {"passed": passed, "hidden_evaluated": False},
        "checks": checks,
        "reasons": [name for name, result in checks.items() if not result],
        "summary": asdict(candidate),
        "baseline_summary_sha256": args.baseline_report_sha256,
        "candidate_summary_sha256": args.candidate_report_sha256,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "development_records_sha256": prediction["development_records_sha256"],
        "predictions_sha256": args.predictions_sha256,
        "prediction_completion_sha256": args.prediction_completion_sha256,
        "evaluation_completion_sha256": args.evaluation_completion_sha256,
        "evaluation_input_sha256": evaluation_completion["evaluation_input_sha256"],
        "candidate_manifest_sha256": prediction["candidate_manifest_sha256"],
        "checkpoint_manifest_sha256": prediction["checkpoint_manifest_sha256"],
        "checkpoint_content_sha256": prediction["checkpoint_content_sha256"],
        "training_run_id": args.training_run_id,
    }
    _write_once(args.output, document)
    print(json.dumps({"candidate_id": candidate_id, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
