"""Record an immutable development rejection without downloading model shards."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from bookforge.fidelity_benchmark import population_contract_from_manifest
from bookforge.fidelity_schema import DatasetSplit

from .development_eligibility import (
    DevelopmentEligibilityError,
    _approved_json,
    _population_matches,
    _summary,
    decide_development_eligibility,
    validate_baseline_evidence_chain,
    validate_development_evidence_chain,
)
from .integrity import canonical_json_bytes
from .merged_candidate import validate_merged_candidate_declaration


def _write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
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
    parser.add_argument("--baseline-completion", type=Path, required=True)
    parser.add_argument("--baseline-completion-sha256", required=True)
    parser.add_argument("--baseline-candidate-manifest-sha256", required=True)
    parser.add_argument("--baseline-engine-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_manifest = validate_merged_candidate_declaration(
        args.candidate_manifest,
        config_path=args.config,
        expected_manifest_sha256=args.candidate_manifest_sha256,
        expected_dataset_manifest_sha256=args.dataset_manifest_sha256,
    )
    if candidate_manifest["training_run_id"] != args.training_run_id:
        raise DevelopmentEligibilityError("candidate declaration uses another training run")
    candidate_id = str(candidate_manifest["candidate_id"])
    config_sha256 = str(candidate_manifest["config_sha256"])
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

    prediction, evaluation = validate_development_evidence_chain(
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
    if prediction["candidate_manifest_sha256"] != args.candidate_manifest_sha256:
        raise DevelopmentEligibilityError("prediction used another candidate declaration")
    validate_baseline_evidence_chain(
        baseline_report_path=args.baseline_report,
        baseline_report_sha256=args.baseline_report_sha256,
        baseline_completion_path=args.baseline_completion,
        baseline_completion_sha256=args.baseline_completion_sha256,
        baseline_candidate_manifest_sha256=args.baseline_candidate_manifest_sha256,
        baseline_engine_sha256=args.baseline_engine_sha256,
        dataset_manifest_path=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        development_records_sha256=prediction["development_records_sha256"],
    )
    checks = decide_development_eligibility(candidate, baseline)
    if all(checks.values()):
        raise DevelopmentEligibilityError(
            "passing candidates require byte-complete development eligibility verification"
        )
    document: dict[str, object] = {
        "schema_version": "bookforge-jax-development-rejection-v1",
        "status": "rejected",
        "candidate_id": candidate_id,
        "stage": "development",
        "eligibility_decision": {
            "passed": False,
            "hidden_evaluated": False,
            "release_authorized": False,
        },
        "checks": checks,
        "reasons": [name for name, result in checks.items() if not result],
        "summary": asdict(candidate),
        "baseline_summary_sha256": args.baseline_report_sha256,
        "baseline_completion_sha256": args.baseline_completion_sha256,
        "baseline_candidate_manifest_sha256": args.baseline_candidate_manifest_sha256,
        "baseline_engine_sha256": args.baseline_engine_sha256,
        "candidate_summary_sha256": args.candidate_report_sha256,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "development_records_sha256": prediction["development_records_sha256"],
        "predictions_sha256": args.predictions_sha256,
        "prediction_completion_sha256": args.prediction_completion_sha256,
        "evaluation_completion_sha256": args.evaluation_completion_sha256,
        "evaluation_input_sha256": evaluation["evaluation_input_sha256"],
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "checkpoint_manifest_sha256": prediction["checkpoint_manifest_sha256"],
        "checkpoint_content_sha256": prediction["checkpoint_content_sha256"],
        "training_run_id": args.training_run_id,
        "checkpoint_bytes_downloaded": False,
    }
    _write_once(args.output, document)
    print(json.dumps({"candidate_id": candidate_id, "passed": False}, indent=2))


if __name__ == "__main__":
    main()
