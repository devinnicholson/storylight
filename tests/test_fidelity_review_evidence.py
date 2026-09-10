from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from storylight.fidelity_benchmark import CandidateIdentity
from storylight.fidelity_review_evidence import (
    HUMAN_REVIEW_ATTESTATION,
    HUMAN_REVIEW_POPULATION_SHA256,
    build_contest_evidence,
    build_human_review_evidence,
    validate_contest_evidence,
    validate_human_review_evidence,
)
from storylight.planner_benchmark import CONTEST_CASES

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT / "research/datasets/story-fidelity-v1/human-review-population.json"


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        candidate_id="candidate-review-test",
        candidate_manifest_sha256="a" * 64,
        engine_sha256="b" * 64,
        model_revision="sha256:" + "b" * 64,
    )


def _contest_report() -> dict[str, object]:
    cases = []
    for index, benchmark_case in enumerate(CONTEST_CASES):
        semantic_count = 4
        forbidden_count = 5 if index == 0 else 0
        cases.append(
            {
                "case_id": benchmark_case.case_id,
                "valid": True,
                "automatic_semantic_pass": True,
                "semantic_checks": [
                    {"label": f"semantic-{check}", "pass": True}
                    for check in range(semantic_count)
                ],
                "forbidden_checks": [
                    {"label": f"forbidden-{check}", "pass": True}
                    for check in range(forbidden_count)
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "runtime": {
            "suite": "contest",
            "case_count": 20,
            "exit_code": 0,
            "candidate_revision": _candidate().model_revision,
        },
        "acceptance": {
            "all_outputs_schema_valid": True,
            "automatic_semantic_pass": True,
        },
        "privacy": {
            "model_and_artifacts_local_only": True,
            "fixtures_are_synthetic": True,
            "modal_or_cloud_called": False,
        },
        "cases": cases,
    }


def test_contest_evidence_requires_exact_fixed_suite_population() -> None:
    candidate = _candidate()
    document = build_contest_evidence(
        candidate,
        _contest_report(),
        source_report_sha256="c" * 64,
    )
    assert document["candidate_identity"] == asdict(candidate)
    assert validate_contest_evidence(document, candidate)

    report = _contest_report()
    report["cases"][0]["semantic_checks"][0]["pass"] = False
    with pytest.raises(ValueError, match="failed semantic"):
        build_contest_evidence(candidate, report, source_report_sha256="c" * 64)

    report = _contest_report()
    report["cases"][0]["case_id"] = "invented_contest_case"
    with pytest.raises(ValueError, match="population differs"):
        build_contest_evidence(candidate, report, source_report_sha256="c" * 64)

    report = _contest_report()
    report["runtime"]["candidate_revision"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="fixed local"):
        build_contest_evidence(candidate, report, source_report_sha256="c" * 64)


def test_human_review_retains_only_ids_and_requires_25_passes() -> None:
    candidate = _candidate()
    population = json.loads(POPULATION.read_text())
    decisions = [
        {**record, "passed": True} for record in population["records"]
    ]
    document = build_human_review_evidence(
        candidate,
        decisions,
        population=population,
        population_contract_sha256=HUMAN_REVIEW_POPULATION_SHA256,
        source_decisions_sha256="d" * 64,
        attestation=HUMAN_REVIEW_ATTESTATION,
    )
    assert validate_human_review_evidence(document, candidate)
    assert "decisions" not in document

    decisions[0]["passed"] = False
    failed = build_human_review_evidence(
        candidate,
        decisions,
        population=population,
        population_contract_sha256=HUMAN_REVIEW_POPULATION_SHA256,
        source_decisions_sha256="d" * 64,
        attestation=HUMAN_REVIEW_ATTESTATION,
    )
    assert not validate_human_review_evidence(failed, candidate)

    decisions[0]["passed"] = True
    decisions[0]["passage_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="locked population"):
        build_human_review_evidence(
            candidate,
            decisions,
            population=population,
            population_contract_sha256=HUMAN_REVIEW_POPULATION_SHA256,
            source_decisions_sha256="d" * 64,
            attestation=HUMAN_REVIEW_ATTESTATION,
        )
