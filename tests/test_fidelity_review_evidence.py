from __future__ import annotations

from dataclasses import asdict

import pytest

from bookforge.fidelity_benchmark import CandidateIdentity
from bookforge.fidelity_review_evidence import (
    HUMAN_REVIEW_ATTESTATION,
    build_contest_evidence,
    build_human_review_evidence,
    validate_contest_evidence,
    validate_human_review_evidence,
)


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        candidate_id="candidate-review-test",
        candidate_manifest_sha256="a" * 64,
        engine_sha256="b" * 64,
        model_revision="sha256:" + "b" * 64,
    )


def _contest_report() -> dict[str, object]:
    cases = []
    for index in range(20):
        semantic_count = 4
        forbidden_count = 5 if index == 0 else 0
        cases.append(
            {
                "case_id": f"contest_case_{index:02d}",
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
        "runtime": {"suite": "contest", "case_count": 20, "exit_code": 0},
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


def test_human_review_retains_only_ids_and_requires_25_passes() -> None:
    candidate = _candidate()
    decisions = [
        {"record_id": f"locked_review_{index:02d}", "passed": True}
        for index in range(25)
    ]
    document = build_human_review_evidence(
        candidate,
        decisions,
        source_decisions_sha256="d" * 64,
        attestation=HUMAN_REVIEW_ATTESTATION,
    )
    assert validate_human_review_evidence(document, candidate)
    assert "decisions" not in document

    decisions[0]["passed"] = False
    failed = build_human_review_evidence(
        candidate,
        decisions,
        source_decisions_sha256="d" * 64,
        attestation=HUMAN_REVIEW_ATTESTATION,
    )
    assert not validate_human_review_evidence(failed, candidate)
