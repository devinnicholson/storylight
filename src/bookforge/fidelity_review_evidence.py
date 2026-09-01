"""Trusted contest-suite and locked human-review evidence for fidelity promotion."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from bookforge.fidelity_benchmark import CandidateIdentity

HUMAN_REVIEW_ATTESTATION = "I_REVIEWED_25_LOCKED_ADVERSARIAL_EXAMPLES"
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_RECORD_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,127}\Z")


def record_ids_sha256(record_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{record_id}\n" for record_id in sorted(record_ids)).encode()
    ).hexdigest()


def build_contest_evidence(
    candidate: CandidateIdentity,
    benchmark: Mapping[str, object],
    *,
    source_report_sha256: str,
) -> dict[str, object]:
    if _SHA256.fullmatch(source_report_sha256) is None:
        raise ValueError("contest source report SHA-256 is invalid")
    runtime = benchmark.get("runtime")
    acceptance = benchmark.get("acceptance")
    privacy = benchmark.get("privacy")
    cases = benchmark.get("cases")
    if (
        benchmark.get("schema_version") != "1.0"
        or not isinstance(runtime, dict)
        or runtime.get("suite") != "contest"
        or runtime.get("case_count") != 20
        or runtime.get("exit_code") != 0
        or not isinstance(acceptance, dict)
        or acceptance.get("all_outputs_schema_valid") is not True
        or acceptance.get("automatic_semantic_pass") is not True
        or not isinstance(privacy, dict)
        or privacy.get("model_and_artifacts_local_only") is not True
        or privacy.get("fixtures_are_synthetic") is not True
        or privacy.get("modal_or_cloud_called") is not False
        or not isinstance(cases, list)
        or len(cases) != 20
    ):
        raise ValueError("contest benchmark did not pass the fixed local 20-case suite")

    case_ids: list[str] = []
    semantic_checks = 0
    forbidden_checks = 0
    for case in cases:
        if not isinstance(case, dict) or case.get("valid") is not True:
            raise ValueError("contest benchmark contains an invalid case")
        case_id = case.get("case_id")
        semantic = case.get("semantic_checks")
        forbidden = case.get("forbidden_checks")
        if (
            not isinstance(case_id, str)
            or _RECORD_ID.fullmatch(case_id) is None
            or not isinstance(semantic, list)
            or not isinstance(forbidden, list)
            or case.get("automatic_semantic_pass") is not True
            or any(
                not isinstance(check, dict) or check.get("pass") is not True
                for check in semantic
            )
            or any(
                not isinstance(check, dict) or check.get("pass") is not True
                for check in forbidden
            )
        ):
            raise ValueError("contest benchmark contains a failed semantic or privacy check")
        case_ids.append(case_id)
        semantic_checks += len(semantic)
        forbidden_checks += len(forbidden)
    if len(set(case_ids)) != 20 or semantic_checks != 80 or forbidden_checks != 5:
        raise ValueError(
            "contest benchmark population differs from "
            "20 cases, 80 requirements, 5 forbiddens"
        )
    return {
        "schema_version": "story-fidelity-contest-suite-v1",
        "candidate_identity": asdict(candidate),
        "passed": True,
        "cases": 20,
        "semantic_checks": 80,
        "forbidden_checks": 5,
        "case_ids_sha256": record_ids_sha256(case_ids),
        "source_report_sha256": source_report_sha256,
    }


def build_human_review_evidence(
    candidate: CandidateIdentity,
    decisions: Sequence[Mapping[str, object]],
    *,
    source_decisions_sha256: str,
    attestation: str,
) -> dict[str, object]:
    if _SHA256.fullmatch(source_decisions_sha256) is None:
        raise ValueError("human-review source SHA-256 is invalid")
    if attestation != HUMAN_REVIEW_ATTESTATION:
        raise ValueError("human-review attestation is missing")
    if len(decisions) != 25:
        raise ValueError("human review must contain exactly 25 locked decisions")
    record_ids: list[str] = []
    failed: list[str] = []
    for decision in decisions:
        if set(decision) != {"record_id", "passed"}:
            raise ValueError("human-review decisions may contain only record_id and passed")
        record_id = decision.get("record_id")
        passed = decision.get("passed")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID.fullmatch(record_id) is None
            or type(passed) is not bool
        ):
            raise ValueError("human-review decision is malformed")
        record_ids.append(record_id)
        if not passed:
            failed.append(record_id)
    if len(set(record_ids)) != 25:
        raise ValueError("human-review record IDs must be unique")
    return {
        "schema_version": "story-fidelity-human-review-v1",
        "candidate_identity": asdict(candidate),
        "passed": not failed,
        "review_protocol": "locked-adversarial-v1",
        "records_reviewed": 25,
        "record_ids_sha256": record_ids_sha256(record_ids),
        "failed_record_ids": sorted(failed),
        "source_decisions_sha256": source_decisions_sha256,
        "reviewer_attestation": HUMAN_REVIEW_ATTESTATION,
        "retains_passages": False,
        "retains_model_outputs": False,
    }


def validate_contest_evidence(
    document: Mapping[str, object], candidate: CandidateIdentity
) -> bool:
    expected_fields = {
        "schema_version",
        "candidate_identity",
        "passed",
        "cases",
        "semantic_checks",
        "forbidden_checks",
        "case_ids_sha256",
        "source_report_sha256",
    }
    return (
        set(document) == expected_fields
        and document.get("schema_version") == "story-fidelity-contest-suite-v1"
        and document.get("candidate_identity") == asdict(candidate)
        and document.get("passed") is True
        and document.get("cases") == 20
        and document.get("semantic_checks") == 80
        and document.get("forbidden_checks") == 5
        and isinstance(document.get("case_ids_sha256"), str)
        and _SHA256.fullmatch(str(document["case_ids_sha256"])) is not None
        and isinstance(document.get("source_report_sha256"), str)
        and _SHA256.fullmatch(str(document["source_report_sha256"])) is not None
    )


def validate_human_review_evidence(
    document: Mapping[str, object], candidate: CandidateIdentity
) -> bool:
    expected_fields = {
        "schema_version",
        "candidate_identity",
        "passed",
        "review_protocol",
        "records_reviewed",
        "record_ids_sha256",
        "failed_record_ids",
        "source_decisions_sha256",
        "reviewer_attestation",
        "retains_passages",
        "retains_model_outputs",
    }
    return (
        set(document) == expected_fields
        and document.get("schema_version") == "story-fidelity-human-review-v1"
        and document.get("candidate_identity") == asdict(candidate)
        and document.get("passed") is True
        and document.get("review_protocol") == "locked-adversarial-v1"
        and document.get("records_reviewed") == 25
        and isinstance(document.get("record_ids_sha256"), str)
        and _SHA256.fullmatch(str(document["record_ids_sha256"])) is not None
        and document.get("failed_record_ids") == []
        and isinstance(document.get("source_decisions_sha256"), str)
        and _SHA256.fullmatch(str(document["source_decisions_sha256"])) is not None
        and document.get("reviewer_attestation") == HUMAN_REVIEW_ATTESTATION
        and document.get("retains_passages") is False
        and document.get("retains_model_outputs") is False
    )
