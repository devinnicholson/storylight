"""Trusted contest-suite and locked human-review evidence for fidelity promotion."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from storylight.fidelity_benchmark import CandidateIdentity
from storylight.planner_benchmark import CONTEST_CASES

HUMAN_REVIEW_ATTESTATION = "I_REVIEWED_25_LOCKED_ADVERSARIAL_EXAMPLES"
HUMAN_REVIEW_POPULATION_SHA256 = (
    "26e34c0d27fb0ed32008156c2e3663a12222073d60a10123a7080f36a9c11703"
)
HUMAN_REVIEW_DATASET_SHA256 = (
    "e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de"
)
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_RECORD_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,127}\Z")


def record_ids_sha256(record_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{record_id}\n" for record_id in sorted(record_ids)).encode()
    ).hexdigest()


def passage_hashes_sha256(passage_hashes: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{digest}\n" for digest in sorted(passage_hashes)).encode()
    ).hexdigest()


def validate_human_review_population(population: Mapping[str, object]) -> None:
    expected_fields = {
        "schema_version",
        "dataset_manifest_sha256",
        "passage_hashes_sha256",
        "privacy",
        "record_ids_sha256",
        "records",
        "records_reviewed",
        "selection_protocol",
        "split",
    }
    records = population.get("records")
    privacy = population.get("privacy")
    if (
        set(population) != expected_fields
        or population.get("schema_version")
        != "story-fidelity-human-review-population-v1"
        or population.get("selection_protocol")
        != "one-per-category-plus-critical-counterfactuals-v1"
        or population.get("split") != "development"
        or population.get("records_reviewed") != 25
        or _SHA256.fullmatch(str(population.get("dataset_manifest_sha256"))) is None
        or privacy != {"passages_retained": False, "model_outputs_retained": False}
        or not isinstance(records, list)
        or len(records) != 25
    ):
        raise ValueError("human-review population contract is invalid")
    pairs: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"record_id", "passage_sha256"}:
            raise ValueError("human-review population record is malformed")
        record_id = record.get("record_id")
        passage_sha256 = record.get("passage_sha256")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID.fullmatch(record_id) is None
            or not isinstance(passage_sha256, str)
            or _SHA256.fullmatch(passage_sha256) is None
        ):
            raise ValueError("human-review population record is malformed")
        pairs.append((record_id, passage_sha256))
    if len(set(pairs)) != 25 or len({record_id for record_id, _ in pairs}) != 25:
        raise ValueError("human-review population records must be unique")
    if population.get("record_ids_sha256") != record_ids_sha256(
        [record_id for record_id, _ in pairs]
    ) or population.get("passage_hashes_sha256") != passage_hashes_sha256(
        [digest for _, digest in pairs]
    ):
        raise ValueError("human-review population aggregate hashes are invalid")


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
        or runtime.get("candidate_revision") != candidate.model_revision
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
    expected_case_ids = {case.case_id for case in CONTEST_CASES}
    if (
        set(case_ids) != expected_case_ids
        or semantic_checks != 80
        or forbidden_checks != 5
    ):
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
    population: Mapping[str, object],
    population_contract_sha256: str,
    source_decisions_sha256: str,
    attestation: str,
) -> dict[str, object]:
    validate_human_review_population(population)
    if population_contract_sha256 != HUMAN_REVIEW_POPULATION_SHA256:
        raise ValueError("human-review population is not the repository-locked contract")
    if _SHA256.fullmatch(source_decisions_sha256) is None:
        raise ValueError("human-review source SHA-256 is invalid")
    if attestation != HUMAN_REVIEW_ATTESTATION:
        raise ValueError("human-review attestation is missing")
    if len(decisions) != 25:
        raise ValueError("human review must contain exactly 25 locked decisions")
    population_records = population["records"]
    assert isinstance(population_records, list)
    expected_pairs = {
        (str(record["record_id"]), str(record["passage_sha256"]))
        for record in population_records
        if isinstance(record, dict)
    }
    record_pairs: list[tuple[str, str]] = []
    failed: list[str] = []
    for decision in decisions:
        if set(decision) != {"record_id", "passage_sha256", "passed"}:
            raise ValueError(
                "human-review decisions must contain record_id, passage_sha256, and passed"
            )
        record_id = decision.get("record_id")
        passage_sha256 = decision.get("passage_sha256")
        passed = decision.get("passed")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID.fullmatch(record_id) is None
            or not isinstance(passage_sha256, str)
            or _SHA256.fullmatch(passage_sha256) is None
            or type(passed) is not bool
        ):
            raise ValueError("human-review decision is malformed")
        record_pairs.append((record_id, passage_sha256))
        if not passed:
            failed.append(record_id)
    if set(record_pairs) != expected_pairs or len(set(record_pairs)) != 25:
        raise ValueError("human-review decisions differ from the locked population")
    return {
        "schema_version": "story-fidelity-human-review-v1",
        "candidate_identity": asdict(candidate),
        "passed": not failed,
        "review_protocol": "locked-adversarial-v1",
        "records_reviewed": 25,
        "dataset_manifest_sha256": population["dataset_manifest_sha256"],
        "population_contract_sha256": population_contract_sha256,
        "record_ids_sha256": population["record_ids_sha256"],
        "passage_hashes_sha256": population["passage_hashes_sha256"],
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
        "dataset_manifest_sha256",
        "population_contract_sha256",
        "record_ids_sha256",
        "passage_hashes_sha256",
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
        and document.get("population_contract_sha256")
        == HUMAN_REVIEW_POPULATION_SHA256
        and document.get("record_ids_sha256")
        == "e6612fe1281b5d5d539b4cd30897740c1ea6e1a2805407ac3455708863ee86c6"
        and document.get("passage_hashes_sha256")
        == "0505e0f6132ec5df69ad8a9f3ec0c5c66c00ebc4431c48db94c1d5137d7dc3d9"
        and document.get("dataset_manifest_sha256") == HUMAN_REVIEW_DATASET_SHA256
        and document.get("failed_record_ids") == []
        and isinstance(document.get("source_decisions_sha256"), str)
        and _SHA256.fullmatch(str(document["source_decisions_sha256"])) is not None
        and document.get("reviewer_attestation") == HUMAN_REVIEW_ATTESTATION
        and document.get("retains_passages") is False
        and document.get("retains_model_outputs") is False
    )
