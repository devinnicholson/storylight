"""Build content-bound Story Fidelity promotion evidence from approved inputs."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    FidelitySummary,
    PromotionDecision,
    RuntimeEvidence,
    candidate_identity_from_manifest,
    decide_promotion,
    population_contract_from_manifest,
)
from bookforge.fidelity_manifest import sha256_path
from bookforge.fidelity_schema import DatasetSplit

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _secure_document(path: Path, *, expected_sha256: str, label: str) -> dict[str, object]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} approved SHA-256 is invalid")
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if sha256_path(path) != expected_sha256:
        raise ValueError(f"{label} differs from its approved SHA-256")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _identity(document: Mapping[str, object], label: str) -> CandidateIdentity:
    value = document.get("candidate_identity")
    if not isinstance(value, dict):
        raise ValueError(f"{label} has no candidate identity")
    manifest_sha256 = value.get("candidate_manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise ValueError(f"{label} has no candidate manifest digest")
    return candidate_identity_from_manifest(value, manifest_sha256=manifest_sha256)


def _summary(
    document: Mapping[str, object],
    *,
    label: str,
    split: DatasetSplit,
    dataset_manifest_sha256: str,
) -> tuple[FidelitySummary, CandidateIdentity, str | None]:
    if document.get("schema_version") != "story-fidelity-evaluation-v1":
        raise ValueError(f"{label} has an unsupported schema version")
    if document.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError(f"{label} was evaluated against another dataset manifest")
    privacy = document.get("privacy")
    if privacy != {"passages_recorded": False, "outputs_recorded": False}:
        raise ValueError(f"{label} did not preserve the evaluation privacy contract")
    raw_summary = document.get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError(f"{label} has no fidelity summary")
    try:
        summary = FidelitySummary(**raw_summary)
    except TypeError as error:
        raise ValueError(f"{label} fidelity summary is invalid") from error
    if summary.split != split.value or summary.surface != "raw":
        raise ValueError(f"{label} has the wrong split or output surface")
    receipt_sha = document.get("custody_receipt_sha256")
    if split is DatasetSplit.HIDDEN:
        if not isinstance(receipt_sha, str) or _SHA256.fullmatch(receipt_sha) is None:
            raise ValueError(f"{label} is not bound to hidden custody evidence")
    elif receipt_sha is not None:
        raise ValueError(f"{label} unexpectedly contains hidden custody evidence")
    return summary, _identity(document, label), receipt_sha


def _candidate_bound_pass(
    document: Mapping[str, object],
    *,
    label: str,
    candidate: CandidateIdentity,
    schema_version: str,
) -> bool:
    if document.get("schema_version") != schema_version:
        raise ValueError(f"{label} has an unsupported schema version")
    if _identity(document, label) != candidate:
        raise ValueError(f"{label} identifies another candidate")
    return document.get("passed") is True


def _runtime(document: Mapping[str, object], *, candidate: CandidateIdentity) -> RuntimeEvidence:
    if (
        set(document) != {"schema_version", "candidate_identity", "runtime"}
        or document.get("schema_version") != "story-fidelity-runtime-v1"
    ):
        raise ValueError("runtime evidence has an unsupported schema version")
    if _identity(document, "runtime evidence") != candidate:
        raise ValueError("runtime evidence identifies another candidate")
    raw = document.get("runtime")
    if not isinstance(raw, dict):
        raise ValueError("runtime evidence has no telemetry summary")
    try:
        return RuntimeEvidence(**raw)
    except TypeError as error:
        raise ValueError("runtime telemetry fields are invalid") from error


def _shadow(
    document: Mapping[str, object],
    *,
    candidate: CandidateIdentity,
    run_id: str,
    training_run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    runtime_sha256: str,
) -> str:
    expected_fields = {
        "schema_version",
        "stage",
        "producer",
        "run_id",
        "training_run_id",
        "config_sha256",
        "dataset_manifest_sha256",
        "status",
        "inputs",
        "candidate_id",
        "candidate_manifest_sha256",
        "candidate_identity",
        "shadow_status",
        "evidence_sha256",
    }
    if set(document) != expected_fields or (
        document.get("schema_version") != "1.0"
        or document.get("stage") != "jetson-shadow"
        or document.get("producer") != "bookforge-jetson-shadow-recorder"
        or document.get("run_id") != run_id
        or document.get("training_run_id") != training_run_id
        or document.get("config_sha256") != config_sha256
        or document.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("status") != "succeeded"
    ):
        raise ValueError("Jetson shadow evidence is not a successful artifact for this run")
    if (
        document.get("candidate_id") != candidate.candidate_id
        or document.get("candidate_manifest_sha256") != candidate.candidate_manifest_sha256
        or _identity(document, "Jetson shadow evidence") != candidate
    ):
        raise ValueError("Jetson shadow evidence identifies another candidate or engine")
    inputs = document.get("inputs")
    evidence = document.get("evidence_sha256")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"int4-export"}
        or _SHA256.fullmatch(str(inputs.get("int4-export"))) is None
        or document.get("shadow_status") not in {"passed", "rejected"}
        or not isinstance(evidence, dict)
        or set(evidence) != {"runtime", "candidate_manifest", "hidden_summary"}
        or evidence.get("runtime") != runtime_sha256
        or evidence.get("candidate_manifest") != candidate.candidate_manifest_sha256
        or _SHA256.fullmatch(str(evidence.get("hidden_summary"))) is None
    ):
        raise ValueError("Jetson shadow evidence has invalid dependency bindings")
    return str(document["shadow_status"])


def build_gate_artifact(
    *,
    run_id: str,
    config_sha256: str,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    candidate_hidden_path: Path,
    candidate_hidden_sha256: str,
    baseline_hidden_path: Path,
    baseline_hidden_sha256: str,
    candidate_development_path: Path,
    candidate_development_sha256: str,
    baseline_development_path: Path,
    baseline_development_sha256: str,
    runtime_path: Path,
    runtime_sha256: str,
    contest_path: Path,
    contest_sha256: str,
    human_review_path: Path,
    human_review_sha256: str,
    jetson_shadow_path: Path,
    jetson_shadow_sha256: str,
) -> dict[str, object]:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", run_id) is None:
        raise ValueError("run ID is invalid")
    if _SHA256.fullmatch(config_sha256) is None:
        raise ValueError("configuration SHA-256 is invalid")

    dataset_document = _secure_document(
        dataset_manifest_path,
        expected_sha256=dataset_manifest_sha256,
        label="dataset manifest",
    )
    candidate_manifest = _secure_document(
        candidate_manifest_path,
        expected_sha256=candidate_manifest_sha256,
        label="candidate manifest",
    )
    candidate = candidate_identity_from_manifest(
        candidate_manifest,
        manifest_sha256=candidate_manifest_sha256,
    )
    training_run_id = candidate_manifest.get("training_run_id")
    if (
        not isinstance(training_run_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", training_run_id) is None
    ):
        raise ValueError("candidate manifest has an invalid training run ID")
    if candidate_manifest.get("source_dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("candidate manifest was built from another dataset manifest")
    if candidate_manifest.get("source_config_sha256") != config_sha256:
        raise ValueError("candidate manifest was trained with another configuration")
    if dataset_document.get("dataset_id") != "story-fidelity-v1":
        raise ValueError("dataset manifest has an unexpected dataset identity")

    approved = {
        "candidate_hidden_summary": (
            candidate_hidden_path,
            candidate_hidden_sha256,
        ),
        "baseline_hidden_summary": (baseline_hidden_path, baseline_hidden_sha256),
        "candidate_development_summary": (
            candidate_development_path,
            candidate_development_sha256,
        ),
        "baseline_development_summary": (
            baseline_development_path,
            baseline_development_sha256,
        ),
        "runtime": (runtime_path, runtime_sha256),
        "contest": (contest_path, contest_sha256),
        "human_review": (human_review_path, human_review_sha256),
        "jetson_shadow": (jetson_shadow_path, jetson_shadow_sha256),
    }
    documents = {
        name: _secure_document(path, expected_sha256=digest, label=name.replace("_", " "))
        for name, (path, digest) in approved.items()
    }

    hidden_candidate, hidden_candidate_identity, candidate_receipt = _summary(
        documents["candidate_hidden_summary"],
        label="candidate hidden summary",
        split=DatasetSplit.HIDDEN,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    hidden_baseline, hidden_baseline_identity, baseline_receipt = _summary(
        documents["baseline_hidden_summary"],
        label="baseline hidden summary",
        split=DatasetSplit.HIDDEN,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    development_candidate, development_candidate_identity, _ = _summary(
        documents["candidate_development_summary"],
        label="candidate development summary",
        split=DatasetSplit.DEVELOPMENT,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    development_baseline, development_baseline_identity, _ = _summary(
        documents["baseline_development_summary"],
        label="baseline development summary",
        split=DatasetSplit.DEVELOPMENT,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    if hidden_candidate_identity != candidate or development_candidate_identity != candidate:
        raise ValueError("candidate summaries identify another candidate or serving engine")
    if hidden_baseline_identity != development_baseline_identity:
        raise ValueError("baseline summaries do not identify the same baseline engine")
    if candidate_receipt != baseline_receipt:
        raise ValueError("hidden summaries were not evaluated under the same custody state")

    runtime = _runtime(documents["runtime"], candidate=candidate)
    contest_passed = _candidate_bound_pass(
        documents["contest"],
        label="contest evidence",
        candidate=candidate,
        schema_version="story-fidelity-contest-suite-v1",
    )
    human_review_passed = _candidate_bound_pass(
        documents["human_review"],
        label="human review evidence",
        candidate=candidate,
        schema_version="story-fidelity-human-review-v1",
    )
    shadow_status = _shadow(
        documents["jetson_shadow"],
        candidate=candidate,
        run_id=run_id,
        training_run_id=training_run_id,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        runtime_sha256=runtime_sha256,
    )

    hidden_population = population_contract_from_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        split=DatasetSplit.HIDDEN,
    )
    development_population = population_contract_from_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        split=DatasetSplit.DEVELOPMENT,
    )
    decision = decide_promotion(
        hidden_candidate,
        baseline=hidden_baseline,
        development_candidate=development_candidate,
        development_baseline=development_baseline,
        runtime=runtime,
        contest_suite_passed=contest_passed,
        human_review_passed=human_review_passed,
        population=hidden_population,
        development_population=development_population,
    )
    if shadow_status != "passed":
        checks = {**decision.checks, "jetson_shadow": False}
        decision = PromotionDecision(
            passed=False,
            reasons=tuple(name for name, passed in checks.items() if not passed),
            checks=checks,
        )
    evidence_sha256 = {
        name: digest for name, (_, digest) in approved.items() if name != "jetson_shadow"
    }
    evidence_sha256["candidate_manifest"] = candidate_manifest_sha256
    evidence_sha256["dataset_manifest"] = dataset_manifest_sha256
    return {
        "schema_version": "1.0",
        "stage": "gate",
        "producer": "bookforge-fidelity-gate-builder",
        "run_id": run_id,
        "training_run_id": training_run_id,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "status": "passed" if decision.passed else "rejected",
        "inputs": {"jetson-shadow": jetson_shadow_sha256},
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_identity": asdict(candidate),
        "baseline_identity": asdict(hidden_baseline_identity),
        "hidden_custody_receipt_sha256": candidate_receipt,
        "evidence_sha256": dict(sorted(evidence_sha256.items())),
        "decision": {
            "passed": decision.passed,
            "reasons": list(decision.reasons),
            "checks": dict(sorted(decision.checks.items())),
        },
    }


def write_gate_artifact(path: Path, artifact: Mapping[str, object]) -> str:
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_path(path)
