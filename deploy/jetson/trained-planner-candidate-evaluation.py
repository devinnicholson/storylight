#!/usr/bin/env python3
"""Gate a Jetson candidate's development report before hidden evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bookforge.fidelity_benchmark import (
    CRITICAL_CATEGORIES,
    DEFAULT_PROMOTION_THRESHOLDS,
    FidelitySummary,
    candidate_identity_from_manifest,
    population_contract_from_manifest,
)
from bookforge.fidelity_schema import DatasetSplit

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ACCEPTED_ENGINE_SHA256 = "95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
_REPORT_FIELDS = {
    "schema_version",
    "split",
    "candidate_identity",
    "dataset_manifest_sha256",
    "custody_receipt_sha256",
    "privacy",
    "summary",
}


def _development_checks(
    candidate: FidelitySummary,
    baseline: FidelitySummary,
) -> dict[str, bool]:
    thresholds = DEFAULT_PROMOTION_THRESHOLDS
    baseline_failures = baseline.records * (1 - baseline.exact_example_pass_rate)
    candidate_failures = candidate.records * (1 - candidate.exact_example_pass_rate)
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
        "semantic_atom_recall": (candidate.semantic_atom_recall >= thresholds.semantic_atom_recall),
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
            candidate.exact_example_pass_rate - baseline.exact_example_pass_rate
            >= thresholds.minimum_development_improvement
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} checksum is not a lowercase SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} checksum changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _summary(
    document: dict[str, Any],
    *,
    label: str,
    dataset_manifest_sha256: str,
) -> FidelitySummary:
    if (
        set(document) != _REPORT_FIELDS
        or document.get("schema_version") != "story-fidelity-evaluation-v1"
        or document.get("split") != DatasetSplit.DEVELOPMENT.value
        or document.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("custody_receipt_sha256") is not None
        or document.get("privacy") != {"passages_recorded": False, "outputs_recorded": False}
    ):
        raise ValueError(f"{label} is not a development-only endpoint report")
    raw = document.get("summary")
    if not isinstance(raw, dict):
        raise ValueError(f"{label} has no fidelity summary")
    try:
        summary = FidelitySummary(**raw)
    except TypeError as error:
        raise ValueError(f"{label} has a malformed fidelity summary") from error
    if summary.surface != "raw" or summary.split != DatasetSplit.DEVELOPMENT.value:
        raise ValueError(f"{label} has the wrong output surface or split")
    return summary


def build_development_gate(
    *,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    candidate_report_path: Path,
    candidate_report_sha256: str,
    baseline_report_path: Path,
    baseline_report_sha256: str,
) -> dict[str, object]:
    candidate_manifest = _document(
        candidate_manifest_path,
        candidate_manifest_sha256,
        "candidate manifest",
    )
    identity = candidate_identity_from_manifest(
        candidate_manifest,
        manifest_sha256=candidate_manifest_sha256,
    )
    training_run_id = candidate_manifest.get("training_run_id")
    if (
        candidate_manifest.get("source_dataset_manifest_sha256") != dataset_manifest_sha256
        or not isinstance(training_run_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", training_run_id) is None
    ):
        raise ValueError("candidate manifest has invalid training or dataset lineage")
    candidate_document = _document(
        candidate_report_path,
        candidate_report_sha256,
        "candidate development report",
    )
    baseline_document = _document(
        baseline_report_path,
        baseline_report_sha256,
        "baseline development report",
    )
    candidate_identity = candidate_document.get("candidate_identity")
    if not isinstance(candidate_identity, dict) or candidate_identity != asdict(identity):
        raise ValueError("candidate development report identifies another serving engine")
    baseline_identity = baseline_document.get("candidate_identity")
    if (
        not isinstance(baseline_identity, dict)
        or not str(baseline_identity.get("candidate_id", "")).startswith("accepted-baseline-")
        or baseline_identity.get("engine_sha256") != _ACCEPTED_ENGINE_SHA256
        or baseline_identity.get("model_revision") != f"sha256:{_ACCEPTED_ENGINE_SHA256}"
        or _SHA256.fullmatch(str(baseline_identity.get("candidate_manifest_sha256"))) is None
    ):
        raise ValueError("baseline development report is not the accepted engine")
    candidate = _summary(
        candidate_document,
        label="candidate development report",
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    baseline = _summary(
        baseline_document,
        label="baseline development report",
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    population = population_contract_from_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        split=DatasetSplit.DEVELOPMENT,
    )
    for label, summary in (("candidate", candidate), ("baseline", baseline)):
        if (
            summary.records != population.records
            or summary.counterfactual_pairs != population.pairs
            or summary.record_ids_sha256 != population.record_ids_sha256
            or dict(summary.category_record_counts) != dict(population.category_record_counts)
        ):
            raise ValueError(f"{label} development report uses another population")
    checks = _development_checks(candidate, baseline)
    passed = all(checks.values())
    return {
        "schema_version": "1.0",
        "producer": "bookforge-jetson-candidate-development-gate",
        "status": "passed" if passed else "rejected",
        "candidate_identity": asdict(identity),
        "training_run_id": training_run_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "candidate_report_sha256": candidate_report_sha256,
        "baseline_report_sha256": baseline_report_sha256,
        "checks": checks,
        "reasons": [name for name, result in checks.items() if not result],
    }


def _write_once(path: Path, document: dict[str, object]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_development_gate(
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        dataset_manifest_path=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        candidate_report_path=args.candidate_report,
        candidate_report_sha256=args.candidate_report_sha256,
        baseline_report_path=args.baseline_report,
        baseline_report_sha256=args.baseline_report_sha256,
    )
    _write_once(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    if document["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
