#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    FidelitySummary,
    candidate_identity_from_manifest,
    population_contract_from_manifest,
    summarize_evaluations,
)
from bookforge.fidelity_evaluation import concept_vocabulary, evaluate_surface
from bookforge.fidelity_manifest import sha256_path, validate_manifest
from bookforge.fidelity_orchestration import (
    FidelityRun,
    file_sha256,
    locked_fidelity_run,
    stage_plan,
)
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.formatting import format_training_record
from training.jax_fidelity.integrity import canonical_json_bytes
from training.jax_fidelity.roundtrip_smoke import contract_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage one durable Story Fidelity Lab run")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan")

    init = subparsers.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--dataset-manifest", type=Path, required=True)
    init.add_argument("--baseline-commit", required=True)
    init.add_argument("--baseline-engine-sha256", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)

    begin = subparsers.add_parser("begin")
    begin.add_argument("--state", type=Path, required=True)
    begin.add_argument("--stage", required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--state", type=Path, required=True)
    complete.add_argument("--stage", required=True)
    complete.add_argument("--artifact", type=Path, required=True)

    fail = subparsers.add_parser("fail")
    fail.add_argument("--state", type=Path, required=True)
    fail.add_argument("--stage", required=True)
    fail.add_argument("--detail", required=True)

    local = subparsers.add_parser(
        "run-local",
        help="execute the deterministic no-spend stages through compatibility packaging",
    )
    local.add_argument("--state", type=Path, required=True)
    local.add_argument("--artifacts-directory", type=Path, required=True)
    local.add_argument("--config", type=Path, required=True)
    local.add_argument("--manifest", type=Path, required=True)
    local.add_argument("--smoke-fixture", type=Path, required=True)

    baseline = subparsers.add_parser(
        "record-baseline",
        help="verify accepted-engine development and hidden reports and complete baseline",
    )
    baseline.add_argument("--state", type=Path, required=True)
    baseline.add_argument("--dataset-manifest", type=Path, required=True)
    baseline.add_argument("--baseline-manifest", type=Path, required=True)
    baseline.add_argument("--baseline-manifest-sha256", required=True)
    baseline.add_argument("--development-report", type=Path, required=True)
    baseline.add_argument("--development-report-sha256", required=True)
    baseline.add_argument("--hidden-report", type=Path, required=True)
    baseline.add_argument("--hidden-report-sha256", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "plan":
        print(json.dumps({"schema_version": "1.0", "stages": stage_plan()}, indent=2))
        return 0
    if arguments.command == "init":
        if arguments.state.exists():
            raise FileExistsError(f"run state already exists: {arguments.state}")
        run = FidelityRun.create(
            run_id=arguments.run_id,
            config_sha256=file_sha256(arguments.config),
            dataset_manifest_sha256=file_sha256(arguments.dataset_manifest),
            baseline_commit=arguments.baseline_commit,
            baseline_engine_sha256=arguments.baseline_engine_sha256,
        )
        run.write(arguments.state)
        _print_status(run)
        return 0
    if arguments.command == "status":
        _print_status(FidelityRun.read(arguments.state))
        return 0
    if arguments.command == "run-local":
        _run_local(arguments)
        return 0
    if arguments.command == "record-baseline":
        _record_baseline(arguments)
        return 0
    with locked_fidelity_run(arguments.state) as run:
        if arguments.command == "begin":
            run.begin(arguments.stage)
        elif arguments.command == "complete":
            run.complete(arguments.stage, artifact=arguments.artifact)
        elif arguments.command == "fail":
            run.fail(arguments.stage, detail=arguments.detail)
        _print_status(run)
    return 0


def _print_status(run: FidelityRun) -> None:
    next_stage = run.next_stage()
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run.run_id,
                "config_sha256": run.config_sha256,
                "dataset_manifest_sha256": run.dataset_manifest_sha256,
                "next_stage": next_stage.name if next_stage is not None else None,
                "stages": {
                    name: {
                        "status": record.status.value,
                        "artifact_sha256": record.artifact_sha256,
                        "detail": record.detail,
                    }
                    for name, record in run.stages.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_records(path: Path) -> list[FidelityRecord]:
    records = [
        FidelityRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("smoke fixture is empty")
    return records


def _approved_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{label} differs from its approved SHA-256")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _baseline_summary(
    document: dict[str, object],
    *,
    split: str,
    identity: CandidateIdentity,
    run: FidelityRun,
    manifest_path: Path,
) -> str | None:
    if (
        document.get("schema_version") != "story-fidelity-evaluation-v1"
        or document.get("split") != split
        or document.get("candidate_identity") != asdict(identity)
        or document.get("dataset_manifest_sha256") != run.dataset_manifest_sha256
        or document.get("privacy") != {"passages_recorded": False, "outputs_recorded": False}
    ):
        raise ValueError(f"baseline {split} report identity or privacy contract changed")
    raw_summary = document.get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError(f"baseline {split} report has no summary")
    try:
        summary = FidelitySummary(**raw_summary)
    except TypeError as error:
        raise ValueError(f"baseline {split} summary is invalid") from error
    population = population_contract_from_manifest(
        manifest_path,
        expected_manifest_sha256=run.dataset_manifest_sha256,
        split=DatasetSplit(split),
    )
    if (
        summary.surface != "raw"
        or summary.split != split
        or summary.records != population.records
        or summary.record_ids_sha256 != population.record_ids_sha256
        or summary.counterfactual_pairs != population.pairs
        or dict(summary.category_record_counts) != dict(population.category_record_counts)
    ):
        raise ValueError(f"baseline {split} summary population changed")
    receipt = document.get("custody_receipt_sha256")
    if split == "hidden":
        if not isinstance(receipt, str) or re.fullmatch(r"[a-f0-9]{64}", receipt) is None:
            raise ValueError("baseline hidden report has no custody receipt binding")
        return receipt
    if receipt is not None:
        raise ValueError("development report unexpectedly contains hidden custody evidence")
    return None


def _publish_artifact(path: Path, document: dict[str, object]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace stage evidence: {path}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _stage_document(
    run: FidelityRun,
    stage: str,
    *,
    evidence: dict[str, object],
) -> dict[str, object]:
    dependencies = next(item for item in stage_plan() if item["name"] == stage)["dependencies"]
    return {
        "schema_version": "1.0",
        "producer": "bookforge-local-preflight",
        "stage": stage,
        "run_id": run.run_id,
        "config_sha256": run.config_sha256,
        "dataset_manifest_sha256": run.dataset_manifest_sha256,
        "status": "succeeded",
        "inputs": {
            dependency: run.stages[str(dependency)].artifact_sha256 for dependency in dependencies
        },
        "evidence": evidence,
    }


def _local_stage_evidence(
    stage: str,
    *,
    config_path: Path,
    manifest_path: Path,
    smoke_fixture: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    manifest_sha256 = sha256_path(manifest_path)
    manifest = validate_manifest(manifest_path)
    if stage == "dataset":
        return {
            "dataset_manifest_sha256": manifest_sha256,
            "manifest_version": manifest.manifest_version,
            "record_schema_sha256": manifest.record_schema_sha256,
            "generator_source_sha256": manifest.generator_source_sha256,
            "generator_config_sha256": manifest.generator_config_sha256,
            "split_records": {
                split.value: declaration.records for split, declaration in manifest.splits.items()
            },
        }

    records = _load_records(smoke_fixture)
    if len(records) != 32 or {record.split.value for record in records} != {"development"}:
        raise ValueError("CPU smoke requires exactly 32 development records")
    if stage == "cpu-smoke":
        vocabulary = concept_vocabulary(records)
        evaluations = [
            evaluate_surface(
                record,
                record.target.as_wire(),
                surface="raw",
                concept_vocabulary=vocabulary,
            )
            for record in records
        ]
        summary = summarize_evaluations(evaluations)
        if not all(evaluation.exact_example_pass for evaluation in evaluations):
            raise RuntimeError("known-good smoke targets failed deterministic evaluation")
        return {
            "dataset_manifest_sha256": manifest_sha256,
            "fixture_sha256": sha256_path(smoke_fixture),
            "summary": asdict(summary),
        }
    if stage == "compatibility-package":
        formatted = [
            format_training_record(record.model_dump(mode="json", by_alias=True))
            for record in records
        ]
        digest = hashlib.sha256(
            b"".join(canonical_json_bytes(row) for row in formatted)
        ).hexdigest()
        return {
            "dataset_manifest_sha256": manifest_sha256,
            "config_sha256": config.sha256,
            "formatted_smoke_sha256": digest,
            "formatted_records": len(formatted),
            "roundtrip_contract": contract_document(config),
        }
    raise ValueError(f"stage is not a local preflight stage: {stage}")


def _run_local(arguments: argparse.Namespace) -> None:
    allowed = ("dataset", "cpu-smoke", "compatibility-package")
    with locked_fidelity_run(arguments.state) as run:
        config = load_config(arguments.config)
        if config.sha256 != run.config_sha256:
            raise ValueError("run state is not bound to the supplied configuration")
        if sha256_path(arguments.manifest) != run.dataset_manifest_sha256:
            raise ValueError("run state is not bound to the supplied dataset manifest")
        for stage in allowed:
            record = run.stages[stage]
            if record.status.value == "completed":
                continue
            next_stage = run.next_stage()
            if next_stage is None or next_stage.name != stage:
                raise ValueError(f"local preflight cannot advance from {next_stage}")
            run.begin(stage)
            try:
                evidence = _local_stage_evidence(
                    stage,
                    config_path=arguments.config,
                    manifest_path=arguments.manifest,
                    smoke_fixture=arguments.smoke_fixture,
                )
                artifact = arguments.artifacts_directory / f"{stage}.json"
                _publish_artifact(artifact, _stage_document(run, stage, evidence=evidence))
                run.complete(stage, artifact=artifact)
            except Exception as error:
                run.fail(stage, detail=f"{type(error).__name__}: {error}"[:500])
                raise
        _print_status(run)


def _record_baseline(arguments: argparse.Namespace) -> None:
    snapshot = FidelityRun.read(arguments.state)
    manifest = _approved_json(
        arguments.baseline_manifest,
        arguments.baseline_manifest_sha256,
        "baseline manifest",
    )
    identity = candidate_identity_from_manifest(
        manifest,
        manifest_sha256=arguments.baseline_manifest_sha256,
    )
    if identity.engine_sha256 != snapshot.baseline_engine_sha256:
        raise ValueError("baseline manifest does not identify the accepted engine")
    if manifest.get("source_dataset_manifest_sha256") != snapshot.dataset_manifest_sha256:
        raise ValueError("baseline manifest was not bound to this dataset manifest")
    development = _approved_json(
        arguments.development_report,
        arguments.development_report_sha256,
        "baseline development report",
    )
    hidden = _approved_json(
        arguments.hidden_report,
        arguments.hidden_report_sha256,
        "baseline hidden report",
    )
    _baseline_summary(
        development,
        split="development",
        identity=identity,
        run=snapshot,
        manifest_path=arguments.dataset_manifest,
    )
    receipt_sha256 = _baseline_summary(
        hidden,
        split="hidden",
        identity=identity,
        run=snapshot,
        manifest_path=arguments.dataset_manifest,
    )
    with locked_fidelity_run(arguments.state) as run:
        if run != snapshot:
            raise RuntimeError("run state changed while baseline evidence was being verified")
        run.begin("baseline")
        artifact = {
            "schema_version": "1.0",
            "producer": "bookforge-baseline-recorder",
            "stage": "baseline",
            "run_id": run.run_id,
            "config_sha256": run.config_sha256,
            "dataset_manifest_sha256": run.dataset_manifest_sha256,
            "status": "succeeded",
            "inputs": {
                "compatibility-package": run.stages["compatibility-package"].artifact_sha256
            },
            "baseline_identity": asdict(identity),
            "hidden_custody_receipt_sha256": receipt_sha256,
            "evidence_sha256": {
                "baseline_manifest": arguments.baseline_manifest_sha256,
                "development_summary": arguments.development_report_sha256,
                "hidden_summary": arguments.hidden_report_sha256,
                "dataset_manifest": run.dataset_manifest_sha256,
            },
        }
        _publish_artifact(arguments.output, artifact)
        run.complete("baseline", artifact=arguments.output)
        _print_status(run)


if __name__ == "__main__":
    raise SystemExit(main())
