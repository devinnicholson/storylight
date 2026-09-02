#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    resolved = str(import_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    FidelitySummary,
    candidate_identity_from_manifest,
    population_contract_from_manifest,
    summarize_evaluations,
)
from bookforge.fidelity_closure_evidence import (
    validate_reconciliation_evidence,
    validate_terminal_evidence,
)
from bookforge.fidelity_evaluation import concept_vocabulary, evaluate_surface
from bookforge.fidelity_lineage import stable_run_id
from bookforge.fidelity_manifest import sha256_path, validate_manifest
from bookforge.fidelity_orchestration import (
    FidelityRun,
    file_sha256,
    locked_fidelity_run,
    stage_plan,
    typed_stage_document,
    validate_stage_artifact,
)
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.development_eligibility import (
    validate_development_eligibility,
)
from training.jax_fidelity.formatting import format_training_record
from training.jax_fidelity.integrity import (
    canonical_json_bytes,
    canonical_sha256,
    verify_artifact_manifest,
)
from training.jax_fidelity.orbax_receipt import verify_orbax_leaf_receipt
from training.jax_fidelity.release import (
    candidate_id_for_checkpoint,
    candidate_id_from_lineage,
)
from training.jax_fidelity.roundtrip_smoke import (
    contract_document,
    validate_roundtrip_evidence,
)


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

    roundtrip = subparsers.add_parser(
        "record-roundtrip",
        help="verify conversion, numerical, and five-step smoke evidence",
    )
    _add_record_paths(roundtrip)
    roundtrip.add_argument("--config", type=Path, required=True)
    roundtrip.add_argument("--conversion-run", type=Path, required=True)
    roundtrip.add_argument("--conversion-completion", type=Path, required=True)
    roundtrip.add_argument("--hf-to-maxtext-completion", type=Path, required=True)
    roundtrip.add_argument("--maxtext-to-hf-completion", type=Path, required=True)
    roundtrip.add_argument("--roundtrip-evidence", type=Path, required=True)
    roundtrip.add_argument("--exported-checkpoint", type=Path, required=True)
    roundtrip.add_argument("--exported-checkpoint-manifest", type=Path, required=True)
    roundtrip.add_argument("--smoke-training-run", type=Path, required=True)
    roundtrip.add_argument("--smoke-training-completion", type=Path, required=True)
    roundtrip.add_argument("--smoke-adapter", type=Path, required=True)
    roundtrip.add_argument("--smoke-adapter-manifest", type=Path, required=True)

    training = subparsers.add_parser(
        "record-training",
        help="verify one finite full-training release package",
    )
    _add_record_paths(training)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--base-checkpoint-root", type=Path, required=True)
    training.add_argument("--base-checkpoint-receipt", type=Path, required=True)
    training.add_argument("--base-checkpoint-receipt-sha256", required=True)
    training.add_argument("--base-checkpoint-manifest", type=Path, required=True)
    training.add_argument("--base-checkpoint-manifest-sha256", required=True)
    training.add_argument("--tokenizer-manifest", type=Path, required=True)
    training.add_argument("--remote-completion", type=Path, required=True)
    training.add_argument("--remote-completion-sha256", required=True)
    training.add_argument("--training-run", type=Path, required=True)
    training.add_argument("--training-completion", type=Path, required=True)
    training.add_argument("--adapter", type=Path, required=True)
    training.add_argument("--adapter-manifest", type=Path, required=True)
    training.add_argument("--package-root", type=Path, required=True)
    training.add_argument("--package-manifest", type=Path, required=True)
    training.add_argument("--runtime-lock", type=Path, required=True)
    training.add_argument(
        "--backend",
        choices=("vertex-tpu-v6e", "modal-l4x2"),
        required=True,
    )

    evaluation = subparsers.add_parser(
        "record-candidate-evaluation",
        help="bind a development-only decision to merged checkpoint bytes",
    )
    _add_record_paths(evaluation)
    evaluation.add_argument("--development-evaluation", type=Path, required=True)
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--merged-checkpoint", type=Path, required=True)
    evaluation.add_argument("--merged-checkpoint-manifest", type=Path, required=True)

    hf_export = subparsers.add_parser(
        "record-hf-export",
        help="verify and record a complete merged-HF release",
    )
    _add_record_paths(hf_export)
    hf_export.add_argument("--release-manifest", type=Path, required=True)
    hf_export.add_argument("--release-root", type=Path, required=True)
    hf_export.add_argument("--training-run", type=Path, required=True)
    hf_export.add_argument("--training-completion", type=Path, required=True)
    hf_export.add_argument("--roundtrip-evidence", type=Path, required=True)
    hf_export.add_argument("--development-evaluation", type=Path, required=True)

    int4_export = subparsers.add_parser(
        "record-int4-export",
        help="verify a checksum-bound TensorRT Edge-LLM INT4 export",
    )
    _add_record_paths(int4_export)
    int4_export.add_argument("--source-release-manifest", type=Path, required=True)
    int4_export.add_argument("--export-manifest", type=Path, required=True)
    int4_export.add_argument("--export-root", type=Path, required=True)

    for command, help_text in (
        ("record-jetson-shadow", "adopt an exact Jetson shadow stage artifact"),
        ("record-gate", "adopt an exact fidelity gate stage artifact"),
    ):
        adopted = subparsers.add_parser(command, help=help_text)
        _add_record_paths(adopted)
        adopted.add_argument("--source", type=Path, required=True)
        adopted.add_argument("--source-sha256", required=True)

    decision = subparsers.add_parser(
        "record-terminal-decision",
        help="record promotion or baseline retention after the gate",
    )
    _add_record_paths(decision)
    decision.add_argument("--gate-artifact", type=Path, required=True)
    decision.add_argument("--deployment-receipt", type=Path, required=True)
    decision.add_argument("--accepted-engine-health", type=Path, required=True)
    decision.add_argument("--rollback-state", type=Path)

    reconcile = subparsers.add_parser(
        "record-reconciliation",
        help="verify cost/resource closure and finish the campaign",
    )
    _add_record_paths(reconcile)
    reconcile.add_argument("--cost-ledger", type=Path, required=True)
    reconcile.add_argument("--paid-resource-inventory", type=Path, required=True)
    reconcile.add_argument("--deployment-artifact", type=Path, required=True)
    return parser


def _add_record_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


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
    recorder = {
        "record-roundtrip": _record_roundtrip,
        "record-training": _record_training,
        "record-candidate-evaluation": _record_candidate_evaluation,
        "record-hf-export": _record_hf_export,
        "record-int4-export": _record_int4_export,
        "record-jetson-shadow": _record_adopted_stage,
        "record-gate": _record_adopted_stage,
        "record-terminal-decision": _record_terminal_decision,
        "record-reconciliation": _record_reconciliation,
    }.get(arguments.command)
    if recorder is not None:
        recorder(arguments)
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


def _source_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    digest = file_sha256(path)
    return _approved_json(path, digest, label), digest


def _source_digest(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise ValueError(f"{label} must be a nonempty regular file")
    return file_sha256(path)


def _release_canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_run_pair(
    run_document: Mapping[str, object],
    run_sha256: str,
    completion: Mapping[str, object],
    *,
    expected_run_id: str,
    expected_stage: str,
    config_sha256: str,
    dataset_manifest_sha256: str | None,
) -> None:
    if (
        run_document.get("schema_version") != "1.0"
        or run_document.get("run_id") != expected_run_id
        or run_document.get("stage") != expected_stage
        or run_document.get("status") != "planned"
        or run_document.get("config_sha256") != config_sha256
    ):
        raise ValueError(f"{expected_stage} run manifest changed its immutable lineage")
    if (
        dataset_manifest_sha256 is not None
        and run_document.get("dataset_manifest_sha256") != dataset_manifest_sha256
    ):
        raise ValueError(f"{expected_stage} run changed the dataset manifest")
    if (
        completion.get("schema_version") != "1.0"
        or completion.get("run_id") != expected_run_id
        or completion.get("status") != "succeeded"
        or completion.get("run_manifest_sha256") != run_sha256
        or not isinstance(completion.get("artifacts"), list)
        or not completion["artifacts"]
        or not isinstance(completion.get("evidence"), dict)
        or not completion["evidence"]
    ):
        raise ValueError(f"{expected_stage} completion is not successful terminal evidence")


def _completion_covers_manifest(
    completion: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    artifact_root: Path,
    label: str,
) -> None:
    rows = completion.get("artifacts")
    files = manifest.get("files")
    if not isinstance(rows, list) or not isinstance(files, list) or not files:
        raise ValueError(f"{label} has no artifact population")
    declared = {
        row.get("path"): row for row in rows if isinstance(row, dict) and row.get("path")
    }
    if len(declared) != len(rows):
        raise ValueError(f"{label} completion contains duplicate or malformed paths")
    matched: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"{label} manifest contains a malformed file")
        relative = item["path"]
        absolute = str((artifact_root / relative).resolve())
        key = absolute if absolute in declared else relative
        if declared.get(key) != {
            "path": key,
            "sha256": item.get("sha256"),
            "bytes": item.get("bytes"),
        }:
            raise ValueError(f"{label} completion differs from artifact bytes: {relative}")
        matched.add(key)
    if matched != set(declared):
        raise ValueError(f"{label} completion declares files outside its manifest")


def _verify_package_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    rows = manifest.get("files")
    if manifest.get("schema_version") != "1.0" or not isinstance(rows, list) or not rows:
        raise ValueError("portable package manifest is malformed")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("portable package manifest contains a malformed file")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("portable package manifest contains an unsafe path")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"portable package byte verification failed: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"package.manifest.json", "completion.json"}
    }
    if declared != actual:
        raise ValueError("portable package contains undeclared or missing files")
    required = {
        "adapter.manifest.json",
        "runtime.lock.json",
        "training/run.json",
        "training/completion.json",
    }
    if not required.issubset(declared):
        raise ValueError("portable package is missing required training evidence")


def _verify_remote_package_files(
    root: Path, rows: object,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("remote completion has no package file population")
    declared: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("remote completion has a malformed package file")
        relative = Path(row["path"])
        normalized = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or normalized in declared:
            raise ValueError("remote completion has an unsafe package file")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"remote package byte verification failed: {normalized}")
        declared[normalized] = row
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(declared) != actual:
        raise ValueError("remote completion differs from the portable package population")


def _complete_recorded_stage(
    snapshot: FidelityRun,
    *,
    state: Path,
    stage: str,
    artifact: dict[str, object],
    output: Path,
) -> None:
    with locked_fidelity_run(state) as run:
        if run != snapshot:
            raise RuntimeError("run state changed while source evidence was being verified")
        record = run.stages[stage]
        if record.status.value in {"pending", "blocked"}:
            run.begin(stage)
        elif record.status.value != "running":
            raise RuntimeError(f"stage {stage!r} is not recordable from {record.status.value!r}")
        _publish_artifact(output, artifact)
        run.complete(stage, artifact=output)
        _print_status(run)


def _record_roundtrip(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    config = load_config(arguments.config)
    if config.sha256 != run.config_sha256:
        raise ValueError("roundtrip configuration differs from the campaign")
    conversion_run, conversion_run_sha = _source_json(
        arguments.conversion_run, "logit-check run"
    )
    conversion_completion, conversion_completion_sha = _source_json(
        arguments.conversion_completion, "logit-check completion"
    )
    conversion_input_sha = _require_sha256(
        conversion_run.get("dataset_manifest_sha256"),
        "logit-check input manifest SHA-256",
    )
    conversion_id = stable_run_id(
        stage="logit-check",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=conversion_input_sha,
    )
    _validate_run_pair(
        conversion_run,
        conversion_run_sha,
        conversion_completion,
        expected_run_id=conversion_id,
        expected_stage="logit-check",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=None,
    )
    if conversion_completion.get("evidence", {}).get("direction") != "logit-check":
        raise ValueError("conversion completion is not the final logit seam")
    hf_to_maxtext, hf_to_maxtext_sha = _source_json(
        arguments.hf_to_maxtext_completion,
        "HF-to-MaxText completion",
    )
    maxtext_to_hf, maxtext_to_hf_sha = _source_json(
        arguments.maxtext_to_hf_completion,
        "MaxText-to-HF completion",
    )
    for document, direction in (
        (hf_to_maxtext, "hf-to-maxtext"),
        (maxtext_to_hf, "maxtext-to-hf"),
    ):
        evidence = document.get("evidence")
        input_manifest_sha256 = (
            evidence.get("input_manifest_sha256") if isinstance(evidence, dict) else None
        )
        expected_run_id = (
            stable_run_id(
                stage=direction,
                config_sha256=run.config_sha256,
                dataset_manifest_sha256=input_manifest_sha256,
            )
            if isinstance(input_manifest_sha256, str)
            and re.fullmatch(r"[a-f0-9]{64}", input_manifest_sha256)
            else None
        )
        if (
            document.get("schema_version") != "1.0"
            or document.get("status") != "succeeded"
            or document.get("run_id") != expected_run_id
            or not isinstance(document.get("artifacts"), list)
            or not document["artifacts"]
            or not isinstance(evidence, dict)
            or evidence.get("direction") != direction
        ):
            raise ValueError(f"{direction} completion is not successful terminal evidence")

    roundtrip, roundtrip_sha = _source_json(
        arguments.roundtrip_evidence, "roundtrip evidence"
    )
    exported_manifest, exported_manifest_sha = _source_json(
        arguments.exported_checkpoint_manifest,
        "exported checkpoint manifest",
    )
    if roundtrip.get("exported_checkpoint_manifest") != exported_manifest:
        raise ValueError("roundtrip evidence changed the exported checkpoint manifest")
    verify_artifact_manifest(arguments.exported_checkpoint, exported_manifest)
    validate_roundtrip_evidence(
        config,
        roundtrip,
        exported_checkpoint=arguments.exported_checkpoint,
    )

    smoke_run, smoke_run_sha = _source_json(arguments.smoke_training_run, "smoke run")
    smoke_completion, smoke_completion_sha = _source_json(
        arguments.smoke_training_completion, "smoke completion"
    )
    smoke_id = stable_run_id(
        stage="lora-smoke",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )
    _validate_run_pair(
        smoke_run,
        smoke_run_sha,
        smoke_completion,
        expected_run_id=smoke_id,
        expected_stage="lora-smoke",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )
    metadata = smoke_run.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("smoke") is not True:
        raise ValueError("roundtrip canary is not an explicit five-step smoke run")
    smoke_manifest, smoke_manifest_sha = _source_json(
        arguments.smoke_adapter_manifest, "smoke adapter manifest"
    )
    verify_artifact_manifest(arguments.smoke_adapter, smoke_manifest)
    _completion_covers_manifest(
        smoke_completion,
        smoke_manifest,
        artifact_root=arguments.smoke_adapter.resolve(),
        label="smoke adapter",
    )
    lineage = roundtrip.get("lineage")
    expected_lineage = {
        "hf_to_maxtext_completion_sha256": hf_to_maxtext_sha,
        "smoke_completion_sha256": smoke_completion_sha,
        "smoke_run_id": smoke_id,
        "maxtext_to_hf_completion_sha256": maxtext_to_hf_sha,
        "logit_completion_sha256": conversion_completion_sha,
        "logit_run_id": conversion_id,
        "hf_to_maxtext_run_id": hf_to_maxtext.get("run_id"),
        "maxtext_to_hf_run_id": maxtext_to_hf.get("run_id"),
    }
    if not isinstance(lineage, dict) or any(
        lineage.get(name) != value for name, value in expected_lineage.items()
    ):
        raise ValueError("roundtrip lineage differs from its verified terminal evidence")
    artifact = typed_stage_document(
        run,
        "roundtrip",
        status="succeeded",
        fields={"roundtrip_status": "passed"},
        evidence_sha256={
            "conversion_run": conversion_run_sha,
            "conversion_completion": conversion_completion_sha,
            "hf_to_maxtext_completion": hf_to_maxtext_sha,
            "maxtext_to_hf_completion": maxtext_to_hf_sha,
            "roundtrip": roundtrip_sha,
            "exported_checkpoint_manifest": exported_manifest_sha,
            "smoke_training_run": smoke_run_sha,
            "smoke_training_completion": smoke_completion_sha,
            "smoke_adapter_manifest": smoke_manifest_sha,
        },
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="roundtrip",
        artifact=artifact,
        output=arguments.output,
    )


def _record_training(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    config = load_config(arguments.config)
    if config.sha256 != run.config_sha256:
        raise ValueError("training configuration differs from the campaign")
    base_receipt, base_receipt_sha = _source_json(
        arguments.base_checkpoint_receipt, "base checkpoint receipt"
    )
    base_manifest, base_manifest_sha = _source_json(
        arguments.base_checkpoint_manifest, "base checkpoint manifest"
    )
    tokenizer_manifest, tokenizer_manifest_sha = _source_json(
        arguments.tokenizer_manifest, "tokenizer manifest"
    )
    if base_receipt_sha != arguments.base_checkpoint_receipt_sha256:
        raise ValueError("base checkpoint receipt differs from its trusted SHA-256")
    if base_manifest_sha != arguments.base_checkpoint_manifest_sha256:
        raise ValueError("base checkpoint manifest differs from its trusted SHA-256")
    selected_base = verify_orbax_leaf_receipt(
        arguments.base_checkpoint_root,
        base_receipt,
        expected_step=0,
        role="base-maxtext",
    )
    if base_receipt.get("artifact_manifest") != base_manifest:
        raise ValueError("base checkpoint manifest differs from the step-0 Orbax receipt")
    verify_artifact_manifest(selected_base, base_manifest)
    base_content_sha = base_manifest.get("content_sha256")
    if not isinstance(base_content_sha, str) or re.fullmatch(
        r"[a-f0-9]{64}", base_content_sha
    ) is None:
        raise ValueError("base checkpoint has no valid content SHA-256")
    training_id = stable_run_id(
        stage="lora-train",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )
    training_run, training_run_sha = _source_json(arguments.training_run, "training run")
    training_completion, training_completion_sha = _source_json(
        arguments.training_completion, "training completion"
    )
    _validate_run_pair(
        training_run,
        training_run_sha,
        training_completion,
        expected_run_id=training_id,
        expected_stage="lora-train",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )
    metadata = training_run.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("smoke") is not False:
        raise ValueError("training evidence is not the bounded full run")
    training_inputs = metadata.get("inputs")
    expected_checkpoint_inputs = {
        "base_checkpoint": {
            "content_sha256": base_manifest.get("content_sha256"),
            "files": len(base_manifest.get("files", [])),
            "bytes": sum(
                int(row.get("bytes", 0))
                for row in base_manifest.get("files", [])
                if isinstance(row, dict)
            ),
        },
        "tokenizer_checkpoint": {
            "content_sha256": tokenizer_manifest.get("content_sha256"),
            "files": len(tokenizer_manifest.get("files", [])),
            "bytes": sum(
                int(row.get("bytes", 0))
                for row in tokenizer_manifest.get("files", [])
                if isinstance(row, dict)
            ),
        },
    }
    if not isinstance(training_inputs, dict) or any(
        training_inputs.get(name) != binding
        for name, binding in expected_checkpoint_inputs.items()
    ):
        raise ValueError("training run did not use the verified Orbax base and tokenizer")

    adapter_manifest, adapter_manifest_sha = _source_json(
        arguments.adapter_manifest, "adapter manifest"
    )
    verify_artifact_manifest(arguments.adapter, adapter_manifest)
    _completion_covers_manifest(
        training_completion,
        adapter_manifest,
        artifact_root=arguments.adapter.resolve(),
        label="training adapter",
    )
    package_manifest, package_manifest_sha = _source_json(
        arguments.package_manifest, "package manifest"
    )
    package_root = arguments.package_root.resolve()
    _verify_package_manifest(package_root, package_manifest)
    runtime_lock, runtime_lock_sha = _source_json(arguments.runtime_lock, "runtime lock")
    if (
        runtime_lock.get("schema_version") != "1.0"
        or not isinstance(runtime_lock.get("packages"), list)
        or not runtime_lock["packages"]
        or runtime_lock.get("packages_sha256") != canonical_sha256(runtime_lock["packages"])
    ):
        raise ValueError("training runtime lock is incomplete or internally inconsistent")
    completion_evidence = training_completion.get("evidence")
    if (
        not isinstance(completion_evidence, dict)
        or completion_evidence.get("runtime_lock", {}).get("sha256") != runtime_lock_sha
    ):
        raise ValueError("training completion changed the runtime lock")

    package_bindings = {
        "training/run.json": training_run_sha,
        "training/completion.json": training_completion_sha,
        "adapter.manifest.json": adapter_manifest_sha,
        "runtime.lock.json": runtime_lock_sha,
    }
    if any(
        file_sha256(package_root / relative) != digest
        for relative, digest in package_bindings.items()
    ):
        raise ValueError("portable package evidence differs from the verified training inputs")
    verify_artifact_manifest(package_root / "adapter", adapter_manifest)

    remote, remote_sha = _source_json(arguments.remote_completion, "remote completion")
    if remote_sha != arguments.remote_completion_sha256:
        raise ValueError("remote completion differs from its trusted SHA-256")
    portable = remote.get("portable_package")
    source_completion_sha = training_completion.get("source_training_completion_sha256")
    _verify_remote_package_files(package_root, remote.get("files"))
    if (
        remote.get("schema_version") != "1.0"
        or remote.get("run_id") != run.run_id
        or remote.get("training_run_id") != training_id
        or remote.get("status") != "succeeded"
        or remote.get("backend") != arguments.backend
        or remote.get("config_sha256") != run.config_sha256
        or remote.get("dataset_manifest_sha256") != run.dataset_manifest_sha256
        or remote.get("base_checkpoint_manifest_sha256") != base_manifest_sha
        or remote.get("base_checkpoint_receipt_sha256") != base_receipt_sha
        or remote.get("tokenizer_manifest_sha256") != tokenizer_manifest_sha
        or not isinstance(source_completion_sha, str)
        or re.fullmatch(r"[a-f0-9]{64}", source_completion_sha) is None
        or remote.get("source_training_completion_sha256") != source_completion_sha
        or not isinstance(portable, dict)
        or portable.get("training_run_id") != training_id
        or portable.get("training_run_sha256") != training_run_sha
        or portable.get("training_completion_sha256") != training_completion_sha
        or portable.get("adapter_manifest_sha256") != adapter_manifest_sha
        or portable.get("package_manifest_sha256") != package_manifest_sha
        or portable.get("runtime_lock_sha256") != runtime_lock_sha
    ):
        raise ValueError("remote completion does not bind the verified training package")
    artifact = typed_stage_document(
        run,
        "train",
        status="succeeded",
        fields={
            "training_run_id": training_id,
            "backend": arguments.backend,
            "automatic_retries": 0,
        },
        evidence_sha256={
            "remote_completion": remote_sha,
            "training_run": training_run_sha,
            "training_completion": training_completion_sha,
            "adapter_manifest": adapter_manifest_sha,
            "package_manifest": package_manifest_sha,
            "runtime_lock": runtime_lock_sha,
            "base_checkpoint_receipt": base_receipt_sha,
            "base_checkpoint_manifest": base_manifest_sha,
            "base_checkpoint_content": base_content_sha,
            "tokenizer_manifest": tokenizer_manifest_sha,
        },
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="train",
        artifact=artifact,
        output=arguments.output,
    )


def _candidate_id_from_checkpoint(
    run: FidelityRun,
    *,
    config_path: Path,
    checkpoint: Path,
) -> str:
    config = load_config(config_path)
    if config.sha256 != run.config_sha256:
        raise ValueError("campaign configuration is not the repository-pinned configuration")
    return candidate_id_for_checkpoint(
        config_path=config_path,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
        training_run_id=str(run.training_run_id),
        merged_hf_checkpoint=checkpoint,
    )


def _record_candidate_evaluation(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    if run.training_run_id is None:
        raise ValueError("candidate evaluation requires completed training lineage")
    manifest, manifest_sha = _source_json(
        arguments.merged_checkpoint_manifest,
        "merged checkpoint manifest",
    )
    verify_artifact_manifest(arguments.merged_checkpoint, manifest)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("merged checkpoint manifest has no files")
    if any(not isinstance(row, dict) for row in raw_files):
        raise ValueError("merged checkpoint manifest contains malformed files")
    candidate_id = _candidate_id_from_checkpoint(
        run,
        config_path=arguments.config,
        checkpoint=arguments.merged_checkpoint,
    )
    evaluation, evaluation_sha = _source_json(
        arguments.development_evaluation,
        "development evaluation",
    )
    validate_development_eligibility(
        evaluation,
        candidate_id=candidate_id,
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
        training_run_id=run.training_run_id,
    )
    artifact = typed_stage_document(
        run,
        "candidate-eval",
        status="succeeded",
        fields={
            "training_run_id": run.training_run_id,
            "candidate_id": candidate_id,
            "development_eligibility": {"passed": True, "hidden_evaluated": False},
        },
        evidence_sha256={
            "development_summary": evaluation_sha,
            "dataset_manifest": run.dataset_manifest_sha256,
            "merged_checkpoint_manifest": manifest_sha,
        },
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="candidate-eval",
        artifact=artifact,
        output=arguments.output,
    )


def _verify_release_files(root: Path, manifest: Mapping[str, object]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("merged-HF release has no files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("merged-HF release contains a malformed file declaration")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("merged-HF release contains an unsafe path")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"merged-HF release byte verification failed: {relative}")
        declared.add(relative.as_posix())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if (
        actual != declared
        or manifest.get("files_content_sha256") != _release_canonical_sha256(rows)
    ):
        raise ValueError("merged-HF release manifest differs from its checkpoint bytes")


def _record_hf_export(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    release, release_sha = _source_json(arguments.release_manifest, "release manifest")
    if (
        release.get("schema_version") != "1.0"
        or release.get("status") != "succeeded"
        or release.get("release_type") != "merged-hf"
        or release.get("config_sha256") != run.config_sha256
        or release.get("dataset_manifest_sha256") != run.dataset_manifest_sha256
        or release.get("training_run_id") != run.training_run_id
        or release.get("candidate_id") != run.candidate_id
    ):
        raise ValueError("merged-HF release changed the selected candidate lineage")
    _verify_release_files(arguments.release_root.resolve(), release)
    base_model = release.get("base_model")
    files = release.get("files")
    if (
        not isinstance(base_model, dict)
        or not isinstance(files, list)
        or candidate_id_from_lineage(
            config_sha256=run.config_sha256,
            dataset_manifest_sha256=run.dataset_manifest_sha256,
            training_run_id=str(run.training_run_id),
            base_model=base_model,
            files=files,
        )
        != run.candidate_id
    ):
        raise ValueError("merged-HF release bytes do not derive the selected candidate ID")
    evidence_paths = {
        "training_run": arguments.training_run,
        "training_completion": arguments.training_completion,
        "roundtrip": arguments.roundtrip_evidence,
        "development_summary": arguments.development_evaluation,
    }
    evidence = {name: _source_digest(path, name) for name, path in evidence_paths.items()}
    terminal = release.get("terminal_evidence")
    if not isinstance(terminal, dict) or terminal != {
        "training_run": evidence["training_run"],
        "training_completion": evidence["training_completion"],
        "roundtrip": evidence["roundtrip"],
        "evaluation": evidence["development_summary"],
    }:
        raise ValueError("merged-HF release changed its terminal evidence")
    artifact = typed_stage_document(
        run,
        "hf-export",
        status="succeeded",
        fields={
            "training_run_id": run.training_run_id,
            "candidate_id": run.candidate_id,
            "release_manifest_sha256": release_sha,
        },
        evidence_sha256={"release_manifest": release_sha, **evidence},
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="hf-export",
        artifact=artifact,
        output=arguments.output,
    )


def _record_int4_export(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    source, source_sha = _source_json(
        arguments.source_release_manifest,
        "source release manifest",
    )
    if (
        source.get("candidate_id") != run.candidate_id
        or source.get("training_run_id") != run.training_run_id
        or source.get("config_sha256") != run.config_sha256
        or source.get("dataset_manifest_sha256") != run.dataset_manifest_sha256
    ):
        raise ValueError("INT4 source release changed the selected candidate lineage")
    source_files = source.get("files")
    source_base_model = source.get("base_model")
    if (
        not isinstance(source_files, list)
        or not isinstance(source_base_model, dict)
        or source.get("files_content_sha256")
        != _release_canonical_sha256(source_files)
        or candidate_id_from_lineage(
            config_sha256=run.config_sha256,
            dataset_manifest_sha256=run.dataset_manifest_sha256,
            training_run_id=str(run.training_run_id),
            base_model=source_base_model,
            files=source_files,
        )
        != run.candidate_id
    ):
        raise ValueError("INT4 source release bytes do not derive the selected candidate ID")
    exported, exported_sha = _source_json(arguments.export_manifest, "INT4 export manifest")
    if (
        exported.get("schema_version") != "1.0"
        or exported.get("status") != "succeeded"
        or exported.get("candidate_id") != run.candidate_id
        or exported.get("training_run_id") != run.training_run_id
        or exported.get("config_sha256") != run.config_sha256
        or exported.get("dataset_manifest_sha256") != run.dataset_manifest_sha256
        or exported.get("source_release_manifest_sha256") != source_sha
        or exported.get("source_files_content_sha256")
        != source.get("files_content_sha256")
        or exported.get("quantization") != "int4_awq"
        or exported.get("components") != ["thinker"]
        or exported.get("skip_visual") is not True
        or exported.get("skip_audio") is not True
        or exported.get("engine_built_in_cloud") is not False
    ):
        raise ValueError("INT4 export changed the approved text-only release contract")
    calibration = exported.get("calibration_provenance")
    calibration_sha = exported.get("calibration_provenance_sha256")
    if (
        not isinstance(calibration, dict)
        or calibration_sha != _release_canonical_sha256(calibration)
    ):
        raise ValueError("INT4 export has invalid calibration provenance")
    rows = exported.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("INT4 export has no ONNX files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("INT4 export has a malformed file declaration")
        relative = Path(row["path"])
        path = arguments.export_root / "onnx" / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in declared
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"INT4 export byte verification failed: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(arguments.export_root / "onnx").as_posix()
        for path in (arguments.export_root / "onnx").rglob("*")
        if path.is_file()
    }
    if declared != actual:
        raise ValueError("INT4 export contains undeclared or missing files")
    artifact = typed_stage_document(
        run,
        "int4-export",
        status="succeeded",
        fields={
            "training_run_id": run.training_run_id,
            "candidate_id": run.candidate_id,
            "source_release_manifest_sha256": source_sha,
            "export_manifest_sha256": exported_sha,
        },
        evidence_sha256={
            "source_release_manifest": source_sha,
            "export_manifest": exported_sha,
            "calibration_provenance": str(calibration_sha),
        },
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="int4-export",
        artifact=artifact,
        output=arguments.output,
    )


def _publish_exact(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace stage evidence: {destination}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _record_adopted_stage(arguments: argparse.Namespace) -> None:
    stage = "jetson-shadow" if arguments.command == "record-jetson-shadow" else "gate"
    run = FidelityRun.read(arguments.state)
    if file_sha256(arguments.source) != arguments.source_sha256:
        raise ValueError(f"approved {stage} source SHA-256 changed")
    validate_stage_artifact(run, stage, arguments.source)
    with locked_fidelity_run(arguments.state) as active:
        if active != run:
            raise RuntimeError("run state changed while source evidence was being verified")
        record = active.stages[stage]
        if record.status.value in {"pending", "blocked"}:
            active.begin(stage)
        elif record.status.value != "running":
            raise RuntimeError(f"stage {stage!r} is not recordable from {record.status.value!r}")
        _publish_exact(arguments.source, arguments.output)
        active.complete(stage, artifact=arguments.output)
        _print_status(active)


def _record_terminal_decision(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    gate = run.stages["gate"]
    if gate.status.value != "completed" or gate.artifact_status not in {"passed", "rejected"}:
        raise ValueError("terminal decision requires completed gate evidence")
    stage = "promotion" if gate.artifact_status == "passed" else "retain-baseline"
    if run.training_run_id is None or run.candidate_id is None:
        raise ValueError("terminal decision requires fixed training and candidate lineage")
    if gate.candidate_manifest_sha256 is None or gate.artifact_sha256 is None:
        raise ValueError("terminal decision requires checksum-bound gate evidence")
    gate_document, gate_sha = _source_json(arguments.gate_artifact, "gate artifact")
    if gate_sha != gate.artifact_sha256:
        raise ValueError("gate artifact differs from the completed gate evidence")
    validate_stage_artifact(run, "gate", arguments.gate_artifact)
    candidate_identity = gate_document.get("candidate_identity")
    if not isinstance(candidate_identity, dict):
        raise ValueError("gate artifact has no candidate identity")
    model_revision = candidate_identity.get("model_revision")
    if (
        not isinstance(model_revision, str)
        or not model_revision.startswith("sha256:")
        or re.fullmatch(r"[a-f0-9]{64}", model_revision.removeprefix("sha256:")) is None
    ):
        raise ValueError("gate candidate model revision is not engine-addressed")
    candidate_engine_sha256 = model_revision.removeprefix("sha256:")
    receipt, receipt_sha = _source_json(
        arguments.deployment_receipt, "deployment receipt"
    )
    health, health_sha = _source_json(
        arguments.accepted_engine_health, "engine health evidence"
    )
    rollback: Mapping[str, object] | None = None
    evidence: dict[str, str]
    status: str
    if stage == "promotion":
        if arguments.rollback_state is None:
            raise ValueError("promotion requires the durable rollback state")
        rollback, rollback_sha = _source_json(arguments.rollback_state, "rollback state")
        evidence = {
            "promotion_receipt": receipt_sha,
            "post_promotion_health": health_sha,
            "rollback_state": rollback_sha,
        }
        status = "promoted"
    else:
        if arguments.rollback_state is not None:
            raise ValueError("baseline retention does not accept promotion rollback state")
        evidence = {
            "retention_receipt": receipt_sha,
            "accepted_engine_health": health_sha,
        }
        status = "retained"
    validate_terminal_evidence(
        receipt,
        health,
        rollback,
        run_id=run.run_id,
        training_run_id=run.training_run_id,
        candidate_id=run.candidate_id,
        candidate_manifest_sha256=gate.candidate_manifest_sha256,
        gate_artifact_sha256=gate.artifact_sha256,
        baseline_engine_sha256=run.baseline_engine_sha256,
        candidate_engine_sha256=candidate_engine_sha256,
        promoted=stage == "promotion",
    )
    artifact = typed_stage_document(
        run,
        stage,
        status=status,
        fields={
            "training_run_id": run.training_run_id,
            "candidate_id": run.candidate_id,
            "candidate_manifest_sha256": gate.candidate_manifest_sha256,
            "gate_artifact_sha256": gate.artifact_sha256,
        },
        evidence_sha256=evidence,
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage=stage,
        artifact=artifact,
        output=arguments.output,
    )


def _record_reconciliation(arguments: argparse.Namespace) -> None:
    run = FidelityRun.read(arguments.state)
    cost, cost_sha = _source_json(arguments.cost_ledger, "cost ledger")
    inventory, inventory_sha = _source_json(
        arguments.paid_resource_inventory,
        "paid resource inventory",
    )
    validate_reconciliation_evidence(cost, inventory, run_id=run.run_id)
    promoted = run.stages["promotion"].status.value == "completed"
    outcome_stage = "promotion" if promoted else "retain-baseline"
    outcome = run.stages[outcome_stage]
    deployment_sha = _source_digest(arguments.deployment_artifact, "deployment artifact")
    if deployment_sha != outcome.artifact_sha256:
        raise ValueError("deployment artifact differs from the completed terminal decision")
    artifact = typed_stage_document(
        run,
        "reconcile",
        status="succeeded",
        fields={
            "training_run_id": run.training_run_id,
            "candidate_id": run.candidate_id,
            "gate_artifact_sha256": run.stages["gate"].artifact_sha256,
            "deployment_artifact_sha256": deployment_sha,
            "deployment_outcome": "promoted" if promoted else "retained",
            "gross_cost_reconciled": True,
            "active_paid_resources": 0,
        },
        evidence_sha256={
            "cost_ledger": cost_sha,
            "paid_resource_inventory": inventory_sha,
            "deployment_receipt": deployment_sha,
        },
    )
    _complete_recorded_stage(
        run,
        state=arguments.state,
        stage="reconcile",
        artifact=artifact,
        output=arguments.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
