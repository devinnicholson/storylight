import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from bookforge.fidelity_benchmark import FidelitySummary, population_contract_from_manifest
from bookforge.fidelity_lineage import stable_run_id
from bookforge.fidelity_manifest import sha256_path
from bookforge.fidelity_orchestration import (
    FidelityRun,
    FidelityRunError,
    StageStatus,
    locked_fidelity_run,
    stage_plan,
)
from bookforge.fidelity_schema import DatasetSplit

ENGINE_SHA = "9" * 64
CANDIDATE_ID = "fidelity-candidate-v1"
CANDIDATE_MANIFEST_SHA = "7" * 64
CANDIDATE_ENGINE_SHA = "8" * 64
ROOT = Path(__file__).resolve().parents[1]


def _training_run_id(run: FidelityRun) -> str:
    return stable_run_id(
        stage="lora-train",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )


def _run() -> FidelityRun:
    return FidelityRun.create(
        run_id="story-fidelity-v1",
        config_sha256="8" * 64,
        dataset_manifest_sha256="6" * 64,
        baseline_commit="d857327",
        baseline_engine_sha256=ENGINE_SHA,
    )


def _artifact(
    run: FidelityRun,
    tmp_path: Path,
    stage: str,
    *,
    status: str = "succeeded",
    **fields: object,
) -> Path:
    local_evidence: dict[str, object] = {}
    stage_fields: dict[str, object] = {"producer": "test"}
    if stage == "dataset":
        local_evidence = {
            "dataset_manifest_sha256": run.dataset_manifest_sha256,
            "split_records": {"train": 4096, "development": 512, "hidden": 512},
            "record_schema_sha256": "1" * 64,
            "generator_source_sha256": "2" * 64,
            "generator_config_sha256": "3" * 64,
        }
    elif stage == "cpu-smoke":
        local_evidence = {
            "dataset_manifest_sha256": run.dataset_manifest_sha256,
            "fixture_sha256": "4" * 64,
            "summary": {
                "records": 32,
                "schema_valid_rate": 1.0,
                "privacy_pass_rate": 1.0,
                "exact_example_pass_rate": 1.0,
            },
        }
    elif stage == "compatibility-package":
        local_evidence = {
            "dataset_manifest_sha256": run.dataset_manifest_sha256,
            "config_sha256": run.config_sha256,
            "formatted_records": 32,
            "formatted_smoke_sha256": "5" * 64,
            "roundtrip_contract": {"schema_version": "1.0"},
        }
    if local_evidence:
        stage_fields = {
            "producer": "bookforge-local-preflight",
            "evidence": local_evidence,
        }
    if stage == "baseline":
        stage_fields = {
            "producer": "bookforge-baseline-recorder",
            "baseline_identity": {
                "candidate_id": "accepted-baseline",
                "candidate_manifest_sha256": "d" * 64,
                "engine_sha256": run.baseline_engine_sha256,
                "model_revision": f"sha256:{run.baseline_engine_sha256}",
            },
            "hidden_custody_receipt_sha256": "e" * 64,
            "evidence_sha256": {
                "baseline_manifest": "d" * 64,
                "development_summary": "e" * 64,
                "hidden_summary": "f" * 64,
                "dataset_manifest": run.dataset_manifest_sha256,
            },
        }
    training_run_id = run.training_run_id or _training_run_id(run)
    evidence_sha256: dict[str, str]
    if stage == "roundtrip":
        evidence_sha256 = {
            name: "a" * 64
            for name in (
                "conversion_run",
                "conversion_completion",
                "hf_to_maxtext_completion",
                "maxtext_to_hf_completion",
                "roundtrip",
                "exported_checkpoint_manifest",
                "smoke_training_run",
                "smoke_training_completion",
                "smoke_adapter_manifest",
            )
        }
        stage_fields = {
            "producer": "bookforge-jax-roundtrip-recorder",
            "roundtrip_status": "passed",
            "evidence_sha256": evidence_sha256,
        }
    elif stage == "train":
        evidence_sha256 = {
            name: "a" * 64
            for name in (
                "remote_completion",
                "training_run",
                "training_completion",
                "adapter_manifest",
                "package_manifest",
                "runtime_lock",
                "base_checkpoint_receipt",
                "base_checkpoint_manifest",
                "base_checkpoint_content",
                "tokenizer_manifest",
            )
        }
        stage_fields = {
            "producer": "bookforge-jax-training-recorder",
            "training_run_id": training_run_id,
            "backend": "vertex-tpu-v6e",
            "automatic_retries": 0,
            "evidence_sha256": evidence_sha256,
        }
    elif stage == "candidate-eval":
        stage_fields = {
            "producer": "bookforge-candidate-evaluation-recorder",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "development_eligibility": {"passed": True, "hidden_evaluated": False},
            "evidence_sha256": {
                "development_summary": "a" * 64,
                "dataset_manifest": run.dataset_manifest_sha256,
                "merged_checkpoint_manifest": "b" * 64,
            },
        }
    elif stage == "hf-export":
        stage_fields = {
            "producer": "bookforge-hf-release-recorder",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "release_manifest_sha256": "b" * 64,
            "evidence_sha256": {
                "release_manifest": "b" * 64,
                "training_run": "a" * 64,
                "training_completion": "a" * 64,
                "roundtrip": "a" * 64,
                "development_summary": "a" * 64,
            },
        }
    elif stage == "int4-export":
        stage_fields = {
            "producer": "bookforge-int4-export-recorder",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "source_release_manifest_sha256": "b" * 64,
            "export_manifest_sha256": "c" * 64,
            "evidence_sha256": {
                "source_release_manifest": "b" * 64,
                "export_manifest": "c" * 64,
                "calibration_provenance": "d" * 64,
            },
        }
    elif stage == "jetson-shadow":
        stage_fields = {
            "producer": "bookforge-jetson-shadow-recorder",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "shadow_status": "passed",
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "candidate_identity": {
                "candidate_id": CANDIDATE_ID,
                "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
                "engine_sha256": CANDIDATE_ENGINE_SHA,
                "model_revision": f"sha256:{CANDIDATE_ENGINE_SHA}",
            },
            "evidence_sha256": {
                "runtime": "a" * 64,
                "candidate_manifest": CANDIDATE_MANIFEST_SHA,
                "hidden_summary": "b" * 64,
            },
        }
    elif stage == "gate":
        stage_fields = {
            "producer": "test",
            "training_run_id": training_run_id,
        }
    elif stage == "promotion":
        stage_fields = {
            "producer": "bookforge-trained-planner-promotion",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "evidence_sha256": {
                "promotion_receipt": "a" * 64,
                "post_promotion_health": "b" * 64,
                "rollback_state": "c" * 64,
            },
        }
    elif stage == "retain-baseline":
        stage_fields = {
            "producer": "bookforge-baseline-retention",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "evidence_sha256": {
                "retention_receipt": "a" * 64,
                "accepted_engine_health": "b" * 64,
            },
        }
    elif stage == "reconcile":
        completed_outcome = (
            run.stages["promotion"]
            if run.stages["promotion"].status is StageStatus.COMPLETED
            else run.stages["retain-baseline"]
        )
        stage_fields = {
            "producer": "bookforge-fidelity-reconciler",
            "training_run_id": training_run_id,
            "candidate_id": CANDIDATE_ID,
            "deployment_artifact_sha256": completed_outcome.artifact_sha256,
            "evidence_sha256": {
                "cost_ledger": "a" * 64,
                "paid_resource_inventory": "b" * 64,
                "deployment_receipt": "c" * 64,
            },
        }
    path = tmp_path / f"{stage}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "stage": stage,
                "run_id": run.run_id,
                "config_sha256": run.config_sha256,
                "dataset_manifest_sha256": run.dataset_manifest_sha256,
                "status": status,
                "inputs": {
                    dependency: run.stages[dependency].artifact_sha256
                    for dependency in next(item for item in stage_plan() if item["name"] == stage)[
                        "dependencies"
                    ]
                },
                **stage_fields,
                **fields,
            }
        )
    )
    return path


def _gate_fields(*, passed: bool) -> dict[str, object]:
    return {
        "producer": "bookforge-fidelity-gate-builder",
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
        "candidate_identity": {
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "engine_sha256": CANDIDATE_ENGINE_SHA,
            "model_revision": f"sha256:{CANDIDATE_ENGINE_SHA}",
        },
        "baseline_identity": {
            "candidate_id": "accepted-baseline-v1",
            "candidate_manifest_sha256": "d" * 64,
            "engine_sha256": ENGINE_SHA,
            "model_revision": f"sha256:{ENGINE_SHA}",
        },
        "hidden_custody_receipt_sha256": "a" * 64,
        "evidence_sha256": {
            "baseline_development_summary": "1" * 64,
            "baseline_hidden_summary": "2" * 64,
            "candidate_development_summary": "3" * 64,
            "candidate_hidden_summary": "4" * 64,
            "candidate_manifest": CANDIDATE_MANIFEST_SHA,
            "contest": "5" * 64,
            "dataset_manifest": "6" * 64,
            "human_review": "b" * 64,
            "runtime": "c" * 64,
        },
        "decision": {
            "passed": passed,
            "reasons": [] if passed else ["p95_latency"],
            "checks": {"semantics": passed},
        },
    }


def test_run_is_ordered_resumable_and_checksum_bound(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    run = _run()
    run.write(path)

    with locked_fidelity_run(path) as active:
        active.begin("dataset")
        active.complete("dataset", artifact=_artifact(active, tmp_path, "dataset"))

    resumed = FidelityRun.read(path)
    assert resumed.stages["dataset"].status is StageStatus.COMPLETED
    assert resumed.stages["dataset"].artifact_sha256 is not None
    assert resumed.next_stage().name == "cpu-smoke"
    assert path.stat().st_mode & 0o777 == 0o600


def test_paid_and_device_stages_require_exact_ephemeral_approval(tmp_path: Path) -> None:
    run = _run()
    for stage in ("dataset", "cpu-smoke", "compatibility-package", "baseline"):
        run.begin(stage)
        run.complete(stage, artifact=_artifact(run, tmp_path, stage))

    with pytest.raises(FidelityRunError, match="BOOKFORGE_JAX_ROUNDTRIP"):
        run.begin("roundtrip", environment={})
    assert run.stages["roundtrip"].status is StageStatus.BLOCKED

    run.begin(
        "roundtrip",
        environment={"BOOKFORGE_JAX_ROUNDTRIP": "I_APPROVE_THIS_BOUNDED_ROUNDTRIP"},
    )
    assert run.stages["roundtrip"].status is StageStatus.RUNNING


def test_stage_artifact_must_bind_exact_dependency_evidence(tmp_path: Path) -> None:
    run = _run()
    run.begin("dataset")
    run.complete("dataset", artifact=_artifact(run, tmp_path, "dataset"))
    run.begin("cpu-smoke")
    artifact = _artifact(run, tmp_path, "cpu-smoke")
    document = json.loads(artifact.read_text())
    document["inputs"]["dataset"] = "0" * 64
    artifact.write_text(json.dumps(document))

    with pytest.raises(FidelityRunError, match="dependency evidence"):
        run.complete("cpu-smoke", artifact=artifact)


def test_candidate_lineage_cannot_change_after_development_evaluation(
    tmp_path: Path,
) -> None:
    run = _run()
    approvals = {
        "roundtrip": {"BOOKFORGE_JAX_ROUNDTRIP": "I_APPROVE_THIS_BOUNDED_ROUNDTRIP"},
        "train": {"BOOKFORGE_JAX_TRAIN": "I_APPROVE_THIS_BOUNDED_TPU_JOB"},
    }
    for stage in (
        "dataset",
        "cpu-smoke",
        "compatibility-package",
        "baseline",
        "roundtrip",
        "train",
        "candidate-eval",
    ):
        run.begin(stage, environment=approvals.get(stage, {}))
        run.complete(stage, artifact=_artifact(run, tmp_path, stage))
    run.begin("hf-export")
    artifact = _artifact(run, tmp_path, "hf-export")
    document = json.loads(artifact.read_text())
    document["candidate_id"] = "fidelity-another-candidate"
    artifact.write_text(json.dumps(document))

    with pytest.raises(FidelityRunError, match="changed the candidate ID"):
        run.complete("hf-export", artifact=artifact)


def test_local_runner_executes_verified_no_spend_stages(tmp_path: Path) -> None:
    script = ROOT / "scripts/orchestrate_jax_fidelity_lab.py"
    config = ROOT / "experiments/jax-fidelity-lab/config.json"
    state = tmp_path / "run.json"
    environment = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            "--state",
            str(state),
            "--run-id",
            "local-preflight-v1",
            "--config",
            str(config),
            "--dataset-manifest",
            str(ROOT / "datasets/story-fidelity-v1/manifest.json"),
            "--baseline-commit",
            "d857327",
            "--baseline-engine-sha256",
            ENGINE_SHA,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "run-local",
            "--state",
            str(state),
            "--artifacts-directory",
            str(tmp_path / "evidence"),
            "--config",
            str(config),
            "--manifest",
            str(ROOT / "datasets/story-fidelity-v1/manifest.json"),
            "--smoke-fixture",
            str(ROOT / "tests/fixtures/story-fidelity-smoke.jsonl"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    run = FidelityRun.read(state)
    assert [
        run.stages[name].status for name in ("dataset", "cpu-smoke", "compatibility-package")
    ] == [
        StageStatus.COMPLETED,
        StageStatus.COMPLETED,
        StageStatus.COMPLETED,
    ]
    assert run.next_stage().name == "baseline"

    dataset_manifest = ROOT / "datasets/story-fidelity-v1/manifest.json"
    dataset_sha = sha256_path(dataset_manifest)
    baseline_manifest = tmp_path / "baseline-manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "candidate_id": "accepted-baseline",
                "engine_sha256": ENGINE_SHA,
                "model_revision": f"sha256:{ENGINE_SHA}",
                "source_dataset_manifest_sha256": dataset_sha,
            },
            sort_keys=True,
        )
        + "\n"
    )
    baseline_manifest_sha = sha256_path(baseline_manifest)
    identity = {
        "candidate_id": "accepted-baseline",
        "candidate_manifest_sha256": baseline_manifest_sha,
        "engine_sha256": ENGINE_SHA,
        "model_revision": f"sha256:{ENGINE_SHA}",
    }
    reports: dict[str, tuple[Path, str]] = {}
    for split in (DatasetSplit.DEVELOPMENT, DatasetSplit.HIDDEN):
        population = population_contract_from_manifest(
            dataset_manifest,
            expected_manifest_sha256=dataset_sha,
            split=split,
        )
        summary = FidelitySummary(
            surface="raw",
            split=split.value,
            records=population.records,
            record_ids_sha256=population.record_ids_sha256,
            category_record_counts=population.category_record_counts,
            schema_valid_rate=1.0,
            privacy_pass_rate=1.0,
            semantic_atom_recall=1.0,
            exact_example_pass_rate=1.0,
            category_pass_rates={name: 1.0 for name in population.category_record_counts},
            counterfactual_pairs=population.pairs,
            counterfactual_sensitivity=1.0,
            unsupported_concept_rate=0.0,
            pii_leaks=0,
            privacy_term_leaks=0,
            source_echoes=0,
            injection_leaks=0,
            forbidden_hits=0,
        )
        report = tmp_path / f"baseline-{split.value}.json"
        report.write_text(
            json.dumps(
                {
                    "schema_version": "story-fidelity-evaluation-v1",
                    "split": split.value,
                    "candidate_identity": identity,
                    "dataset_manifest_sha256": dataset_sha,
                    "custody_receipt_sha256": "a" * 64 if split is DatasetSplit.HIDDEN else None,
                    "privacy": {"passages_recorded": False, "outputs_recorded": False},
                    "summary": asdict(summary),
                },
                sort_keys=True,
            )
            + "\n"
        )
        reports[split.value] = (report, sha256_path(report))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "record-baseline",
            "--state",
            str(state),
            "--dataset-manifest",
            str(dataset_manifest),
            "--baseline-manifest",
            str(baseline_manifest),
            "--baseline-manifest-sha256",
            baseline_manifest_sha,
            "--development-report",
            str(reports["development"][0]),
            "--development-report-sha256",
            reports["development"][1],
            "--hidden-report",
            str(reports["hidden"][0]),
            "--hidden-report-sha256",
            reports["hidden"][1],
            "--output",
            str(tmp_path / "evidence/baseline.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert FidelityRun.read(state).next_stage().name == "roundtrip"


def test_rejected_gate_retains_baseline_and_still_reconciles(tmp_path: Path) -> None:
    run = _run()
    for spec in stage_plan():
        stage = str(spec["name"])
        if stage == "gate":
            break
        run.begin(
            stage,
            environment={
                str(spec["approval_environment"]): {
                    "roundtrip": "I_APPROVE_THIS_BOUNDED_ROUNDTRIP",
                    "train": "I_APPROVE_THIS_BOUNDED_TPU_JOB",
                    "int4-export": "I_APPROVE_THIS_BOUNDED_GPU_EXPORT",
                    "jetson-shadow": "I_APPROVE_THIS_REVERSIBLE_SHADOW_TEST",
                }.get(stage, ""),
            },
        )
        run.complete(stage, artifact=_artifact(run, tmp_path, stage))

    candidate = "7" * 64
    run.begin("gate")
    gate_artifact = _artifact(
        run,
        tmp_path,
        "gate",
        status="rejected",
        **_gate_fields(passed=False),
    )
    run.complete("gate", artifact=gate_artifact)
    assert run.stages["promotion"].status is StageStatus.SKIPPED

    run.begin("retain-baseline")
    run.complete(
        "retain-baseline",
        artifact=_artifact(
            run,
            tmp_path,
            "retain-baseline",
            status="retained",
            candidate_manifest_sha256=candidate,
            gate_artifact_sha256=run.stages["gate"].artifact_sha256,
        ),
    )
    run.begin("reconcile")
    run.complete(
        "reconcile",
        artifact=_artifact(
            run,
            tmp_path,
            "reconcile",
            gate_artifact_sha256=run.stages["gate"].artifact_sha256,
            deployment_outcome="retained",
            gross_cost_reconciled=True,
            active_paid_resources=0,
        ),
    )
    assert run.next_stage() is None


def test_promotion_rejects_artifact_not_bound_to_passed_gate(tmp_path: Path) -> None:
    run = _run()
    for spec in stage_plan():
        stage = str(spec["name"])
        if stage == "gate":
            break
        environment = {}
        if stage == "roundtrip":
            environment["BOOKFORGE_JAX_ROUNDTRIP"] = "I_APPROVE_THIS_BOUNDED_ROUNDTRIP"
        elif stage == "train":
            environment["BOOKFORGE_JAX_TRAIN"] = "I_APPROVE_THIS_BOUNDED_TPU_JOB"
        elif stage == "int4-export":
            environment["BOOKFORGE_JAX_INT4_EXPORT"] = "I_APPROVE_THIS_BOUNDED_GPU_EXPORT"
        elif stage == "jetson-shadow":
            environment["BOOKFORGE_JAX_JETSON_SHADOW"] = "I_APPROVE_THIS_REVERSIBLE_SHADOW_TEST"
        run.begin(stage, environment=environment)
        run.complete(stage, artifact=_artifact(run, tmp_path, stage))
    candidate = "7" * 64
    run.begin("gate")
    run.complete(
        "gate",
        artifact=_artifact(
            run,
            tmp_path,
            "gate",
            status="passed",
            **_gate_fields(passed=True),
        ),
    )
    run.begin(
        "promotion",
        environment={"BOOKFORGE_JAX_PROMOTE": "I_APPROVE_THIS_VERIFIED_ENGINE"},
    )
    with pytest.raises(FidelityRunError, match="gate evidence"):
        run.complete(
            "promotion",
            artifact=_artifact(
                run,
                tmp_path,
                "promotion",
                status="promoted",
                candidate_manifest_sha256=candidate,
                gate_artifact_sha256="0" * 64,
            ),
        )
