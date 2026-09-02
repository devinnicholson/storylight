# ruff: noqa: E402
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bookforge.fidelity_lineage import stable_run_id
from bookforge.fidelity_orchestration import FidelityRun, StageStatus, stage_plan
from training.jax_fidelity.configuration import EOS_TOKEN_IDS, load_config
from training.jax_fidelity.integrity import artifact_manifest, canonical_sha256, sha256_file
from training.jax_fidelity.manifests import complete_run, start_run
from training.jax_fidelity.orbax_receipt import orbax_leaf_receipt
from training.jax_fidelity.release import candidate_id_for_checkpoint
from training.jax_fidelity.remote_release import package_training_release
from training.jax_fidelity.roundtrip_smoke import (
    REQUIRED_BOOLEAN_CHECKS,
    contract_document,
)

SCRIPT = ROOT / "scripts/orchestrate_jax_fidelity_lab.py"
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"
DATASET = ROOT / "datasets/story-fidelity-v1/manifest.json"


def _campaign(tmp_path: Path, *, through: str) -> tuple[FidelityRun, Path]:
    run = FidelityRun.create(
        run_id="trusted-recorder-v1",
        config_sha256=sha256_file(CONFIG),
        dataset_manifest_sha256=sha256_file(DATASET),
        baseline_commit="d857327",
        baseline_engine_sha256="9" * 64,
    )
    for spec in stage_plan():
        stage = str(spec["name"])
        record = run.stages[stage]
        record.status = StageStatus.COMPLETED
        record.started_at = "2026-09-01T00:00:00+00:00"
        record.finished_at = "2026-09-01T00:00:01+00:00"
        record.artifact_sha256 = f"{len(stage) % 10}" * 64
        record.artifact_bytes = 1
        record.artifact_status = "succeeded"
        if stage == through:
            break
    if run.stages["train"].status is StageStatus.COMPLETED:
        run.training_run_id = stable_run_id(
            stage="lora-train",
            config_sha256=run.config_sha256,
            dataset_manifest_sha256=run.dataset_manifest_sha256,
        )
    state = tmp_path / "run.json"
    run.write(state)
    return run, state


def _environment(**updates: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(updates)
    return environment


def _passing_development_evaluation(
    *, candidate_id: str, dataset_manifest_sha256: str, training_run_id: str
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "stage": "development",
        "config_sha256": sha256_file(CONFIG),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "eligibility_decision": {"passed": True, "hidden_evaluated": False},
        "checks": {"schema_valid": True, "development_improvement": True},
        "reasons": [],
        "baseline_summary_sha256": "a" * 64,
        "candidate_summary_sha256": "b" * 64,
        "development_records_sha256": "d" * 64,
        "predictions_sha256": "e" * 64,
        "prediction_completion_sha256": "f" * 64,
        "evaluation_completion_sha256": "1" * 64,
        "evaluation_input_sha256": canonical_sha256(
            {
                "development_records_sha256": "d" * 64,
                "predictions_sha256": "e" * 64,
                "prediction_completion_sha256": "f" * 64,
            }
        ),
        "candidate_manifest_sha256": "2" * 64,
        "checkpoint_manifest_sha256": "3" * 64,
        "checkpoint_content_sha256": "4" * 64,
        "summary": {
            "surface": "raw",
            "split": "development",
            "records": 512,
            "record_ids_sha256": "c" * 64,
            "category_record_counts": {"action_binding": 512},
            "schema_valid_rate": 1.0,
            "privacy_pass_rate": 1.0,
            "semantic_atom_recall": 0.99,
            "exact_example_pass_rate": 0.97,
            "category_pass_rates": {"action_binding": 0.99},
            "counterfactual_pairs": 0,
            "counterfactual_sensitivity": 1.0,
            "unsupported_concept_rate": 0.0,
            "pii_leaks": 0,
            "privacy_term_leaks": 0,
            "source_echoes": 0,
            "injection_leaks": 0,
            "forbidden_hits": 0,
        },
    }


def test_direct_orchestrator_cli_bootstraps_its_repo_imports() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "plan"],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert json.loads(completed.stdout)["stages"][0]["name"] == "dataset"


def test_roundtrip_recorder_requires_numerical_and_five_step_smoke_evidence(
    tmp_path: Path,
) -> None:
    campaign, state = _campaign(tmp_path, through="baseline")
    config = load_config(CONFIG)

    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "model.safetensors").write_bytes(b"roundtrip")
    exported_manifest = artifact_manifest(exported)
    exported_manifest_path = tmp_path / "exported.manifest.json"
    exported_manifest_path.write_text(json.dumps(exported_manifest, sort_keys=True) + "\n")
    roundtrip = {
        "schema_version": "1.0",
        "contract": contract_document(config),
        "checks": {
            **{name: True for name in REQUIRED_BOOLEAN_CHECKS},
            "eos_token_ids": list(EOS_TOKEN_IDS),
            "forward_kl_divergence": 0.01,
            "logit_comparison": "adapted-maxtext-vs-merged-hf",
        },
        "exported_checkpoint_manifest": exported_manifest,
    }
    roundtrip_path = tmp_path / "roundtrip.json"

    runs = tmp_path / "runs"
    conversion_input_sha = "a" * 64
    conversion_id = stable_run_id(
        stage="logit-check",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=conversion_input_sha,
    )
    conversion_run = start_run(
        runs,
        run_id=conversion_id,
        stage="logit-check",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=conversion_input_sha,
        command=["logit-check"],
    )
    conversion_evidence = runs / conversion_id / "conversion-evidence.json"
    conversion_evidence.write_text('{"status":"succeeded"}\n')
    conversion_completion = complete_run(
        runs,
        run_id=conversion_id,
        status="succeeded",
        artifacts=[conversion_evidence],
        evidence={"direction": "logit-check"},
    )
    conversion_completions: dict[str, Path] = {}
    for direction in ("hf-to-maxtext", "maxtext-to-hf"):
        input_sha = ("c" if direction == "hf-to-maxtext" else "d") * 64
        run_id = stable_run_id(
            stage=direction,
            config_sha256=campaign.config_sha256,
            dataset_manifest_sha256=input_sha,
        )
        start_run(
            runs,
            run_id=run_id,
            stage=direction,
            config_sha256=campaign.config_sha256,
            dataset_manifest_sha256=input_sha,
            command=[direction],
        )
        evidence_path = runs / run_id / "conversion-evidence.json"
        evidence_path.write_text('{"status":"succeeded"}\n')
        conversion_completions[direction] = complete_run(
            runs,
            run_id=run_id,
            status="succeeded",
            artifacts=[evidence_path],
            evidence={"direction": direction, "input_manifest_sha256": input_sha},
        )

    smoke = tmp_path / "smoke-adapter"
    smoke.mkdir()
    (smoke / "adapter.bin").write_bytes(b"five steps")
    smoke_manifest = artifact_manifest(smoke)
    smoke_manifest_path = tmp_path / "smoke.manifest.json"
    smoke_manifest_path.write_text(json.dumps(smoke_manifest, sort_keys=True) + "\n")
    smoke_id = stable_run_id(
        stage="lora-smoke",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=campaign.dataset_manifest_sha256,
    )
    smoke_run = start_run(
        runs,
        run_id=smoke_id,
        stage="lora-smoke",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=campaign.dataset_manifest_sha256,
        command=["train", "steps=5"],
        metadata={"smoke": True},
    )
    smoke_completion = complete_run(
        runs,
        run_id=smoke_id,
        status="succeeded",
        artifacts=[smoke / "adapter.bin"],
        evidence={"runtime_lock": {"sha256": "b" * 64}},
    )
    roundtrip["lineage"] = {
        "hf_to_maxtext_completion_sha256": sha256_file(conversion_completions["hf-to-maxtext"]),
        "smoke_completion_sha256": sha256_file(smoke_completion),
        "smoke_run_id": smoke_id,
        "maxtext_to_hf_completion_sha256": sha256_file(conversion_completions["maxtext-to-hf"]),
        "logit_completion_sha256": sha256_file(conversion_completion),
        "logit_run_id": conversion_id,
        "hf_to_maxtext_run_id": json.loads(conversion_completions["hf-to-maxtext"].read_text())[
            "run_id"
        ],
        "maxtext_to_hf_run_id": json.loads(conversion_completions["maxtext-to-hf"].read_text())[
            "run_id"
        ],
    }
    roundtrip_path.write_text(json.dumps(roundtrip, sort_keys=True) + "\n")
    output = tmp_path / "roundtrip-stage.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "record-roundtrip",
            "--state",
            str(state),
            "--output",
            str(output),
            "--config",
            str(CONFIG),
            "--conversion-run",
            str(conversion_run),
            "--conversion-completion",
            str(conversion_completion),
            "--hf-to-maxtext-completion",
            str(conversion_completions["hf-to-maxtext"]),
            "--maxtext-to-hf-completion",
            str(conversion_completions["maxtext-to-hf"]),
            "--roundtrip-evidence",
            str(roundtrip_path),
            "--exported-checkpoint",
            str(exported),
            "--exported-checkpoint-manifest",
            str(exported_manifest_path),
            "--smoke-training-run",
            str(smoke_run),
            "--smoke-training-completion",
            str(smoke_completion),
            "--smoke-adapter",
            str(smoke),
            "--smoke-adapter-manifest",
            str(smoke_manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(BOOKFORGE_JAX_ROUNDTRIP="I_APPROVE_THIS_BOUNDED_ROUNDTRIP"),
    )
    document = json.loads(output.read_text())
    assert document["roundtrip_status"] == "passed"
    assert set(document["evidence_sha256"]) >= {
        "smoke_training_run",
        "smoke_training_completion",
        "smoke_adapter_manifest",
    }
    assert FidelityRun.read(state).next_stage().name == "train"


def test_candidate_recorder_derives_identity_from_merged_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    campaign, state = _campaign(tmp_path, through="train")
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"candidate")
    manifest = artifact_manifest(merged)
    manifest_path = tmp_path / "merged.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    candidate_id = candidate_id_for_checkpoint(
        config_path=CONFIG,
        dataset_manifest_sha256=campaign.dataset_manifest_sha256,
        training_run_id=str(campaign.training_run_id),
        merged_hf_checkpoint=merged,
    )
    evaluation = tmp_path / "development-evaluation.json"
    evaluation.write_text(
        json.dumps(
            _passing_development_evaluation(
                candidate_id=candidate_id,
                dataset_manifest_sha256=campaign.dataset_manifest_sha256,
                training_run_id=str(campaign.training_run_id),
            ),
            sort_keys=True,
        )
        + "\n"
    )
    output = tmp_path / "candidate-eval-stage.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "record-candidate-evaluation",
            "--state",
            str(state),
            "--output",
            str(output),
            "--config",
            str(CONFIG),
            "--development-evaluation",
            str(evaluation),
            "--merged-checkpoint",
            str(merged),
            "--merged-checkpoint-manifest",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    resumed = FidelityRun.read(state)
    assert resumed.candidate_id == candidate_id
    assert resumed.next_stage().name == "hf-export"


def test_training_recorder_accepts_exact_portable_completion_lineage(
    tmp_path: Path,
) -> None:
    campaign, state = _campaign(tmp_path, through="roundtrip")
    training_id = stable_run_id(
        stage="lora-train",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=campaign.dataset_manifest_sha256,
    )
    runs = tmp_path / "runs"
    base_root = tmp_path / "base-maxtext"
    base = base_root / "roundtrip" / "checkpoints" / "0" / "items"
    base.mkdir(parents=True)
    (base / "checkpoint").write_bytes(b"verified step-zero Orbax base")
    tokenizer = tmp_path / "tokenizer-snapshot"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n")
    base_manifest = artifact_manifest(base)
    tokenizer_manifest = artifact_manifest(tokenizer)
    base_manifest_path = tmp_path / "base.manifest.json"
    tokenizer_manifest_path = tmp_path / "tokenizer.manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, sort_keys=True) + "\n")
    tokenizer_manifest_path.write_text(json.dumps(tokenizer_manifest, sort_keys=True) + "\n")
    base_receipt_path = tmp_path / "base.receipt.json"
    base_receipt_path.write_text(
        json.dumps(
            orbax_leaf_receipt(
                base_root,
                base,
                expected_step=0,
                role="base-maxtext",
            ),
            sort_keys=True,
        )
        + "\n",
    )
    checkpoint_inputs = {
        "base_checkpoint": {
            "content_sha256": base_manifest["content_sha256"],
            "files": len(base_manifest["files"]),
            "bytes": sum(row["bytes"] for row in base_manifest["files"]),
        },
        "tokenizer_checkpoint": {
            "content_sha256": tokenizer_manifest["content_sha256"],
            "files": len(tokenizer_manifest["files"]),
            "bytes": sum(row["bytes"] for row in tokenizer_manifest["files"]),
        },
    }
    start_run(
        runs,
        run_id=training_id,
        stage="lora-train",
        config_sha256=campaign.config_sha256,
        dataset_manifest_sha256=campaign.dataset_manifest_sha256,
        command=["train"],
        metadata={"smoke": False, "inputs": checkpoint_inputs},
    )
    adapter = tmp_path / "adapter-source"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"trained adapter")
    packages = [{"name": "jax", "version": "0.8.2"}]
    runtime_lock = tmp_path / "runtime.lock.json"
    runtime_lock.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "packages": packages,
                "packages_sha256": canonical_sha256(packages),
            },
            sort_keys=True,
        )
        + "\n"
    )
    original_completion = complete_run(
        runs,
        run_id=training_id,
        status="succeeded",
        artifacts=[adapter / "adapter.bin"],
        evidence={"runtime_lock": {"sha256": sha256_file(runtime_lock)}},
    )
    package = tmp_path / "portable"
    portable = package_training_release(
        output_directory=adapter,
        run_directory=runs,
        training_run_id=training_id,
        runtime_lock=runtime_lock,
        destination=package,
    )
    remote = tmp_path / "remote-completion.json"
    remote.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": campaign.run_id,
                "training_run_id": training_id,
                "status": "succeeded",
                "backend": "modal-l40s",
                "config_sha256": campaign.config_sha256,
                "dataset_manifest_sha256": campaign.dataset_manifest_sha256,
                "base_checkpoint_manifest_sha256": sha256_file(base_manifest_path),
                "base_checkpoint_receipt_sha256": sha256_file(base_receipt_path),
                "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
                "source_training_completion_sha256": sha256_file(original_completion),
                "portable_package": portable,
                "files": artifact_manifest(package)["files"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    output = tmp_path / "training-stage.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "record-training",
        "--state",
        str(state),
        "--output",
        str(output),
        "--remote-completion",
        str(remote),
        "--remote-completion-sha256",
        sha256_file(remote),
        "--config",
        str(CONFIG),
        "--base-checkpoint-root",
        str(base_root),
        "--base-checkpoint-receipt",
        str(base_receipt_path),
        "--base-checkpoint-receipt-sha256",
        sha256_file(base_receipt_path),
        "--base-checkpoint-manifest",
        str(base_manifest_path),
        "--base-checkpoint-manifest-sha256",
        sha256_file(base_manifest_path),
        "--tokenizer-manifest",
        str(tokenizer_manifest_path),
        "--training-run",
        str(package / "training/run.json"),
        "--training-completion",
        str(package / "training/completion.json"),
        "--adapter",
        str(package / "adapter"),
        "--adapter-manifest",
        str(package / "adapter.manifest.json"),
        "--package-root",
        str(package),
        "--package-manifest",
        str(package / "package.manifest.json"),
        "--runtime-lock",
        str(package / "runtime.lock.json"),
        "--backend",
        "modal-l40s",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=_environment(BOOKFORGE_JAX_TRAIN="I_APPROVE_THIS_BOUNDED_TPU_JOB"),
    )
    assert FidelityRun.read(state).training_run_id == training_id
    document = json.loads(output.read_text())
    assert document["evidence_sha256"]["base_checkpoint_receipt"] == sha256_file(base_receipt_path)
    assert document["evidence_sha256"]["base_checkpoint_manifest"] == sha256_file(
        base_manifest_path
    )
    assert document["evidence_sha256"]["base_checkpoint_content"] == base_manifest["content_sha256"]
    assert document["evidence_sha256"]["remote_completion"] == sha256_file(remote)
    bad_command = command.copy()
    bad_command[bad_command.index("--remote-completion-sha256") + 1] = "0" * 64
    rejected = subprocess.run(
        bad_command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(BOOKFORGE_JAX_TRAIN="I_APPROVE_THIS_BOUNDED_TPU_JOB"),
    )
    assert rejected.returncode != 0
    assert "remote completion differs from its trusted SHA-256" in rejected.stderr
