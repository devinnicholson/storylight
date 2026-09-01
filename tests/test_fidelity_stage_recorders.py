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
from training.jax_fidelity.integrity import artifact_manifest, sha256_file
from training.jax_fidelity.manifests import complete_run, start_run
from training.jax_fidelity.release import candidate_id_for_checkpoint
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
            evidence={"direction": direction},
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
        "hf_to_maxtext_completion_sha256": sha256_file(
            conversion_completions["hf-to-maxtext"]
        ),
        "smoke_completion_sha256": sha256_file(smoke_completion),
        "smoke_run_id": smoke_id,
        "maxtext_to_hf_completion_sha256": sha256_file(
            conversion_completions["maxtext-to-hf"]
        ),
        "logit_completion_sha256": sha256_file(conversion_completion),
        "logit_run_id": conversion_id,
        "hf_to_maxtext_run_id": json.loads(
            conversion_completions["hf-to-maxtext"].read_text()
        )["run_id"],
        "maxtext_to_hf_run_id": json.loads(
            conversion_completions["maxtext-to-hf"].read_text()
        )["run_id"],
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
        env=_environment(
            BOOKFORGE_JAX_ROUNDTRIP="I_APPROVE_THIS_BOUNDED_ROUNDTRIP"
        ),
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
            {
                "schema_version": "1.0",
                "candidate_id": candidate_id,
                "stage": "development",
                "dataset_manifest_sha256": campaign.dataset_manifest_sha256,
                "training_run_id": campaign.training_run_id,
                "eligibility_decision": {"passed": True, "hidden_evaluated": False},
                "summary": {"records": 512, "surface": "raw"},
            },
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
