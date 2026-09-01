"""One-attempt Modal L40S fallback for the pinned Story Fidelity JAX run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/config.json"
PLAN_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-plan-2026-09.json"
APP_NAME = "bookforge-jax-fidelity"
GPU = "L40S"
TIMEOUT_SECONDS = 2_700
MAX_CONTAINERS = 1
BUDGET_MONTH = "2026-09"
WORKSPACE_HARD_STOP_USD = 28.0
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z][a-z0-9-]{7,62}$")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_INPUT_ROOT = Path("/inputs")
_SCRATCH_ROOT = Path("/scratch")
_RELEASE_ROOT = Path("/releases")


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _pinned_image_uri() -> str:
    config = _json_object(CONFIG_PATH)
    versions = config.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("JAX config has no versions object")
    image = versions.get("container_image")
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise ValueError("JAX container image must be pinned by sha256 digest")
    return image


def _maxtext_revision() -> str:
    versions = _json_object(CONFIG_PATH).get("versions")
    if not isinstance(versions, dict):
        raise ValueError("JAX config has no versions object")
    revision = versions.get("maxtext_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("MaxText revision must be a full Git commit")
    return revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _approval_token(
    run_id: str,
    config_sha256: str,
    dataset_sha256: str,
    prepared_sha256: str,
    input_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_sha256: str,
    *,
    smoke: bool,
) -> str:
    mode = "smoke" if smoke else "train"
    return (
        f"APPROVE_MODAL_JAX_RUN:{run_id}:{config_sha256}:{dataset_sha256}:"
        f"{prepared_sha256}:{input_manifest_sha256}:{checkpoint_manifest_sha256}:"
        f"{tokenizer_manifest_sha256}:{gcp_rejection_sha256}:{mode}"
    )


def _validate_request(
    request: dict[str, object],
) -> tuple[str, str, str, str, str, str, str, str, bool]:
    run_id = request.get("run_id")
    config_sha = request.get("config_sha256")
    dataset_sha = request.get("dataset_manifest_sha256")
    prepared_sha = request.get("prepared_train_sha256")
    input_manifest_sha = request.get("input_manifest_sha256")
    checkpoint_manifest_sha = request.get("base_checkpoint_manifest_sha256")
    tokenizer_manifest_sha = request.get("tokenizer_manifest_sha256")
    rejection_sha = request.get("gcp_rejection_sha256")
    smoke = request.get("smoke")
    approval = request.get("approval_token")
    rejection = request.get("gcp_rejection")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid Modal JAX run ID")
    for name, value in (
        ("config_sha256", config_sha),
        ("dataset_manifest_sha256", dataset_sha),
        ("prepared_train_sha256", prepared_sha),
        ("input_manifest_sha256", input_manifest_sha),
        ("base_checkpoint_manifest_sha256", checkpoint_manifest_sha),
        ("tokenizer_manifest_sha256", tokenizer_manifest_sha),
        ("gcp_rejection_sha256", rejection_sha),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if type(smoke) is not bool:
        raise ValueError("smoke must be a boolean")
    if not isinstance(rejection, dict):
        raise ValueError("Modal fallback requires immutable GCP rejection evidence")
    if (
        rejection.get("schema_version") != "1.0"
        or rejection.get("producer") != "bookforge-gcp-jax-submitter"
        or rejection.get("run_id") != run_id
        or rejection.get("status") != "rejected-pre-billable"
        or rejection.get("submission_intent_created") is not False
        or rejection.get("custom_job_created") is not False
        or not isinstance(rejection.get("spec_sha256"), str)
        or _SHA256.fullmatch(str(rejection["spec_sha256"])) is None
        or not isinstance(rejection.get("reason"), str)
        or not str(rejection["reason"]).strip()
    ):
        raise ValueError("GCP rejection evidence is not pre-billable or run-bound")
    expected_bindings = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_train_sha256": prepared_sha,
        "input_manifest_sha256": input_manifest_sha,
        "base_checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
    }
    if rejection.get("input_bindings") != expected_bindings or rejection.get(
        "input_bindings_sha256"
    ) != _canonical_sha256(expected_bindings):
        raise ValueError("GCP rejection is not bound to the requested staged inputs")
    rejection_bytes = (json.dumps(rejection, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(rejection_bytes).hexdigest() != rejection_sha:
        raise ValueError("GCP rejection evidence checksum changed")
    expected = _approval_token(
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        checkpoint_manifest_sha,
        tokenizer_manifest_sha,
        rejection_sha,
        smoke=smoke,
    )
    if approval != expected:
        raise ValueError("Modal fallback approval token is not exact")
    return (
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        checkpoint_manifest_sha,
        tokenizer_manifest_sha,
        rejection_sha,
        smoke,
    )


jax_image = (
    modal.Image.from_registry(_pinned_image_uri())
    .apt_install("git", "ca-certificates")
    .run_commands(
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /opt/MaxText",
        f"git -C /opt/MaxText checkout {_maxtext_revision()}",
        f'test "$(git -C /opt/MaxText rev-parse HEAD)" = "{_maxtext_revision()}"',
        'test -z "$(git -C /opt/MaxText status --porcelain)"',
    )
    .pip_install_from_requirements(REPOSITORY_ROOT / "training/jax_fidelity/requirements.lock")
    .pip_install("jax[cuda12]==0.11.0")
    .add_local_dir(REPOSITORY_ROOT / "training", "/opt/bookforge/training", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "src", "/opt/bookforge/src", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "infra/gcp/jax", "/opt/bookforge/infra/gcp/jax", copy=True)
    .run_commands(
        "PYTHONPATH=/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --write-lock /opt/bookforge/runtime.lock.json",
        "PYTHONPATH=/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --lock /opt/bookforge/runtime.lock.json",
    )
    .env({"PYTHONPATH": "/opt/bookforge:/opt/bookforge/src", "JAX_PLATFORMS": "cuda"})
)

input_volume = modal.Volume.from_name("bookforge-jax-fidelity-inputs", create_if_missing=False)
scratch_volume = modal.Volume.from_name("bookforge-jax-fidelity-scratch", create_if_missing=False)
release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=jax_image,
    gpu=GPU,
    cpu=8,
    memory=65_536,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=MAX_CONTAINERS,
    volumes={
        str(_INPUT_ROOT): input_volume,
        str(_SCRATCH_ROOT): scratch_volume,
        str(_RELEASE_ROOT): release_volume,
    },
    secrets=[modal.Secret.from_name("bookforge-hf-read")],
)
def run_finite(request: dict[str, object]) -> dict[str, object]:
    """Run once; a reused or partially completed run ID fails closed."""

    (
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        checkpoint_manifest_sha,
        tokenizer_manifest_sha,
        _,
        smoke,
    ) = _validate_request(request)
    input_directory = _INPUT_ROOT / run_id
    scratch_directory = _SCRATCH_ROOT / run_id
    release_directory = _RELEASE_ROOT / run_id
    completion = release_directory / "completion.json"
    if scratch_directory.exists() or release_directory.exists():
        raise RuntimeError("run ID already has scratch or release state and cannot be retried")
    config = input_directory / "config.json"
    manifest = input_directory / "dataset" / "manifest.json"
    prepared = input_directory / "prepared" / "train.jsonl"
    checkpoint = input_directory / "checkpoint"
    tokenizer_checkpoint = input_directory / "tokenizer"
    checkpoint_manifest_path = input_directory / "checkpoint.manifest.json"
    tokenizer_manifest_path = input_directory / "tokenizer.manifest.json"
    from infra.gcp.jax.vertex_entrypoint import _verify_input_population

    _verify_input_population(
        input_directory,
        run_id=run_id,
        expected_manifest_sha256=input_manifest_sha,
    )
    if _sha256(config) != config_sha or _sha256(manifest) != dataset_sha:
        raise RuntimeError("Modal input contract hash changed")
    if _sha256(prepared) != prepared_sha:
        raise RuntimeError("prepared training data hash changed")
    if _sha256(checkpoint_manifest_path) != checkpoint_manifest_sha:
        raise RuntimeError("base checkpoint manifest hash changed")
    if _sha256(tokenizer_manifest_path) != tokenizer_manifest_sha:
        raise RuntimeError("tokenizer manifest hash changed")

    scratch_directory.mkdir(parents=True, exist_ok=False)
    output_directory = scratch_directory / "output"
    run_directory = scratch_directory / "runs"
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import validate_dataset_manifest, verify_artifact_manifest
    from training.jax_fidelity.manifests import stable_run_id
    from training.jax_fidelity.remote_release import package_training_release
    from training.jax_fidelity.runtime import approval_token

    experiment = load_config(config)
    validated = validate_dataset_manifest(
        manifest,
        expected_manifest_sha256=dataset_sha,
        required_split_records=experiment.dataset["required_split_records"],
    )
    verify_artifact_manifest(checkpoint, _json_object(checkpoint_manifest_path))
    verify_artifact_manifest(tokenizer_checkpoint, _json_object(tokenizer_manifest_path))
    stage = "lora-smoke" if smoke else "lora-train"
    training_run_id = stable_run_id(
        stage=stage,
        config_sha256=config_sha,
        dataset_manifest_sha256=validated.manifest_sha256,
    )
    environment = os.environ.copy()
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage=stage,
        run_id=training_run_id,
        config_sha256=config_sha,
        input_sha256=prepared_sha,
    )
    command = [
        "python3",
        "-m",
        "training.jax_fidelity.train",
        "--config",
        str(config),
        "--dataset-manifest",
        str(manifest),
        "--dataset-manifest-sha256",
        dataset_sha,
        "--prepared-train-jsonl",
        str(prepared),
        "--prepared-train-sha256",
        prepared_sha,
        "--base-checkpoint",
        str(checkpoint),
        "--hf-tokenizer-checkpoint",
        str(tokenizer_checkpoint),
        "--output-directory",
        str(output_directory),
        "--run-directory",
        str(run_directory),
        "--maxtext-root",
        "/opt/MaxText",
        "--execute",
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=environment, timeout=2_640)

    training_completion_path = run_directory / training_run_id / "completion.json"
    training_completion = _json_object(training_completion_path)
    if (
        training_completion.get("status") != "succeeded"
        or training_completion.get("run_id") != training_run_id
        or not training_completion.get("artifacts")
        or not training_completion.get("evidence")
    ):
        raise RuntimeError("training has no nonempty successful terminal evidence")
    package_evidence = package_training_release(
        output_directory=output_directory,
        run_directory=run_directory,
        training_run_id=training_run_id,
        runtime_lock="/opt/bookforge/runtime.lock.json",
        destination=release_directory,
    )
    files: list[dict[str, object]] = []
    for source in sorted(item for item in release_directory.rglob("*") if item.is_file()):
        if source.is_symlink():
            raise RuntimeError("release artifacts may not be symbolic links")
        relative = source.relative_to(release_directory)
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "training_run_id": training_run_id,
        "status": "succeeded",
        "backend": "modal-l40s",
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "input_manifest_sha256": input_manifest_sha,
        "base_checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
        "source_training_completion_sha256": _sha256(training_completion_path),
        "portable_package": package_evidence,
        "files": files,
    }
    temporary = completion.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, completion)
    release_volume.commit()
    scratch_volume.commit()
    return payload


def _authoritative_workspace_total() -> float:
    completed = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Modal billing report was not a list")
    return sum(float(row["Cost"]) for row in rows)


@app.local_entrypoint()
def run_cli(
    run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    base_checkpoint_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_evidence: str,
    approval_token_value: str,
    smoke: bool = True,
) -> None:
    plan = _json_object(PLAN_PATH)
    if plan.get("status") != "plan-only" or plan.get("automatic_retries") != 0:
        raise RuntimeError("Modal plan is not safe to execute")
    ceiling = plan.get("gross_ceiling_usd")
    if not isinstance(ceiling, int | float) or ceiling <= 0:
        raise RuntimeError("Modal gross ceiling is invalid")
    if plan.get("gross_ceiling_policy") != "declared-estimate-not-provider-enforced":
        raise RuntimeError("Modal gross ceiling policy is missing or ambiguous")
    workspace_total = _authoritative_workspace_total()
    if plan.get("budget_month") != BUDGET_MONTH:
        raise RuntimeError("Modal plan is not for the current approved budget month")
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("Modal plan is outside its approved budget month")
    hard_stop = plan.get("workspace_hard_stop_usd")
    if hard_stop != WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace hard stop changed")
    if workspace_total + float(ceiling) > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace budget has insufficient headroom")
    rejection_path = Path(gcp_rejection_evidence)
    if not rejection_path.is_file() or rejection_path.is_symlink():
        raise RuntimeError("GCP rejection evidence must be a regular local file")
    rejection = _json_object(rejection_path)
    rejection_sha256 = _sha256(rejection_path)
    expected = _approval_token(
        run_id,
        config_sha256,
        dataset_manifest_sha256,
        prepared_train_sha256,
        input_manifest_sha256,
        base_checkpoint_manifest_sha256,
        tokenizer_manifest_sha256,
        rejection_sha256,
        smoke=smoke,
    )
    if approval_token_value != expected:
        raise RuntimeError("exact Modal approval token is required")
    request: dict[str, object] = {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "prepared_train_sha256": prepared_train_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "base_checkpoint_manifest_sha256": base_checkpoint_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "gcp_rejection": rejection,
        "gcp_rejection_sha256": rejection_sha256,
        "smoke": smoke,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    from infra.gcp.jax.modal_reconciliation import append_reconciliation

    result: dict[str, object] | None = None
    status = "remote-error"
    postrun_total: float | None = None
    postrun_error: str | None = None
    try:
        result = run_finite.remote(request)
        status = "succeeded"
    finally:
        try:
            postrun_total = _authoritative_workspace_total()
        except Exception as error:
            postrun_error = f"{type(error).__name__}: {error}"
        append_reconciliation(
            LEDGER_PATH,
            attempt_id=f"jax:{run_id}",
            stage="jax-training",
            workspace_before_usd=workspace_total,
            workspace_after_usd=postrun_total,
            declared_ceiling_usd=float(ceiling),
            status=status,
            result=result,
            postrun_report_error=postrun_error,
        )
    if result is None:
        raise RuntimeError("Modal training returned no result")
    print(
        json.dumps(
            {
                "authoritative_workspace_total_usd": workspace_total,
                "full_call_ceiling_usd": ceiling,
                "full_call_ceiling_provider_enforced": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
