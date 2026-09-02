"""One finite CUDA prediction pass over the public development population."""

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
APP_NAME = "bookforge-jax-development-prediction"
GPU = "L40S"
MAX_CONTAINERS = 1
RETRIES = 0
TIMEOUT_SECONDS = 1_200
BUDGET_MONTH = "2026-09"
WORKSPACE_HARD_STOP_USD = 28.0
FULL_CALL_CEILING_USD = 1.25
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
_INPUT_ROOT = Path("/inputs/prediction")
_OUTPUT_ROOT = Path("/releases/prediction")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")
_DIGEST_IMAGE = re.compile(r"nvcr\.io/nvidia/pytorch:[^@]+@sha256:[0-9a-f]{64}\Z")


def _pinned_nvidia_image() -> str:
    image = os.environ.get("BOOKFORGE_NVIDIA_PYTORCH_IMAGE", "")
    if _DIGEST_IMAGE.fullmatch(image) is None:
        raise RuntimeError(
            "BOOKFORGE_NVIDIA_PYTORCH_IMAGE must be an NVIDIA PyTorch image pinned by digest"
        )
    return image


NVIDIA_PYTORCH_IMAGE = _pinned_nvidia_image()

prediction_image = (
    modal.Image.from_registry(NVIDIA_PYTORCH_IMAGE)
    .pip_install(
        "transformers==5.13.0",
        "huggingface-hub==1.26.0",
        "safetensors==0.8.0",
        "pydantic==2.13.4",
    )
    .add_local_dir(REPOSITORY_ROOT / "src", "/opt/bookforge/src", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "training", "/opt/bookforge/training", copy=True)
    .add_local_file(
        REPOSITORY_ROOT / "infra/gcp/jax/stage_modal_prediction_inputs.py",
        "/opt/bookforge/infra/gcp/jax/stage_modal_prediction_inputs.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/bookforge:/opt/bookforge/src",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
)

input_volume = modal.Volume.from_name("bookforge-jax-fidelity-inputs", create_if_missing=False)
release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approval_token(
    *,
    run_id: str,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    development_records_sha256: str,
    candidate_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_content_sha256: str,
    input_manifest_sha256: str,
    batch_size: int,
) -> str:
    return (
        f"APPROVE_MODAL_JAX_PREDICTION:{run_id}:{candidate_id}:{config_sha256}:"
        f"{dataset_manifest_sha256}:{development_records_sha256}:"
        f"{candidate_manifest_sha256}:{checkpoint_manifest_sha256}:"
        f"{checkpoint_content_sha256}:{input_manifest_sha256}:{batch_size}"
    )


def _validate_request(request: dict[str, object]) -> tuple[str, str, dict[str, object], str, int]:
    run_id = request.get("run_id")
    candidate_id = request.get("candidate_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid Modal prediction run ID")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("invalid content-addressed prediction candidate ID")
    hash_names = (
        "config_sha256",
        "dataset_manifest_sha256",
        "development_records_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
        "input_manifest_sha256",
    )
    hashes: dict[str, str] = {}
    for name in hash_names:
        value = request.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
        hashes[name] = value
    batch_size = request.get("batch_size")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer in 1..16")
    bindings: dict[str, object] = {
        "candidate_id": candidate_id,
        "config_sha256": hashes["config_sha256"],
        "dataset_manifest_sha256": hashes["dataset_manifest_sha256"],
        "development_records_sha256": hashes["development_records_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "checkpoint_manifest_sha256": hashes["checkpoint_manifest_sha256"],
        "checkpoint_content_sha256": hashes["checkpoint_content_sha256"],
    }
    expected = _approval_token(
        run_id=run_id,
        candidate_id=candidate_id,
        input_manifest_sha256=hashes["input_manifest_sha256"],
        batch_size=batch_size,
        **{name: str(value) for name, value in bindings.items() if name != "candidate_id"},
    )
    if request.get("approval_token") != expected:
        raise ValueError("Modal development-prediction approval token is not exact")
    return run_id, candidate_id, bindings, hashes["input_manifest_sha256"], batch_size


def _write_once(path: Path, content: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


@app.function(
    image=prediction_image,
    gpu=GPU,
    cpu=4,
    memory=32_768,
    timeout=TIMEOUT_SECONDS,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
    volumes={"/inputs": input_volume, "/releases": release_volume},
)
def predict_finite(request: dict[str, object]) -> dict[str, object]:
    """Generate all 512 development outputs once, with no network model access."""

    run_id, candidate_id, bindings, input_manifest_sha, batch_size = _validate_request(request)
    input_directory = _INPUT_ROOT / run_id
    output_directory = _OUTPUT_ROOT / run_id
    if output_directory.exists():
        raise RuntimeError("prediction output prefix already has terminal or partial state")

    from infra.gcp.jax.stage_modal_prediction_inputs import (
        validate_public_dataset,
        verify_staged_inputs,
    )
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import (
        artifact_manifest,
        canonical_json_bytes,
        canonical_sha256,
    )
    from training.jax_fidelity.merged_candidate import validate_merged_candidate_manifest
    from training.jax_fidelity.predict import generate_predictions, load_records

    verify_staged_inputs(
        input_directory,
        run_id=run_id,
        expected_manifest_sha256=input_manifest_sha,
        expected_bindings=bindings,
    )
    config_path = input_directory / "config.json"
    dataset_manifest = input_directory / "dataset/manifest.json"
    records_path = input_directory / "dataset/development.jsonl"
    candidate_manifest = input_directory / "candidate/candidate.manifest.json"
    checkpoint = input_directory / "candidate/merged-hf"
    config = load_config(config_path)
    dataset_sha, records_sha = validate_public_dataset(dataset_manifest, records_path)
    if config.sha256 != bindings["config_sha256"]:
        raise RuntimeError("prediction config checksum changed")
    if dataset_sha != bindings["dataset_manifest_sha256"]:
        raise RuntimeError("prediction dataset manifest checksum changed")
    if records_sha != bindings["development_records_sha256"]:
        raise RuntimeError("prediction development record checksum changed")
    candidate = validate_merged_candidate_manifest(
        candidate_manifest,
        checkpoint,
        config_path=config_path,
        expected_manifest_sha256=str(bindings["candidate_manifest_sha256"]),
        expected_config_sha256=config.sha256,
        expected_dataset_manifest_sha256=dataset_sha,
        expected_candidate_id=candidate_id,
    )
    checkpoint_document = artifact_manifest(checkpoint)
    if (
        canonical_sha256(checkpoint_document) != bindings["checkpoint_manifest_sha256"]
        or checkpoint_document["content_sha256"] != bindings["checkpoint_content_sha256"]
        or candidate["candidate_id"] != candidate_id
    ):
        raise RuntimeError("prediction checkpoint manifest or lineage changed")

    output_directory.mkdir(parents=True, exist_ok=False)
    intent_path = output_directory / "intent.json"
    _write_once(
        intent_path,
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "status": "prediction-intent-recorded",
                "retry_allowed": False,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "input_manifest_sha256": input_manifest_sha,
                "bindings": bindings,
                "batch_size": batch_size,
            }
        ),
        mode=0o600,
    )
    release_volume.commit()
    records = load_records(records_path)
    predictions = generate_predictions(
        records,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_output_tokens=config.production["completion_budget_tokens"],
    )
    if len(predictions) != 512 or len({row["record_id"] for row in predictions}) != 512:
        raise RuntimeError("prediction worker did not produce the complete development population")
    predictions_path = output_directory / "predictions.jsonl"
    _write_once(predictions_path, b"".join(canonical_json_bytes(row) for row in predictions))
    release_volume.commit()
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (intent_path, predictions_path)
    ]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l40s-cuda",
        "run_id": run_id,
        "candidate_id": candidate_id,
        "input_manifest_sha256": input_manifest_sha,
        **bindings,
        "batch_size": batch_size,
        "maximum_output_tokens": config.production["completion_budget_tokens"],
        "predictions": len(predictions),
        "split": "development",
        "hidden_evaluated": False,
        "passages_retained": False,
        "files": files,
    }
    completion = output_directory / "completion.json"
    _write_once(completion, canonical_json_bytes(payload))
    release_volume.commit()
    return {**payload, "completion_sha256": _sha256(completion)}


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
        raise RuntimeError("Modal billing report was not a JSON list")
    return sum(float(row["Cost"]) for row in rows)


@app.local_entrypoint()
def predict_cli(
    run_id: str,
    candidate_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    development_records_sha256: str,
    candidate_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_content_sha256: str,
    input_manifest_sha256: str,
    approval_token_value: str,
    batch_size: int = 8,
) -> None:
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("Modal prediction authorization is outside its budget month")
    workspace_before = _authoritative_workspace_total()
    if workspace_before + FULL_CALL_CEILING_USD > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace budget has insufficient prediction headroom")
    request: dict[str, object] = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "development_records_sha256": development_records_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_content_sha256": checkpoint_content_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "batch_size": batch_size,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    from infra.gcp.jax.modal_reconciliation import (
        append_reconciliation,
        assert_attempt_available,
        reserve_attempt,
    )

    attempt_id = f"jax-prediction:{run_id}"
    assert_attempt_available(LEDGER_PATH, attempt_id=attempt_id)
    reserve_attempt(LEDGER_PATH, attempt_id=attempt_id, stage="jax-development-prediction")
    result: dict[str, object] | None = None
    status = "remote-error"
    workspace_after: float | None = None
    postrun_error: str | None = None
    try:
        result = predict_finite.remote(request)
        status = "succeeded"
    finally:
        try:
            workspace_after = _authoritative_workspace_total()
        except Exception as error:
            postrun_error = f"{type(error).__name__}: {error}"
        append_reconciliation(
            LEDGER_PATH,
            attempt_id=attempt_id,
            stage="jax-development-prediction",
            workspace_before_usd=workspace_before,
            workspace_after_usd=workspace_after,
            declared_ceiling_usd=FULL_CALL_CEILING_USD,
            status=status,
            result=result,
            postrun_report_error=postrun_error,
        )
    if result is None:
        raise RuntimeError("Modal prediction returned no result")
    print(json.dumps(result, indent=2, sort_keys=True))
