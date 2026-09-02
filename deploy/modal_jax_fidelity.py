"""One-attempt two-L4 fallback for the pinned Story Fidelity JAX run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import modal

from deploy.modal_jax_image import (
    JAX_IMAGE,
    offline_environment,
    pinned_image_uri,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/config.json"
PLAN_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-plan-2026-09.json"
APP_NAME = "bookforge-jax-fidelity"
GPU = "L4:2"
BACKEND = "modal-l4x2"
TIMEOUT_SECONDS = 3_600
TRAINING_TIMEOUT_SECONDS = 3_180
PUBLICATION_RESERVE_SECONDS = 600
MIN_INLINE_FINALIZATION_SECONDS = 300
FINALIZE_TIMEOUT_SECONDS = 1_800
MAX_CONTAINERS = 1
BUDGET_MONTH = "2026-09"
WORKSPACE_HARD_STOP_USD = 28.0
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z][a-z0-9-]{7,62}$")
_RUN_DATE = re.compile(r"(?:^|-)(20\d{6})(?:-|$)")
_GCP_PROJECT = "your-gcp-project"
_GCP_REGION = "us-east1"
_GCP_CREATE_METHOD = "google.cloud.aiplatform.v1.JobService.CreateCustomJob"
_GCP_UNAVAILABLE_PRODUCER = "bookforge-gcp-jax-unavailability-recorder"
_GCP_UNAVAILABLE_ABSENCE = "billing-disabled-plus-empty-create-audit-log-400d"
_GCP_IMAGE_TAG = re.compile(
    r"^us-east1-docker\.pkg\.dev/your-gcp-project/"
    r"bookforge-jax/trainer:[0-9a-f]{20}$"
)
_INPUT_ROOT = Path("/inputs")
_SCRATCH_ROOT = Path("/scratch")
_RELEASE_ROOT = Path("/releases")


def _pinned_image_uri() -> str:
    return pinned_image_uri()


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _recent_run_id(run_id: str, *, now: datetime) -> bool:
    match = _RUN_DATE.search(run_id)
    if match is None:
        return False
    try:
        stamped = datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return False
    age = now.date() - stamped
    return 0 <= age.days <= 7


def _gcp_audit_filter(run_id: str) -> str:
    return " AND ".join(
        (
            'logName="projects/your-gcp-project/logs/'
            'cloudaudit.googleapis.com%2Factivity"',
            'protoPayload.serviceName="aiplatform.googleapis.com"',
            f'protoPayload.methodName="{_GCP_CREATE_METHOD}"',
            f'protoPayload.request.customJob.displayName="{run_id}"',
        )
    )


def _validate_gcp_unavailability_rejection(
    rejection: dict[str, object],
    *,
    run_id: str,
    expected_bindings: dict[str, object],
    smoke: bool,
) -> bool:
    """Validate the no-image, billing-disabled fallback evidence contract."""

    if "spec_sha256" in rejection:
        return False
    try:
        checked_at = datetime.fromisoformat(str(rejection["checked_at"]))
    except (KeyError, ValueError):
        return False
    if checked_at.tzinfo is None:
        return False
    age = datetime.now(UTC) - checked_at.astimezone(UTC)
    if not (0 <= age.total_seconds() <= 1_800) or not _recent_run_id(
        run_id, now=datetime.now(UTC)
    ):
        return False
    billing = rejection.get("billing_verification")
    audit = rejection.get("audit_absence")
    container = rejection.get("container_image")
    resource = rejection.get("intended_vertex_resource")
    if not all(isinstance(value, dict) for value in (billing, audit, container, resource)):
        return False
    assert isinstance(billing, dict)
    assert isinstance(audit, dict)
    assert isinstance(container, dict)
    assert isinstance(resource, dict)
    source_sha = container.get("image_source_sha256")
    image_plan_sha = container.get("image_build_plan_sha256")
    tagged_uri = container.get("intended_tagged_uri")
    if (
        rejection.get("schema_version") != "1.0"
        or rejection.get("producer") != _GCP_UNAVAILABLE_PRODUCER
        or rejection.get("status") != "rejected-pre-billable"
        or rejection.get("rejection_kind") != "gcp-unavailable-before-image-build"
        or rejection.get("project") != _GCP_PROJECT
        or rejection.get("region") != _GCP_REGION
        or rejection.get("run_id") != run_id
        or rejection.get("submission_intent_created") is not False
        or rejection.get("custom_job_created") is not False
        or rejection.get("job_absence_verified") is not True
        or rejection.get("run_id_absence_basis") != _GCP_UNAVAILABLE_ABSENCE
        or rejection.get("fallback_allowed") is not True
        or rejection.get("queries_read_only") is not True
        or rejection.get("remote_mutation") is not False
        or not isinstance(rejection.get("reason"), str)
        or not str(rejection["reason"]).strip()
        or billing.get("active_project") != _GCP_PROJECT
        or not isinstance(billing.get("active_account"), str)
        or not str(billing["active_account"]).strip()
        or billing.get("billing_enabled") is not False
        or not isinstance(billing.get("billing_account_name"), str)
        or audit.get("log") != "cloudaudit.googleapis.com/activity"
        or audit.get("service_name") != "aiplatform.googleapis.com"
        or audit.get("method_name") != _GCP_CREATE_METHOD
        or audit.get("display_name") != run_id
        or audit.get("filter") != _gcp_audit_filter(run_id)
        or audit.get("freshness") != "400d"
        or audit.get("maximum_run_id_age_days") != 7
        or audit.get("matching_entries") != []
        or not isinstance(source_sha, str)
        or _SHA256.fullmatch(source_sha) is None
        or not isinstance(image_plan_sha, str)
        or _SHA256.fullmatch(image_plan_sha) is None
        or not isinstance(tagged_uri, str)
        or _GCP_IMAGE_TAG.fullmatch(tagged_uri) is None
        or tagged_uri.rsplit(":", 1)[-1] != source_sha[:20]
        or container.get("runnable_digest_uri") is not None
        or container.get("digest_resolved") is not False
        or container.get("build_attempted") is not False
        or container.get("build_intent_created") is not False
        or container.get("build_receipt_created") is not False
        or container.get("build_admission") != "blocked-billing-disabled"
    ):
        return False
    service_account = resource.get("service_account")
    input_prefix = resource.get("input_prefix")
    release_prefix = resource.get("release_prefix")
    resource_container = resource.get("container")
    return not (
        resource.get("backend") != "vertex-custom-job"
        or resource.get("project") != _GCP_PROJECT
        or resource.get("region") != _GCP_REGION
        or resource.get("display_name") != run_id
        or resource.get("create_method") != _GCP_CREATE_METHOD
        or resource.get("create_url")
        != (
            "https://us-east1-aiplatform.googleapis.com/v1/projects/"
            "your-gcp-project/locations/us-east1/customJobs"
        )
        or resource.get("machine_type") != "ct6e-standard-1t"
        or resource.get("tpu_chips") != 1
        or resource.get("replicas") != 1
        or resource.get("timeout_seconds") != 2_700
        or resource.get("automatic_retries") != 0
        or resource.get("endpoint_created") is not False
        or not isinstance(service_account, str)
        or not service_account.endswith(
            "@your-gcp-project.iam.gserviceaccount.com"
        )
        or not isinstance(input_prefix, str)
        or not input_prefix.startswith("gs://")
        or not input_prefix.endswith(f"/inputs/{run_id}")
        or not isinstance(release_prefix, str)
        or not release_prefix.startswith("gs://")
        or not release_prefix.endswith(f"/releases/{run_id}")
        or input_prefix.removeprefix("gs://").split("/", 1)[0]
        == release_prefix.removeprefix("gs://").split("/", 1)[0]
        or resource.get("input_bindings") != expected_bindings
        or resource.get("smoke") is not smoke
        or resource_container
        != {
            "image_source_sha256": source_sha,
            "image_build_plan_sha256": image_plan_sha,
            "intended_tagged_uri": tagged_uri,
            "runnable_digest_uri": None,
            "digest_resolved": False,
            "build_attempted": False,
        }
        or rejection.get("intended_vertex_resource_sha256")
        != _canonical_sha256(resource)
    )


def _write_once_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bounded_training_timeout(elapsed_seconds: float) -> int:
    """Reserve a fixed tail of the Modal deadline for durable publication."""

    available = math.floor(
        TIMEOUT_SECONDS - PUBLICATION_RESERVE_SECONDS - elapsed_seconds
    )
    if available < 1:
        raise RuntimeError("Modal deadline has no safe training window remaining")
    return min(TRAINING_TIMEOUT_SECONDS, available)


def _finalize_approval_token(
    run_id: str, input_manifest_sha256: str, gcp_rejection_sha256: str
) -> str:
    return (
        f"APPROVE_MODAL_JAX_FINALIZE:{run_id}:{input_manifest_sha256}:"
        f"{gcp_rejection_sha256}"
    )


def _attempt_document(
    *,
    run_id: str,
    training_run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_receipt_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_sha256: str,
    smoke: bool,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "producer": "bookforge-modal-jax-full-trainer",
        "status": "started",
        "run_id": run_id,
        "training_run_id": training_run_id,
        "backend": BACKEND,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "prepared_train_sha256": prepared_train_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "base_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "base_checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "gcp_rejection_sha256": gcp_rejection_sha256,
        "smoke": smoke,
        "automatic_retries": 0,
    }


def _safe_release_rows(
    staging: Path, completion: dict[str, object]
) -> dict[str, dict[str, object]]:
    raw_rows = completion.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise RuntimeError("staged release completion has no files")
    rows: dict[str, dict[str, object]] = {}
    for raw in raw_rows:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("path"), str)
            or type(raw.get("bytes")) is not int
            or int(raw["bytes"]) < 0
            or not isinstance(raw.get("sha256"), str)
            or _SHA256.fullmatch(str(raw["sha256"])) is None
        ):
            raise RuntimeError("staged release contains a malformed file row")
        pure = PurePosixPath(str(raw["path"]))
        relative = pure.as_posix()
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or relative != raw["path"]
            or relative == "completion.json"
            or relative in rows
        ):
            raise RuntimeError("staged release contains an unsafe or duplicate path")
        source = staging.joinpath(*pure.parts)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != raw["bytes"]
            or _sha256(source) != raw["sha256"]
        ):
            raise RuntimeError(f"staged release file changed: {relative}")
        rows[relative] = dict(raw)
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path != staging / "completion.json"
    }
    if actual != set(rows):
        raise RuntimeError("staged release has undeclared or missing files")
    return rows


def _copy_release_file_once(
    source: Path,
    destination: Path,
    row: dict[str, object],
    *,
    trusted_root: Path,
) -> None:
    try:
        relative_parent = destination.parent.relative_to(trusted_root)
    except ValueError as error:
        raise RuntimeError("release destination escaped its trusted root") from error
    current = trusted_root
    for part in relative_parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("release destination contains a symbolic-link parent")
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise RuntimeError("release destination parent is not a directory")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    digest = hashlib.sha256()
    try:
        try:
            with source.open("rb") as reader, os.fdopen(
                descriptor, "wb", closefd=False
            ) as writer:
                for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    try:
        verified = (
            destination.stat().st_size == row["bytes"]
            and digest.hexdigest() == row["sha256"]
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if not verified:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"release copy failed verification: {row['path']}")


def _publish_staged_release(
    staging: Path, destination: Path, *, commit: Callable[[], object]
) -> None:
    """Resume a two-phase, create-only publication and write completion last."""

    completion_source = staging / "completion.json"
    completion = _json_object(completion_source)
    rows = _safe_release_rows(staging, completion)
    if destination.is_symlink():
        raise RuntimeError("release destination may not be a symbolic link")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise RuntimeError("release destination contains a symbolic link")
    existing = {
        path.relative_to(destination).as_posix(): path
        for path in destination.rglob("*")
        if path.is_file()
    }
    unknown = set(existing) - {*rows, "completion.json"}
    if unknown:
        raise RuntimeError("release destination contains undeclared files")
    for relative, path in existing.items():
        if relative == "completion.json":
            continue
        row = rows[relative]
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or _sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"existing release file conflicts: {relative}")
    completion_destination = destination / "completion.json"
    if completion_destination.exists() or completion_destination.is_symlink():
        if (
            not completion_destination.is_file()
            or completion_destination.is_symlink()
            or completion_destination.read_bytes() != completion_source.read_bytes()
            or set(existing) != {*rows, "completion.json"}
        ):
            raise RuntimeError("existing release completion conflicts with staged evidence")
        return
    for relative, row in rows.items():
        if relative in existing:
            continue
        pure = PurePosixPath(relative)
        _copy_release_file_once(
            staging.joinpath(*pure.parts),
            destination.joinpath(*pure.parts),
            row,
            trusted_root=destination,
        )
    commit()
    _copy_release_file_once(
        completion_source,
        completion_destination,
        {
            "path": "completion.json",
            "bytes": completion_source.stat().st_size,
            "sha256": _sha256(completion_source),
        },
        trusted_root=destination,
    )
    commit()


def _approval_token(
    run_id: str,
    config_sha256: str,
    dataset_sha256: str,
    prepared_sha256: str,
    input_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_receipt_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_sha256: str,
    *,
    smoke: bool,
) -> str:
    mode = "smoke" if smoke else "train"
    return (
        f"APPROVE_MODAL_JAX_RUN:{run_id}:{config_sha256}:{dataset_sha256}:"
        f"{prepared_sha256}:{input_manifest_sha256}:{checkpoint_manifest_sha256}:"
        f"{checkpoint_receipt_sha256}:{tokenizer_manifest_sha256}:"
        f"{gcp_rejection_sha256}:{mode}"
    )


def _validate_request(
    request: dict[str, object],
) -> tuple[str, str, str, str, str, str, str, str, str, bool]:
    run_id = request.get("run_id")
    config_sha = request.get("config_sha256")
    dataset_sha = request.get("dataset_manifest_sha256")
    prepared_sha = request.get("prepared_train_sha256")
    input_manifest_sha = request.get("input_manifest_sha256")
    checkpoint_manifest_sha = request.get("base_checkpoint_manifest_sha256")
    checkpoint_receipt_sha = request.get("base_checkpoint_receipt_sha256")
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
        ("base_checkpoint_receipt_sha256", checkpoint_receipt_sha),
        ("tokenizer_manifest_sha256", tokenizer_manifest_sha),
        ("gcp_rejection_sha256", rejection_sha),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if type(smoke) is not bool:
        raise ValueError("smoke must be a boolean")
    if not isinstance(rejection, dict):
        raise ValueError("Modal fallback requires immutable GCP rejection evidence")
    expected_bindings = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_train_sha256": prepared_sha,
        "input_manifest_sha256": input_manifest_sha,
        "base_checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "base_checkpoint_receipt_sha256": checkpoint_receipt_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
    }
    if rejection.get("producer") == "bookforge-gcp-jax-submitter":
        rejection_valid = (
            rejection.get("schema_version") == "1.0"
            and rejection.get("run_id") == run_id
            and rejection.get("status") == "rejected-pre-billable"
            and rejection.get("submission_intent_created") is False
            and rejection.get("custom_job_created") is False
            and rejection.get("job_absence_verified") is True
            and rejection.get("run_id_absence_basis")
            in {"vertex-list", "billing-disabled-plus-empty-create-audit-log"}
            and rejection.get("fallback_allowed") is True
            and isinstance(rejection.get("spec_sha256"), str)
            and _SHA256.fullmatch(str(rejection["spec_sha256"])) is not None
            and isinstance(rejection.get("reason"), str)
            and bool(str(rejection["reason"]).strip())
        )
    else:
        rejection_valid = _validate_gcp_unavailability_rejection(
            rejection,
            run_id=run_id,
            expected_bindings=expected_bindings,
            smoke=smoke,
        )
    if not rejection_valid:
        raise ValueError("GCP rejection evidence is not pre-billable or run-bound")
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
        checkpoint_receipt_sha,
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
        checkpoint_receipt_sha,
        tokenizer_manifest_sha,
        rejection_sha,
        smoke,
    )


input_volume = modal.Volume.from_name("bookforge-jax-fidelity-inputs", create_if_missing=False)
scratch_volume = modal.Volume.from_name("bookforge-jax-fidelity-scratch", create_if_missing=False)
release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
app = modal.App(APP_NAME)


def _validate_gpu_preflight(path: Path) -> dict[str, object]:
    payload = _json_object(path)
    mesh = payload.get("mesh_shape")
    if (
        payload.get("hardware") != "gpu"
        or payload.get("devices") != 2
        or payload.get("platform") != "gpu"
        or payload.get("memory_fraction") != "0.95"
        or payload.get("ici_fsdp_parallelism") != -1
        or not isinstance(mesh, dict)
        or mesh.get("fsdp") != 2
    ):
        raise RuntimeError("durable GPU preflight evidence changed")
    return payload


def _verify_v2_prepared_evidence(
    input_directory: Path,
    *,
    experiment: object,
    input_manifest: dict[str, object],
    config_sha256: str,
    prepared_sha256: str,
    tokenizer_manifest_sha256: str,
) -> dict[str, object] | None:
    """Delegate v2 verification to the provider-independent semantic gate."""

    from training.jax_fidelity.prepared_staging import (
        verify_v2_prepared_training_input,
    )

    return verify_v2_prepared_training_input(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha256,
        prepared_sha256=prepared_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
    )


def _verify_v3_recovery_evidence(
    input_directory: Path,
    *,
    experiment: object,
    input_manifest: dict[str, object],
    config_sha256: str,
    prepared_sha256: str,
    tokenizer_manifest_sha256: str,
) -> dict[str, object] | None:
    """Dispatch the separate fail-closed v3 recovery verifier."""

    from training.jax_fidelity.recovery_staging import verify_recovery_training_input

    return verify_recovery_training_input(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha256,
        prepared_sha256=prepared_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
    )


def _verify_training_completion_acceptance(
    training_completion: dict[str, object],
    *,
    experiment_id: str,
    smoke: bool,
    expected_steps: int,
    expected_rank: int,
    expected_lora_pair_count: int,
    approved_maxtext_patch_sha256: str,
    v3_acceptance: Mapping[str, object] | None = None,
    output_directory: Path | None = None,
) -> dict[str, object] | None:
    """Re-run the v3 learning gate before durable publication."""

    from training.jax_fidelity.learning_evidence import (
        verify_tensorboard_learning,
        verify_v3_terminal_acceptance,
    )
    from training.jax_fidelity.orbax_receipt import (
        discover_orbax_items,
        lora_checkpoint_evidence,
    )

    evidence = training_completion.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("training completion has no evidence object")
    learning = evidence.get("learning")
    adapter = evidence.get("terminal_adapter")
    if not isinstance(learning, dict) or not isinstance(adapter, dict):
        if experiment_id.endswith("-v3-canary"):
            raise RuntimeError("v3 training completion lacks terminal learning evidence")
        return None
    if experiment_id.endswith("-v3-canary") and output_directory is not None:
        if v3_acceptance is None:
            raise RuntimeError("v3 publication requires immutable learning thresholds")
        actual_learning = verify_tensorboard_learning(
            output_directory,
            expected_steps=expected_steps,
            v3_acceptance=v3_acceptance,
            require_full_v3=not smoke,
        )
        terminal_adapter = discover_orbax_items(
            output_directory,
            expected_step=expected_steps - 1,
        )
        actual_adapter = lora_checkpoint_evidence(
            terminal_adapter,
            expected_rank=expected_rank,
            expected_pair_count=expected_lora_pair_count,
            expected_step=expected_steps - 1,
            approved_maxtext_patch_sha256=approved_maxtext_patch_sha256,
        )
        if learning != actual_learning or adapter != actual_adapter:
            raise RuntimeError("v3 terminal evidence differs from durable training bytes")
        learning = actual_learning
        adapter = actual_adapter
    accepted = verify_v3_terminal_acceptance(
        experiment_id=experiment_id,
        smoke=smoke,
        expected_steps=expected_steps,
        expected_rank=expected_rank,
        expected_lora_pair_count=expected_lora_pair_count,
        approved_maxtext_patch_sha256=approved_maxtext_patch_sha256,
        learning_evidence=learning,
        adapter_evidence=adapter,
    )
    if accepted is not None and evidence.get("learnability_acceptance") != accepted:
        raise RuntimeError("v3 terminal acceptance receipt changed before publication")
    return accepted


def _finalize_completed_scratch(
    *,
    run_id: str,
    training_run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_receipt_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_sha256: str,
    smoke: bool,
    scratch_directory: Path,
    release_directory: Path,
    scratch_commit: Callable[[], object],
    release_commit: Callable[[], object],
    experiment_id: str = "",
    expected_steps: int = 0,
    expected_rank: int = 0,
    expected_lora_pair_count: int = 0,
    approved_maxtext_patch_sha256: str = "",
    v3_acceptance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish one completed durable training scratch tree without retraining."""

    expected_attempt = _attempt_document(
        run_id=run_id,
        training_run_id=training_run_id,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        prepared_train_sha256=prepared_train_sha256,
        input_manifest_sha256=input_manifest_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        checkpoint_receipt_sha256=checkpoint_receipt_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        gcp_rejection_sha256=gcp_rejection_sha256,
        smoke=smoke,
    )
    attempt_path = scratch_directory / "attempt.json"
    gpu_preflight_path = scratch_directory / "gpu-preflight.json"
    if _json_object(attempt_path) != expected_attempt:
        raise RuntimeError("durable training attempt evidence changed")
    gpu_preflight = _validate_gpu_preflight(gpu_preflight_path)
    output_directory = scratch_directory / "output"
    run_directory = scratch_directory / "runs"
    training_completion_path = run_directory / training_run_id / "completion.json"
    training_completion = _json_object(training_completion_path)
    if (
        training_completion.get("status") != "succeeded"
        or training_completion.get("run_id") != training_run_id
        or not training_completion.get("artifacts")
        or not training_completion.get("evidence")
    ):
        raise RuntimeError("durable scratch has no successful training completion")
    _verify_training_completion_acceptance(
        training_completion,
        experiment_id=experiment_id,
        smoke=smoke,
        expected_steps=expected_steps,
        expected_rank=expected_rank,
        expected_lora_pair_count=expected_lora_pair_count,
        approved_maxtext_patch_sha256=approved_maxtext_patch_sha256,
        v3_acceptance=v3_acceptance,
        output_directory=output_directory,
    )

    from training.jax_fidelity.remote_release import package_training_release

    staging = scratch_directory / "finalized-release"
    staging_completion = staging / "completion.json"
    if staging.exists() or staging.is_symlink():
        if not staging.is_dir() or staging.is_symlink():
            raise RuntimeError("durable finalization staging is unsafe")
        if not staging_completion.exists() and not staging_completion.is_symlink():
            # Only derivative, completion-less staging is disposable. The paid
            # training output and its terminal evidence live outside this subtree.
            shutil.rmtree(staging)
            scratch_commit()
        elif not staging_completion.is_file() or staging_completion.is_symlink():
            raise RuntimeError("durable finalization completion is unsafe")
    if staging_completion.is_file():
        payload = _json_object(staging_completion)
        expected_identity = {
            "schema_version": "1.0",
            "run_id": run_id,
            "training_run_id": training_run_id,
            "status": "succeeded",
            "backend": BACKEND,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "base_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "base_checkpoint_receipt_sha256": checkpoint_receipt_sha256,
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "gcp_rejection_sha256": gcp_rejection_sha256,
            "source_training_completion_sha256": _sha256(training_completion_path),
            "attempt_sha256": _sha256(attempt_path),
            "gpu_preflight_sha256": _sha256(gpu_preflight_path),
        }
        if any(payload.get(name) != value for name, value in expected_identity.items()):
            raise RuntimeError("durable finalization staging identity changed")
        if payload.get("gpu_preflight") != gpu_preflight:
            raise RuntimeError("durable finalization GPU evidence changed")
        _safe_release_rows(staging, payload)
    else:
        package_evidence = package_training_release(
            output_directory=output_directory,
            run_directory=run_directory,
            training_run_id=training_run_id,
            runtime_lock="/opt/bookforge/runtime.lock.json",
            destination=staging,
        )
        provider_evidence = staging / "provider"
        provider_evidence.mkdir(mode=0o700)
        release_attempt = provider_evidence / "attempt.json"
        release_gpu_preflight = provider_evidence / "gpu-preflight.json"
        shutil.copyfile(attempt_path, release_attempt, follow_symlinks=False)
        shutil.copyfile(gpu_preflight_path, release_gpu_preflight, follow_symlinks=False)
        release_attempt.chmod(0o400)
        release_gpu_preflight.chmod(0o400)
        files: list[dict[str, object]] = []
        for source in sorted(item for item in staging.rglob("*") if item.is_file()):
            if source.is_symlink():
                raise RuntimeError("release artifacts may not be symbolic links")
            relative = source.relative_to(staging)
            files.append(
                {
                    "path": relative.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "training_run_id": training_run_id,
            "status": "succeeded",
            "backend": BACKEND,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "base_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "base_checkpoint_receipt_sha256": checkpoint_receipt_sha256,
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "gcp_rejection_sha256": gcp_rejection_sha256,
            "source_training_completion_sha256": _sha256(training_completion_path),
            "portable_package": package_evidence,
            "attempt_sha256": _sha256(release_attempt),
            "gpu_preflight_sha256": _sha256(release_gpu_preflight),
            "gpu_preflight": gpu_preflight,
            "files": files,
        }
        _write_once_json(staging_completion, payload)
        scratch_commit()
    _publish_staged_release(staging, release_directory, commit=release_commit)
    return payload


@app.function(
    image=JAX_IMAGE,
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
)
def run_finite(request: dict[str, object]) -> dict[str, object]:
    """Run once; a reused or partially completed run ID fails closed."""

    started = time.monotonic()
    (
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        checkpoint_manifest_sha,
        checkpoint_receipt_sha,
        tokenizer_manifest_sha,
        rejection_sha,
        smoke,
    ) = _validate_request(request)
    input_volume.reload()
    scratch_volume.reload()
    release_volume.reload()
    input_directory = _INPUT_ROOT / run_id
    scratch_directory = _SCRATCH_ROOT / run_id
    release_directory = _RELEASE_ROOT / run_id
    if scratch_directory.exists() or release_directory.exists():
        raise RuntimeError("run ID already has scratch or release state and cannot be retried")
    config = input_directory / "config.json"
    manifest = input_directory / "dataset" / "manifest.json"
    prepared = input_directory / "prepared" / "train.jsonl"
    tokenizer_checkpoint = input_directory / "tokenizer"
    tokenizer_manifest_path = input_directory / "tokenizer.manifest.json"
    from infra.gcp.jax.vertex_entrypoint import verify_base_orbax, verify_input_population

    input_manifest = verify_input_population(
        input_directory,
        run_id=run_id,
        expected_manifest_sha256=input_manifest_sha,
    )
    if _sha256(config) != config_sha or _sha256(manifest) != dataset_sha:
        raise RuntimeError("Modal input contract hash changed")
    if _sha256(prepared) != prepared_sha:
        raise RuntimeError("prepared training data hash changed")
    if _sha256(tokenizer_manifest_path) != tokenizer_manifest_sha:
        raise RuntimeError("tokenizer manifest hash changed")

    scratch_directory.mkdir(parents=True, exist_ok=False)
    output_directory = scratch_directory / "output"
    run_directory = scratch_directory / "runs"
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import validate_dataset_manifest, verify_artifact_manifest
    from training.jax_fidelity.manifests import stable_run_id
    from training.jax_fidelity.modal_gpu_preflight import run_two_gpu_fsdp_preflight
    from training.jax_fidelity.runtime import approval_token

    experiment = load_config(config)
    validated = validate_dataset_manifest(
        manifest,
        expected_manifest_sha256=dataset_sha,
        required_split_records=experiment.dataset["required_split_records"],
    )
    checkpoint = verify_base_orbax(
        input_directory,
        input_manifest=input_manifest,
        expected_manifest_sha256=checkpoint_manifest_sha,
        expected_receipt_sha256=checkpoint_receipt_sha,
    )
    verify_artifact_manifest(tokenizer_checkpoint, _json_object(tokenizer_manifest_path))
    _verify_v2_prepared_evidence(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha,
        prepared_sha256=prepared_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
    )
    _verify_v3_recovery_evidence(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha,
        prepared_sha256=prepared_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
    )
    stage = "lora-smoke" if smoke else "lora-train"
    training_run_id = stable_run_id(
        stage=stage,
        config_sha256=config_sha,
        dataset_manifest_sha256=validated.manifest_sha256,
    )
    attempt = _attempt_document(
        run_id=run_id,
        training_run_id=training_run_id,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        prepared_train_sha256=prepared_sha,
        input_manifest_sha256=input_manifest_sha,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        checkpoint_receipt_sha256=checkpoint_receipt_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        gcp_rejection_sha256=rejection_sha,
        smoke=smoke,
    )
    attempt_path = scratch_directory / "attempt.json"
    _write_once_json(attempt_path, attempt)
    scratch_volume.commit()
    environment = offline_environment(os.environ.copy())
    gpu_preflight = run_two_gpu_fsdp_preflight(
        config_path=config,
        maxtext_root=Path("/opt/MaxText"),
        environment=environment,
    )
    gpu_preflight_path = scratch_directory / "gpu-preflight.json"
    _write_once_json(gpu_preflight_path, gpu_preflight)
    scratch_volume.commit()
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
    training_timeout = _bounded_training_timeout(time.monotonic() - started)
    subprocess.run(command, check=True, env=environment, timeout=training_timeout)

    training_completion_path = run_directory / training_run_id / "completion.json"
    training_completion = _json_object(training_completion_path)
    if (
        training_completion.get("status") != "succeeded"
        or training_completion.get("run_id") != training_run_id
        or not training_completion.get("artifacts")
        or not training_completion.get("evidence")
    ):
        raise RuntimeError("training has no nonempty successful terminal evidence")
    # This is the durability boundary for paid compute. From this point onward a
    # CPU-only finalizer can publish the exact successful checkpoint without retraining.
    scratch_volume.commit()
    if TIMEOUT_SECONDS - (time.monotonic() - started) < MIN_INLINE_FINALIZATION_SECONDS:
        raise RuntimeError(
            "training is durable but the inline publication window is exhausted; "
            "run finalize-only recovery"
        )
    return _finalize_completed_scratch(
        run_id=run_id,
        training_run_id=training_run_id,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        prepared_train_sha256=prepared_sha,
        input_manifest_sha256=input_manifest_sha,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        checkpoint_receipt_sha256=checkpoint_receipt_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        gcp_rejection_sha256=rejection_sha,
        smoke=smoke,
        scratch_directory=scratch_directory,
        release_directory=release_directory,
        scratch_commit=scratch_volume.commit,
        release_commit=release_volume.commit,
        experiment_id=experiment.experiment_id,
        expected_steps=(
            experiment.training["smoke_steps"]
            if smoke
            else experiment.training["steps"]
        ),
        expected_rank=experiment.training["rank"],
        expected_lora_pair_count=experiment.training.get(
            "expected_lora_pair_count", 1
        ),
        approved_maxtext_patch_sha256=experiment.training.get(
            "approved_maxtext_patch_sha256", ""
        ),
        v3_acceptance=(
            experiment.recovery["learnability_acceptance"]
            if experiment.experiment_id.endswith("-v3-canary")
            else None
        ),
    )


@app.function(
    image=JAX_IMAGE,
    cpu=4,
    memory=16_384,
    timeout=FINALIZE_TIMEOUT_SECONDS,
    retries=0,
    max_containers=MAX_CONTAINERS,
    volumes={
        str(_INPUT_ROOT): input_volume,
        str(_SCRATCH_ROOT): scratch_volume,
        str(_RELEASE_ROOT): release_volume,
    },
)
def finalize_finite(request: dict[str, object]) -> dict[str, object]:
    """Publish a completed durable scratch tree; this function cannot train."""

    (
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        checkpoint_manifest_sha,
        checkpoint_receipt_sha,
        tokenizer_manifest_sha,
        rejection_sha,
        smoke,
    ) = _validate_request(request)
    expected_recovery_approval = _finalize_approval_token(
        run_id, input_manifest_sha, rejection_sha
    )
    if request.get("finalize_approval_token") != expected_recovery_approval:
        raise ValueError("exact Modal JAX finalize-only approval token is required")
    input_volume.reload()
    scratch_volume.reload()
    release_volume.reload()
    input_directory = _INPUT_ROOT / run_id
    scratch_directory = _SCRATCH_ROOT / run_id
    release_directory = _RELEASE_ROOT / run_id
    if not scratch_directory.is_dir() or scratch_directory.is_symlink():
        raise RuntimeError("finalize-only recovery requires durable scratch state")

    from infra.gcp.jax.vertex_entrypoint import verify_base_orbax, verify_input_population
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import validate_dataset_manifest, verify_artifact_manifest
    from training.jax_fidelity.manifests import stable_run_id

    input_manifest = verify_input_population(
        input_directory,
        run_id=run_id,
        expected_manifest_sha256=input_manifest_sha,
    )
    config = input_directory / "config.json"
    dataset_manifest = input_directory / "dataset/manifest.json"
    prepared = input_directory / "prepared/train.jsonl"
    tokenizer_manifest_path = input_directory / "tokenizer.manifest.json"
    tokenizer_checkpoint = input_directory / "tokenizer"
    if _sha256(config) != config_sha or _sha256(dataset_manifest) != dataset_sha:
        raise RuntimeError("finalize-only staged input contract changed")
    if _sha256(prepared) != prepared_sha:
        raise RuntimeError("finalize-only prepared training data changed")
    if _sha256(tokenizer_manifest_path) != tokenizer_manifest_sha:
        raise RuntimeError("finalize-only tokenizer manifest changed")
    experiment = load_config(config)
    validated = validate_dataset_manifest(
        dataset_manifest,
        expected_manifest_sha256=dataset_sha,
        required_split_records=experiment.dataset["required_split_records"],
    )
    verify_base_orbax(
        input_directory,
        input_manifest=input_manifest,
        expected_manifest_sha256=checkpoint_manifest_sha,
        expected_receipt_sha256=checkpoint_receipt_sha,
    )
    verify_artifact_manifest(tokenizer_checkpoint, _json_object(tokenizer_manifest_path))
    _verify_v2_prepared_evidence(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha,
        prepared_sha256=prepared_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
    )
    _verify_v3_recovery_evidence(
        input_directory,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=config_sha,
        prepared_sha256=prepared_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
    )
    stage = "lora-smoke" if smoke else "lora-train"
    training_run_id = stable_run_id(
        stage=stage,
        config_sha256=config_sha,
        dataset_manifest_sha256=validated.manifest_sha256,
    )
    return _finalize_completed_scratch(
        run_id=run_id,
        training_run_id=training_run_id,
        config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        prepared_train_sha256=prepared_sha,
        input_manifest_sha256=input_manifest_sha,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        checkpoint_receipt_sha256=checkpoint_receipt_sha,
        tokenizer_manifest_sha256=tokenizer_manifest_sha,
        gcp_rejection_sha256=rejection_sha,
        smoke=smoke,
        scratch_directory=scratch_directory,
        release_directory=release_directory,
        scratch_commit=scratch_volume.commit,
        release_commit=release_volume.commit,
        experiment_id=experiment.experiment_id,
        expected_steps=(
            experiment.training["smoke_steps"]
            if smoke
            else experiment.training["steps"]
        ),
        expected_rank=experiment.training["rank"],
        expected_lora_pair_count=experiment.training.get(
            "expected_lora_pair_count", 1
        ),
        approved_maxtext_patch_sha256=experiment.training.get(
            "approved_maxtext_patch_sha256", ""
        ),
        v3_acceptance=(
            experiment.recovery["learnability_acceptance"]
            if experiment.experiment_id.endswith("-v3-canary")
            else None
        ),
    )


def _authoritative_workspace_total() -> float:
    completed = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return _parse_modal_billing_total(completed.stdout)


def _parse_modal_billing_total(payload: str) -> float:
    """Accept Modal's current and legacy billing fields without ambiguity."""

    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise RuntimeError("Modal billing report was not a list")
    costs: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Modal billing report entry was not an object")
        current = row.get("cost")
        legacy = row.get("Cost")
        if current is None and legacy is None:
            raise RuntimeError("Modal billing report entry had no cost")
        if current is not None and legacy is not None and str(current) != str(legacy):
            raise RuntimeError("Modal billing report entry had conflicting costs")
        cost = float(current if current is not None else legacy)
        if not math.isfinite(cost) or cost < 0:
            raise RuntimeError("Modal billing report entry had an invalid cost")
        costs.append(cost)
    return math.fsum(costs)


@app.local_entrypoint()
def run_cli(
    run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    base_checkpoint_manifest_sha256: str,
    base_checkpoint_receipt_sha256: str,
    tokenizer_manifest_sha256: str,
    gcp_rejection_evidence: str,
    approval_token_value: str,
    smoke: bool = False,
    finalize_only: bool = False,
    finalize_approval_token_value: str = "",
) -> None:
    plan = _json_object(PLAN_PATH)
    if (
        plan.get("status") != "plan-only"
        or plan.get("gpu") != GPU
        or plan.get("container_count") != MAX_CONTAINERS
        or plan.get("function_calls") != 1
        or plan.get("timeout_seconds") != TIMEOUT_SECONDS
        or plan.get("automatic_retries") != 0
        or plan.get("publication_reserve_seconds") != PUBLICATION_RESERVE_SECONDS
        or plan.get("finalize_recovery_timeout_seconds") != FINALIZE_TIMEOUT_SECONDS
        or plan.get("finalize_recovery_gpu") is not None
        or plan.get("finalize_recovery_function_calls_max") != 1
        or plan.get("finalize_recovery_automatic_retries") != 0
        or plan.get("finalize_recovery_web_endpoint") is not False
    ):
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
        base_checkpoint_receipt_sha256,
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
        "base_checkpoint_receipt_sha256": base_checkpoint_receipt_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "gcp_rejection": rejection,
        "gcp_rejection_sha256": rejection_sha256,
        "smoke": smoke,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    if finalize_only:
        expected_finalize = _finalize_approval_token(
            run_id, input_manifest_sha256, rejection_sha256
        )
        if finalize_approval_token_value != expected_finalize:
            raise RuntimeError("exact finalize-only approval token is required")
        request["finalize_approval_token"] = finalize_approval_token_value
        result = finalize_finite.remote(request)
        completion_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        print(
            json.dumps(
                {
                    "authoritative_workspace_total_usd": workspace_total,
                    "finalize_only": True,
                    "finalize_only_gpu": None,
                    "trusted_completion_sha256": hashlib.sha256(
                        completion_bytes
                    ).hexdigest(),
                    "result": result,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if finalize_approval_token_value:
        raise RuntimeError("finalize approval is only valid with --finalize-only")
    from infra.gcp.jax.modal_reconciliation import (
        append_reconciliation,
        assert_attempt_available,
        reserve_attempt,
    )

    attempt_id = f"jax:{run_id}"
    assert_attempt_available(LEDGER_PATH, attempt_id=attempt_id)
    reserve_attempt(LEDGER_PATH, attempt_id=attempt_id, stage="jax-training")

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
            attempt_id=attempt_id,
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
    completion_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    print(
        json.dumps(
            {
                "authoritative_workspace_total_usd": workspace_total,
                "full_call_ceiling_usd": ceiling,
                "full_call_ceiling_provider_enforced": False,
                "trusted_completion_sha256": hashlib.sha256(completion_bytes).hexdigest(),
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
