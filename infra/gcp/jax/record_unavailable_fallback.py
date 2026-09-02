#!/usr/bin/env python3
"""Record a read-only GCP-unavailable decision before falling back to Modal.

This path is intentionally separate from Vertex job submission.  It is used when
the active Bookforge project has billing disabled, so the digest-pinned training
image cannot be built and no complete CustomJob specification exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from job_plan import (
    MACHINE_TYPE,
    PROJECT_ID,
    REGION,
    REPLICAS,
    TIMEOUT_SECONDS,
    TPU_CHIPS,
)

PRODUCER = "bookforge-gcp-jax-unavailability-recorder"
RUN_ID_ABSENCE_BASIS = "billing-disabled-plus-empty-create-audit-log-400d"
AUDIT_FRESHNESS = "400d"
MAXIMUM_RUN_ID_AGE_DAYS = 7
CREATE_METHOD = "google.cloud.aiplatform.v1.JobService.CreateCustomJob"
IMAGE_REPOSITORY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/bookforge-jax/trainer"
_RUN_ID = re.compile(r"^[a-z][a-z0-9-]{7,62}$")
_RUN_DATE = re.compile(r"(?:^|-)(20\d{6})(?:-|$)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TAGGED_IMAGE = re.compile(
    rf"^{re.escape(IMAGE_REPOSITORY)}:([0-9a-f]{{20}})$"
)
_DIGEST_IMAGE = re.compile(r"^[a-z0-9][a-z0-9./_:-]*@sha256:[0-9a-f]{64}$")
_BINDING_NAMES = {
    "config_sha256",
    "dataset_manifest_sha256",
    "prepared_train_sha256",
    "input_manifest_sha256",
    "base_checkpoint_manifest_sha256",
    "base_checkpoint_receipt_sha256",
    "tokenizer_manifest_sha256",
}
Run = Callable[..., subprocess.CompletedProcess[str]]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id_is_recent(run_id: str, *, now: dt.datetime) -> bool:
    match = _RUN_DATE.search(run_id)
    if match is None:
        return False
    try:
        stamped = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return False
    age = now.date() - stamped
    return dt.timedelta(0) <= age <= dt.timedelta(days=MAXIMUM_RUN_ID_AGE_DAYS)


def _audit_filter(run_id: str) -> str:
    return " AND ".join(
        (
            f'logName="projects/{PROJECT_ID}/logs/'
            'cloudaudit.googleapis.com%2Factivity"',
            'protoPayload.serviceName="aiplatform.googleapis.com"',
            f'protoPayload.methodName="{CREATE_METHOD}"',
            f'protoPayload.request.customJob.displayName="{run_id}"',
        )
    )


def _gcloud_json(runner: Run, *arguments: str) -> object:
    completed = runner(
        ["gcloud", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"read-only gcloud query failed ({' '.join(arguments)}): {detail}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("read-only gcloud query returned invalid JSON") from error


def _validated_bindings(values: dict[str, str]) -> dict[str, str]:
    if set(values) != _BINDING_NAMES:
        raise ValueError("exactly seven named input bindings are required")
    for name, value in values.items():
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    return values


def _validated_image_plan(
    path: Path, *, state_directory: Path
) -> tuple[dict[str, object], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("image build plan must be a regular file")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("image build plan must contain a JSON object")
    source_sha = plan.get("source_sha256")
    tagged_uri = plan.get("tagged_image_uri")
    match = _TAGGED_IMAGE.fullmatch(str(tagged_uri))
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("mode") != "plan-only"
        or plan.get("project") != PROJECT_ID
        or plan.get("region") != REGION
        or plan.get("automatic_retries") != 0
        or plan.get("remote_mutation") is not False
        or not isinstance(source_sha, str)
        or _SHA256.fullmatch(source_sha) is None
        or match is None
        or match.group(1) != source_sha[:20]
    ):
        raise ValueError("image build plan identity or policy is invalid")
    if "@sha256:" in str(tagged_uri):
        raise ValueError("unbuilt image target must not masquerade as a digest URI")
    provider_command = plan.get("provider_build_command")
    source_files = plan.get("source_files")
    builder_image = plan.get("builder_image")
    if (
        not isinstance(provider_command, list)
        or provider_command[:3] != ["gcloud", "builds", "submit"]
        or not all(isinstance(item, str) for item in provider_command)
        or not isinstance(source_files, list)
        or not source_files
        or _canonical_sha256(source_files) != source_sha
        or not isinstance(builder_image, str)
        or _DIGEST_IMAGE.fullmatch(builder_image) is None
    ):
        raise ValueError("image build plan has no bounded provider command")
    command_sha = hashlib.sha256(
        json.dumps(provider_command, separators=(",", ":")).encode()
    ).hexdigest()
    expected_approval = (
        f"APPROVE_GCP_JAX_IMAGE_BUILD:{source_sha}:"
        f"{hashlib.sha256(builder_image.encode()).hexdigest()}:{command_sha}"
    )
    if plan.get("approval_token") != expected_approval:
        raise ValueError("image build plan approval binding is invalid")
    if state_directory.is_symlink() or state_directory.exists() and (
        not state_directory.is_dir() or state_directory.is_symlink()
    ):
        raise ValueError("image build state path must be a regular directory")
    for suffix in ("submission-intent.json", "submission-receipt.json"):
        build_state = state_directory / f"image-{source_sha}.{suffix}"
        if build_state.exists() or build_state.is_symlink():
            raise RuntimeError("image build state exists; fallback would be ambiguous")
    return plan, _sha256(path)


def _resource_intent(
    *,
    run_id: str,
    service_account: str,
    scratch_uri: str,
    release_uri: str,
    input_bindings: dict[str, str],
    image_plan: dict[str, object],
    image_plan_sha256: str,
    smoke: bool,
) -> dict[str, object]:
    input_prefix = f"{scratch_uri.rstrip('/')}/inputs/{run_id}"
    release_prefix = f"{release_uri.rstrip('/')}/releases/{run_id}"
    return {
        "backend": "vertex-custom-job",
        "project": PROJECT_ID,
        "region": REGION,
        "display_name": run_id,
        "create_method": CREATE_METHOD,
        "create_url": (
            f"https://{REGION}-aiplatform.googleapis.com/v1/projects/"
            f"{PROJECT_ID}/locations/{REGION}/customJobs"
        ),
        "machine_type": MACHINE_TYPE,
        "tpu_chips": TPU_CHIPS,
        "replicas": REPLICAS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "automatic_retries": 0,
        "endpoint_created": False,
        "service_account": service_account,
        "input_prefix": input_prefix,
        "release_prefix": release_prefix,
        "input_bindings": input_bindings,
        "smoke": smoke,
        "container": {
            "image_source_sha256": image_plan["source_sha256"],
            "image_build_plan_sha256": image_plan_sha256,
            "intended_tagged_uri": image_plan["tagged_image_uri"],
            "runnable_digest_uri": None,
            "digest_resolved": False,
            "build_attempted": False,
        },
    }


def collect_rejection(
    *,
    run_id: str,
    input_bindings: dict[str, str],
    service_account: str,
    scratch_uri: str,
    release_uri: str,
    image_plan_path: Path,
    image_build_state_directory: Path,
    smoke: bool,
    runner: Run = subprocess.run,
    now: dt.datetime | None = None,
) -> dict[str, object]:
    """Collect read-only evidence and return one Modal fallback authorization."""

    checked_at = now or dt.datetime.now(dt.UTC)
    if checked_at.tzinfo is None:
        raise ValueError("evidence time must be timezone-aware")
    checked_at = checked_at.astimezone(dt.UTC)
    if _RUN_ID.fullmatch(run_id) is None or not _run_id_is_recent(
        run_id, now=checked_at
    ):
        raise ValueError("run ID must contain a valid date from the last seven days")
    bindings = _validated_bindings(dict(input_bindings))
    expected_account_suffix = f"@{PROJECT_ID}.iam.gserviceaccount.com"
    if not service_account.endswith(expected_account_suffix):
        raise ValueError("service account must belong to the Bookforge project")
    if not scratch_uri.startswith("gs://") or not release_uri.startswith("gs://"):
        raise ValueError("scratch and release locations must be gs:// URIs")
    if (
        scratch_uri.removeprefix("gs://").split("/", 1)[0]
        == release_uri.removeprefix("gs://").split("/", 1)[0]
    ):
        raise ValueError("scratch and release buckets must be separate")
    image_plan, image_plan_sha = _validated_image_plan(
        image_plan_path, state_directory=image_build_state_directory
    )
    configuration = _gcloud_json(runner, "config", "list", "--format=json")
    core = configuration.get("core") if isinstance(configuration, dict) else None
    if not isinstance(core, dict) or core.get("project") != PROJECT_ID:
        raise RuntimeError("active gcloud project is not the Bookforge project")
    account = core.get("account")
    if not isinstance(account, str) or not account.strip():
        raise RuntimeError("active gcloud account is unavailable")
    billing = _gcloud_json(
        runner,
        "beta",
        "billing",
        "projects",
        "describe",
        PROJECT_ID,
        "--format=json",
    )
    if not isinstance(billing, dict) or billing.get("billingEnabled") is not False:
        raise RuntimeError("fallback path requires billing to be explicitly disabled")
    audit_filter = _audit_filter(run_id)
    audit_rows = _gcloud_json(
        runner,
        "logging",
        "read",
        audit_filter,
        f"--project={PROJECT_ID}",
        f"--freshness={AUDIT_FRESHNESS}",
        "--limit=1",
        "--format=json",
    )
    if audit_rows != []:
        raise RuntimeError("CreateCustomJob audit absence was not established")
    resource = _resource_intent(
        run_id=run_id,
        service_account=service_account,
        scratch_uri=scratch_uri,
        release_uri=release_uri,
        input_bindings=bindings,
        image_plan=image_plan,
        image_plan_sha256=image_plan_sha,
        smoke=smoke,
    )
    return {
        "schema_version": "1.0",
        "producer": PRODUCER,
        "status": "rejected-pre-billable",
        "rejection_kind": "gcp-unavailable-before-image-build",
        "project": PROJECT_ID,
        "region": REGION,
        "run_id": run_id,
        "checked_at": checked_at.isoformat(),
        "submission_intent_created": False,
        "custom_job_created": False,
        "job_absence_verified": True,
        "run_id_absence_basis": RUN_ID_ABSENCE_BASIS,
        "fallback_allowed": True,
        "reason": (
            "Billing is disabled for the active project; no image build was "
            "attempted and no digest-pinned Vertex training image exists for this plan."
        ),
        "billing_verification": {
            "active_project": PROJECT_ID,
            "active_account": account,
            "billing_enabled": False,
            "billing_account_name": billing.get("billingAccountName", ""),
        },
        "audit_absence": {
            "log": "cloudaudit.googleapis.com/activity",
            "service_name": "aiplatform.googleapis.com",
            "method_name": CREATE_METHOD,
            "display_name": run_id,
            "filter": audit_filter,
            "freshness": AUDIT_FRESHNESS,
            "maximum_run_id_age_days": MAXIMUM_RUN_ID_AGE_DAYS,
            "matching_entries": [],
        },
        "container_image": {
            "image_source_sha256": image_plan["source_sha256"],
            "image_build_plan_sha256": image_plan_sha,
            "intended_tagged_uri": image_plan["tagged_image_uri"],
            "runnable_digest_uri": None,
            "digest_resolved": False,
            "build_attempted": False,
            "build_intent_created": False,
            "build_receipt_created": False,
            "build_admission": "blocked-billing-disabled",
        },
        "intended_vertex_resource": resource,
        "intended_vertex_resource_sha256": _canonical_sha256(resource),
        "input_bindings": bindings,
        "input_bindings_sha256": _canonical_sha256(bindings),
        "queries_read_only": True,
        "remote_mutation": False,
    }


def _write_once(path: Path, document: dict[str, object]) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("evidence parent must be an existing regular directory")
    rendered = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--prepared-train-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-receipt-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--scratch-uri", required=True)
    parser.add_argument("--release-uri", required=True)
    parser.add_argument("--image-build-plan", type=Path, required=True)
    parser.add_argument("--image-build-state-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    bindings = {
        "config_sha256": args.config_sha256,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "prepared_train_sha256": args.prepared_train_sha256,
        "input_manifest_sha256": args.input_manifest_sha256,
        "base_checkpoint_manifest_sha256": args.base_checkpoint_manifest_sha256,
        "base_checkpoint_receipt_sha256": args.base_checkpoint_receipt_sha256,
        "tokenizer_manifest_sha256": args.tokenizer_manifest_sha256,
    }
    evidence = collect_rejection(
        run_id=args.run_id,
        input_bindings=bindings,
        service_account=args.service_account,
        scratch_uri=args.scratch_uri,
        release_uri=args.release_uri,
        image_plan_path=args.image_build_plan,
        image_build_state_directory=args.image_build_state_directory,
        smoke=args.smoke,
    )
    _write_once(args.output, evidence)
    print(
        json.dumps(
            {
                "evidence": str(args.output),
                "sha256": _sha256(args.output),
                "status": evidence["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
