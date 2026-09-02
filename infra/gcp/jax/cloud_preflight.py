#!/usr/bin/env python3
"""Collect and evaluate read-only Vertex JAX admission evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from job_plan import MACHINE_TYPE, PROJECT_ID, REGION

REQUIRED_SERVICES = {
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
}
Run = Callable[..., subprocess.CompletedProcess[str]]


def _run_json(runner: Run, *arguments: str) -> object:
    completed = runner(
        ["gcloud", *arguments], check=False, capture_output=True, text=True, timeout=30
    )
    if completed.returncode != 0:
        return {
            "collection_error": {
                "command": ["gcloud", *arguments],
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "collection_error": {
                "command": ["gcloud", *arguments],
                "returncode": completed.returncode,
                "stderr": f"invalid JSON response: {error.msg}",
            }
        }


def collect_snapshots(
    *,
    run_id: str,
    scratch_bucket: str,
    release_bucket: str,
    quota_id: str,
    runner: Run = subprocess.run,
) -> dict[str, object]:
    """Issue only describe/list commands and return their unmodified JSON results."""

    commands = {
        "configuration": ("config", "list", "--format=json"),
        "billing": ("beta", "billing", "projects", "describe", PROJECT_ID, "--format=json"),
        "services": (
            "services",
            "list",
            "--enabled",
            f"--project={PROJECT_ID}",
            "--format=json(config.name)",
        ),
        "jobs": (
            "ai",
            "custom-jobs",
            "list",
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--filter=displayName={run_id}",
            "--format=json(name,displayName,state)",
        ),
        "scratch_bucket": (
            "storage",
            "buckets",
            "describe",
            f"gs://{scratch_bucket}",
            "--format=json",
        ),
        "release_bucket": (
            "storage",
            "buckets",
            "describe",
            f"gs://{release_bucket}",
            "--format=json",
        ),
        "quota": (
            "beta",
            "quotas",
            "info",
            "describe",
            quota_id,
            "--service=aiplatform.googleapis.com",
            f"--project={PROJECT_ID}",
            "--format=json",
        ),
    }
    return {name: _run_json(runner, *command) for name, command in commands.items()}


def _bucket_is_private(document: object, *, lifecycle_required: bool) -> bool:
    if not isinstance(document, dict):
        return False
    eligibility = document.get("quotaIncreaseEligibility")
    if isinstance(eligibility, dict) and eligibility.get("ineligibilityReason"):
        return False
    private = (
        document.get("public_access_prevention") == "enforced"
        and document.get("uniform_bucket_level_access") is True
    )
    lifecycle = document.get("lifecycle_config")
    has_lifecycle = isinstance(lifecycle, dict) and bool(lifecycle.get("rule"))
    return private and (has_lifecycle if lifecycle_required else not has_lifecycle)


def _quota_available(document: object, *, minimum: float) -> bool:
    if not isinstance(document, dict):
        return False
    infos = document.get("dimensionsInfos")
    if not isinstance(infos, list):
        return False
    for info in infos:
        if not isinstance(info, dict):
            continue
        dimensions = info.get("dimensions")
        locations = info.get("applicableLocations")
        details = info.get("details")
        region_matches = (
            isinstance(dimensions, dict) and dimensions.get("region") == REGION
        ) or (isinstance(locations, list) and REGION in locations)
        if not region_matches:
            continue
        if not isinstance(details, dict):
            continue
        value = details.get("value")
        try:
            if float(value) >= minimum:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _credits_verified(
    attestation: dict[str, object], *, run_id: str, billing: object
) -> bool:
    if not isinstance(billing, dict):
        return False
    try:
        verified_at = dt.datetime.fromisoformat(str(attestation["verified_at"]))
    except (KeyError, ValueError):
        return False
    if verified_at.tzinfo is None:
        return False
    age = dt.datetime.now(dt.UTC) - verified_at
    return (
        dt.timedelta(0) <= age <= dt.timedelta(hours=24)
        and attestation.get("project") == PROJECT_ID
        and attestation.get("run_id") == run_id
        and attestation.get("billing_account_name") == billing.get("billingAccountName")
        and attestation.get("status") == "verified-promotional-credit-balance"
    )


def evaluate(
    snapshots: dict[str, object],
    *,
    run_id: str,
    credits_attestation: dict[str, object],
    spec_sha256: str,
    input_bindings_sha256: str,
) -> dict[str, object]:
    """Fail closed unless every free/read-only admission check is evidenced."""

    configuration = snapshots.get("configuration")
    core = configuration.get("core") if isinstance(configuration, dict) else None
    checks = {
        "active_project": isinstance(core, dict) and core.get("project") == PROJECT_ID,
        "active_account": isinstance(core, dict) and bool(core.get("account")),
        "billing_enabled": isinstance(snapshots.get("billing"), dict)
        and snapshots["billing"].get("billingEnabled") is True,
        "billing_account_linked": isinstance(snapshots.get("billing"), dict)
        and bool(snapshots["billing"].get("billingAccountName")),
        "services_enabled": REQUIRED_SERVICES.issubset(
            {
                row.get("config", {}).get("name")
                for row in snapshots.get("services", [])
                if isinstance(row, dict) and isinstance(row.get("config"), dict)
            }
        ),
        "run_id_absent": snapshots.get("jobs") == [],
        "scratch_private_with_expiry": _bucket_is_private(
            snapshots.get("scratch_bucket"), lifecycle_required=True
        ),
        "release_private_without_expiry": _bucket_is_private(
            snapshots.get("release_bucket"), lifecycle_required=False
        ),
        "regional_tpu_quota": _quota_available(snapshots.get("quota"), minimum=1),
        "credits_manually_verified": _credits_verified(
            credits_attestation, run_id=run_id, billing=snapshots.get("billing")
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": "1.0",
        "mode": "read-only-preflight",
        "project": PROJECT_ID,
        "region": REGION,
        "machine_type": MACHINE_TYPE,
        "run_id": run_id,
        "spec_sha256": spec_sha256,
        "input_bindings_sha256": input_bindings_sha256,
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "checks": checks,
        "ready": not failed,
        "failed_checks": failed,
        "remote_mutation": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-bucket", required=True)
    parser.add_argument("--release-bucket", required=True)
    parser.add_argument("--quota-id", required=True)
    parser.add_argument("--credits-attestation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    credits = json.loads(args.credits_attestation.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("run_id") != args.run_id:
        raise ValueError("preflight plan identity changed")
    snapshots = collect_snapshots(
        run_id=args.run_id,
        scratch_bucket=args.scratch_bucket,
        release_bucket=args.release_bucket,
        quota_id=args.quota_id,
    )
    print(
        json.dumps(
            evaluate(
                snapshots,
                run_id=args.run_id,
                credits_attestation=credits,
                spec_sha256=str(plan.get("spec_sha256", "")),
                input_bindings_sha256=str(plan.get("input_bindings_sha256", "")),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
