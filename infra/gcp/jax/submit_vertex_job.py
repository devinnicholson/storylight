#!/usr/bin/env python3
"""Submit one planned Vertex job after read-only preflight and exact approval."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from job_plan import PROJECT_ID, REGION, approval_token

_BILLING_DISABLED_ABSENCE = "billing-disabled-plus-empty-create-audit-log"


class PreflightRejected(RuntimeError):
    """The job was rejected before any billable resource was requested."""

    def __init__(
        self,
        message: str,
        *,
        fallback_allowed: bool = True,
        job_absence_verified: bool = False,
        run_id_absence_basis: str | None = None,
    ) -> None:
        super().__init__(message)
        self.fallback_allowed = fallback_allowed
        self.job_absence_verified = job_absence_verified
        self.run_id_absence_basis = run_id_absence_basis


def _gcloud(*arguments: str) -> str:
    completed = subprocess.run(
        ["gcloud", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _validated_plan(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plan must contain a JSON object")
    if value.get("mode") != "plan-only":
        raise ValueError("only an unsubmitted plan may be submitted")
    if value.get("project") != PROJECT_ID or value.get("region") != REGION:
        raise ValueError("plan project or region changed")
    body = value.get("custom_job")
    if not isinstance(body, dict):
        raise ValueError("plan has no custom job body")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if value.get("spec_sha256") != digest:
        raise ValueError("custom job spec hash changed")
    run_id = value.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("plan has no run ID")
    if value.get("approval_token") != approval_token(run_id, digest):
        raise ValueError("plan approval token is inconsistent")
    bindings = value.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("plan has no input bindings")
    bindings_digest = hashlib.sha256(
        json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if value.get("input_bindings_sha256") != bindings_digest:
        raise ValueError("plan input binding hash changed")
    job_spec = body.get("jobSpec")
    if not isinstance(job_spec, dict):
        raise ValueError("plan job specification is malformed")
    workers = job_spec.get("workerPoolSpecs")
    if not isinstance(workers, list) or len(workers) != 1 or not isinstance(workers[0], dict):
        raise ValueError("plan worker pool is malformed")
    container = workers[0].get("containerSpec")
    if not isinstance(container, dict):
        raise ValueError("plan worker container is malformed")
    arguments = container.get("args")
    if not isinstance(arguments, list):
        raise ValueError("plan worker arguments are malformed")
    parsed = {
        str(arguments[index]).removeprefix("--").replace("-", "_"): arguments[index + 1]
        for index in range(0, len(arguments) - 1)
        if isinstance(arguments[index], str) and str(arguments[index]).startswith("--")
    }
    if any(parsed.get(name) != expected for name, expected in bindings.items()):
        raise ValueError("plan input bindings differ from worker arguments")
    return value


def _preflight(plan: dict[str, object], admission: dict[str, object]) -> None:
    run_id = str(plan["run_id"])
    active_project = _gcloud("config", "get-value", "project")
    if active_project != PROJECT_ID:
        raise PreflightRejected(
            f"active gcloud project must be {PROJECT_ID}", fallback_allowed=False
        )
    billing = json.loads(
        _gcloud("beta", "billing", "projects", "describe", PROJECT_ID, "--format=json")
    )
    if not isinstance(billing, dict) or billing.get("billingEnabled") is not True:
        checks = admission.get("checks")
        absence_basis = admission.get("run_id_absence_basis")
        absence_verified = (
            isinstance(checks, dict)
            and checks.get("run_id_absent") is True
            and absence_basis == _BILLING_DISABLED_ABSENCE
        )
        raise PreflightRejected(
            "project billing is not enabled",
            fallback_allowed=absence_verified,
            job_absence_verified=absence_verified,
            run_id_absence_basis=absence_basis if absence_verified else None,
        )
    jobs = json.loads(
        _gcloud(
            "ai",
            "custom-jobs",
            "list",
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--filter=displayName={run_id}",
            "--format=json",
        )
    )
    if jobs != []:
        raise PreflightRejected(
            "run ID already exists or the job lookup was not empty",
            fallback_allowed=False,
        )
    credits_verification = os.environ.get("BOOKFORGE_GCP_CREDITS_VERIFIED")
    if credits_verification != f"VERIFIED:{run_id}":
        raise PreflightRejected(
            "promotional-credit verification is missing for this run",
            job_absence_verified=True,
            run_id_absence_basis="vertex-list",
        )
    if os.environ.get("BOOKFORGE_GCP_JAX_APPROVAL") != plan["approval_token"]:
        raise PreflightRejected(
            "exact one-purpose GCP approval token is missing",
            job_absence_verified=True,
            run_id_absence_basis="vertex-list",
        )


def _validated_admission_evidence(path: Path, plan: dict[str, object]) -> dict[str, object]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise PreflightRejected(
            "cloud admission evidence must be an object", fallback_allowed=False
        )
    exact = {
        "mode": "read-only-preflight",
        "project": PROJECT_ID,
        "region": REGION,
        "run_id": plan["run_id"],
        "spec_sha256": plan["spec_sha256"],
        "input_bindings_sha256": plan["input_bindings_sha256"],
        "remote_mutation": False,
    }
    if any(evidence.get(key) != value for key, value in exact.items()):
        raise PreflightRejected(
            "cloud admission evidence is stale or belongs to another plan",
            fallback_allowed=False,
        )
    checks = evidence.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(type(value) is not bool for value in checks.values())
    ):
        raise PreflightRejected(
            "cloud admission evidence has malformed checks", fallback_allowed=False
        )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if (
        evidence.get("ready") is not (not failed_checks)
        or evidence.get("failed_checks") != failed_checks
    ):
        raise PreflightRejected(
            "cloud admission evidence has inconsistent check results",
            fallback_allowed=False,
        )
    try:
        checked_at = dt.datetime.fromisoformat(str(evidence["checked_at"]))
    except (KeyError, ValueError) as error:
        raise PreflightRejected(
            "cloud admission evidence has no valid timestamp", fallback_allowed=False
        ) from error
    if checked_at.tzinfo is None:
        raise PreflightRejected(
            "cloud admission evidence timestamp has no timezone", fallback_allowed=False
        )
    age = dt.datetime.now(dt.UTC) - checked_at
    if age < dt.timedelta(0) or age > dt.timedelta(minutes=15):
        raise PreflightRejected(
            "cloud admission evidence is older than 15 minutes", fallback_allowed=False
        )
    if "run_id_absent" in failed_checks:
        raise PreflightRejected(
            "read-only admission did not prove the run ID absent",
            fallback_allowed=False,
        )
    if "run_id_absent" not in checks:
        raise PreflightRejected(
            "cloud admission evidence omitted the run-ID absence check",
            fallback_allowed=False,
        )
    absence_basis = evidence.get("run_id_absence_basis")
    if checks.get("run_id_absent") is True and absence_basis not in {
        "vertex-list",
        _BILLING_DISABLED_ABSENCE,
    }:
        raise PreflightRejected(
            "cloud admission evidence has no valid run-ID absence basis",
            fallback_allowed=False,
        )
    if absence_basis == _BILLING_DISABLED_ABSENCE and evidence.get(
        "run_id_absence_evidence"
    ) != {
        "audit_filter": (
            'logName="projects/'
            f'{PROJECT_ID}/logs/cloudaudit.googleapis.com%2Factivity" AND '
            'protoPayload.serviceName="aiplatform.googleapis.com" AND '
            'protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob" AND '
            f'protoPayload.request.customJob.displayName="{plan["run_id"]}"'
        ),
        "freshness": "400d",
        "maximum_run_id_age_days": 7,
    }:
        raise PreflightRejected(
            "cloud admission evidence has malformed CustomJob audit evidence",
            fallback_allowed=False,
        )
    return evidence


def _require_ready_admission(evidence: dict[str, object]) -> None:
    if evidence["ready"] is True:
        return
    failed_checks = evidence["failed_checks"]
    if not isinstance(failed_checks, list):
        raise PreflightRejected(
            "cloud admission evidence has malformed failed checks",
            fallback_allowed=False,
            job_absence_verified=False,
        )
    checks = evidence.get("checks")
    absence_verified = isinstance(checks, dict) and checks.get("run_id_absent") is True
    raise PreflightRejected(
        "read-only cloud admission failed: " + ", ".join(failed_checks),
        fallback_allowed=absence_verified,
        job_absence_verified=absence_verified,
        run_id_absence_basis=(
            str(evidence["run_id_absence_basis"]) if absence_verified else None
        ),
    )


def _write_once_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_preflight_rejection(
    state_directory: Path, plan: dict[str, object], rejection: PreflightRejected
) -> Path:
    path = state_directory / f"{plan['run_id']}.preflight-rejection.json"
    _write_once_json(
        path,
        {
            "schema_version": "1.0",
            "producer": "bookforge-gcp-jax-submitter",
            "run_id": plan["run_id"],
            "spec_sha256": plan["spec_sha256"],
            "input_bindings": plan["input_bindings"],
            "input_bindings_sha256": plan["input_bindings_sha256"],
            "status": "rejected-pre-billable",
            "submission_intent_created": False,
            "custom_job_created": False if rejection.job_absence_verified else None,
            "job_absence_verified": rejection.job_absence_verified,
            "run_id_absence_basis": rejection.run_id_absence_basis,
            "fallback_allowed": (
                rejection.fallback_allowed and rejection.job_absence_verified
            ),
            "reason": str(rejection),
        },
    )
    return path


def _create_intent(state_directory: Path, plan: dict[str, object]) -> Path:
    state_directory.mkdir(parents=True, exist_ok=True)
    path = state_directory / f"{plan['run_id']}.submission-intent.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": plan["run_id"],
                "spec_sha256": plan["spec_sha256"],
                "status": "submission-intent-recorded",
                "retry_allowed": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return path


def _assert_no_unreconciled_paid_attempt(state_directory: Path) -> None:
    """Block every new paid request while any prior intent lacks final cost evidence."""

    if not state_directory.exists():
        return
    for intent in sorted(state_directory.glob("*.submission-intent.json")):
        run_id = intent.name.removesuffix(".submission-intent.json")
        reconciliation = state_directory / f"{run_id}.billing-reconciliation.json"
        if not reconciliation.is_file() or reconciliation.is_symlink():
            raise RuntimeError(
                f"unreconciled paid Vertex attempt blocks new submissions: {run_id}"
            )
        try:
            document = json.loads(reconciliation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            message = f"invalid Vertex reconciliation blocks spending: {run_id}"
            raise RuntimeError(message) from error
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != "1.0"
            or document.get("producer") != "bookforge-gcp-jax-reconciler"
            or document.get("status") != "reconciled"
            or document.get("run_id") != run_id
            or document.get("retry_allowed") is not False
            or any(
                not isinstance(document.get(name), str)
                or len(str(document[name])) != 64
                or any(character not in "0123456789abcdef" for character in str(document[name]))
                for name in ("job_evidence_sha256", "billing_evidence_sha256")
            )
        ):
            raise RuntimeError(f"invalid Vertex reconciliation blocks spending: {run_id}")
        intent_sha256 = hashlib.sha256(intent.read_bytes()).hexdigest()
        if document.get("submission_intent_sha256") != intent_sha256:
            raise RuntimeError(f"stale Vertex reconciliation blocks spending: {run_id}")
        expected_evidence = {
            "job": state_directory / f"{run_id}.terminal-job-evidence.json",
            "billing": state_directory / f"{run_id}.final-billing-evidence.json",
        }
        for kind, evidence_path in expected_evidence.items():
            if (
                document.get(f"{kind}_evidence_path") != evidence_path.name
                or not evidence_path.is_file()
                or evidence_path.is_symlink()
                or document.get(f"{kind}_evidence_sha256")
                != hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            ):
                raise RuntimeError(
                    f"missing or changed Vertex {kind} evidence blocks spending: {run_id}"
                )


def submit(
    plan_path: Path, state_directory: Path, admission_evidence_path: Path
) -> dict[str, object]:
    plan = _validated_plan(plan_path)
    _assert_no_unreconciled_paid_attempt(state_directory)
    try:
        admission = _validated_admission_evidence(admission_evidence_path, plan)
        _preflight(plan, admission)
        _require_ready_admission(admission)
    except PreflightRejected as error:
        _record_preflight_rejection(state_directory, plan, error)
        raise
    intent = _create_intent(state_directory, plan)
    token = _gcloud("auth", "print-access-token")
    request = urllib.request.Request(
        str(plan["create_url"]),
        data=json.dumps(plan["custom_job"], separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.loads(response.read())
    receipt = intent.with_name(intent.name.replace("submission-intent", "submission-receipt"))
    _write_once_json(receipt, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--admission-evidence", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = _validated_plan(args.plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(
        json.dumps(
            submit(args.plan, args.state_directory, args.admission_evidence),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
