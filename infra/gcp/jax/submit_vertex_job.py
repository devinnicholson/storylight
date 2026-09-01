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


class PreflightRejected(RuntimeError):
    """The job was rejected before any billable resource was requested."""

    def __init__(self, message: str, *, fallback_allowed: bool = True) -> None:
        super().__init__(message)
        self.fallback_allowed = fallback_allowed


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


def _preflight(plan: dict[str, object]) -> None:
    active_project = _gcloud("config", "get-value", "project")
    if active_project != PROJECT_ID:
        raise PreflightRejected(f"active gcloud project must be {PROJECT_ID}")
    billing = json.loads(
        _gcloud("beta", "billing", "projects", "describe", PROJECT_ID, "--format=json")
    )
    if not isinstance(billing, dict) or billing.get("billingEnabled") is not True:
        raise PreflightRejected("project billing is not enabled")
    run_id = str(plan["run_id"])
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
        raise PreflightRejected("promotional-credit verification is missing for this run")
    if os.environ.get("BOOKFORGE_GCP_JAX_APPROVAL") != plan["approval_token"]:
        raise PreflightRejected("exact one-purpose GCP approval token is missing")


def _validated_admission_evidence(path: Path, plan: dict[str, object]) -> dict[str, object]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise PreflightRejected("cloud admission evidence must be an object")
    exact = {
        "mode": "read-only-preflight",
        "project": PROJECT_ID,
        "region": REGION,
        "run_id": plan["run_id"],
        "spec_sha256": plan["spec_sha256"],
        "input_bindings_sha256": plan["input_bindings_sha256"],
        "ready": True,
        "remote_mutation": False,
    }
    if any(evidence.get(key) != value for key, value in exact.items()):
        raise PreflightRejected("cloud admission evidence is stale or belongs to another plan")
    checks = evidence.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise PreflightRejected("cloud admission evidence contains a failed check")
    try:
        checked_at = dt.datetime.fromisoformat(str(evidence["checked_at"]))
    except (KeyError, ValueError) as error:
        raise PreflightRejected("cloud admission evidence has no valid timestamp") from error
    if checked_at.tzinfo is None:
        raise PreflightRejected("cloud admission evidence timestamp has no timezone")
    age = dt.datetime.now(dt.UTC) - checked_at
    if age < dt.timedelta(0) or age > dt.timedelta(minutes=15):
        raise PreflightRejected("cloud admission evidence is older than 15 minutes")
    return evidence


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
            "custom_job_created": False,
            "fallback_allowed": rejection.fallback_allowed,
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
        _validated_admission_evidence(admission_evidence_path, plan)
        _preflight(plan)
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
