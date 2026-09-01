#!/usr/bin/env python3
"""Submit one planned Vertex job after read-only preflight and exact approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from job_plan import PROJECT_ID, REGION, approval_token


class PreflightRejected(RuntimeError):
    """The job was rejected before any billable resource was requested."""


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
        raise PreflightRejected("run ID already exists or the job lookup was not empty")
    credits_verification = os.environ.get("BOOKFORGE_GCP_CREDITS_VERIFIED")
    if credits_verification != f"VERIFIED:{run_id}":
        raise PreflightRejected("promotional-credit verification is missing for this run")
    if os.environ.get("BOOKFORGE_GCP_JAX_APPROVAL") != plan["approval_token"]:
        raise PreflightRejected("exact one-purpose GCP approval token is missing")


def _write_once_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_preflight_rejection(
    state_directory: Path, plan: dict[str, object], reason: str
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
            "reason": reason,
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


def submit(plan_path: Path, state_directory: Path) -> dict[str, object]:
    plan = _validated_plan(plan_path)
    try:
        _preflight(plan)
    except PreflightRejected as error:
        _record_preflight_rejection(state_directory, plan, str(error))
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
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = _validated_plan(args.plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(json.dumps(submit(args.plan, args.state_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
