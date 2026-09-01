#!/usr/bin/env python3
"""Record final job and billing evidence for one Vertex attempt without cloud mutation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_evidence_once(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"canonical reconciliation evidence already exists: {destination}")
    encoded = source.read_bytes()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def reconcile(
    *,
    run_id: str,
    state_directory: Path,
    job_evidence_path: Path,
    billing_evidence_path: Path,
) -> dict[str, object]:
    intent_path = state_directory / f"{run_id}.submission-intent.json"
    output = state_directory / f"{run_id}.billing-reconciliation.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Vertex attempt is already reconciled")
    intent = _json(intent_path)
    if intent.get("run_id") != run_id or intent.get("retry_allowed") is not False:
        raise ValueError("submission intent identity changed")
    job = _json(job_evidence_path)
    found = job.get("custom_job_found")
    if found is True:
        if job.get("displayName") != run_id or job.get("state") not in TERMINAL_STATES:
            raise ValueError("Vertex job evidence is not the requested terminal job")
    elif not (
        found is False
        and job.get("status") == "not-found-after-authoritative-search"
        and job.get("displayName") == run_id
    ):
        raise ValueError("ambiguous Vertex job evidence cannot reconcile an attempt")
    billing = _json(billing_evidence_path)
    try:
        gross = float(billing["gross_cost_usd"])
        credits = float(billing["credits_applied_usd"])
        net = float(billing["net_cost_usd"])
        observed = dt.datetime.fromisoformat(str(billing["observed_at"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("billing evidence is incomplete") from error
    if (
        billing.get("run_id") != run_id
        or billing.get("status") != "verified-final-provider-cost"
        or not billing.get("source")
        or observed.tzinfo is None
        or any(not math.isfinite(value) or value < 0 for value in (gross, credits, net))
        or credits > gross
        or abs(max(0.0, gross - credits) - net) > 0.000001
    ):
        raise ValueError("billing evidence is not final, exact, or internally consistent")
    canonical_job = state_directory / f"{run_id}.terminal-job-evidence.json"
    canonical_billing = state_directory / f"{run_id}.final-billing-evidence.json"
    _copy_evidence_once(job_evidence_path, canonical_job)
    _copy_evidence_once(billing_evidence_path, canonical_billing)
    document: dict[str, object] = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-reconciler",
        "status": "reconciled",
        "run_id": run_id,
        "spec_sha256": intent.get("spec_sha256"),
        "submission_intent_sha256": _sha256(intent_path),
        "retry_allowed": False,
        "job_evidence_path": canonical_job.name,
        "job_evidence_sha256": _sha256(canonical_job),
        "billing_evidence_path": canonical_billing.name,
        "billing_evidence_sha256": _sha256(canonical_billing),
        "gross_cost_usd": gross,
        "credits_applied_usd": credits,
        "net_cost_usd": net,
        "remote_state_retained": True,
        "automatic_remote_deletion": False,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--job-evidence", type=Path, required=True)
    parser.add_argument("--billing-evidence", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            reconcile(
                run_id=args.run_id,
                state_directory=args.state_directory,
                job_evidence_path=args.job_evidence,
                billing_evidence_path=args.billing_evidence,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
