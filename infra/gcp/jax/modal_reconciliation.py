"""Append local evidence for finite Modal calls without mutating remote artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_attempt_available(path: Path | str, *, attempt_id: str) -> None:
    """Refuse a paid call already recorded or following unresolved billing evidence."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Modal reconciliation ledger is malformed")
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("Modal reconciliation ledger entry is malformed")
        if row.get("attempt_id") == attempt_id:
            raise ValueError(f"Modal attempt was already started: {attempt_id}")
        if row.get("workspace_after_usd") is None or row.get("postrun_report_error") is not None:
            raise ValueError("an earlier Modal attempt has unresolved billing evidence")


def reserve_attempt(path: Path | str, *, attempt_id: str, stage: str) -> None:
    """Atomically reserve a paid attempt before invoking the provider."""

    if not attempt_id or not stage:
        raise ValueError("attempt_id and stage are required")
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if ledger_path.exists():
            document = json.loads(ledger_path.read_text(encoding="utf-8"))
            entries = document.get("entries") if isinstance(document, dict) else None
            if not isinstance(entries, list):
                raise ValueError("Modal reconciliation ledger is malformed")
        else:
            document = {"schema_version": "1.0", "entries": []}
            entries = document["entries"]
        if entries:
            for row in entries:
                if not isinstance(row, dict):
                    raise ValueError("Modal reconciliation ledger entry is malformed")
                if row.get("attempt_id") == attempt_id:
                    raise ValueError(f"Modal attempt was already started: {attempt_id}")
                if row.get("status") == "reserved" or row.get("workspace_after_usd") is None:
                    raise ValueError("an earlier Modal attempt has unresolved billing evidence")
        entries.append(
            {
                "attempt_id": attempt_id,
                "stage": stage,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
                    "+00:00", "Z"
                ),
                "status": "reserved",
                "remote_state_retained": True,
                "automatic_remote_deletion": False,
            }
        )
        temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, ledger_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_reconciliation(
    path: Path | str,
    *,
    attempt_id: str,
    stage: str,
    workspace_before_usd: float,
    workspace_after_usd: float | None,
    declared_ceiling_usd: float,
    status: str,
    result: dict[str, object] | None,
    postrun_report_error: str | None = None,
) -> dict[str, Any]:
    """Append one unique attempt to a process-safe local reconciliation ledger."""

    if not attempt_id or not stage:
        raise ValueError("attempt_id and stage are required")
    if status not in {"succeeded", "remote-error"}:
        raise ValueError("status must be succeeded or remote-error")
    if not math.isfinite(workspace_before_usd) or workspace_before_usd < 0:
        raise ValueError("pre-run workspace total must be finite and non-negative")
    if workspace_after_usd is not None and (
        not math.isfinite(workspace_after_usd) or workspace_after_usd < 0
    ):
        raise ValueError("post-run workspace total must be finite and non-negative")
    if not math.isfinite(declared_ceiling_usd) or declared_ceiling_usd <= 0:
        raise ValueError("declared ceiling must be finite and positive")
    entry: dict[str, Any] = {
        "attempt_id": attempt_id,
        "stage": stage,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": status,
        "workspace_before_usd": workspace_before_usd,
        "workspace_after_usd": workspace_after_usd,
        "reported_delta_usd": (
            None if workspace_after_usd is None else workspace_after_usd - workspace_before_usd
        ),
        "declared_ceiling_usd": declared_ceiling_usd,
        "declared_ceiling_provider_enforced": False,
        "remote_state_retained": True,
        "automatic_remote_deletion": False,
        "result_sha256": _canonical_sha256(result) if result is not None else None,
        "postrun_report_error": postrun_report_error,
    }
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if ledger_path.exists():
            document = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
                raise ValueError("Modal reconciliation ledger is malformed")
        else:
            document = {"schema_version": "1.0", "entries": []}
        entries = document["entries"]
        matches = [
            index
            for index, row in enumerate(entries)
            if isinstance(row, dict) and row.get("attempt_id") == attempt_id
        ]
        if len(matches) == 1 and entries[matches[0]].get("status") == "reserved":
            entries[matches[0]] = entry
        elif matches:
            raise ValueError(f"Modal attempt is already reconciled: {attempt_id}")
        else:
            entries.append(entry)
        temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, ledger_path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return entry
