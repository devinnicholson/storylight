"""Append local evidence for finite Modal calls without mutating remote artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

_PROVISIONAL_BILLING_STATUS = "provisional-provider-app-cost"
_SETTLED_BILLING_STATUS = "settled-provider-app-cost"
_SETTLEMENT_STATUS = "verified-provider-app-cost"
_BILLING_REPORT_ARGV = ("modal", "billing", "report", "--for", "this month", "--json")
_BILLING_REPORT_SCOPE = "this month"
_BILLING_REPORT_TIMEOUT_SECONDS = 60
_APP_ID_ALIASES = ("object_id", "Object ID", "app_id", "App ID")
_DESCRIPTION_ALIASES = ("description", "Description")
_COST_ALIASES = ("cost", "Cost")
_APP_ID_PATTERN = re.compile(r"ap-[A-Za-z0-9]{22}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

# A historical entry without provider-app evidence is unresolved by default. A
# reviewed migration may add only the canonical SHA-256 of an immutable legacy
# entry here; descriptions, dates, stages, or other broad predicates are never
# sufficient to bypass settlement.
# These are the exact 23 historical rows already committed at 279989d before
# provider-row settlement existed. Any field change invalidates the waiver.
_GRANDFATHERED_LEGACY_ENTRY_SHA256S: frozenset[str] = frozenset(
    {
        "f692a6128ac8e0bb8126ecc443bbd0630ed4b322d55e475a559b271208dc1311",
        "063c565f64a43fdc701b84ee878b6d0f287e3de03ca0eaa8aa1b53ff56044605",
        "43115c410a7e905f13c0c573b364d7c633975d58654d59ced020b589a9b8d321",
        "0914088952e8d513b80210f375f11565a8866e399e8e9dcd73949282240c327f",
        "3a569a1bb9a3cd5c9f297819382a7f14c02c6ed27bcb821e6025d3c6e5bd506f",
        "21514f79fee7cee4a6d5693b5ff467ed21339496203100a01eb0334708f2f3d8",
        "f893d93061e5b582612b837c77cc15fe11a18f5771299387d24fd4eb9ba42f4d",
        "8034dd1d8eaba8e9d45f80b337c6bb5180cb8cadbfe01eca1a64a6de0c0a7a3c",
        "488619afb33c6a0e5df2d04b3a5682a3cef6e88816d57632eccb6a1e2dccb7c0",
        "2f7387ed5c0d75b9317108b44e618db171710f408a659b4ffb5556dc858cdc5c",
        "43df32f411556d4da60cdb1f87aaaace1e4b5628842ba832cd30e9ee6a763882",
        "6830a21c729350b87eb44f3317d30af060971f1501f9da3c5150bca35eb04f23",
        "1d8834ab228979df58c6284a88be4243207ea406cc59b5df900a29afea761ded",
        "028afe75b1fa86d93e788a05cf18f8e85dd1c900a54029c4c038cc26bf62054f",
        "4d78f506ae6521e4f032cb1704beecbb7cad51c8724a99c803c828dc3711b84b",
        "728541f17cec17b3ddd567fcba171867a78e90aa2034c1aefcfb5d48e7ae6b9c",
        "ad13b58c65a3ac398f9928214de8dbdfc909b473482f84585557e384413bd54a",
        "101efcc43007f8fc011a86306edc6a36d7b2d4f6412e081a3f37c77f9802aa68",
        "739c33ee4367ef453dd21b264aa80a57e66bbad50bada8080079f3cee9bcdee7",
        "6b24769a3623defc1d0a408f29d04eb239cf469b810ee41a1cce9be841a49f29",
        "26c279551795a54cf7abfd5213ddafe429245886909eb7b7cecdd1fa7b13ac73",
        "0a4b23f6ce3c7dae9e58a25a8aa70da64afe3864d7e6a2d3dacb817c42f1509d",
        "af392762ce8ebe40697a5e04d4b6d6a1e4a363bad4f0c60199ad9fc1654062d3",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_decimal_cost(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("billing report entry has an invalid cost")
    try:
        cost = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("billing report entry has an invalid cost") from error
    if not cost.is_finite() or cost < 0:
        raise ValueError("billing report entry has an invalid cost")
    return cost


def _alias_value(
    row: dict[str, object],
    aliases: tuple[str, ...],
    *,
    label: str,
    normalize,
):
    found = [(key, normalize(row[key])) for key in aliases if key in row]
    if not found:
        raise ValueError(f"billing report entry has no {label}")
    first = found[0][1]
    if any(value != first for _, value in found[1:]):
        raise ValueError(f"billing report entry has conflicting {label} aliases")
    return first


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("billing report entry has an invalid string field")
    return value


def _normalized_report_row(row: object) -> tuple[dict[str, object], str, str, Decimal]:
    if not isinstance(row, dict):
        raise ValueError("billing report entry is not an object")
    app_id = _alias_value(
        row, _APP_ID_ALIASES, label="app ID", normalize=_nonempty_string
    )
    description = _alias_value(
        row, _DESCRIPTION_ALIASES, label="description", normalize=_nonempty_string
    )
    cost = _alias_value(row, _COST_ALIASES, label="cost", normalize=_parse_decimal_cost)
    return row, app_id, description, cost


def _report_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f"billing report object has duplicate key: {key}")
        row[key] = value
    return row


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"billing report contains non-finite JSON value: {value}")


def _select_provider_app_row(
    payload: bytes | str,
    *,
    provider_app_id: str,
    provider_app_description: str,
) -> tuple[str, dict[str, object], str, Decimal]:
    raw = _report_bytes(payload)
    try:
        report = json.loads(
            raw,
            object_pairs_hook=_report_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("billing report payload is not valid JSON") from error
    if not isinstance(report, list):
        raise ValueError("billing report payload is not a JSON list")

    matches: list[tuple[dict[str, object], Decimal]] = []
    for raw_row in report:
        row, app_id, description, cost = _normalized_report_row(raw_row)
        if app_id == provider_app_id and description == provider_app_description:
            matches.append((row, cost))
    if len(matches) != 1:
        raise ValueError(
            "billing report must contain exactly one row for the exact "
            "provider app ID and description"
        )
    selected_row, cost = matches[0]
    return hashlib.sha256(raw).hexdigest(), selected_row, _canonical_sha256(selected_row), cost


def _report_bytes(payload: bytes | str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    raise ValueError("billing report payload must be bytes or text")


def _billing_report_relative_path(
    ledger_path: Path, *, attempt_id: str, provider_app_id: str
) -> PurePosixPath:
    attempt_digest = hashlib.sha256(attempt_id.encode()).hexdigest()[:20]
    return PurePosixPath(
        f"{ledger_path.name}.artifacts",
        f"{attempt_digest}.{provider_app_id}.modal-billing-report.json",
    )


def _write_billing_report_artifact(
    ledger_path: Path,
    *,
    attempt_id: str,
    provider_app_id: str,
    payload: bytes,
) -> dict[str, object]:
    relative = _billing_report_relative_path(
        ledger_path, attempt_id=attempt_id, provider_app_id=provider_app_id
    )
    directory = ledger_path.parent / relative.parts[0]
    if directory.is_symlink():
        raise ValueError("Modal billing artifact directory may not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    path = ledger_path.parent.joinpath(*relative.parts)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return {
        "path": relative.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_billing_report_artifact(
    ledger_path: Path,
    *,
    entry: dict[str, Any],
    settlement: dict[str, object],
) -> bytes:
    binding = settlement.get("billing_report_artifact")
    app_id = settlement.get("provider_app_id")
    attempt_id = entry.get("attempt_id")
    if not isinstance(binding, dict) or not isinstance(app_id, str):
        raise ValueError("Modal billing report artifact binding is missing")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("Modal billing attempt identity is missing")
    expected = _billing_report_relative_path(
        ledger_path, attempt_id=attempt_id, provider_app_id=app_id
    )
    relative_value = binding.get("path")
    if not isinstance(relative_value, str):
        raise ValueError("Modal billing report artifact path is invalid")
    relative = PurePosixPath(relative_value)
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Modal billing report artifact path is invalid")
    path = ledger_path.parent.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Modal billing report artifact is missing or unsafe")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400:
        raise ValueError("Modal billing report artifact is not immutable")
    payload = path.read_bytes()
    if (
        binding.get("bytes") != len(payload)
        or binding.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("Modal billing report artifact binding changed")
    return payload


def _workspace_observation(entry: dict[str, Any]) -> dict[str, Any]:
    """Copy the immediate workspace reading without treating it as final cost."""

    return {
        "status": "provisional",
        "observed_at": entry.get("recorded_at"),
        "before_usd": entry.get("workspace_before_usd"),
        "after_usd": entry.get("workspace_after_usd"),
        "delta_usd": entry.get("reported_delta_usd"),
        "report_error": entry.get("postrun_report_error"),
    }


def _new_billing(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _PROVISIONAL_BILLING_STATUS,
        "workspace_observation": _workspace_observation(entry),
        "settlements": [],
    }


def _billing_is_settled(entry: dict[str, Any], *, ledger_path: Path) -> bool:
    """Return true only for a structurally complete provider-app settlement."""

    billing = entry.get("billing")
    if not isinstance(billing, dict):
        return _canonical_sha256(entry) in _GRANDFATHERED_LEGACY_ENTRY_SHA256S
    if billing.get("status") != _SETTLED_BILLING_STATUS:
        return False
    workspace = billing.get("workspace_observation")
    settlements = billing.get("settlements")
    if not isinstance(workspace, dict) or workspace.get("status") != "provisional":
        return False
    if workspace != _workspace_observation(entry):
        return False
    if not isinstance(settlements, list) or len(settlements) != 1:
        return False
    settlement = settlements[0]
    if not isinstance(settlement, dict) or settlement.get("status") != _SETTLEMENT_STATUS:
        return False
    app_id = settlement.get("provider_app_id")
    description = settlement.get("provider_app_description")
    cost = settlement.get("provider_app_cost_usd")
    if not isinstance(app_id, str) or not _APP_ID_PATTERN.fullmatch(app_id):
        return False
    if not isinstance(description, str) or not description or description != description.strip():
        return False
    try:
        parsed_cost = _parse_decimal_cost(cost)
        selected_row, row_app_id, row_description, row_cost = _normalized_report_row(
            settlement.get("selected_row")
        )
    except ValueError:
        return False
    if (row_app_id, row_description, row_cost) != (app_id, description, parsed_cost):
        return False
    if settlement.get("selected_row_sha256") != _canonical_sha256(selected_row):
        return False
    report_sha256 = settlement.get("billing_report_sha256")
    if not isinstance(report_sha256, str) or not _SHA256_PATTERN.fullmatch(report_sha256):
        return False
    if settlement.get("billing_report_query_argv") != list(_BILLING_REPORT_ARGV):
        return False
    if settlement.get("billing_report_scope") != _BILLING_REPORT_SCOPE:
        return False
    try:
        raw_report = _read_billing_report_artifact(
            ledger_path, entry=entry, settlement=settlement
        )
        raw_sha256, raw_row, raw_row_sha256, raw_cost = _select_provider_app_row(
            raw_report,
            provider_app_id=app_id,
            provider_app_description=description,
        )
    except (OSError, ValueError):
        return False
    if (
        raw_sha256 != report_sha256
        or raw_row != selected_row
        or raw_row_sha256 != settlement.get("selected_row_sha256")
        or raw_cost != parsed_cost
    ):
        return False
    observed_at = settlement.get("observed_at")
    if not isinstance(observed_at, str):
        return False
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if observed.tzinfo is None or observed.utcoffset() is None:
        return False
    declared_ceiling = entry.get("declared_ceiling_usd")
    try:
        ceiling = _parse_decimal_cost(declared_ceiling)
    except ValueError:
        return False
    return settlement.get("declared_ceiling_exceeded") is (parsed_cost > ceiling)


def _assert_prior_billing_settled(entry: dict[str, Any], *, ledger_path: Path) -> None:
    if entry.get("status") == "reserved" or not _billing_is_settled(
        entry, ledger_path=ledger_path
    ):
        raise ValueError("an earlier Modal attempt has unresolved or provisional billing evidence")


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
        _assert_prior_billing_settled(row, ledger_path=ledger_path)


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
        for row in entries:
            if not isinstance(row, dict):
                raise ValueError("Modal reconciliation ledger entry is malformed")
            if row.get("attempt_id") == attempt_id:
                raise ValueError(f"Modal attempt was already started: {attempt_id}")
            _assert_prior_billing_settled(row, ledger_path=ledger_path)
        entries.append(
            {
                "attempt_id": attempt_id,
                "stage": stage,
                "recorded_at": _utc_now(),
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
    evidence: dict[str, object] | None = None,
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
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("reconciliation evidence must be an object")
    recorded_at = _utc_now()
    entry: dict[str, Any] = {
        "attempt_id": attempt_id,
        "stage": stage,
        "recorded_at": recorded_at,
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
    if evidence is not None:
        entry["evidence"] = evidence
    entry["billing"] = _new_billing(entry)
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


def settle_attempt(
    path: Path | str,
    *,
    attempt_id: str,
    provider_app_id: str,
    provider_app_description: str,
    billing_report: bytes | str,
) -> dict[str, Any]:
    """Select and attach one exact row from a raw Modal billing report.

    The settlement is added to the finalized attempt under an exclusive ledger
    lock. Existing workspace totals and their delta are never replaced or
    reclassified as final provider cost. Cost and observation time are derived
    here rather than accepted from the caller.
    """

    if not attempt_id:
        raise ValueError("attempt_id is required")
    if not isinstance(provider_app_id, str) or not _APP_ID_PATTERN.fullmatch(
        provider_app_id
    ):
        raise ValueError("provider_app_id must be an exact valid Modal app ID")
    if (
        not isinstance(provider_app_description, str)
        or not provider_app_description
        or provider_app_description != provider_app_description.strip()
    ):
        raise ValueError("provider_app_description must be a non-empty trimmed string")
    raw_report = _report_bytes(billing_report)
    report_sha256, selected_row, selected_row_sha256, provider_cost = (
        _select_provider_app_row(
            raw_report,
            provider_app_id=provider_app_id,
            provider_app_description=provider_app_description,
        )
    )

    ledger_path = Path(path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Modal reconciliation ledger does not exist: {ledger_path}")
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
            raise ValueError("Modal reconciliation ledger is malformed")
        entries = document["entries"]
        matches = [
            row
            for row in entries
            if isinstance(row, dict) and row.get("attempt_id") == attempt_id
        ]
        if len(matches) != 1:
            if not matches:
                raise ValueError(f"Modal attempt was not found: {attempt_id}")
            raise ValueError(f"Modal attempt identity is ambiguous: {attempt_id}")
        entry = matches[0]
        if entry.get("status") not in {"succeeded", "remote-error"}:
            raise ValueError(f"Modal attempt is not finalized: {attempt_id}")

        existing_settlement: dict[str, object] | None = None
        for row in entries:
            if not isinstance(row, dict):
                raise ValueError("Modal reconciliation ledger entry is malformed")
            billing = row.get("billing")
            settlements = billing.get("settlements") if isinstance(billing, dict) else None
            if not isinstance(settlements, list):
                continue
            for settlement in settlements:
                if not isinstance(settlement, dict):
                    raise ValueError("Modal billing settlement is malformed")
                if settlement.get("provider_app_id") == provider_app_id:
                    if row is entry and existing_settlement is None:
                        existing_settlement = settlement
                    else:
                        raise ValueError(
                            f"Modal provider app is already settled: {provider_app_id}"
                        )

        billing = entry.get("billing")
        if billing is None:
            # Legacy finalized rows had only the immediate workspace observation.
            billing = _new_billing(entry)
            entry["billing"] = billing
        if not isinstance(billing, dict):
            raise ValueError("Modal billing evidence is malformed")
        if billing.get("workspace_observation") != _workspace_observation(entry):
            raise ValueError("Modal workspace billing observation was modified")
        settlements = billing.get("settlements")
        if not isinstance(settlements, list):
            raise ValueError("Modal billing settlements are malformed")
        declared_ceiling = entry.get("declared_ceiling_usd")
        try:
            declared_ceiling_decimal = _parse_decimal_cost(declared_ceiling)
        except ValueError as error:
            raise ValueError("Modal attempt has an invalid declared ceiling") from error
        if existing_settlement is None:
            if billing.get("status") != _PROVISIONAL_BILLING_STATUS or settlements:
                raise ValueError(f"Modal attempt is already settled: {attempt_id}")
            report_artifact = _write_billing_report_artifact(
                ledger_path,
                attempt_id=attempt_id,
                provider_app_id=provider_app_id,
                payload=raw_report,
            )
            settlement = {
                "status": _SETTLEMENT_STATUS,
                "provider_app_id": provider_app_id,
                "provider_app_description": provider_app_description,
                "provider_app_cost_usd": format(provider_cost, "f"),
                "observed_at": _utc_now(),
                "billing_report_query_argv": list(_BILLING_REPORT_ARGV),
                "billing_report_scope": _BILLING_REPORT_SCOPE,
                "billing_report_sha256": report_sha256,
                "billing_report_artifact": report_artifact,
                "selected_row": selected_row,
                "selected_row_sha256": selected_row_sha256,
                "declared_ceiling_exceeded": provider_cost > declared_ceiling_decimal,
            }
            settlements.append(settlement)
            billing["status"] = _SETTLED_BILLING_STATUS
        else:
            if "billing_report_artifact" in existing_settlement:
                raise ValueError(f"Modal attempt is already settled: {attempt_id}")
            previous_report_sha256 = existing_settlement.get(
                "billing_report_sha256"
            )
            if (
                billing.get("status") != _SETTLED_BILLING_STATUS
                or settlements != [existing_settlement]
                or existing_settlement.get("status") != _SETTLEMENT_STATUS
                or existing_settlement.get("provider_app_description")
                != provider_app_description
                or existing_settlement.get("provider_app_cost_usd")
                != format(provider_cost, "f")
                or not isinstance(previous_report_sha256, str)
                or _SHA256_PATTERN.fullmatch(previous_report_sha256) is None
                or existing_settlement.get("selected_row") != selected_row
                or existing_settlement.get("selected_row_sha256") != selected_row_sha256
                or existing_settlement.get("declared_ceiling_exceeded")
                is not (provider_cost > declared_ceiling_decimal)
            ):
                raise ValueError("existing Modal settlement does not match the raw report")
            existing_settlement["migrated_from_unretained_billing_report_sha256"] = (
                previous_report_sha256
            )
            existing_settlement["billing_report_sha256"] = report_sha256
            existing_settlement["observed_at"] = _utc_now()
            existing_settlement["billing_report_artifact"] = (
                _write_billing_report_artifact(
                    ledger_path,
                    attempt_id=attempt_id,
                    provider_app_id=provider_app_id,
                    payload=raw_report,
                )
            )
        if not _billing_is_settled(entry, ledger_path=ledger_path):
            raise ValueError("Modal billing settlement did not produce valid final evidence")

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


def settle_reconciliation(
    path: Path | str,
    *,
    attempt_id: str,
    provider_app_id: str,
    provider_app_description: str,
    billing_report: bytes | str,
) -> dict[str, Any]:
    """Compatibility-friendly descriptive alias for :func:`settle_attempt`."""

    return settle_attempt(
        path,
        attempt_id=attempt_id,
        provider_app_id=provider_app_id,
        provider_app_description=provider_app_description,
        billing_report=billing_report,
    )


def settle_attempt_from_modal_cli(
    path: Path | str,
    *,
    attempt_id: str,
    provider_app_id: str,
    provider_app_description: str,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the fixed Modal report command and settle its exact provider row."""

    completed = subprocess.run(
        _BILLING_REPORT_ARGV,
        check=True,
        capture_output=True,
        env=environment,
        timeout=_BILLING_REPORT_TIMEOUT_SECONDS,
    )
    return settle_attempt(
        path,
        attempt_id=attempt_id,
        provider_app_id=provider_app_id,
        provider_app_description=provider_app_description,
        billing_report=completed.stdout,
    )
