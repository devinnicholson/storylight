#!/usr/bin/env python3
"""Read-only readiness gate for a Jetson trained-planner acceptance window."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import urllib.request
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--accepted-engine", type=Path, required=True)
    parser.add_argument("--expected-accepted-engine-sha256", required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--cold-start-evidence", type=Path)
    parser.add_argument("--cold-start-evidence-sha256")
    parser.add_argument("--minimum-available-mib", type=int, default=768)
    parser.add_argument("--oom-lookback-hours", type=int, default=168)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False, timeout=15)


def get_json(url: str) -> object:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return json.load(response)


def parse_environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"').strip("'")
    return result


def evidence_is_fresh(
    path: Path | None,
    expected_sha: str | None,
    engine_sha: str,
    latest_oom_usec: int,
    minimum_available_mib: int,
) -> tuple[bool, str, int]:
    if path is None or expected_sha is None:
        return False, "historical OOM requires checksum-bound cold-start evidence", 0
    if path.is_symlink() or not path.is_file() or sha256(path) != expected_sha:
        return False, "cold-start evidence file or checksum is invalid", 0
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, "cold-start evidence is not valid JSON", 0
    required = {
        "schema_version": "1.0",
        "producer": "bookforge-cold-start-recorder",
        "status": "passed",
        "accepted_engine_sha256": engine_sha,
        "oom_events": 0,
        "swap_events": 0,
        "projector_flow_passed": True,
        "restoration_demonstrated": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        return False, "cold-start evidence does not prove a clean exact restoration", 0
    completed = value.get("completed_realtime_usec")
    available = value.get("minimum_available_memory_mib")
    ready = value.get("planner_ready_seconds")
    if (
        not isinstance(completed, int)
        or completed <= latest_oom_usec
        or not isinstance(available, (int, float))
        or isinstance(available, bool)
        or available < minimum_available_mib
        or not isinstance(ready, (int, float))
        or isinstance(ready, bool)
        or ready <= 0
        or ready > 90
    ):
        return False, "cold-start evidence is stale or outside readiness bounds", 0
    return True, "checksum-bound post-OOM cold start passed", completed


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", args.user):
        raise SystemExit("invalid --user")
    if not re.fullmatch(r"[a-f0-9]{64}", args.expected_accepted_engine_sha256):
        raise SystemExit("invalid accepted engine checksum")
    if bool(args.candidate_id) != bool(args.candidate_manifest_sha256):
        raise SystemExit("candidate id and manifest checksum must be supplied together")
    if args.candidate_id and (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", args.candidate_id)
        or not re.fullmatch(r"[a-f0-9]{64}", args.candidate_manifest_sha256)
    ):
        raise SystemExit("invalid candidate identity")
    if bool(args.cold_start_evidence) != bool(args.cold_start_evidence_sha256):
        raise SystemExit("cold-start evidence and checksum must be supplied together")
    plan = {
        "schema_version": "1.0",
        "artifact_type": "bookforge-jetson-acceptance-preflight",
        "mode": "candidate" if args.candidate_id else "accepted-noop",
        "read_only": True,
        "probes": [
            "accepted-engine-identity-and-permissions",
            "exact-routing-and-rollback-identity",
            "tooling-provenance-and-candidate-root",
            "historical-oom-and-checksum-bound-cold-start",
            "available-memory-and-swap",
            "planner-readiness",
            "projector-kiosk-and-connected-display",
            "service-restart-events",
        ],
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "planned"}, sort_keys=True))
        return

    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    engine = args.accepted_engine
    try:
        info = engine.lstat()
        engine_digest = sha256(engine)
        engine_safe = (
            stat.S_ISREG(info.st_mode)
            and not engine.is_symlink()
            and info.st_size > 0
            and not info.st_mode & 0o022
            and engine_digest == args.expected_accepted_engine_sha256
        )
        record(
            "accepted_engine",
            engine_safe,
            {"sha256": engine_digest, "mode": oct(stat.S_IMODE(info.st_mode)), "uid": info.st_uid},
        )
    except OSError as error:
        record("accepted_engine", False, str(error))

    config_path = Path("/etc/bookforge/bookforge.env")
    try:
        environment = parse_environment(config_path)
        expected_revision = f"sha256:{args.expected_accepted_engine_sha256}"
        routing_ok = (
            environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND") == "tensorrt_slots"
            and environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION") == expected_revision
            and environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL") == "http://127.0.0.1:11435"
        )
        record(
            "exact_routing_identity",
            routing_ok,
            {"revision": environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION")},
        )
    except OSError as error:
        record("exact_routing_identity", False, str(error))

    active_state = Path("/var/lib/bookforge/trained-planner/active.env")
    record(
        "rollback_state_clear",
        not active_state.exists() and not active_state.is_symlink(),
        "no promotion is active" if not active_state.exists() else str(active_state),
    )
    current = Path("/usr/local/lib/bookforge/trained-planner-tooling/current")
    receipt = Path("/var/lib/bookforge/trained-planner/tooling-current.json")
    try:
        receipt_value = json.loads(receipt.read_text())
        manifest_path = current / "tooling.manifest.json"
        manifest_value = json.loads(manifest_path.read_text())
        current_info = current.lstat()
        payloads_ok = True
        file_records = manifest_value.get("files")
        if not isinstance(file_records, list) or not file_records:
            raise ValueError("installed tooling manifest has no file records")
        for file_record in file_records:
            if not isinstance(file_record, dict):
                raise ValueError("installed tooling manifest has an invalid file record")
            relative = Path(str(file_record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("installed tooling manifest has an unsafe path")
            file_path = current / relative
            file_info = file_path.lstat()
            payloads_ok = payloads_ok and (
                not file_path.is_symlink()
                and stat.S_ISREG(file_info.st_mode)
                and file_info.st_uid == 0
                and file_info.st_gid == 0
                and stat.S_IMODE(file_info.st_mode) == file_record.get("mode")
                and file_info.st_size == file_record.get("bytes")
                and sha256(file_path) == file_record.get("sha256")
            )
        provenance_ok = (
            current.is_symlink()
            and current_info.st_uid == 0
            and current_info.st_gid == 0
            and manifest_path.is_file()
            and sha256(manifest_path) == receipt_value.get("tooling_manifest_sha256")
            and manifest_value.get("source_commit") == receipt_value.get("source_commit")
            and manifest_value.get("source_commit_verified") is True
            and receipt_value.get("source_commit_verified") is True
            and manifest_value.get("source_manifest_sha256")
            == receipt_value.get("source_manifest_sha256")
            and hashlib.sha256(canonical(file_records)).hexdigest()
            == manifest_value.get("source_manifest_sha256")
            and payloads_ok
            and receipt_value.get("artifact_type") == "bookforge-jetson-tooling-install-receipt"
            and re.fullmatch(r"[a-f0-9]{40}", str(receipt_value.get("source_commit", "")))
        )
        record("tooling_provenance", bool(provenance_ok), receipt_value)
    except (OSError, ValueError) as error:
        record("tooling_provenance", False, str(error))

    candidate_root = Path("/var/lib/bookforge/trained-planner-candidates")
    root_ok = candidate_root.is_dir() and not candidate_root.is_symlink()
    if args.candidate_id:
        candidate = candidate_root / args.candidate_id
        verifier = current / "install-trained-planner-candidate.sh"
        result = command(
            str(verifier),
            "--bundle",
            str(candidate),
            "--expected-manifest-sha256",
            args.candidate_manifest_sha256,
            "--verify-only",
            "--installed-layout",
        )
        root_ok = root_ok and result.returncode == 0
        candidate_detail: object = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    else:
        candidate_detail = "candidate root exists for a future immutable candidate"
    record("candidate_root", root_ok, candidate_detail)

    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        meminfo[key] = int(raw.strip().split()[0])
    available_mib = meminfo.get("MemAvailable", 0) / 1024
    swap_active = len(Path("/proc/swaps").read_text().splitlines()) > 1
    record(
        "memory",
        available_mib >= args.minimum_available_mib and not swap_active,
        {
            "available_mib": round(available_mib, 1),
            "minimum_mib": args.minimum_available_mib,
            "swap_active": swap_active,
        },
    )

    journal = command(
        "journalctl",
        "--user",
        "-u",
        "bookforge-tensorrt-planner.service",
        "--since",
        f"{args.oom_lookback_hours} hours ago",
        "-o",
        "json",
        "--no-pager",
    )
    latest_oom_usec = 0
    restart_timestamps: list[int] = []
    for raw in journal.stdout.splitlines():
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        message = str(event.get("MESSAGE", "")).casefold()
        timestamp = int(event.get("__REALTIME_TIMESTAMP", 0))
        if "oom-kill" in message or "out of memory" in message or "killed process" in message:
            latest_oom_usec = max(latest_oom_usec, timestamp)
        if "scheduled restart job" in message or "main process exited" in message:
            restart_timestamps.append(timestamp)
    if latest_oom_usec:
        cold_ok, cold_detail, evidence_completed_usec = evidence_is_fresh(
            args.cold_start_evidence,
            args.cold_start_evidence_sha256,
            args.expected_accepted_engine_sha256,
            latest_oom_usec,
            args.minimum_available_mib,
        )
    else:
        cold_ok = True
        cold_detail = "no historical OOM in the requested lookback"
        evidence_completed_usec = 0
    record(
        "historical_oom_and_cold_start",
        journal.returncode == 0 and cold_ok,
        {"latest_oom_realtime_usec": latest_oom_usec, "detail": cold_detail},
    )
    unaccounted_restarts = [
        timestamp
        for timestamp in restart_timestamps
        if not evidence_completed_usec or timestamp > evidence_completed_usec
    ]
    record(
        "planner_restart_events",
        not unaccounted_restarts,
        {
            "events_in_lookback": len(restart_timestamps),
            "events_after_cold_start_evidence": len(unaccounted_restarts),
        },
    )

    accepted_service = command(
        "systemctl", "--user", "is-active", "bookforge-tensorrt-planner.service"
    )
    kiosk_service = command("systemctl", "--user", "is-active", "bookforge-kiosk.service")
    try:
        planner_ready = get_json("http://127.0.0.1:11435/v1/models")
        api_ready = get_json("http://127.0.0.1:8080/readyz")
        projector = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            .open("http://127.0.0.1:8080/projector", timeout=5)
            .status
        )
        endpoint_ok = (
            bool(planner_ready)
            and isinstance(api_ready, dict)
            and api_ready.get("ready") is True
            and projector == 200
        )
        endpoint_detail: object = {"api": api_ready, "projector_status": projector}
    except Exception as error:  # a preflight failure is evidence, not a traceback
        endpoint_ok, endpoint_detail = False, str(error)
    record(
        "planner_and_projector",
        accepted_service.stdout.strip() == "active" and endpoint_ok,
        endpoint_detail,
    )
    connected = [
        path
        for path in Path("/sys/class/drm").glob("*/status")
        if path.read_text().strip() == "connected"
    ]
    record(
        "kiosk_and_display",
        kiosk_service.stdout.strip() == "active" and bool(connected),
        [str(path) for path in connected],
    )

    passed = all(bool(check["passed"]) for check in checks.values())
    print(
        json.dumps(
            {**plan, "status": "passed" if passed else "blocked", "checks": checks}, sort_keys=True
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
