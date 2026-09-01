#!/usr/bin/env python3
"""Restart the accepted planner once and record checksum-bound cold-start evidence."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pwd
import re
import signal
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def approval_token(action: dict[str, object]) -> str:
    action_sha256 = hashlib.sha256(canonical(action)).hexdigest()
    return f"RECORD_BOOKFORGE_TRAINED_PLANNER_COLD_START:{action_sha256}"


def loopback_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("cold-start probes require a loopback HTTP base URL")
    return value.rstrip("/")


def parse_environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"').strip("'")
    return result


def route_is_accepted(
    environment: dict[str, str], *, engine_sha256: str, planner_base_url: str
) -> bool:
    return (
        environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND") == "tensorrt_slots"
        and environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION")
        == f"sha256:{engine_sha256}"
        and environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL") == planner_base_url
    )


def memory_available_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable value")


def swap_counters() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, raw = line.split()
        if key in {"pswpin", "pswpout"}:
            values[key] = int(raw)
    if set(values) != {"pswpin", "pswpout"}:
        raise RuntimeError("/proc/vmstat has no complete swap counters")
    return values["pswpin"], values["pswpout"]


def journal_failures(user: str, since_realtime_usec: int) -> tuple[int, int]:
    since = f"@{since_realtime_usec / 1_000_000:.6f}"
    completed = user_command(
        user,
        "journalctl",
        "--user",
        "-u",
        "bookforge-tensorrt-planner.service",
        "--since",
        since,
        "-o",
        "cat",
        "--no-pager",
    )
    if completed.returncode != 0:
        raise RuntimeError("could not inspect the planner journal after restart")
    messages = completed.stdout.casefold().splitlines()
    oom_events = sum(
        "oom-kill" in line or "out of memory" in line or "killed process" in line
        for line in messages
    )
    restart_failures = sum(
        "failed with result" in line or "main process exited" in line for line in messages
    )
    return oom_events, restart_failures


def request_json(
    url: str, *, payload: dict[str, object] | None = None, timeout: float = 5
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"loopback endpoint returned a non-object: {url}")
    return value


def user_command(user: str, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    account = pwd.getpwnam(user)
    return subprocess.run(
        [
            "runuser",
            "-u",
            user,
            "--",
            "env",
            f"XDG_RUNTIME_DIR=/run/user/{account.pw_uid}",
            f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{account.pw_uid}/bus",
            *args,
        ],
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(document))
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--accepted-engine", type=Path, required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--config", type=Path, default=Path("/etc/bookforge/bookforge.env"))
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--minimum-available-mib", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-token")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _SHA256.fullmatch(args.expected_engine_sha256) is None:
        raise SystemExit("--expected-engine-sha256 must be a lowercase SHA-256")
    if re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", args.user) is None:
        raise SystemExit("--user is invalid")
    if not 1 <= args.timeout_seconds <= 180:
        raise SystemExit("--timeout-seconds must be between 1 and 180")
    if args.minimum_available_mib < 512:
        raise SystemExit("--minimum-available-mib must be at least 512")
    planner_base = loopback_base_url(args.planner_base_url)
    api_base = loopback_base_url(args.api_base_url)
    action = {
        "user": args.user,
        "accepted_engine": str(args.accepted_engine.expanduser().absolute()),
        "expected_engine_sha256": args.expected_engine_sha256,
        "config": str(args.config.expanduser().absolute()),
        "planner_base_url": planner_base,
        "api_base_url": api_base,
        "timeout_seconds": args.timeout_seconds,
        "minimum_available_memory_mib": args.minimum_available_mib,
        "output": str(args.output.expanduser().absolute()),
        "service": "bookforge-tensorrt-planner.service",
    }
    action_sha256 = hashlib.sha256(canonical(action)).hexdigest()
    token = approval_token(action)
    args.accepted_engine = Path(action["accepted_engine"])
    args.config = Path(action["config"])
    args.output = Path(action["output"])
    plan = {
        "schema_version": "1.0",
        "producer": "bookforge-cold-start-recorder",
        "mode": "plan-only",
        "accepted_engine_sha256": args.expected_engine_sha256,
        "planner_base_url": planner_base,
        "api_base_url": api_base,
        "timeout_seconds": args.timeout_seconds,
        "minimum_available_memory_mib": args.minimum_available_mib,
        "service_restart": True,
        "action_sha256": action_sha256,
        "approval_token": token,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.approval_token != token:
        raise SystemExit("--approval-token must match the exact dry-run token")
    if os.geteuid() != 0:
        raise SystemExit("--execute must run with sudo to inspect root state safely")
    try:
        pwd.getpwnam(args.user)
    except KeyError as error:
        raise SystemExit("--user does not identify an existing account") from error
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("cold-start evidence output is write-once")
    evidence_root = Path("/var/lib/bookforge/trained-planner/evidence")
    evidence_root_info = evidence_root.lstat()
    if (
        not args.output.is_relative_to(evidence_root)
        or evidence_root.is_symlink()
        or not stat.S_ISDIR(evidence_root_info.st_mode)
        or evidence_root_info.st_uid != 0
        or evidence_root_info.st_gid != 0
        or evidence_root_info.st_mode & 0o077
    ):
        raise SystemExit("cold-start evidence must remain below the safe root-owned evidence root")
    engine = args.accepted_engine
    engine_info = engine.lstat()
    if (
        engine.is_symlink()
        or not stat.S_ISREG(engine_info.st_mode)
        or engine_info.st_uid != 0
        or engine_info.st_gid != 0
        or engine_info.st_mode & 0o222
        or sha256(engine) != args.expected_engine_sha256
    ):
        raise SystemExit("accepted engine bytes differ from the approved identity")
    config_info = args.config.lstat()
    if (
        args.config.is_symlink()
        or not stat.S_ISREG(config_info.st_mode)
        or config_info.st_uid != 0
        or config_info.st_gid != 0
        or config_info.st_mode & 0o077
    ):
        raise SystemExit("Bookforge root environment is missing or unsafe")
    config_sha256_before = sha256(args.config)
    before_route = parse_environment(args.config)
    if not route_is_accepted(
        before_route,
        engine_sha256=args.expected_engine_sha256,
        planner_base_url=planner_base,
    ):
        raise SystemExit("Bookforge is not routed to the approved accepted engine")

    started_realtime_usec = time.time_ns() // 1_000
    swap_before = swap_counters()
    minimum_memory = memory_available_mib()
    service_window_open = True

    def ensure_accepted_service() -> None:
        if not service_window_open:
            return
        active = user_command(
            args.user, "systemctl", "--user", "is-active", "bookforge-tensorrt-planner.service"
        )
        if active.stdout.strip() != "active":
            user_command(
                args.user,
                "systemctl",
                "--user",
                "start",
                "bookforge-tensorrt-planner.service",
            )

    def interrupted(_signal: int, _frame: object) -> None:
        raise SystemExit("cold-start window interrupted; accepted service restoration requested")

    atexit.register(ensure_accepted_service)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    restarted = user_command(
        args.user,
        "systemctl",
        "--user",
        "restart",
        "bookforge-tensorrt-planner.service",
    )
    if restarted.returncode != 0:
        raise RuntimeError(f"accepted planner restart failed: {restarted.stderr.strip()}")
    deadline = time.monotonic() + args.timeout_seconds
    planner_model: dict[str, object] | None = None
    while time.monotonic() < deadline:
        minimum_memory = min(minimum_memory, memory_available_mib())
        try:
            planner_model = request_json(f"{planner_base}/v1/models", timeout=1)
            break
        except (OSError, TimeoutError, ValueError, RuntimeError):
            time.sleep(0.25)
    if planner_model is None:
        raise RuntimeError("accepted TensorRT planner did not become ready before timeout")
    planner_ready_seconds = (time.time_ns() // 1_000 - started_realtime_usec) / 1_000_000

    api_ready = request_json(f"{api_base}/readyz")
    prepared = request_json(
        f"{api_base}/v1/live-scene-planner/prepare",
        payload={
            "text": "A silver moth opens a tiny book and one blue star rises above it.",
            "visual_style": "luminous paper theater",
            "seed": 20260901,
            "session_id": "bookforge-cold-start-acceptance",
        },
        timeout=15,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{api_base}/projector", timeout=5) as response:
        projector_status = response.status

    after_route = parse_environment(args.config)
    config_sha256_after = sha256(args.config)
    engine_sha256_after = sha256(engine)
    swap_after = swap_counters()
    oom_events, restart_failures = journal_failures(args.user, started_realtime_usec)
    minimum_memory = min(minimum_memory, memory_available_mib())
    swap_events = max(0, swap_after[0] - swap_before[0]) + max(0, swap_after[1] - swap_before[1])
    planning_ms = prepared.get("planning_ms")
    plan_revision = prepared.get("revision")
    projector_flow_passed = (
        api_ready.get("ready") is True
        and projector_status == 200
        and isinstance(planning_ms, (int, float))
        and not isinstance(planning_ms, bool)
        and planning_ms >= 0
        and plan_revision == f"sha256:{args.expected_engine_sha256}"
    )
    restoration_demonstrated = (
        route_is_accepted(
            after_route,
            engine_sha256=args.expected_engine_sha256,
            planner_base_url=planner_base,
        )
        and user_command(
            args.user,
            "systemctl",
            "--user",
            "is-active",
            "bookforge-tensorrt-planner.service",
        ).stdout.strip()
        == "active"
        and config_sha256_after == config_sha256_before
        and engine_sha256_after == args.expected_engine_sha256
    )
    passed = (
        oom_events == 0
        and restart_failures == 0
        and swap_events == 0
        and minimum_memory >= args.minimum_available_mib
        and planner_ready_seconds <= args.timeout_seconds
        and projector_flow_passed
        and restoration_demonstrated
    )
    completed_realtime_usec = time.time_ns() // 1_000
    evidence = {
        "schema_version": "1.0",
        "producer": "bookforge-cold-start-recorder",
        "status": "passed" if passed else "blocked",
        "accepted_engine_sha256": args.expected_engine_sha256,
        "action_sha256": action_sha256,
        "started_realtime_usec": started_realtime_usec,
        "completed_realtime_usec": completed_realtime_usec,
        "planner_ready_seconds": round(planner_ready_seconds, 3),
        "minimum_available_memory_mib": round(minimum_memory, 1),
        "oom_events": oom_events,
        "restart_failures": restart_failures,
        "swap_events": swap_events,
        "projector_flow_passed": projector_flow_passed,
        "restoration_demonstrated": restoration_demonstrated,
        "engine_sha256_after": engine_sha256_after,
        "planner_model_response_sha256": hashlib.sha256(canonical(planner_model)).hexdigest(),
        "planner_prepare_response_sha256": hashlib.sha256(canonical(prepared)).hexdigest(),
        "config_sha256": config_sha256_after,
        "config_sha256_before": config_sha256_before,
    }
    write_once(args.output, evidence)
    print(json.dumps({**evidence, "evidence_sha256": sha256(args.output)}, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
