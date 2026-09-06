"""Bounded subprocess lifecycle for an explicitly authorized single Modal trial."""

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def require(condition):
    if not condition:
        raise ValueError("trial lifecycle verification failed")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    write(path, data)


def write(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def supervise(
    output, command, *, app, attempt_path, authority, work_seconds, cleanup_seconds, authorize, cli
):
    """Claim once, supervise one child, then prove the exact app has no active work."""
    authorize()
    require(output.is_absolute() and output.resolve() == output)
    output.mkdir(mode=0o700)
    before = json.loads(cli("app", "list", "--json"))
    require(not any(row["description"] == app and row["state"] != "stopped" for row in before))
    previous = {row["app_id"] for row in before}
    authorize()
    save(attempt_path, authority)
    save(output / "authorization.json", authority)
    started, process = time.monotonic(), None
    record = {"returncode": 1, "external_app_shutdown_verified": False}

    def interrupted(*_):
        raise InterruptedError("trial supervisor interrupted")

    prior_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["returncode"] = process.wait(timeout=work_seconds)
    finally:
        record["work_wall_seconds"] = time.monotonic() - started
        deadline = min(started + work_seconds + cleanup_seconds, time.monotonic() + cleanup_seconds)
        try:
            app_id = read(output / "app.json")["app_id"] if (output / "app.json").exists() else None
            while app_id is None and time.monotonic() + 21 < deadline:
                try:
                    rows = json.loads(cli("app", "list", "--json"))
                except (OSError, ValueError, subprocess.SubprocessError):
                    rows = []
                matches = [
                    row
                    for row in rows
                    if row["description"] == app and row["app_id"] not in previous
                ]
                require(len(matches) <= 1)
                if matches:
                    app_id = matches[0]["app_id"]
                else:
                    time.sleep(0.2)
            require(app_id and re.fullmatch(r"ap-[A-Za-z0-9]+", app_id))
            record["app_id"] = app_id
            require(time.monotonic() + 10 < deadline)
            try:
                cli("app", "stop", "--yes", app_id, timeout=10)
                record["stop_returncode"] = 0
            except subprocess.CalledProcessError as error:
                record["stop_returncode"] = error.returncode
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            while time.monotonic() + 10 < deadline:
                apps = json.loads(cli("app", "list", "--json"))
                containers = json.loads(cli("container", "list", "--json"))
                exact = [row for row in apps if row["app_id"] == app_id]
                require(len(exact) == 1 and all("app_id" in row for row in containers))
                if (
                    exact[0]["state"] == "stopped"
                    and int(exact[0]["tasks"]) == 0
                    and not any(row["app_id"] == app_id for row in containers)
                ):
                    save(output / "shutdown.json", {"app": exact[0], "active_containers": 0})
                    record["external_app_shutdown_verified"] = True
                    break
                time.sleep(0.2)
            require(record["external_app_shutdown_verified"])
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            record["total_wall_seconds"] = time.monotonic() - started
            save(output / "supervisor.json", record)
            signal.signal(signal.SIGTERM, prior_handler)
    require(
        record["returncode"] == 0
        and record["work_wall_seconds"] <= work_seconds
        and record["total_wall_seconds"] <= work_seconds + cleanup_seconds
    )
    return record
