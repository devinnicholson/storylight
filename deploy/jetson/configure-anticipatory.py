#!/usr/bin/env python3
"""Enable the supervised, loopback-only GKE bridge without exposing the edge API."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path("/etc/storylight/storylight.env")


def replacement_values(*, enabled: bool, port: int) -> dict[str, str]:
    if not 1024 <= port <= 65535:
        raise ValueError("Bridge port must be between 1024 and 65535")
    if not enabled:
        return {"STORYLIGHT_ANTICIPATORY_BACKEND": "disabled"}
    return {
        "STORYLIGHT_ANTICIPATORY_BACKEND": "gke",
        "STORYLIGHT_ANTICIPATORY_URL": f"http://127.0.0.1:{port}",
        "STORYLIGHT_ANTICIPATORY_ALLOW_LOOPBACK_HTTP": "true",
        "STORYLIGHT_ANTICIPATORY_AUDIENCE": "",
    }


def environment_value(content: str, name: str) -> str | None:
    result = None
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            result = value.strip().strip("\"'")
    return result


def rewrite_environment(content: str, values: dict[str, str]) -> str:
    lines = [line for line in content.splitlines() if line.partition("=")[0].strip() not in values]
    return "\n".join([*lines, *(f"{key}={value}" for key, value in values.items())]) + "\n"


def read_config(path: Path, *, owner_uid: int = 0) -> str:
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != owner_uid or parent.st_mode & 0o022:
        raise ValueError("Configuration directory has unsafe ownership, permissions, or type")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != owner_uid
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise ValueError("Configuration must be a single-link root-owned mode-600 regular file")
        content = stream.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError("Configuration is unexpectedly large")
    return content


def write_private_file(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".storylight-env-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_json(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=2) as response:
        value = json.loads(response.read(65537))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON health response")
    return value


def verify_bridge(port: int) -> None:
    health = read_json(f"http://127.0.0.1:{port}/health")
    if health != {
        "ready": True,
        "service": "storylight-anticipatory",
        "privacy_boundary": "sanitized_scene_spec_v1",
    }:
        raise ValueError("The loopback port is not the expected Storylight GKE bridge")


def restart_api(user: str) -> None:
    subprocess.run(
        ["/usr/bin/systemctl", "restart", f"storylight@{user}.service"], check=True, timeout=40
    )


def wait_ready(enabled: bool) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            ready = read_json("http://127.0.0.1:8080/readyz")
            runtime = read_json("http://127.0.0.1:8080/v1/prepared-projections/runtime")
            if ready.get("ready") is True and runtime.get("enabled") is enabled:
                return
        except (OSError, ValueError):
            pass  # A restarting service briefly refuses connections.
        time.sleep(1)
    raise RuntimeError("The API did not report the requested next-page configuration")


def apply_configuration(path: Path, original: str, updated: str, *, restart, verify) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        prefix="storylight.env.before-anticipatory-", dir=path.parent
    )
    backup = Path(backup_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    write_private_file(path, updated)
    try:
        restart()
        verify()
    except (Exception, KeyboardInterrupt):
        if path.read_text(encoding="utf-8") != updated:
            raise RuntimeError(
                f"Configuration changed concurrently; recover from {backup}"
            ) from None
        write_private_file(path, original)
        restart()
        print(f"Previous configuration restored. Private backup: {backup}")
        raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run with sudo /usr/bin/python3 -I; no password is stored")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", args.user):
        parser.error("Select a valid non-root service user")
    if pwd.getpwnam(args.user).pw_uid == 0:
        parser.error("The Storylight API must not run as root")
    values = replacement_values(enabled=not args.disable, port=args.port)
    original = read_config(CONFIG)
    if not args.disable:
        if environment_value(original, "STORYLIGHT_LIVE_SCENE_PLANNER") != "model":
            parser.error("Keep the accepted local model planner configured before enabling GKE")
        verify_bridge(args.port)
    updated = rewrite_environment(original, values)
    for key, value in values.items():
        print(f"{key}={value}")
    if args.dry_run:
        print("Dry run passed. No configuration, service, GPU, or credentials changed.")
        return
    backup = apply_configuration(
        CONFIG,
        original,
        updated,
        restart=lambda: restart_api(args.user),
        verify=lambda: wait_ready(not args.disable),
    )
    print(f"Private rollback copy: {backup}")
    print("Next-page configuration ready. No GPU was started; cloud warmup remains explicit.")


if __name__ == "__main__":
    main()
