#!/usr/bin/env python3
"""Capture allowlisted device evidence without reading credentials into output."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def sensor_files(pattern: str) -> dict[str, str]:
    values = {}
    for path in sorted(Path("/sys").glob(pattern)):
        try:
            values[str(path)] = path.read_text().strip()
        except OSError:
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--server-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    import storylight.tensorrt_slot_client as deployed

    api_pid = command(
        "systemctl", "show", "storylight@operator.service", "-p", "MainPID", "--value"
    )
    server_pid = command(
        "systemctl",
        "--user",
        "show",
        "storylight-tensorrt-planner.service",
        "-p",
        "MainPID",
        "--value",
    )
    environment = (Path("/proc") / api_pid / "environ").read_bytes()
    allowed = {
        "STORYLIGHT_LIVE_SCENE_PLANNER_BACKEND",
        "STORYLIGHT_LIVE_SCENE_PLANNER_MAX_OUTPUT_TOKENS",
        "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_REVISION",
        "STORYLIGHT_LIVE_SCENE_PLANNER_BASE_URL",
        "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_NAME",
    }
    settings = {
        key: value
        for item in environment.decode().split("\0")
        if "=" in item
        for key, value in [item.split("=", 1)]
        if key in allowed
    }
    deployed_files = {
        path.name: file_digest(path) for path in sorted(Path(deployed.__file__).parent.glob("*.py"))
    }
    snapshot = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "l4t": Path("/etc/nv_tegra_release").read_text(),
        "packages": command("dpkg-query", "-W", "nvidia-jetpack", "libnvinfer10"),
        "power_mode": command("nvpmodel", "-q"),
        "server_revision": command("git", "-C", str(args.server_repo), "rev-parse", "HEAD"),
        "server_pid": int(server_pid),
        "server_started": command(
            "systemctl",
            "--user",
            "show",
            "storylight-tensorrt-planner.service",
            "-p",
            "ActiveEnterTimestamp",
            "--value",
        ),
        "engine_files": {
            path.name: file_digest(path)
            for path in sorted(args.engine_dir.iterdir())
            if path.is_file()
        },
        "deployed_files": deployed_files,
        "accepted_prompt_sha256": hashlib.sha256(
            deployed.TENSORRT_SLOT_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "planner_config": settings,
        "temperatures_millicelsius": sensor_files("class/thermal/thermal_zone*/temp"),
        "thermal_types": sensor_files("class/thermal/thermal_zone*/type"),
        "cpu_frequencies_khz": sensor_files("devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"),
        "device_frequencies_hz": sensor_files("class/devfreq/*/cur_freq"),
        "memory_kib": {
            line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":")[0] in {"MemTotal", "MemAvailable", "SwapTotal"}
        },
    }
    rendered = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    provenance = {
        "code_revision": args.code_revision,
        "deployment_sha256": hashlib.sha256(
            json.dumps(deployed_files, sort_keys=True).encode()
        ).hexdigest(),
        "engine_sha256": snapshot["engine_files"]["llm.engine"],
        "environment_sha256": hashlib.sha256(environment).hexdigest(),
        "snapshot_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hardware.json").write_text(rendered)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
