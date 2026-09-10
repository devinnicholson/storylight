from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/jetson/record-trained-planner-cold-start.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("storylight_cold_start", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cold_start_dry_run_is_non_mutating_and_exactly_bound(tmp_path: Path) -> None:
    engine_sha = "a" * 64
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--user",
            "demo",
            "--accepted-engine",
            str(tmp_path / "engine"),
            "--expected-engine-sha256",
            engine_sha,
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "plan-only"
    assert plan["service_restart"] is True
    assert plan["approval_token"].startswith("RECORD_STORYLIGHT_TRAINED_PLANNER_COLD_START:")
    assert len(plan["approval_token"].rsplit(":", 1)[1]) == 64
    assert not output.exists()

    changed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--user",
            "demo",
            "--accepted-engine",
            str(tmp_path / "engine"),
            "--expected-engine-sha256",
            engine_sha,
            "--minimum-available-mib",
            "1024",
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(changed.stdout)["approval_token"] != plan["approval_token"]


def test_cold_start_route_requires_exact_backend_revision_and_loopback() -> None:
    module = _load_module()
    engine_sha = "b" * 64
    environment = {
        "STORYLIGHT_LIVE_SCENE_PLANNER_BACKEND": "tensorrt_slots",
        "STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_REVISION": f"sha256:{engine_sha}",
        "STORYLIGHT_LIVE_SCENE_PLANNER_BASE_URL": "http://127.0.0.1:11435",
    }
    assert module.route_is_accepted(
        environment,
        engine_sha256=engine_sha,
        planner_base_url="http://127.0.0.1:11435",
    )
    environment["STORYLIGHT_LIVE_SCENE_PLANNER_MODEL_REVISION"] = "sha256:" + "c" * 64
    assert not module.route_is_accepted(
        environment,
        engine_sha256=engine_sha,
        planner_base_url="http://127.0.0.1:11435",
    )
    with pytest.raises(ValueError, match="loopback"):
        module.loopback_base_url("https://example.com")
