import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_simple_scene_trial as supervisor  # noqa: E402


def test_metadata_requires_exact_non_snapshot_resources():
    function = {
        "image_id": supervisor.harness.cold.IMAGE_ID,
        "resources": {
            "gpu_config": {"count": 1, "gpu_type": "L4"},
            "memory_mb": 65536,
            "memory_mb_max": 65536,
            "milli_cpu": 8000,
            "milli_cpu_max": 8000,
        },
        "autoscaler_settings": {"max_containers": 1, "scaledown_window": 90},
        "max_inputs": 1,
        "startup_timeout_secs": 120,
        "timeout_secs": 60,
        "routing_region": "us-east",
        "scheduler_placement": {"regions": ["us"]},
        "is_class": True,
    }
    data = {"ranked_functions": [{"function": function}]}
    assert supervisor.validate_metadata(data)["max_inputs"] == 1
    for changes in (
        {"max_inputs": 2},
        {"checkpointing_enabled": True},
        {"experimental_options": {"enable_gpu_snapshot": "True"}},
        {"timeout_secs": 180},
        {"image_id": "different"},
        {"autoscaler_settings": {"max_containers": 2, "scaledown_window": 90}},
    ):
        invalid = copy.deepcopy(data)
        invalid["ranked_functions"][0]["function"].update(changes)
        with pytest.raises(ValueError):
            supervisor.validate_metadata(invalid)


@pytest.mark.parametrize("mode", ["complete", "timeout", "deployment_failure"])
def test_supervisor_preflight_claim_deadline_and_exact_shutdown(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    manifest_path = tmp_path / "manifest.json"
    manifest = supervisor.harness.prepare_manifest()
    manifest.update(status="authorized", expires_at=7000)
    supervisor.write(manifest_path, manifest)
    monkeypatch.setattr(supervisor, "MANIFEST", manifest_path)
    authorization_dir = tmp_path / "authorization"
    authorization_dir.mkdir(mode=0o700)
    ledger = tmp_path / "ledger.json"
    supervisor.write(
        ledger,
        {
            "envelope": {
                "run_cap_usd": 10,
                "usage_before_lab_usd": 10,
                "monthly_credit_usd": 30,
                "authorized_paid_usd": 0,
                "reserve_usd": 2,
            },
            "estimated_usage_usd": 1.75,
            "reservations": {"reservation:simple-scenes-20260905-a": 1.75},
        },
    )
    authorization = authorization_dir / "authorization.json"
    supervisor.write(
        authorization,
        {
            "schema_version": 1,
            "manifest_sha256": supervisor.harness.legacy.file_hash(manifest_path),
            "reservation_id": "reservation:simple-scenes-20260905-a",
            "reserved_usd": 1.75,
            "maximum_operations": 12,
            "ledger_sha256": supervisor.harness.legacy.file_hash(ledger),
        },
    )
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    args = SimpleNamespace(
        manifest=manifest_path,
        authorization=authorization,
        authorization_sha256=supervisor.harness.legacy.file_hash(authorization),
        ledger=ledger,
        output=output,
        python=sys.executable,
    )
    state = {"clock": 0, "deployed": False, "stopped": False, "final": 0}
    monkeypatch.setattr(supervisor.time, "time", lambda: 1000 + state["clock"])
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: state["clock"])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda seconds: state.update(clock=state["clock"] + seconds)
    )
    monkeypatch.setattr(supervisor.harness, "token_preflight", lambda: {"max_tokens": 101})
    commands, processes, killed, reaped = [], {}, [], []
    app = {
        "app_id": "ap-NewTrial",
        "description": supervisor.APP,
        "state": "deployed",
        "tasks": "0",
    }

    class Process:
        def __init__(self, command, *, stdout, **kwargs):
            commands.append(command)
            self.pid = 5000 + len(commands)
            self.returncode = 0
            processes[self.pid] = self
            if "deploy" in command:
                assert (authorization_dir / "supervisor-attempt.json").exists()
                state["deployed"] = True
                if mode == "deployment_failure":
                    self.returncode = 1
            elif "list" in command and "app" in command:
                rows = []
                if mode == "deployment_failure" and state["deployed"] and not state["stopped"]:
                    attempt = state.get("recovery_reads", 0)
                    state["recovery_reads"] = attempt + 1
                    if attempt < 3:
                        # First read fails, second hangs, third cannot see the new app yet.
                        self.returncode = (1, None, 0)[attempt]
                        stdout.write(b"[]")
                        return
                if state["deployed"]:
                    row = dict(app)
                    if state["stopped"]:
                        row.update(state="stopped", tasks="1" if state["final"] == 0 else "0")
                        state["final"] += 1
                    rows = [row]
                stdout.write(json.dumps(rows).encode())
            elif "list" in command and "container" in command:
                stdout.write(b"[]")
            elif "--metadata" in command:
                supervisor.write(Path(command[-1]), {"app_id": "ap-NewTrial", "verified": True})
            elif "logs" in command:
                stdout.write(b'{"simple_scene_stage":{"phase":"initialization","state":"start"}}\n')
                self.returncode = None
            elif "--run" in command:
                self.returncode = None if mode == "timeout" else 0
            elif "stop" in command:
                assert command[-1] == "ap-NewTrial"
                state["stopped"] = True

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert self.returncode is not None and timeout == 1
            reaped.append(self.pid)
            return self.returncode

    monkeypatch.setattr(supervisor.subprocess, "Popen", Process)

    def kill(pid, signal):
        killed.append(pid)
        processes[pid].returncode = -9

    monkeypatch.setattr(supervisor.os, "killpg", kill)
    trial = supervisor.Supervisor(args)
    # Ledger mismatch must fail before inventory, deployment or any paid operation.
    original_ledger = ledger.read_bytes()
    ledger.write_bytes(original_ledger + b" ")
    with pytest.raises(ValueError):
        trial.work()
    assert not commands
    ledger.write_bytes(original_ledger)
    if mode in {"timeout", "deployment_failure"}:
        with pytest.raises(TimeoutError if mode == "timeout" else ValueError):
            trial.work()
    else:
        trial.work()
    trial.cleanup()
    assert trial.record["external_app_shutdown_verified"] is True
    assert trial.record["stop_returncode"] == 0
    assert trial.record["total_wall_seconds"] <= 660
    assert sum("--run" in command for command in commands) == (mode != "deployment_failure")
    assert sum("deploy" in command for command in commands) == 1
    assert state["final"] == 2
    assert killed
    if mode == "deployment_failure":
        assert state["recovery_reads"] == 4 and len(reaped) == 2
        assert trial.record["recovery_read_failures"] == 2
        assert trial.record["stop_started_wall_seconds"] < 45
    if mode == "timeout":
        assert 600 <= trial.record["stop_started_wall_seconds"] < 601
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == supervisor.harness.legacy.file_hash(
        ledger
    )
