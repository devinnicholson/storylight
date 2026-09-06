import hashlib
import json
import subprocess
import time

import pytest

from scripts import run_gcp_klein_trial as trial


def manifest_file(tmp_path):
    identity, _ = trial.benchmark.cold.frozen_cases()
    identity.update(gpu="NVIDIA RTX PRO 6000 Blackwell", capability=[12, 0])
    sources = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in {
            "worker": trial.ROOT / "deploy/gcp_klein_worker/app.py",
            "runtime": trial.ROOT / "deploy/klein_scene_runtime.py",
            "weights": trial.ROOT / "deploy/gcp_klein_worker/klein_weights.py",
        }.items()
    }
    value = trial.benchmark.prepare_manifest(
        "bookforge-klein-qualification-20260906-a", identity, sources
    )
    value.update(status="authorized", expires_at=int(time.time()) + 1800)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


def test_ambiguous_deployment_is_deleted_and_cleanup_failure_is_visible(tmp_path, monkeypatch):
    manifest = manifest_file(tmp_path)
    image = trial.IMAGE_PREFIX + "a" * 64
    commands = []

    def control(args, timeout=30):
        commands.append(args)
        if args[:2] == ["run", "deploy"]:
            raise subprocess.TimeoutExpired("deploy", timeout)
        return []

    monkeypatch.setattr(trial, "gcloud", control)
    service = "bookforge-klein-qualification-20260906-e"
    with pytest.raises(RuntimeError, match="deletion could not be verified"):
        trial.run(image, manifest, tmp_path / "closed", service=service)
    closure = json.loads((tmp_path / "closed/closure.json").read_text())
    assert closure["service_deleted_and_absent"] is False
    assert closure["service_absent_at_observation"] is True
    assert closure["late_creation_cleanup_required"] is True
    assert closure["failure"] == "TimeoutExpired"
    assert any(command[:3] == ["run", "deploy", service] for command in commands)
    assert any(command[:4] == ["run", "services", "delete", service] for command in commands)
    scope = json.loads((tmp_path / "closed/scope.json").read_text())
    assert scope["service"] == service
    assert scope["experiment_id"] == "bookforge-klein-qualification-20260906-a"

    def failed_delete(args, timeout=30):
        if args[:3] == ["run", "services", "delete"] or (
            args[:3] == ["run", "services", "list"] and len(commands) > 5
        ):
            raise RuntimeError("unavailable")
        return control(args, timeout)

    monkeypatch.setattr(trial, "gcloud", failed_delete)
    with pytest.raises(RuntimeError, match="deletion could not be verified"):
        trial.run(image, manifest, tmp_path / "unclosed")
    closure = json.loads((tmp_path / "unclosed/closure.json").read_text())
    assert closure["service_deleted_and_absent"] is False
    assert closure["cleanup_error"] == "RuntimeError"

    binding = {
        "role": "roles/run.invoker",
        "members": [f"serviceAccount:{trial.ACCOUNT}"],
    }

    def completed_control(args, timeout=30):
        if args[:3] == ["run", "services", "list"]:
            return []
        if args[:3] == ["run", "services", "get-iam-policy"]:
            return {"bindings": [binding] if binding else []}
        return {}

    monkeypatch.setattr(trial, "gcloud", completed_control)
    monkeypatch.setattr(trial, "verify_deployment", lambda *args: ("https://test.run.app", "trial"))
    monkeypatch.setattr(trial, "execute", lambda *args: subprocess.CompletedProcess([], 1, "", ""))
    sleeps = []
    monkeypatch.setattr(trial.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="client did not complete"):
        trial.run(image, manifest, tmp_path / "client-failed")
    closure = json.loads((tmp_path / "client-failed/closure.json").read_text())
    assert closure["service_deleted_and_absent"] is True
    assert closure["late_creation_cleanup_required"] is False
    assert sleeps == [30, 30, 30, 30]

    for invalid in ({}, {**binding, "condition": {"expression": "true"}}):
        binding = invalid
        sleeps.clear()
        output = tmp_path / ("conditional" if invalid else "missing-grant")
        with pytest.raises(ValueError, match="invoker grant is missing"):
            trial.run(image, manifest, output)
        assert sleeps == []
        assert not (output / "client-exit.json").exists()
        assert json.loads((output / "closure.json").read_text())["service_deleted_and_absent"]

    binding = {"role": "roles/run.invoker", "members": [f"serviceAccount:{trial.ACCOUNT}"]}
    elapsed = [0]
    monkeypatch.setattr(trial.time, "monotonic", lambda: elapsed[0])

    def delayed_sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] = trial.LIFETIME_SECONDS - 30

    monkeypatch.setattr(trial.time, "sleep", delayed_sleep)
    with pytest.raises(TimeoutError, match="IAM propagation"):
        trial.run(image, manifest, tmp_path / "deadline")
    assert sleeps == [30]
    assert not (tmp_path / "deadline/client-exit.json").exists()
    closure = json.loads((tmp_path / "deadline/closure.json").read_text())
    assert closure["service_deleted_and_absent"]


def test_existing_service_is_never_replaced_or_deleted(tmp_path, monkeypatch):
    service = "bookforge-klein-qualification-20260906-e"
    manifest = manifest_file(tmp_path)
    commands = []

    def control(args, timeout=30):
        commands.append(args)
        return [{"metadata": {"name": service}}]

    monkeypatch.setattr(trial, "gcloud", control)
    with pytest.raises(ValueError, match="existing service"):
        trial.run(trial.IMAGE_PREFIX + "a" * 64, manifest, tmp_path / "output", service=service)
    assert commands == [["run", "services", "list", "--region", trial.REGION]]
