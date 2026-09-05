# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import modal_jax_fidelity


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_training_process_records_timeout_and_killed_returncode(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    log_path = scratch / "failure-evidence/training-subprocess.log"
    command = [
        sys.executable,
        "-c",
        "import os, time; os.write(1, b'STARTED\\n'); time.sleep(30)",
    ]

    evidence, error = modal_jax_fidelity._run_training_process(
        command,
        environment={},
        timeout_seconds=0.2,
        log_path=log_path,
        scratch_directory=scratch,
    )

    assert isinstance(error, subprocess.TimeoutExpired)
    assert evidence["timed_out"] is True
    assert evidence["timeout_seconds"] == 0.2
    assert isinstance(evidence["returncode"], int)
    assert evidence["returncode"] < 0
    log = evidence["log"]
    assert isinstance(log, dict)
    assert str(log["tail"]).endswith("STARTED\n")
    assert log["sha256"] == hashlib.sha256(log_path.read_bytes()).hexdigest()


def test_failure_envelope_is_complete_and_committed_before_original_error_is_raised(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    attempt = scratch / "attempt.json"
    preflight = scratch / "gpu-preflight.json"
    completion = scratch / "runs/training-run/completion.json"
    _write(attempt, b'{"status":"started"}\n')
    _write(preflight, b'{"devices":2,"mesh":{"fsdp":2}}\n')
    _write(completion, b'{"run_id":"training-run","status":"failed"}\n')
    log_path = scratch / "failure-evidence/training-subprocess.log"
    _write(log_path, b"ValueError: root failure\n")

    runtime_lock = tmp_path / "runtime.lock.json"
    lock_sha = _write(runtime_lock, b'{"packages":["jax"]}\n')
    patch = tmp_path / "approved.patch"
    patch_sha = _write(patch, b"approved patch\n")
    train_source = tmp_path / "MaxText/src/maxtext/trainers/pre_train/train.py"
    train_sha = _write(train_source, b"patched train\n")
    utils_source = tmp_path / "MaxText/src/maxtext/utils/train_utils.py"
    utils_sha = _write(utils_source, b"patched utils\n")
    provenance = {
        "bookforge_source_manifest": {
            "path": str(tmp_path / "source.manifest.json"),
            "bytes": len(b'{"files":[]}\n'),
            "sha256": _write(tmp_path / "source.manifest.json", b'{"files":[]}\n'),
            "expected_sha256": hashlib.sha256(b'{"files":[]}\n').hexdigest(),
            "file_count": 0,
            "files_sha256": hashlib.sha256(b"[]").hexdigest(),
        },
        "runtime_lock": {
            "path": str(runtime_lock),
            "bytes": runtime_lock.stat().st_size,
            "sha256": lock_sha,
        },
        "approved_maxtext_patch": {
            "path": str(patch),
            "bytes": patch.stat().st_size,
            "sha256": patch_sha,
            "expected_sha256": patch_sha,
        },
        "maxtext": {
            "root": str(tmp_path / "MaxText"),
            "expected_revision": "a" * 40,
            "observed_revision": "a" * 40,
            "patched_sources": [
                {
                    "module": "maxtext.trainers.pre_train.train",
                    "path": str(train_source),
                    "snapshot_path": "maxtext/trainers/pre_train/train.py",
                    "bytes": train_source.stat().st_size,
                    "sha256": train_sha,
                },
                {
                    "module": "maxtext.utils.train_utils",
                    "path": str(utils_source),
                    "snapshot_path": "maxtext/utils/train_utils.py",
                    "bytes": utils_source.stat().st_size,
                    "sha256": utils_sha,
                },
            ],
        },
    }
    process_evidence = {
        "returncode": 7,
        "timed_out": False,
        "timeout_seconds": 3180,
        "duration_seconds": 1.25,
        "log": {
            **modal_jax_fidelity._relative_file_binding(log_path, trusted_root=scratch),
            "streams": "stdout+stderr",
            "stream_bytes": log_path.stat().st_size,
            "truncated": False,
            "limit_bytes": modal_jax_fidelity._FAILURE_LOG_LIMIT_BYTES,
            "tail": log_path.read_text(),
        },
    }
    original = subprocess.CalledProcessError(7, ["python3", "trainer.py"])
    commits: list[dict[str, object]] = []

    def commit() -> None:
        failure_path = scratch / "failure.json"
        assert failure_path.is_file()
        commits.append(json.loads(failure_path.read_text()))

    with pytest.raises(subprocess.CalledProcessError) as raised:
        modal_jax_fidelity._persist_failure_and_raise(
            original,
            run_id="outer-run",
            training_run_id="training-run",
            scratch_directory=scratch,
            attempt_path=attempt,
            gpu_preflight_path=preflight,
            training_completion_path=completion,
            process_evidence=process_evidence,
            runtime_provenance=provenance,
            scratch_commit=commit,
        )

    assert raised.value is original
    assert len(commits) == 1
    failure = commits[0]
    assert failure["status"] == "failed"
    assert failure["error"] == {
        "type": "CalledProcessError",
        "message": str(original),
    }
    assert failure["process"]["returncode"] == 7
    assert failure["process"]["timed_out"] is False
    assert failure["attempt"]["sha256"] == hashlib.sha256(attempt.read_bytes()).hexdigest()
    assert failure["gpu_preflight"]["sha256"] == hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert failure["training_completion"]["present"] is True
    assert failure["training_completion"]["status"] == "failed"
    assert (
        failure["training_completion"]["sha256"]
        == hashlib.sha256(completion.read_bytes()).hexdigest()
    )
    assert failure["runtime_lock"]["sha256"] == lock_sha
    assert failure["approved_maxtext_patch"]["sha256"] == patch_sha
    assert failure["approved_maxtext_patch"]["expected_sha256"] == patch_sha
    assert failure["bookforge_source_manifest"]["verified"] is True
    assert failure["bookforge_source_manifest"]["file_count"] == 0
    assert failure["maxtext"]["expected_revision"] == "a" * 40
    assert failure["maxtext"]["observed_revision"] == "a" * 40
    source_rows = {row["module"]: row for row in failure["maxtext"]["patched_sources"]}
    assert source_rows["maxtext.trainers.pre_train.train"]["sha256"] == train_sha
    assert source_rows["maxtext.utils.train_utils"]["sha256"] == utils_sha
    for row in (
        failure["runtime_lock"],
        failure["approved_maxtext_patch"],
        *failure["maxtext"]["patched_sources"],
        failure["bookforge_source_manifest"],
    ):
        snapshot = scratch / row["path"]
        assert snapshot.is_file()
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == row["sha256"]
