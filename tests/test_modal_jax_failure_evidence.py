# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import modal_jax_fidelity, modal_jax_image


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _packaged_source_fixture(root: Path) -> tuple[dict[str, str], dict[str, object]]:
    for local_relative, _container_relative in (
        modal_jax_image.PACKAGED_BOOKFORGE_DIRECTORIES
    ):
        (root / local_relative).mkdir(parents=True, exist_ok=True)
    _write(root / "training/kept.py", b"kept\n")
    _write(root / "training/__pycache__/ignored.pyc", b"ignored\n")
    _write(root / "deploy/worker.py", b"worker\n")
    for local_relative, _container_relative in modal_jax_image.PACKAGED_BOOKFORGE_FILES:
        _write(root / local_relative, local_relative.encode())
    manifest = modal_jax_fidelity.packaged_bookforge_source_manifest(root)
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    manifest_path = root / "source.manifest.json"
    manifest_sha256 = _write(manifest_path, payload)
    return (
        {
            "BOOKFORGE_SOURCE_MANIFEST": str(manifest_path),
            "BOOKFORGE_SOURCE_MANIFEST_SHA256": manifest_sha256,
        },
        manifest,
    )


def test_packaged_source_manifest_is_exact_and_ignores_only_build_artifacts(
    tmp_path: Path,
) -> None:
    environment, manifest = _packaged_source_fixture(tmp_path)
    paths = {str(row["path"]) for row in manifest["files"]}

    assert "training/kept.py" in paths
    assert "deploy/worker.py" in paths
    assert "training/__pycache__/ignored.pyc" not in paths
    assert manifest["file_count"] == len(manifest["files"])
    assert modal_jax_fidelity._verify_bookforge_source_manifest(
        environment, container_root=tmp_path
    )["files_sha256"] == manifest["files_sha256"]

    (tmp_path / "src/unlisted.py").write_text("drift\n")
    with pytest.raises(RuntimeError, match="inventory changed"):
        modal_jax_fidelity._verify_bookforge_source_manifest(
            environment, container_root=tmp_path
        )


def test_training_process_retains_bounded_combined_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(modal_jax_fidelity, "_FAILURE_LOG_LIMIT_BYTES", 96)
    monkeypatch.setattr(modal_jax_fidelity, "_FAILURE_LOG_TAIL_BYTES", 48)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    log_path = scratch / "failure-evidence/training-subprocess.log"
    command = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'A' * 200); "
        "os.write(2, b'ROOT-CAUSE-SENTINEL\\n'); raise SystemExit(7)",
    ]

    evidence, error = modal_jax_fidelity._run_training_process(
        command,
        environment={},
        timeout_seconds=5,
        log_path=log_path,
        scratch_directory=scratch,
    )

    assert isinstance(error, subprocess.CalledProcessError)
    assert error.returncode == 7
    assert evidence["returncode"] == 7
    assert evidence["timed_out"] is False
    log = evidence["log"]
    assert isinstance(log, dict)
    assert log["path"] == "failure-evidence/training-subprocess.log"
    assert log["streams"] == "stdout+stderr"
    assert log["stream_bytes"] == 220
    assert log["bytes"] == 96
    assert log["limit_bytes"] == 96
    assert log["truncated"] is True
    assert str(log["tail"]).endswith("ROOT-CAUSE-SENTINEL\n")
    assert log["sha256"] == hashlib.sha256(log_path.read_bytes()).hexdigest()
    assert log_path.read_bytes().endswith(b"ROOT-CAUSE-SENTINEL\n")
    streamed = capfd.readouterr()
    assert streamed.out.endswith("ROOT-CAUSE-SENTINEL\n")


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


def test_training_process_returns_spawn_failure_for_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OSError("spawn blocked")

    def fail_spawn(*args: object, **kwargs: object) -> None:
        raise original

    monkeypatch.setattr(modal_jax_fidelity.subprocess, "Popen", fail_spawn)
    evidence, error = modal_jax_fidelity._run_training_process(
        ["python3", "trainer.py"],
        environment={},
        timeout_seconds=5,
        log_path=tmp_path / "failure-evidence/training-subprocess.log",
        scratch_directory=tmp_path,
    )

    assert error is original
    assert evidence["status"] == "spawn-failed"
    assert evidence["log"] == {"present": False}


def test_log_capture_failure_does_not_replace_the_training_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_log_write(path: Path, payload: bytes) -> None:
        raise OSError("scratch unavailable")

    monkeypatch.setattr(modal_jax_fidelity, "_write_once_bytes", fail_log_write)
    evidence, error = modal_jax_fidelity._run_training_process(
        [sys.executable, "-c", "raise SystemExit(9)"],
        environment={},
        timeout_seconds=5,
        log_path=tmp_path / "failure-evidence/training-subprocess.log",
        scratch_directory=tmp_path,
    )

    assert isinstance(error, subprocess.CalledProcessError)
    assert error.returncode == 9
    assert evidence["log"]["present"] is False
    assert evidence["capture_errors"] == [
        {"type": "OSError", "message": "scratch unavailable"}
    ]


def test_runtime_provenance_binds_revision_patch_lock_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    maxtext = tmp_path / "MaxText"
    source_hashes: dict[str, str] = {}
    for module, relative, _snapshot in modal_jax_fidelity._PATCHED_MAXTEXT_SOURCES:
        source_hashes[module] = _write(maxtext / relative, module.encode())
    runtime_lock = tmp_path / "runtime.lock.json"
    _write(runtime_lock, b'{"packages":[]}\n')
    patch = tmp_path / "approved.patch"
    patch_sha = _write(patch, b"approved patch\n")
    experiment = SimpleNamespace(
        versions={"maxtext_revision": revision},
        training={"approved_maxtext_patch_sha256": patch_sha},
    )
    bookforge_root = tmp_path / "bookforge"
    source_environment, source_manifest = _packaged_source_fixture(bookforge_root)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command == ["git", "-C", str(maxtext), "rev-parse", "HEAD"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=f"{revision}\n")

    monkeypatch.setattr(modal_jax_fidelity.subprocess, "run", fake_run)
    provenance = modal_jax_fidelity._collect_training_runtime_provenance(
        experiment,
        maxtext_root=maxtext,
        environment={
            **source_environment,
            "BOOKFORGE_JAX_RUNTIME_LOCK": str(runtime_lock),
            "BOOKFORGE_MAXTEXT_APPROVED_PATCH": str(patch),
        },
        bookforge_root=bookforge_root,
    )

    assert (
        provenance["runtime_lock"]["sha256"]
        == hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
    )
    assert provenance["approved_maxtext_patch"]["sha256"] == patch_sha
    assert provenance["approved_maxtext_patch"]["expected_sha256"] == patch_sha
    assert provenance["maxtext"]["expected_revision"] == revision
    assert provenance["maxtext"]["observed_revision"] == revision
    assert provenance["bookforge_source_manifest"]["file_count"] == source_manifest[
        "file_count"
    ]
    observed_sources = {
        row["module"]: row["sha256"] for row in provenance["maxtext"]["patched_sources"]
    }
    assert observed_sources == source_hashes
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    persisted = modal_jax_fidelity._persist_runtime_provenance(
        provenance,
        run_id="outer-run",
        training_run_id="training-run",
        expected_source_manifest_sha256=source_environment[
            "BOOKFORGE_SOURCE_MANIFEST_SHA256"
        ],
        scratch_directory=scratch,
    )
    provenance_path = scratch / "runtime-provenance.json"
    source_snapshot = scratch / "bookforge-source.manifest.json"
    assert provenance_path.stat().st_mode & 0o777 == 0o400
    assert source_snapshot.stat().st_mode & 0o777 == 0o400
    assert persisted["bookforge_source_manifest"]["sha256"] == source_environment[
        "BOOKFORGE_SOURCE_MANIFEST_SHA256"
    ]
    assert (
        modal_jax_fidelity._validate_runtime_provenance(
            scratch,
            run_id="outer-run",
            training_run_id="training-run",
            expected_source_manifest_sha256=source_environment[
                "BOOKFORGE_SOURCE_MANIFEST_SHA256"
            ],
        )
        == persisted
    )
    source_snapshot.chmod(0o600)
    source_snapshot.write_text("{}\n")
    with pytest.raises(RuntimeError, match="checksum changed"):
        modal_jax_fidelity._validate_runtime_provenance(
            scratch,
            run_id="outer-run",
            training_run_id="training-run",
            expected_source_manifest_sha256=source_environment[
                "BOOKFORGE_SOURCE_MANIFEST_SHA256"
            ],
        )


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


def test_failure_persistence_never_masks_the_original_error(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    attempt = scratch / "attempt.json"
    _write(attempt, b'{}\n')
    original = RuntimeError("root cause")

    def fail_commit() -> None:
        raise OSError("volume commit failed")

    with pytest.raises(RuntimeError) as raised:
        modal_jax_fidelity._persist_failure_and_raise(
            original,
            run_id="outer-run",
            training_run_id="training-run",
            scratch_directory=scratch,
            attempt_path=attempt,
            gpu_preflight_path=scratch / "gpu-preflight.json",
            training_completion_path=scratch / "runs/training-run/completion.json",
            process_evidence=None,
            runtime_provenance=None,
            scratch_commit=fail_commit,
            failure_stage="gpu-preflight",
        )

    assert raised.value is original
    assert "volume commit failed" in "\n".join(original.__notes__)
    failure = json.loads((scratch / "failure.json").read_text())
    assert failure["failure_stage"] == "gpu-preflight"
    assert failure["process"] == {"status": "not-started"}


def test_expected_missing_stage_artifacts_do_not_create_capture_errors(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    attempt = scratch / "attempt.json"
    _write(attempt, b'{}\n')

    modal_jax_fidelity._persist_failure_envelope(
        RuntimeError("preflight failed"),
        run_id="outer-run",
        training_run_id="training-run",
        scratch_directory=scratch,
        attempt_path=attempt,
        gpu_preflight_path=scratch / "gpu-preflight.json",
        training_completion_path=scratch / "runs/training-run/completion.json",
        process_evidence=None,
        runtime_provenance=None,
        scratch_commit=lambda: None,
        failure_stage="gpu-preflight",
    )

    failure = json.loads((scratch / "failure.json").read_text())
    assert "capture_errors" not in failure
    assert failure["gpu_preflight"] == {
        "present": False,
        "path": "gpu-preflight.json",
    }
    assert failure["training_completion"] == {
        "present": False,
        "path": "runs/training-run/completion.json",
    }
