"""Build a portable, immutable package from one completed training run."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .integrity import artifact_manifest, canonical_json_bytes, sha256_file


class RemoteReleaseError(RuntimeError):
    """Training output cannot be represented as a portable release."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteReleaseError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RemoteReleaseError(f"{path} must contain a JSON object")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def package_training_release(
    *,
    output_directory: Path | str,
    run_directory: Path | str,
    training_run_id: str,
    runtime_lock: Path | str,
    destination: Path | str,
) -> dict[str, Any]:
    """Copy a completed run into a location-independent, write-once package."""

    outputs = Path(output_directory).resolve()
    run_root = Path(run_directory).resolve() / training_run_id
    lock = Path(runtime_lock).resolve()
    target = Path(destination).resolve()
    if target.exists() or target.is_symlink():
        raise RemoteReleaseError(f"portable release already exists: {target}")
    run_path = run_root / "run.json"
    completion_path = run_root / "completion.json"
    if not run_path.is_file() or run_path.is_symlink():
        raise RemoteReleaseError("training run manifest is missing or unsafe")
    if not completion_path.is_file() or completion_path.is_symlink():
        raise RemoteReleaseError("training completion is missing or unsafe")
    if not lock.is_file() or lock.is_symlink():
        raise RemoteReleaseError("runtime lock is missing or unsafe")

    run = _json_object(run_path)
    completion = _json_object(completion_path)
    if (
        run.get("run_id") != training_run_id
        or completion.get("run_id") != training_run_id
        or completion.get("status") != "succeeded"
        or completion.get("run_manifest_sha256") != sha256_file(run_path)
    ):
        raise RemoteReleaseError("training run and completion lineage is inconsistent")

    adapter_manifest = artifact_manifest(outputs)
    declared = {row["path"]: row for row in adapter_manifest["files"]}
    portable_rows: list[dict[str, Any]] = []
    completion_rows = completion.get("artifacts")
    if not isinstance(completion_rows, list) or not completion_rows:
        raise RemoteReleaseError("training completion has no artifacts")
    seen: set[str] = set()
    for row in completion_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RemoteReleaseError("training completion artifact row is malformed")
        source = Path(row["path"])
        if not source.is_absolute():
            raise RemoteReleaseError("source training completion paths must be absolute")
        try:
            relative = source.resolve().relative_to(outputs)
        except ValueError as error:
            raise RemoteReleaseError("training artifact escaped the output directory") from error
        portable = relative.as_posix()
        if portable in seen or declared.get(portable) != {
            "path": portable,
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
        }:
            raise RemoteReleaseError("training completion does not match adapter bytes")
        seen.add(portable)
        portable_rows.append(dict(declared[portable]))
    if seen != set(declared):
        raise RemoteReleaseError("training completion does not declare every adapter artifact")

    portable_completion = dict(completion)
    portable_completion["artifacts"] = portable_rows
    evidence = portable_completion.get("evidence")
    runtime_evidence = evidence.get("runtime_lock") if isinstance(evidence, dict) else None
    if not isinstance(runtime_evidence, dict) or runtime_evidence.get("sha256") != sha256_file(
        lock
    ):
        raise RemoteReleaseError("training completion is not bound to the runtime lock")
    evidence = dict(evidence)
    portable_runtime_evidence = dict(runtime_evidence)
    portable_runtime_evidence["path"] = "runtime.lock.json"
    evidence["runtime_lock"] = portable_runtime_evidence
    portable_completion["evidence"] = evidence

    target.mkdir(parents=True, exist_ok=False)
    try:
        adapter = target / "adapter"
        shutil.copytree(outputs, adapter, symlinks=False)
        for path in adapter.rglob("*"):
            if path.is_symlink():
                raise RemoteReleaseError("adapter output contains a symbolic link")
        (target / "training").mkdir()
        shutil.copyfile(run_path, target / "training" / "run.json", follow_symlinks=False)
        _write_once(target / "training" / "completion.json", portable_completion)
        _write_once(target / "adapter.manifest.json", adapter_manifest)
        shutil.copyfile(lock, target / "runtime.lock.json", follow_symlinks=False)
        package_manifest = artifact_manifest(target)
        _write_once(target / "package.manifest.json", package_manifest)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {
        "schema_version": "1.0",
        "status": "succeeded",
        "training_run_id": training_run_id,
        "package_manifest_sha256": sha256_file(target / "package.manifest.json"),
        "adapter_manifest_sha256": sha256_file(target / "adapter.manifest.json"),
        "training_run_sha256": sha256_file(target / "training" / "run.json"),
        "training_completion_sha256": sha256_file(target / "training" / "completion.json"),
        "runtime_lock_sha256": sha256_file(target / "runtime.lock.json"),
    }
