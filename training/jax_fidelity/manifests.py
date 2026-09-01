"""Append-only run and completion evidence for the offline training campaign."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookforge.fidelity_lineage import stable_run_id as _stable_run_id

from .integrity import canonical_json_bytes, sha256_file


class ManifestError(RuntimeError):
    """Run evidence is inconsistent, incomplete, or would be overwritten."""


def stable_run_id(*, stage: str, config_sha256: str, dataset_manifest_sha256: str) -> str:
    return _stable_run_id(
        stage=stage,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_once(path: Path, document: Mapping[str, Any]) -> str:
    """Create an immutable JSON file, allowing only byte-identical replays."""

    encoded = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ManifestError(f"could not inspect existing manifest {path}: {error}") from error
        if existing != encoded:
            raise ManifestError(f"refusing to replace immutable manifest: {path}") from None
        return sha256_file(path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def start_run(
    run_directory: Path | str,
    *,
    run_id: str,
    stage: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    command: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the run contract before any model-heavy operation begins."""

    if not run_id or "/" in run_id or ".." in run_id:
        raise ManifestError("run_id must be one safe path component")
    run_dir = Path(run_directory).resolve() / run_id
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        raise ManifestError("completed run IDs are terminal and may not be reused")
    document = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage": stage,
        "status": "planned",
        "created_at": _timestamp(),
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "command": list(command),
        "metadata": dict(metadata or {}),
    }
    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # created_at is evidence, not part of idempotency comparison.
        document["created_at"] = existing.get("created_at")
    _write_once(manifest_path, document)
    return manifest_path


def complete_run(
    run_directory: Path | str,
    *,
    run_id: str,
    status: str,
    artifacts: Sequence[Path | str],
    evidence: Mapping[str, Any],
) -> Path:
    """Verify outputs, then write completion.json as the final operation."""

    if status not in {"succeeded", "failed", "rejected"}:
        raise ManifestError("terminal status must be succeeded, failed, or rejected")
    run_dir = Path(run_directory).resolve() / run_id
    run_manifest_path = run_dir / "run.json"
    if not run_manifest_path.is_file():
        raise ManifestError("run manifest must exist before completion")
    artifact_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise ManifestError(f"completion artifact is not a regular file: {path}")
        artifact_rows.append(
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    document = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "completed_at": _timestamp(),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "artifacts": artifact_rows,
        "evidence": dict(evidence),
    }
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        document["completed_at"] = existing.get("completed_at")
    _write_once(completion_path, document)
    return completion_path
