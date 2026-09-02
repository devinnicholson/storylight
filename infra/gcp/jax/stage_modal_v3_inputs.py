#!/usr/bin/env python3
"""Plan or stage the public immutable overlay for a Modal v3 recovery clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from infra.gcp.jax.full_input_v3 import (
    OVERLAY_PATHS,
    OVERLAY_PREFIX,
    build_overlay_manifest,
    canonical_bytes,
    manifest_sha256,
    overlay_approval_token,
)

VOLUME_NAME = "bookforge-jax-fidelity-inputs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"v3 staging input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"v3 staging input is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"v3 staging JSON must contain one object: {path}")
    return value


def build_stage_plan(
    *,
    target_run_id: str,
    sources: dict[str, Path],
    source_tokenizer_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Return the recovery overlay, normalized sources, and a no-write plan."""

    if set(sources) != set(OVERLAY_PATHS):
        raise ValueError("v3 staging sources must match the approved public paths")
    normalized: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for relative in OVERLAY_PATHS:
        supplied = sources[relative]
        if supplied.is_symlink() or not supplied.is_file():
            raise ValueError(f"v3 staging source is not a regular file: {supplied}")
        path = supplied.resolve(strict=True)
        normalized[relative] = path
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = build_overlay_manifest(
        target_run_id=target_run_id,
        files=rows,
        config=_json_object(normalized["config.json"]),
        dataset_manifest=_json_object(normalized["dataset/manifest.json"]),
        recovery_manifest=_json_object(normalized["recovery/inputs.manifest.json"]),
        source_tokenizer_manifest_sha256=source_tokenizer_manifest_sha256,
    )
    digest = manifest_sha256(manifest)
    plan = {
        "schema_version": "1.0",
        "producer": "bookforge-modal-jax-v3-recovery-overlay-staging-plan",
        "status": "plan-only",
        "volume": VOLUME_NAME,
        "target_run_id": target_run_id,
        "prefix": f"{OVERLAY_PREFIX}/{target_run_id}",
        "overlay_manifest_sha256": digest,
        "overlay_manifest": manifest,
        "approval_token": overlay_approval_token(
            target_run_id=target_run_id, overlay_manifest_sha256=digest
        ),
        "host_upload_bytes": sum(row["bytes"] for row in rows),
        "host_upload_paths": list(OVERLAY_PATHS),
        "cached_model_upload_bytes": 0,
        "remote_mutation": False,
    }
    return manifest, normalized, plan


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _listed_names(stdout: str) -> set[str]:
    try:
        entries = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Modal v3 overlay listing was not JSON") from error
    if not isinstance(entries, list):
        raise RuntimeError("Modal v3 overlay listing was not a list")
    result: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            raw = entry
        elif isinstance(entry, dict):
            normalized = {str(key).casefold(): value for key, value in entry.items()}
            raw = next(
                (
                    value
                    for key in ("path", "name", "filename")
                    if isinstance((value := normalized.get(key)), str)
                ),
                "",
            )
        else:
            raise RuntimeError("Modal v3 overlay listing contains an invalid entry")
        parts = raw.strip("/").split("/") if raw.strip("/") else []
        if parts:
            result.add(parts[-1])
    return result


def stage_overlay(
    *,
    target_run_id: str,
    manifest: dict[str, Any],
    sources: dict[str, Path],
    approval_token_value: str,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    """Upload verified public files and publish the recovery manifest last."""

    digest = manifest_sha256(manifest)
    expected = overlay_approval_token(target_run_id=target_run_id, overlay_manifest_sha256=digest)
    if approval_token_value != expected:
        raise ValueError("v3 recovery overlay staging approval token is not exact")
    if (
        manifest.get("target_run_id") != target_run_id
        or manifest.get("prefix") != f"{OVERLAY_PREFIX}/{target_run_id}"
        or set(sources) != set(OVERLAY_PATHS)
    ):
        raise ValueError("v3 recovery overlay changed after planning")
    prefix = f"/{OVERLAY_PREFIX}/{target_run_id}"
    root_listing = runner(["modal", "volume", "ls", VOLUME_NAME, "/", "--json"])
    if OVERLAY_PREFIX in _listed_names(root_listing.stdout):
        listing = runner(["modal", "volume", "ls", VOLUME_NAME, f"/{OVERLAY_PREFIX}", "--json"])
        if target_run_id in _listed_names(listing.stdout):
            raise RuntimeError("v3 recovery overlay already contains state and cannot be retried")
    rows = {row["path"]: row for row in manifest["files"]}
    for relative in OVERLAY_PATHS:
        source = sources[relative]
        row = rows[relative]
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != row["bytes"]
            or _sha256(source) != row["sha256"]
        ):
            raise RuntimeError(f"v3 overlay source changed after planning: {relative}")
        runner(["modal", "volume", "put", VOLUME_NAME, str(source), f"{prefix}/{relative}"])
    with tempfile.TemporaryDirectory(prefix="bookforge-v3-recovery-overlay-") as raw:
        local = Path(raw) / "staging.manifest.json"
        descriptor = os.open(local, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        remote = f"{prefix}/staging.manifest.json"
        runner(["modal", "volume", "put", VOLUME_NAME, str(local), remote])
        readback = Path(raw) / "readback.manifest.json"
        runner(["modal", "volume", "get", VOLUME_NAME, remote, str(readback)])
        if _sha256(readback) != digest or readback.read_bytes() != canonical_bytes(manifest):
            raise RuntimeError("v3 overlay manifest failed read-back verification")
    return {
        "schema_version": "1.0",
        "producer": "bookforge-modal-jax-v3-recovery-overlay-stager",
        "status": "staged",
        "volume": VOLUME_NAME,
        "target_run_id": target_run_id,
        "prefix": f"{OVERLAY_PREFIX}/{target_run_id}",
        "overlay_manifest_sha256": digest,
        "files": len(OVERLAY_PATHS),
        "bytes": sum(int(row["bytes"]) for row in manifest["files"]),
        "manifest_uploaded_last": True,
        "cached_model_upload_bytes": 0,
        "overwrite_enabled": False,
    }


def _sources(args: argparse.Namespace) -> dict[str, Path]:
    recovery = args.recovery_directory
    return {
        "config.json": args.config,
        "dataset/manifest.json": args.dataset_manifest,
        "dataset/train.jsonl": args.train,
        "dataset/development.jsonl": args.development,
        "recovery/inputs.manifest.json": recovery / "inputs.manifest.json",
        "recovery/overfit-canary.source.jsonl": recovery / "overfit-canary.source.jsonl",
        "recovery/overfit-canary.train.jsonl": recovery / "overfit-canary.train.jsonl",
        "recovery/public-probe.source.jsonl": recovery / "public-probe.source.jsonl",
        "recovery/public-probe.teacher.jsonl": recovery / "public-probe.teacher.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--recovery-directory", type=Path, required=True)
    parser.add_argument("--source-tokenizer-manifest-sha256", required=True)
    parser.add_argument("--approval-token")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest, sources, plan = build_stage_plan(
        target_run_id=args.target_run_id,
        sources=_sources(args),
        source_tokenizer_manifest_sha256=args.source_tokenizer_manifest_sha256,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.execute:
        if args.approval_token is None:
            parser.error("--execute requires --approval-token")
        print(
            json.dumps(
                stage_overlay(
                    target_run_id=args.target_run_id,
                    manifest=manifest,
                    sources=sources,
                    approval_token_value=args.approval_token,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
