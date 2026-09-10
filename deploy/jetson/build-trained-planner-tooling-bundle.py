#!/usr/bin/env python3
"""Build a checksum-bound Jetson acceptance-tooling directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

TOOL_FILES = (
    "build-gemma4-tensorrt-edge-engine.sh",
    "build-trained-planner-candidate.sh",
    "download-gcs-fidelity-tensorrt-candidate.py",
    "emit-accepted-planner-identity.sh",
    "install-trained-planner-candidate.sh",
    "install-trained-planner-tooling.py",
    "preflight-trained-planner-acceptance.py",
    "promote-trained-planner.sh",
    "record-trained-planner-cold-start.py",
    "record-trained-planner-terminal-evidence.py",
    "record-baseline-retention.sh",
    "rollback-trained-planner.sh",
    "run-tensorrt-edge-server.sh",
    "run-trained-planner-candidate-evaluation.sh",
    "run-trained-planner-shadow.sh",
    "trained-planner-candidate-evaluation.py",
    "trained-planner-shadow-evidence.py",
    "wait-tensorrt-planner-ready.sh",
    "systemd/storylight-trained-planner-candidate@.service",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-uncommitted-source", action="store_true")
    return parser.parse_args()


def source_commit_verified(source_root: Path, source_commit: str) -> tuple[bool, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *TOOL_FILES],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    verified = (
        head.returncode == 0
        and head.stdout.strip() == source_commit
        and status.returncode == 0
        and not status.stdout.strip()
    )
    if verified:
        return True, "source files are clean at the declared Git commit"
    detail = (
        f"HEAD={head.stdout.strip() or '<unavailable>'}; "
        f"source_status={status.stdout.strip() or '<command-failed>'}"
    )
    return False, detail


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_commit):
        raise SystemExit("--source-commit must be a lowercase 40-character Git SHA")
    source_root = args.source_root.resolve()
    commit_verified, commit_detail = source_commit_verified(source_root, args.source_commit)
    if not commit_verified and not args.allow_uncommitted_source:
        raise SystemExit(f"tooling source is not clean at --source-commit: {commit_detail}")
    records: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for relative in TOOL_FILES:
        path = source_root / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise SystemExit(f"tooling source must be a regular non-symlink: {relative}")
        payload = path.read_bytes()
        if not payload:
            raise SystemExit(f"tooling source is empty: {relative}")
        mode = 0o444 if relative.startswith("systemd/") else 0o555
        payloads[relative] = payload
        records.append(
            {"path": relative, "bytes": len(payload), "mode": mode, "sha256": sha256_bytes(payload)}
        )
    source_manifest_sha256 = sha256_bytes(canonical(records))
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "storylight-jetson-trained-planner-tooling",
        "source_commit": args.source_commit,
        "source_commit_verified": commit_verified,
        "source_manifest_sha256": source_manifest_sha256,
        "files": records,
    }
    manifest_bytes = canonical(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, payload in payloads.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            os.chmod(target, 0o444 if relative.startswith("systemd/") else 0o555)
        (temporary / "tooling.manifest.json").write_bytes(manifest_bytes)
        os.chmod(temporary / "tooling.manifest.json", 0o444)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "bundle": str(output),
                "manifest_sha256": manifest_sha256,
                "source_commit": args.source_commit,
                "source_commit_verified": commit_verified,
                "source_manifest_sha256": source_manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
