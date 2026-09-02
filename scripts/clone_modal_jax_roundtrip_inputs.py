#!/usr/bin/env python3
"""Clone a verified public Modal roundtrip population under a new config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

VOLUME_NAME = "bookforge-jax-fidelity-inputs"
APPROVAL_ENVIRONMENT = "BOOKFORGE_MODAL_JAX_ROUNDTRIP_CLONE_APPROVAL"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{rendered}\n".encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError("input manifest contains an invalid path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise ValueError(f"input manifest contains an unsafe path: {raw!r}")
    if any(part.lower() == "hidden" or "hidden" in part.lower() for part in path.parts):
        raise ValueError("hidden data may not be cloned into Modal")
    return raw


def cloned_manifest(
    source_bytes: bytes,
    *,
    source_manifest_sha256: str,
    source_run_id: str,
    target_run_id: str,
    config: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a source manifest and bind the clone to the new config."""

    if _SHA256.fullmatch(source_manifest_sha256) is None:
        raise ValueError("source manifest SHA-256 is invalid")
    if _sha256_bytes(source_bytes) != source_manifest_sha256:
        raise ValueError("source input manifest SHA-256 changed")
    if _RUN_ID.fullmatch(source_run_id) is None or _RUN_ID.fullmatch(target_run_id) is None:
        raise ValueError("roundtrip run ID is invalid")
    if source_run_id == target_run_id:
        raise ValueError("source and target run IDs must differ")
    try:
        document = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source input manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("source input manifest must contain an object")
    if (
        document.get("schema_version") != "1.0"
        or document.get("status") != "complete"
        or document.get("purpose") != "hf-maxtext-roundtrip-smoke"
        or document.get("run_id") != source_run_id
    ):
        raise ValueError("source input manifest identity changed")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source input manifest has no files")

    config_bytes = config.read_bytes()
    updated_rows: list[dict[str, object]] = []
    source_paths: list[str] = []
    seen: set[str] = set()
    config_count = 0
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ValueError("source input manifest contains an invalid file row")
        relative = _safe_relative_path(raw_row.get("path"))
        if relative in seen:
            raise ValueError(f"source input manifest repeats {relative}")
        seen.add(relative)
        size = raw_row.get("bytes")
        sha = raw_row.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(sha, str):
            raise ValueError(f"source input manifest metadata is invalid for {relative}")
        if _SHA256.fullmatch(sha) is None:
            raise ValueError(f"source input manifest SHA-256 is invalid for {relative}")
        if relative == "config.json":
            config_count += 1
            updated_rows.append(
                {"path": relative, "bytes": len(config_bytes), "sha256": _sha256(config)}
            )
        else:
            source_paths.append(relative)
            updated_rows.append({"path": relative, "bytes": size, "sha256": sha})
    if config_count != 1:
        raise ValueError("source input manifest must declare exactly one config.json")

    cloned = dict(document)
    cloned["run_id"] = target_run_id
    cloned["files"] = updated_rows
    return cloned, source_paths


def approval_token(
    source_run_id: str,
    target_run_id: str,
    source_manifest_sha256: str,
    target_manifest_sha256: str,
) -> str:
    return (
        f"APPROVE_MODAL_JAX_ROUNDTRIP_CLONE:{VOLUME_NAME}:{source_run_id}:"
        f"{target_run_id}:{source_manifest_sha256}:{target_manifest_sha256}"
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _top_level_entries() -> set[str]:
    completed = _run(["modal", "volume", "ls", VOLUME_NAME, "/", "--json"])
    entries = json.loads(completed.stdout)
    if not isinstance(entries, list):
        raise RuntimeError("Modal volume listing was not a list")
    result: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            raw = entry
        elif isinstance(entry, dict):
            raw = next(
                (
                    value
                    for key in ("path", "name", "filename")
                    if isinstance((value := entry.get(key)), str)
                ),
                "",
            )
        else:
            raise RuntimeError("Modal volume listing contains an invalid entry")
        if raw.strip("/"):
            result.add(raw.strip("/").split("/", 1)[0])
    return result


def execute_clone(
    *,
    source_run_id: str,
    target_run_id: str,
    source_manifest_sha256: str,
    target_manifest: dict[str, Any],
    source_paths: list[str],
    config: Path,
) -> dict[str, object]:
    encoded = _canonical_json_bytes(target_manifest)
    target_manifest_sha256 = _sha256_bytes(encoded)
    expected = approval_token(
        source_run_id,
        target_run_id,
        source_manifest_sha256,
        target_manifest_sha256,
    )
    if os.environ.get(APPROVAL_ENVIRONMENT) != expected:
        raise RuntimeError("exact Modal roundtrip clone approval token is required")
    if target_run_id in _top_level_entries():
        raise RuntimeError("target Modal input prefix already contains state")

    for relative in source_paths:
        _run(
            [
                "modal",
                "volume",
                "cp",
                VOLUME_NAME,
                f"/{source_run_id}/{relative}",
                f"/{target_run_id}/{relative}",
            ]
        )
    _run(
        [
            "modal",
            "volume",
            "put",
            VOLUME_NAME,
            str(config),
            f"/{target_run_id}/config.json",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="bookforge-modal-roundtrip-clone-") as raw:
        manifest_path = Path(raw) / "inputs.manifest.json"
        manifest_path.write_bytes(encoded)
        _run(
            [
                "modal",
                "volume",
                "put",
                VOLUME_NAME,
                str(manifest_path),
                f"/{target_run_id}/inputs.manifest.json",
            ]
        )
        receipt = Path(raw) / "receipt.json"
        _run(
            [
                "modal",
                "volume",
                "get",
                VOLUME_NAME,
                f"/{target_run_id}/inputs.manifest.json",
                str(receipt),
            ]
        )
        if not receipt.is_file() or _sha256(receipt) != target_manifest_sha256:
            raise RuntimeError("cloned Modal manifest failed read-back verification")
    return {
        "schema_version": "1.0",
        "status": "staged",
        "source_run_id": source_run_id,
        "run_id": target_run_id,
        "volume": VOLUME_NAME,
        "source_input_manifest_sha256": source_manifest_sha256,
        "input_manifest_sha256": target_manifest_sha256,
        "files": len(target_manifest["files"]),
        "manifest_uploaded_last": True,
        "hidden_data_uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--source-input-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite clone evidence: {args.output}")

    with tempfile.TemporaryDirectory(prefix="bookforge-modal-roundtrip-source-") as raw:
        source_manifest_path = Path(raw) / "inputs.manifest.json"
        _run(
            [
                "modal",
                "volume",
                "get",
                VOLUME_NAME,
                f"/{args.source_run_id}/inputs.manifest.json",
                str(source_manifest_path),
            ]
        )
        target_manifest, source_paths = cloned_manifest(
            source_manifest_path.read_bytes(),
            source_manifest_sha256=args.source_input_manifest_sha256,
            source_run_id=args.source_run_id,
            target_run_id=args.target_run_id,
            config=args.config,
        )

    encoded = _canonical_json_bytes(target_manifest)
    target_manifest_sha256 = _sha256_bytes(encoded)
    if args.execute:
        result = execute_clone(
            source_run_id=args.source_run_id,
            target_run_id=args.target_run_id,
            source_manifest_sha256=args.source_input_manifest_sha256,
            target_manifest=target_manifest,
            source_paths=source_paths,
            config=args.config,
        )
    else:
        result = {
            "schema_version": "1.0",
            "status": "plan-only",
            "source_run_id": args.source_run_id,
            "run_id": args.target_run_id,
            "volume": VOLUME_NAME,
            "source_input_manifest_sha256": args.source_input_manifest_sha256,
            "input_manifest_sha256": target_manifest_sha256,
            "approval_token": approval_token(
                args.source_run_id,
                args.target_run_id,
                args.source_input_manifest_sha256,
                target_manifest_sha256,
            ),
            "files": len(target_manifest["files"]),
            "remote_mutation": False,
            "hidden_data_uploaded": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
