#!/usr/bin/env python3
"""Plan one immutable Modal v3 recovery clone without remote mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from infra.gcp.jax.full_input_v3 import (  # noqa: E402
    build_full_training_manifest,
    clone_approval_token,
    manifest_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required v3 clone input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"required v3 clone input is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"required v3 clone input must be a JSON object: {path}")
    return value


def build_plan(
    *,
    source_input_manifest_path: Path,
    roundtrip_release: Path,
    overlay_manifest_path: Path,
    target_run_id: str,
) -> dict[str, Any]:
    source_manifest = _json_object(source_input_manifest_path)
    completion_path = roundtrip_release / "completion.json"
    receipt_path = roundtrip_release / "evidence/base-orbax.receipt.json"
    base_manifest_path = roundtrip_release / "base-orbax.manifest.json"
    completion = _json_object(completion_path)
    receipt = _json_object(receipt_path)
    base_manifest = _json_object(base_manifest_path)
    overlay = _json_object(overlay_manifest_path)
    source_run_id = source_manifest.get("run_id")
    if not isinstance(source_run_id, str):
        raise ValueError("source input manifest has no run ID")
    source_manifest_sha = _sha256(source_input_manifest_path)
    completion_sha = _sha256(completion_path)
    overlay_sha = _sha256(overlay_manifest_path)
    target_manifest, tokenizer_paths, overlay_paths = build_full_training_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_manifest_sha,
        source_input_manifest=source_manifest,
        roundtrip_completion_sha256=completion_sha,
        roundtrip_completion=completion,
        base_receipt=receipt,
        base_manifest=base_manifest,
        overlay_manifest_sha256=overlay_sha,
        overlay_manifest=overlay,
        target_run_id=target_run_id,
    )
    target_sha = manifest_sha256(target_manifest)
    return {
        "schema_version": "1.0",
        "producer": "bookforge-modal-jax-full-input-v3-recovery-clone-planner",
        "status": "plan-only",
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_manifest_sha,
        "roundtrip_completion_sha256": completion_sha,
        "overlay_manifest_sha256": overlay_sha,
        "target_run_id": target_run_id,
        "target_manifest_sha256": target_sha,
        "target_manifest": target_manifest,
        "source_tokenizer_allowlist": list(tokenizer_paths),
        "overlay_allowlist": list(overlay_paths),
        "approval_token": clone_approval_token(
            source_run_id=source_run_id,
            source_input_manifest_sha256=source_manifest_sha,
            roundtrip_completion_sha256=completion_sha,
            overlay_manifest_sha256=overlay_sha,
            target_run_id=target_run_id,
            target_manifest_sha256=target_sha,
        ),
        "prepared_mapping": {
            "source": "recovery/overfit-canary.train.jsonl",
            "destination": "prepared/train.jsonl",
        },
        "cached_model_upload_bytes": 0,
        "remote_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input-manifest", type=Path, required=True)
    parser.add_argument("--roundtrip-release", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite v3 clone plan: {args.output}")
    plan = build_plan(
        source_input_manifest_path=args.source_input_manifest,
        roundtrip_release=args.roundtrip_release,
        overlay_manifest_path=args.overlay_manifest,
        target_run_id=args.target_run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
