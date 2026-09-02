#!/usr/bin/env python3
"""Stage one immutable JAX input population into the named Modal volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from stage_inputs import (
    build_full_training_input_manifest,
    canonical_bytes,
    verify_full_training_sources,
)

VOLUME_NAME = "bookforge-jax-fidelity-inputs"
APPROVAL_ENVIRONMENT = "BOOKFORGE_MODAL_JAX_STAGE_APPROVAL"
Run = Callable[..., subprocess.CompletedProcess[str]]


def approval_token(run_id: str, manifest_sha256: str) -> str:
    return f"APPROVE_MODAL_JAX_STAGE:{VOLUME_NAME}:{run_id}:{manifest_sha256}"


def _run(command: list[str], *, runner: Run = subprocess.run) -> subprocess.CompletedProcess[str]:
    return runner(command, check=True, capture_output=True, text=True, timeout=600)


def stage_inputs(
    *,
    run_id: str,
    manifest: dict[str, object],
    sources: dict[str, Path],
    runner: Run = subprocess.run,
) -> dict[str, object]:
    verify_full_training_sources(manifest, sources)
    encoded = canonical_bytes(manifest)
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    token = approval_token(run_id, manifest_sha256)
    if os.environ.get(APPROVAL_ENVIRONMENT) != token:
        raise RuntimeError("exact Modal input-staging approval token is required")
    listing = _run(["modal", "volume", "ls", VOLUME_NAME, "/", "--json"], runner=runner)
    try:
        entries = json.loads(listing.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Modal volume listing was not JSON") from error
    if not isinstance(entries, list):
        raise RuntimeError("Modal volume listing was not a list")
    top_level: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict):
            value = ""
            for key in ("path", "name", "filename"):
                candidate = entry.get(key)
                if isinstance(candidate, str):
                    value = candidate
                    break
        else:
            raise RuntimeError("Modal volume listing contains an invalid entry")
        if value.strip("/"):
            top_level.add(value.strip("/").split("/", 1)[0])
    if run_id in top_level:
        raise RuntimeError("Modal input run prefix already contains state")
    for relative, source in sorted(sources.items()):
        _run(
            ["modal", "volume", "put", VOLUME_NAME, str(source), f"/{run_id}/{relative}"],
            runner=runner,
        )
    with tempfile.TemporaryDirectory(prefix="bookforge-modal-stage-") as raw:
        manifest_path = Path(raw) / "inputs.manifest.json"
        manifest_path.write_bytes(encoded)
        _run(
            [
                "modal",
                "volume",
                "put",
                VOLUME_NAME,
                str(manifest_path),
                f"/{run_id}/inputs.manifest.json",
            ],
            runner=runner,
        )
        receipt = Path(raw) / "receipt.json"
        _run(
            [
                "modal",
                "volume",
                "get",
                VOLUME_NAME,
                f"/{run_id}/inputs.manifest.json",
                str(receipt),
            ],
            runner=runner,
        )
        if not receipt.is_file():
            raise RuntimeError("staged Modal input manifest was not readable")
        receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
        if receipt_sha256 != manifest_sha256:
            raise RuntimeError("staged Modal input manifest failed read-back verification")
    return {
        "schema_version": "1.0",
        "status": "staged",
        "run_id": run_id,
        "volume": VOLUME_NAME,
        "input_manifest_sha256": manifest_sha256,
        "files": len(sources),
        "manifest_uploaded_last": True,
        "overwrite_enabled": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging evidence: {args.output}")
    manifest, sources = build_full_training_input_manifest(
        run_id=args.run_id,
        config=args.config,
        dataset_manifest=args.dataset_manifest,
        prepared_train=args.prepared_train,
        checkpoint=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_receipt=args.checkpoint_receipt,
        tokenizer=args.tokenizer,
        tokenizer_manifest=args.tokenizer_manifest,
    )
    manifest_sha256 = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    if args.execute:
        document = stage_inputs(run_id=args.run_id, manifest=manifest, sources=sources)
    else:
        document = {
            "schema_version": "1.0",
            "status": "plan-only",
            "run_id": args.run_id,
            "volume": VOLUME_NAME,
            "input_manifest_sha256": manifest_sha256,
            "approval_token": approval_token(args.run_id, manifest_sha256),
            "manifest": manifest,
            "remote_mutation": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
