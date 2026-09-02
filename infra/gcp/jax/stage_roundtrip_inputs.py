#!/usr/bin/env python3
"""Stage one immutable HF-to-MaxText roundtrip population into Modal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from stage_inputs import build_input_manifest, canonical_bytes, public_dataset_sources

VOLUME_NAME = "bookforge-jax-fidelity-inputs"
APPROVAL_ENVIRONMENT = "BOOKFORGE_MODAL_JAX_ROUNDTRIP_STAGE_APPROVAL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_roundtrip_input_manifest(
    *,
    run_id: str,
    config: Path,
    dataset_manifest: Path,
    prepared_train: Path,
    hf_snapshot: Path,
    hf_snapshot_manifest: Path,
    tokenizer: Path,
    tokenizer_manifest: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Build the ordinary cloud population plus the manifest's public splits."""

    document, sources = build_input_manifest(
        run_id=run_id,
        config=config,
        dataset_manifest=dataset_manifest,
        prepared_train=prepared_train,
        checkpoint=hf_snapshot,
        checkpoint_manifest=hf_snapshot_manifest,
        checkpoint_receipt=None,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        require_base_orbax=False,
    )
    for relative, source in public_dataset_sources(dataset_manifest).items():
        if relative in sources:
            raise ValueError(f"duplicate staged input path: {relative}")
        sources[relative] = source
    document["purpose"] = "hf-maxtext-roundtrip-smoke"
    document["files"] = [
        {
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
        for relative, source in sorted(sources.items())
    ]
    return document, sources


def approval_token(run_id: str, manifest_sha256: str) -> str:
    return f"APPROVE_MODAL_JAX_ROUNDTRIP_STAGE:{VOLUME_NAME}:{run_id}:{manifest_sha256}"


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
    try:
        entries = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Modal volume listing was not JSON") from error
    if not isinstance(entries, list):
        raise RuntimeError("Modal volume listing was not a list")
    result: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict):
            value = next(
                (
                    candidate
                    for key in ("path", "name", "filename")
                    if isinstance((candidate := entry.get(key)), str)
                ),
                "",
            )
        else:
            raise RuntimeError("Modal volume listing contains an invalid entry")
        if value.strip("/"):
            result.add(value.strip("/").split("/", 1)[0])
    return result


def stage_roundtrip_inputs(
    *,
    run_id: str,
    manifest: dict[str, object],
    sources: dict[str, Path],
) -> dict[str, object]:
    encoded = canonical_bytes(manifest)
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    token = approval_token(run_id, manifest_sha256)
    if os.environ.get(APPROVAL_ENVIRONMENT) != token:
        raise RuntimeError("exact Modal roundtrip staging approval token is required")
    if run_id in _top_level_entries():
        raise RuntimeError("Modal input run prefix already contains state")
    for relative, source in sorted(sources.items()):
        _run(["modal", "volume", "put", VOLUME_NAME, str(source), f"/{run_id}/{relative}"])
    with tempfile.TemporaryDirectory(prefix="bookforge-modal-roundtrip-stage-") as raw:
        local_manifest = Path(raw) / "inputs.manifest.json"
        local_manifest.write_bytes(encoded)
        _run(
            [
                "modal",
                "volume",
                "put",
                VOLUME_NAME,
                str(local_manifest),
                f"/{run_id}/inputs.manifest.json",
            ]
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
            ]
        )
        if (
            not receipt.is_file()
            or hashlib.sha256(receipt.read_bytes()).hexdigest() != manifest_sha256
        ):
            raise RuntimeError("staged Modal roundtrip manifest failed read-back verification")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--hf-snapshot", type=Path, required=True)
    parser.add_argument("--hf-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite staging evidence: {args.output}")
    manifest, sources = build_roundtrip_input_manifest(
        run_id=args.run_id,
        config=args.config,
        dataset_manifest=args.dataset_manifest,
        prepared_train=args.prepared_train,
        hf_snapshot=args.hf_snapshot,
        hf_snapshot_manifest=args.hf_snapshot_manifest,
        tokenizer=args.tokenizer,
        tokenizer_manifest=args.tokenizer_manifest,
    )
    manifest_sha256 = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    if args.execute:
        result = stage_roundtrip_inputs(run_id=args.run_id, manifest=manifest, sources=sources)
    else:
        result = {
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
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
