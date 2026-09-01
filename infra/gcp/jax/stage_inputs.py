#!/usr/bin/env python3
"""Build and optionally upload one checksum-bound Vertex input population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ID = "your-gcp-project"
APPROVAL_ENVIRONMENT = "BOOKFORGE_GCP_JAX_STAGE_APPROVAL"
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"artifact directory is not a regular directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"artifact contains a symbolic link: {path}")
        yield path


def build_input_manifest(
    *,
    run_id: str,
    config: Path,
    dataset_manifest: Path,
    prepared_train: Path,
    checkpoint: Path,
    checkpoint_manifest: Path,
    tokenizer: Path,
    tokenizer_manifest: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must be an immutable lowercase slug")
    sources = {
        "config.json": config,
        "dataset/manifest.json": dataset_manifest,
        "prepared/train.jsonl": prepared_train,
        "checkpoint.manifest.json": checkpoint_manifest,
        "tokenizer.manifest.json": tokenizer_manifest,
    }
    for prefix, root in (("checkpoint", checkpoint), ("tokenizer", tokenizer)):
        sources.update(
            (f"{prefix}/{path.relative_to(root).as_posix()}", path) for path in _files(root)
        )
    entries: list[dict[str, object]] = []
    for relative, source in sorted(sources.items()):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"input is not a regular file: {source}")
        entries.append(
            {
                "path": relative,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    document: dict[str, object] = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": run_id,
        "status": "complete",
        "files": entries,
    }
    return document, sources


def canonical_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def approval_token(run_id: str, manifest_sha256: str) -> str:
    return f"APPROVE_GCP_JAX_STAGE:{run_id}:{manifest_sha256}"


def _gcs_location(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("destination must be a non-root gs:// URI")
    return parsed.netloc, parsed.path.strip("/")


def upload_inputs(
    *,
    destination: str,
    manifest: dict[str, object],
    sources: dict[str, Path],
) -> None:
    from google.cloud import storage

    bucket_name, prefix = _gcs_location(destination)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    bucket.reload()
    if bucket.iam_configuration.public_access_prevention != "enforced":
        raise RuntimeError("input bucket must enforce public-access prevention")
    if bucket.iam_configuration.uniform_bucket_level_access_enabled is not True:
        raise RuntimeError("input bucket must use uniform bucket-level access")
    if list(client.list_blobs(bucket, prefix=f"{prefix}/", max_results=1)):
        raise RuntimeError("input run prefix already contains objects")
    for relative, source in sorted(sources.items()):
        bucket.blob(f"{prefix}/{relative}").upload_from_filename(
            source,
            if_generation_match=0,
        )
    bucket.blob(f"{prefix}/inputs.manifest.json").upload_from_string(
        canonical_bytes(manifest),
        content_type="application/json",
        if_generation_match=0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging evidence: {args.output}")
    _, prefix = _gcs_location(args.destination)
    if not prefix.endswith(f"/inputs/{args.run_id}"):
        raise ValueError("destination must end with /inputs/{run_id}")
    document, sources = build_input_manifest(
        run_id=args.run_id,
        config=args.config,
        dataset_manifest=args.dataset_manifest,
        prepared_train=args.prepared_train,
        checkpoint=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        tokenizer=args.tokenizer,
        tokenizer_manifest=args.tokenizer_manifest,
    )
    encoded = canonical_bytes(document)
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    token = approval_token(args.run_id, manifest_sha256)
    plan = {
        "schema_version": "1.0",
        "mode": "plan-only" if not args.execute else "executed",
        "run_id": args.run_id,
        "destination": args.destination,
        "input_manifest_sha256": manifest_sha256,
        "approval_token": token,
        "manifest": document,
    }
    if args.execute:
        if os.environ.get(APPROVAL_ENVIRONMENT) != token:
            raise RuntimeError("exact input-staging approval token is required")
        upload_inputs(destination=args.destination, manifest=document, sources=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(plan, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
