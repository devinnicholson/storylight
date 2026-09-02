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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def public_dataset_sources(manifest_path: Path) -> dict[str, Path]:
    """Resolve and verify every public split declared by the dataset manifest."""

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is not valid JSON") from error
    splits = document.get("splits") if isinstance(document, dict) else None
    if not isinstance(splits, dict) or not splits:
        raise ValueError("dataset manifest has no splits")
    sources: dict[str, Path] = {}
    root = manifest_path.resolve().parent
    for name, declaration in sorted(splits.items()):
        if not isinstance(declaration, dict):
            raise ValueError(f"dataset split {name!r} is malformed")
        raw_relative = declaration.get("path")
        if raw_relative is None:
            continue
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError(f"dataset split {name!r} has an invalid path")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"dataset split {name!r} escaped the manifest directory")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError(f"dataset split {name!r} may not be a symbolic link")
        source = unresolved.resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"dataset split {name!r} escaped the manifest directory") from error
        if not source.is_file():
            raise ValueError(f"dataset split {name!r} is not a regular file")
        if sha256_file(source) != declaration.get("sha256"):
            raise ValueError(f"dataset split {name!r} SHA-256 mismatch")
        sources[f"dataset/{relative.as_posix()}"] = source
    if not sources:
        raise ValueError("staging requires accessible public dataset splits")
    return sources


def _json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return document


def _v2_prepared_binding(
    *,
    config: Path,
    dataset_manifest: Path,
    prepared_train: Path,
    tokenizer_manifest: Path,
    preparation_manifest: Path | None,
    prepared_validation: Path | None,
) -> tuple[dict[str, object] | None, dict[str, Path]]:
    evidence = (preparation_manifest, prepared_validation)
    try:
        config_document = _json_object(config, "training configuration")
    except ValueError:
        if any(path is not None for path in evidence):
            raise
        return None, {}
    experiment_id = config_document.get("experiment_id")
    is_v2 = isinstance(experiment_id, str) and experiment_id.endswith("-v2")
    if not is_v2:
        if any(path is not None for path in evidence):
            raise ValueError("preparation evidence is accepted only for a v2 experiment")
        return None, {}
    if preparation_manifest is None or prepared_validation is None:
        raise ValueError("v2 staging requires both preparation evidence files")

    dataset_document = _json_object(dataset_manifest, "dataset manifest")
    preparation = _json_object(preparation_manifest, "preparation manifest")
    validation = _json_object(prepared_validation, "prepared validation")
    production = config_document.get("production_contract")
    training = config_document.get("training")
    splits = dataset_document.get("splits")
    train_split = splits.get("train") if isinstance(splits, dict) else None
    if (
        not isinstance(production, dict)
        or not isinstance(training, dict)
        or not isinstance(train_split, dict)
    ):
        raise ValueError("v2 config or dataset contract is malformed")

    config_sha = sha256_file(config)
    prepared_sha = sha256_file(prepared_train)
    preparation_sha = sha256_file(preparation_manifest)
    validation_sha = sha256_file(prepared_validation)
    tokenizer_sha = sha256_file(tokenizer_manifest)
    source_train_sha = train_split.get("sha256")
    prompt_sha = production.get("prompt_contract_sha256")
    policy = training.get("preparation_policy")
    records = preparation.get("prepared_records")
    if (
        not isinstance(source_train_sha, str)
        or _SHA256.fullmatch(source_train_sha) is None
        or not isinstance(prompt_sha, str)
        or _SHA256.fullmatch(prompt_sha) is None
        or not isinstance(policy, str)
        or not policy
        or type(records) is not int
        or records < 1
        or preparation.get("schema_version")
        != "bookforge-jax-training-preparation-v2"
        or preparation.get("policy") != policy
        or preparation.get("source_train_sha256") != source_train_sha
        or preparation.get("prepared_sha256") != prepared_sha
        or preparation.get("prompt_contract_sha256") != prompt_sha
        or preparation.get("assistant_turns_per_record") != 1
        or preparation.get("pair_adjacency_preserved") is not True
    ):
        raise ValueError("v2 preparation manifest is not bound to the staged inputs")

    expected_validation = {
        "schema_version": "bookforge-jax-prepared-validation-v1",
        "status": "passed",
        "config_sha256": config_sha,
        "prepared_sha256": prepared_sha,
        "preparation_manifest_sha256": preparation_sha,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "prompt_contract_sha256": prompt_sha,
        "records": records,
        "assistant_turns_per_record": 1,
        "input_budget_tokens": production.get("input_budget_tokens"),
        "completion_budget_tokens": production.get("completion_budget_tokens"),
        "max_target_length": training.get("max_target_length"),
    }
    if any(validation.get(name) != value for name, value in expected_validation.items()):
        raise ValueError("prepared validation is not bound to the staged inputs")
    maxima = (
        ("maximum_prompt_tokens", production.get("input_budget_tokens")),
        ("maximum_completion_tokens", production.get("completion_budget_tokens")),
        ("maximum_total_tokens", training.get("max_target_length")),
    )
    if any(
        type(validation.get(name)) is not int
        or int(validation[name]) < 1
        or type(limit) is not int
        or int(validation[name]) > limit
        for name, limit in maxima
    ):
        raise ValueError("prepared validation exceeds the v2 token budgets")

    return (
        {
            "policy": policy,
            "records": records,
            "prepared_sha256": prepared_sha,
            "preparation_manifest_sha256": preparation_sha,
            "prepared_validation_sha256": validation_sha,
            "prompt_contract_sha256": prompt_sha,
            "source_train_sha256": source_train_sha,
            "tokenizer_manifest_sha256": tokenizer_sha,
        },
        {
            "prepared/preparation.manifest.json": preparation_manifest,
            "prepared/prepared-validation.json": prepared_validation,
        },
    )


def build_input_manifest(
    *,
    run_id: str,
    config: Path,
    dataset_manifest: Path,
    prepared_train: Path,
    checkpoint: Path,
    checkpoint_manifest: Path,
    checkpoint_receipt: Path | None,
    tokenizer: Path,
    tokenizer_manifest: Path,
    preparation_manifest: Path | None = None,
    prepared_validation: Path | None = None,
    require_base_orbax: bool = True,
) -> tuple[dict[str, object], dict[str, Path]]:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must be an immutable lowercase slug")
    base_orbax: dict[str, object] | None = None
    if require_base_orbax:
        if checkpoint_receipt is None:
            raise ValueError("full training staging requires a base Orbax receipt")
        from training.jax_fidelity.integrity import verify_artifact_manifest
        from training.jax_fidelity.orbax_receipt import verify_orbax_leaf_receipt

        try:
            receipt = json.loads(checkpoint_receipt.read_text(encoding="utf-8"))
            manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("base Orbax receipt or manifest is not valid JSON") from error
        if not isinstance(receipt, dict) or not isinstance(manifest, dict):
            raise ValueError("base Orbax receipt and manifest must be JSON objects")
        leaf = verify_orbax_leaf_receipt(
            checkpoint,
            receipt,
            expected_step=0,
            role="base-maxtext",
        )
        verify_artifact_manifest(leaf, manifest)
        base_orbax = {
            "role": "base-maxtext",
            "expected_step": 0,
            "relative_path": receipt["relative_path"],
            "receipt_sha256": sha256_file(checkpoint_receipt),
            "manifest_sha256": sha256_file(checkpoint_manifest),
            "content_sha256": manifest["content_sha256"],
        }
    elif checkpoint_receipt is not None:
        raise ValueError("non-Orbax staging must not accept an unused checkpoint receipt")
    sources = {
        "config.json": config,
        "dataset/manifest.json": dataset_manifest,
        "prepared/train.jsonl": prepared_train,
        "checkpoint.manifest.json": checkpoint_manifest,
        "tokenizer.manifest.json": tokenizer_manifest,
    }
    if checkpoint_receipt is not None:
        sources["checkpoint.receipt.json"] = checkpoint_receipt
    prepared_binding, prepared_sources = _v2_prepared_binding(
        config=config,
        dataset_manifest=dataset_manifest,
        prepared_train=prepared_train,
        tokenizer_manifest=tokenizer_manifest,
        preparation_manifest=preparation_manifest,
        prepared_validation=prepared_validation,
    )
    sources.update(prepared_sources)
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
    if base_orbax is not None:
        document["base_orbax"] = base_orbax
    if prepared_binding is not None:
        document["prepared_training"] = prepared_binding
    return document, sources


def build_full_training_input_manifest(
    *,
    run_id: str,
    config: Path,
    dataset_manifest: Path,
    prepared_train: Path,
    checkpoint: Path,
    checkpoint_manifest: Path,
    checkpoint_receipt: Path,
    tokenizer: Path,
    tokenizer_manifest: Path,
    preparation_manifest: Path | None = None,
    prepared_validation: Path | None = None,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Build a full-training population including the manifest's public splits."""

    document, sources = build_input_manifest(
        run_id=run_id,
        config=config,
        dataset_manifest=dataset_manifest,
        prepared_train=prepared_train,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_receipt=checkpoint_receipt,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        preparation_manifest=preparation_manifest,
        prepared_validation=prepared_validation,
    )
    for relative, source in public_dataset_sources(dataset_manifest).items():
        if relative in sources:
            raise ValueError(f"duplicate staged input path: {relative}")
        sources[relative] = source
    document["files"] = [
        {
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for relative, source in sorted(sources.items())
    ]
    return document, sources


def canonical_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_full_training_sources(
    manifest: dict[str, object], sources: dict[str, Path]
) -> None:
    """Re-verify a built population immediately before either provider upload."""

    expected_rows = [
        {
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for relative, source in sorted(sources.items())
        if source.is_file() and not source.is_symlink()
    ]
    if len(expected_rows) != len(sources) or manifest.get("files") != expected_rows:
        raise ValueError("full-training source bytes differ from the input manifest")
    prepared_binding, _ = _v2_prepared_binding(
        config=sources["config.json"],
        dataset_manifest=sources["dataset/manifest.json"],
        prepared_train=sources["prepared/train.jsonl"],
        tokenizer_manifest=sources["tokenizer.manifest.json"],
        preparation_manifest=sources.get("prepared/preparation.manifest.json"),
        prepared_validation=sources.get("prepared/prepared-validation.json"),
    )
    if prepared_binding is None:
        if "prepared_training" in manifest:
            raise ValueError("non-v2 input manifest contains a prepared-training binding")
    elif manifest.get("prepared_training") != prepared_binding:
        raise ValueError("v2 prepared-training binding changed")
    receipt_path = sources.get("checkpoint.receipt.json")
    manifest_path = sources.get("checkpoint.manifest.json")
    checkpoint_rows = [
        (Path(relative), source)
        for relative, source in sources.items()
        if Path(relative).parts[:1] == ("checkpoint",)
    ]
    if receipt_path is None or manifest_path is None or not checkpoint_rows:
        raise ValueError("full-training upload requires base Orbax receipt, manifest, and bytes")
    relative, checkpoint_file = checkpoint_rows[0]
    checkpoint_root = checkpoint_file
    for _ in relative.parts[1:]:
        checkpoint_root = checkpoint_root.parent
    from training.jax_fidelity.integrity import verify_artifact_manifest
    from training.jax_fidelity.orbax_receipt import verify_orbax_leaf_receipt

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(artifact, dict):
        raise ValueError("base Orbax receipt and manifest must be JSON objects")
    leaf = verify_orbax_leaf_receipt(
        checkpoint_root,
        receipt,
        expected_step=0,
        role="base-maxtext",
    )
    verify_artifact_manifest(leaf, artifact)
    if manifest.get("base_orbax") != {
        "role": "base-maxtext",
        "expected_step": 0,
        "relative_path": receipt.get("relative_path"),
        "receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": artifact.get("content_sha256"),
    }:
        raise ValueError("full-training input manifest base Orbax binding changed")


def approval_token(run_id: str, manifest_sha256: str) -> str:
    return f"APPROVE_GCP_JAX_STAGE:{run_id}:{manifest_sha256}"


def _gcs_location(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("destination must be a non-root gs:// URI")
    return parsed.netloc, parsed.path.strip("/")


def _is_run_input_prefix(prefix: str, run_id: str) -> bool:
    parts = prefix.split("/")
    return len(parts) >= 2 and parts[-2:] == ["inputs", run_id]


def upload_inputs(
    *,
    destination: str,
    manifest: dict[str, object],
    sources: dict[str, Path],
) -> None:
    verify_full_training_sources(manifest, sources)
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
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--prepared-validation", type=Path)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging evidence: {args.output}")
    _, prefix = _gcs_location(args.destination)
    if not _is_run_input_prefix(prefix, args.run_id):
        raise ValueError("destination must end with /inputs/{run_id}")
    document, sources = build_full_training_input_manifest(
        run_id=args.run_id,
        config=args.config,
        dataset_manifest=args.dataset_manifest,
        prepared_train=args.prepared_train,
        checkpoint=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_receipt=args.checkpoint_receipt,
        tokenizer=args.tokenizer,
        tokenizer_manifest=args.tokenizer_manifest,
        preparation_manifest=args.preparation_manifest,
        prepared_validation=args.prepared_validation,
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
