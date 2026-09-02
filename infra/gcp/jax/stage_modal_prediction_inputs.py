#!/usr/bin/env python3
"""Stage one public development prediction population into a Modal volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.integrity import (
    artifact_manifest,
    canonical_json_bytes,
    sha256_file,
)
from training.jax_fidelity.merged_candidate import validate_merged_candidate_manifest
from training.jax_fidelity.predict import load_records

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VOLUME_NAME = "bookforge-jax-fidelity-inputs"
REMOTE_ROOT = "prediction"
APPROVAL_ENVIRONMENT = "BOOKFORGE_MODAL_JAX_PREDICTION_STAGE_APPROVAL"
PUBLIC_MANIFEST = REPOSITORY_ROOT / "datasets/story-fidelity-v1/manifest.json"
PUBLIC_DEVELOPMENT = REPOSITORY_ROOT / "datasets/story-fidelity-v1/development.jsonl"
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")
Run = Callable[..., subprocess.CompletedProcess[str]]


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_committed_public_sources(manifest: Path, records: Path) -> None:
    expected = (PUBLIC_MANIFEST, PUBLIC_DEVELOPMENT)
    actual = (manifest.resolve(), records.resolve())
    if actual != tuple(path.resolve() for path in expected):
        raise ValueError("prediction staging accepts only the repository public development files")
    for path in expected:
        relative = path.relative_to(REPOSITORY_ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative.as_posix()],
            cwd=REPOSITORY_ROOT,
            check=False,
            timeout=30,
        )
        if clean.returncode != 0:
            raise ValueError(f"public prediction input differs from HEAD: {relative}")


def validate_public_dataset(manifest: Path, records: Path) -> tuple[str, str]:
    document = _json_object(manifest)
    splits = document.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("dataset manifest has no split declarations")
    development = splits.get("development")
    hidden = splits.get("hidden")
    if (
        not isinstance(development, dict)
        or development.get("public") is not True
        or development.get("path") != "development.jsonl"
        or development.get("records") != 512
        or development.get("sha256") != sha256_file(records)
    ):
        raise ValueError("dataset manifest does not declare the exact public development split")
    if (
        not isinstance(hidden, dict)
        or hidden.get("public") is not False
        or hidden.get("path") is not None
    ):
        raise ValueError("dataset manifest must keep the hidden split external and hash-only")
    load_records(records)
    return sha256_file(manifest), sha256_file(records)


def _source_rows(sources: Mapping[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative, source in sorted(sources.items()):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("prediction input path escaped its immutable prefix")
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"prediction input is not a regular file: {source}")
        rows.append(
            {
                "path": candidate.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return rows


def build_prediction_input_manifest(
    *,
    run_id: str,
    config_path: Path,
    dataset_manifest_path: Path,
    development_records_path: Path,
    candidate_directory: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Validate and describe the only bytes permitted in a prediction prefix."""

    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("prediction run ID must be an immutable lowercase slug")
    _require_committed_public_sources(dataset_manifest_path, development_records_path)
    config = load_config(config_path)
    dataset_sha, records_sha = validate_public_dataset(
        dataset_manifest_path,
        development_records_path,
    )
    candidate_root = candidate_directory.resolve()
    candidate_manifest = candidate_root / "candidate.manifest.json"
    checkpoint = candidate_root / "merged-hf"
    candidate_manifest_sha = sha256_file(candidate_manifest)
    candidate = validate_merged_candidate_manifest(
        candidate_manifest,
        checkpoint,
        config_path=config_path,
        expected_manifest_sha256=candidate_manifest_sha,
        expected_config_sha256=config.sha256,
        expected_dataset_manifest_sha256=dataset_sha,
    )
    checkpoint_document = artifact_manifest(checkpoint)
    checkpoint_manifest_sha = _canonical_sha256(checkpoint_document)
    sources: dict[str, Path] = {
        "config.json": config_path,
        "dataset/manifest.json": dataset_manifest_path,
        "dataset/development.jsonl": development_records_path,
        "candidate/candidate.manifest.json": candidate_manifest,
    }
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            sources[f"candidate/merged-hf/{path.relative_to(checkpoint).as_posix()}"] = path
    bindings = {
        "candidate_id": candidate["candidate_id"],
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": dataset_sha,
        "development_records_sha256": records_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_content_sha256": checkpoint_document["content_sha256"],
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "producer": "bookforge-modal-development-prediction-stager",
        "run_id": run_id,
        "status": "complete",
        "prefix": f"{REMOTE_ROOT}/{run_id}",
        "privacy": {
            "split": "development",
            "public_records_only": True,
            "hidden_records_included": False,
        },
        "bindings": bindings,
        "files": _source_rows(sources),
    }
    return manifest, sources


def approval_token(run_id: str, input_manifest_sha256: str) -> str:
    return (
        f"APPROVE_MODAL_JAX_PREDICTION_STAGE:{VOLUME_NAME}:"
        f"{REMOTE_ROOT}/{run_id}:{input_manifest_sha256}"
    )


def _run(command: list[str], *, runner: Run) -> subprocess.CompletedProcess[str]:
    return runner(command, check=True, capture_output=True, text=True, timeout=900)


def _listed_prefixes(stdout: str) -> set[str]:
    try:
        entries = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Modal prediction input listing was not JSON") from error
    if not isinstance(entries, list):
        raise RuntimeError("Modal prediction input listing was not a list")
    values: set[str] = set()
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
            raise RuntimeError("Modal prediction input listing contains an invalid entry")
        cleaned = value.strip("/")
        if cleaned:
            parts = cleaned.split("/")
            values.add(parts[1] if parts[0] == REMOTE_ROOT and len(parts) > 1 else parts[0])
    return values


def stage_prediction_inputs(
    *,
    run_id: str,
    manifest: dict[str, object],
    sources: Mapping[str, Path],
    runner: Run = subprocess.run,
) -> dict[str, object]:
    encoded = canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    token = approval_token(run_id, manifest_sha)
    if os.environ.get(APPROVAL_ENVIRONMENT) != token:
        raise RuntimeError("exact Modal prediction input-staging approval token is required")
    listing = _run(
        ["modal", "volume", "ls", VOLUME_NAME, f"/{REMOTE_ROOT}", "--json"],
        runner=runner,
    )
    if run_id in _listed_prefixes(listing.stdout):
        raise RuntimeError("Modal prediction input prefix already contains state")
    prefix = f"/{REMOTE_ROOT}/{run_id}"
    for relative, source in sorted(sources.items()):
        _run(
            ["modal", "volume", "put", VOLUME_NAME, str(source), f"{prefix}/{relative}"],
            runner=runner,
        )
    with tempfile.TemporaryDirectory(prefix="bookforge-modal-prediction-stage-") as raw:
        local_manifest = Path(raw) / "inputs.manifest.json"
        local_manifest.write_bytes(encoded)
        _run(
            [
                "modal",
                "volume",
                "put",
                VOLUME_NAME,
                str(local_manifest),
                f"{prefix}/inputs.manifest.json",
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
                f"{prefix}/inputs.manifest.json",
                str(receipt),
            ],
            runner=runner,
        )
        if not receipt.is_file() or sha256_file(receipt) != manifest_sha:
            raise RuntimeError("Modal prediction input manifest failed read-back verification")
    return {
        "schema_version": "1.0",
        "status": "staged",
        "run_id": run_id,
        "volume": VOLUME_NAME,
        "prefix": f"{REMOTE_ROOT}/{run_id}",
        "input_manifest_sha256": manifest_sha,
        "bindings": manifest["bindings"],
        "manifest_uploaded_last": True,
        "hidden_records_uploaded": False,
        "overwrite_enabled": False,
    }


def verify_staged_inputs(
    root: Path,
    *,
    run_id: str,
    expected_manifest_sha256: str,
    expected_bindings: Mapping[str, object],
) -> dict[str, Any]:
    manifest_path = root / "inputs.manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("prediction input manifest checksum changed")
    document = _json_object(manifest_path)
    if (
        document.get("schema_version") != "1.0"
        or document.get("producer") != "bookforge-modal-development-prediction-stager"
        or document.get("run_id") != run_id
        or document.get("status") != "complete"
        or document.get("prefix") != f"{REMOTE_ROOT}/{run_id}"
        or document.get("privacy")
        != {
            "split": "development",
            "public_records_only": True,
            "hidden_records_included": False,
        }
        or document.get("bindings") != dict(expected_bindings)
    ):
        raise RuntimeError("prediction input identity, privacy, or hash bindings changed")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("prediction input manifest contains no files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("prediction input file declaration is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise RuntimeError("prediction input file path is unsafe or duplicated")
        declared.add(relative.as_posix())
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise RuntimeError(f"prediction input checksum changed: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != declared:
        raise RuntimeError("prediction prefix contains undeclared or missing files")
    if "dataset/development.jsonl" not in declared or any(
        "hidden" in Path(relative).name.lower() for relative in declared
    ):
        raise RuntimeError("prediction prefix is not development-only")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=PUBLIC_MANIFEST)
    parser.add_argument("--development-records", type=Path, default=PUBLIC_DEVELOPMENT)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite staging evidence: {args.output}")
    manifest, sources = build_prediction_input_manifest(
        run_id=args.run_id,
        config_path=args.config,
        dataset_manifest_path=args.dataset_manifest,
        development_records_path=args.development_records,
        candidate_directory=args.candidate_directory,
    )
    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    if args.execute:
        document = stage_prediction_inputs(
            run_id=args.run_id,
            manifest=manifest,
            sources=sources,
        )
    else:
        document = {
            "schema_version": "1.0",
            "status": "plan-only",
            "run_id": args.run_id,
            "volume": VOLUME_NAME,
            "prefix": f"{REMOTE_ROOT}/{args.run_id}",
            "input_manifest_sha256": manifest_sha,
            "approval_token": approval_token(args.run_id, manifest_sha),
            "bindings": manifest["bindings"],
            "hidden_records_uploaded": False,
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
