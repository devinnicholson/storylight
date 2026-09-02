"""Derive provider-neutral full-training inputs from a verified round-trip release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")


def canonical_bytes(document: object) -> bytes:
    rendered = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return f"{rendered}\n".encode()


def manifest_sha256(document: object) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _safe_rows(document: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} has no files")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
            or not isinstance(row.get("sha256"), str)
            or _SHA256.fullmatch(row["sha256"]) is None
        ):
            raise ValueError(f"{label} contains a malformed file declaration")
        path = PurePosixPath(row["path"])
        relative = path.as_posix()
        if path.is_absolute() or ".." in path.parts or relative != row["path"]:
            raise ValueError(f"{label} contains an unsafe path")
        if relative in result:
            raise ValueError(f"{label} contains a duplicate path")
        result[relative] = dict(row)
    return result


def _selected_input_rows(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exact = {
        "config.json",
        "dataset/manifest.json",
        "prepared/train.jsonl",
        "tokenizer.manifest.json",
    }
    selected = {
        path: row
        for path, row in source.items()
        if path in exact or path.startswith("tokenizer/") or path.startswith("dataset/")
    }
    if any("hidden" in part.casefold() for path in selected for part in PurePosixPath(path).parts):
        raise ValueError("hidden data may not enter the full-training population")
    if not exact.issubset(selected):
        raise ValueError("round-trip inputs lack required full-training files")
    if not any(path.startswith("tokenizer/") for path in selected):
        raise ValueError("round-trip inputs have no tokenizer files")
    if not {
        "dataset/train.jsonl",
        "dataset/development.jsonl",
    }.issubset(selected):
        raise ValueError("round-trip inputs lack public dataset splits")
    return selected


def build_full_training_manifest(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    source_input_manifest: dict[str, Any],
    target_run_id: str,
    roundtrip_completion_sha256: str,
    roundtrip_completion: dict[str, Any],
    base_receipt: dict[str, Any],
    base_manifest: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the exact target manifest and source-input copy allowlist."""

    if (
        _RUN_ID.fullmatch(source_run_id) is None
        or _RUN_ID.fullmatch(target_run_id) is None
        or source_run_id == target_run_id
    ):
        raise ValueError("source and target run IDs must be distinct immutable slugs")
    if (
        _SHA256.fullmatch(source_input_manifest_sha256) is None
        or _SHA256.fullmatch(roundtrip_completion_sha256) is None
        or manifest_sha256(source_input_manifest) != source_input_manifest_sha256
    ):
        raise ValueError("round-trip source manifest checksum changed")
    if (
        source_input_manifest.get("schema_version") != "1.0"
        or source_input_manifest.get("producer") != "bookforge-gcp-jax-input-stager"
        or source_input_manifest.get("status") != "complete"
        or source_input_manifest.get("run_id") != source_run_id
    ):
        raise ValueError("round-trip input manifest identity changed")
    if (
        roundtrip_completion.get("schema_version") != "1.0"
        or roundtrip_completion.get("status") != "succeeded"
        or roundtrip_completion.get("backend") != "modal-l4x2"
        or roundtrip_completion.get("run_id") != source_run_id
        or roundtrip_completion.get("input_manifest_sha256")
        != source_input_manifest_sha256
    ):
        raise ValueError("round-trip release identity changed")

    source_rows = _safe_rows(source_input_manifest, "round-trip input manifest")
    selected = _selected_input_rows(source_rows)
    release_rows = _safe_rows(roundtrip_completion, "round-trip completion")
    base_rows = _safe_rows(base_manifest, "base Orbax manifest")
    relative_leaf = base_receipt.get("relative_path")
    if (
        base_receipt.get("schema_version") != "1.0"
        or base_receipt.get("role") != "base-maxtext"
        or base_receipt.get("expected_step") != 0
        or not isinstance(relative_leaf, str)
        or PurePosixPath(relative_leaf).is_absolute()
        or ".." in PurePosixPath(relative_leaf).parts
        or base_receipt.get("artifact_manifest") != base_manifest
    ):
        raise ValueError("base Orbax receipt identity changed")

    target_rows = [dict(row) for row in selected.values()]
    for path, row in base_rows.items():
        release_path = f"base-orbax/{relative_leaf}/{path}"
        if release_rows.get(release_path) != {**row, "path": release_path}:
            raise ValueError("round-trip completion does not bind every base Orbax byte")
        target_rows.append({**row, "path": f"checkpoint/{relative_leaf}/{path}"})

    receipt_release_path = "evidence/base-orbax.receipt.json"
    manifest_release_path = "base-orbax.manifest.json"
    receipt_row = release_rows.get(receipt_release_path)
    manifest_row = release_rows.get(manifest_release_path)
    if receipt_row is None or manifest_row is None:
        raise ValueError("round-trip completion lacks base Orbax evidence")
    selected_bindings = {
        "config_sha256": selected["config.json"]["sha256"],
        "dataset_manifest_sha256": selected["dataset/manifest.json"]["sha256"],
        "tokenizer_manifest_sha256": selected["tokenizer.manifest.json"]["sha256"],
        "base_orbax_receipt_sha256": receipt_row["sha256"],
        "base_orbax_manifest_sha256": manifest_row["sha256"],
    }
    if any(roundtrip_completion.get(name) != digest for name, digest in selected_bindings.items()):
        raise ValueError("round-trip completion is not linked to the selected source bytes")
    target_rows.extend(
        (
            {**receipt_row, "path": "checkpoint.receipt.json"},
            {**manifest_row, "path": "checkpoint.manifest.json"},
        )
    )
    target_rows.sort(key=lambda row: row["path"])
    target = {
        "schema_version": "1.0",
        "producer": "bookforge-gcp-jax-input-stager",
        "run_id": target_run_id,
        "status": "complete",
        "files": target_rows,
        "base_orbax": {
            "role": "base-maxtext",
            "expected_step": 0,
            "relative_path": relative_leaf,
            "receipt_sha256": receipt_row["sha256"],
            "manifest_sha256": manifest_row["sha256"],
            "content_sha256": base_manifest.get("content_sha256"),
        },
        "derivation": {
            "producer": "bookforge-modal-jax-full-input-clone",
            "source_run_id": source_run_id,
            "source_input_manifest_sha256": source_input_manifest_sha256,
            "roundtrip_completion_sha256": roundtrip_completion_sha256,
            **selected_bindings,
        },
    }
    return target, tuple(sorted(selected))


def approval_token(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    roundtrip_completion_sha256: str,
    target_run_id: str,
    target_manifest_sha256: str,
) -> str:
    return (
        "APPROVE_MODAL_JAX_FULL_INPUT_CLONE:"
        f"{source_run_id}:{source_input_manifest_sha256}:"
        f"{roundtrip_completion_sha256}:{target_run_id}:{target_manifest_sha256}"
    )
