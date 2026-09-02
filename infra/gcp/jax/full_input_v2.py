"""Build an immutable v2 training population from small inputs and a cached base."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"
OVERLAY_PRODUCER = "bookforge-modal-jax-v2-overlay-stager"
TARGET_PRODUCER = "bookforge-gcp-jax-input-stager"
OVERLAY_PREFIX = "full-input-v2"
PREPARATION_POLICY = "balanced-counterfactual-pairs-v1"
PREPARED_RECORDS = 320
OVERLAY_PATHS = (
    "config.json",
    "dataset/development.jsonl",
    "dataset/manifest.json",
    "dataset/train.jsonl",
    "prepared/preparation.manifest.json",
    "prepared/prepared-validation.json",
    "prepared/train.jsonl",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")


def canonical_bytes(document: object) -> bytes:
    rendered = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return f"{rendered}\n".encode()


def manifest_sha256(document: object) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _run_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable run slug")
    return value


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


def build_overlay_manifest(
    *,
    target_run_id: str,
    files: list[dict[str, Any]],
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    preparation_manifest: dict[str, Any],
    prepared_validation: dict[str, Any],
    source_tokenizer_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and describe the small host-originated v2 files."""

    target_run_id = _run_id(target_run_id, "target_run_id")
    tokenizer_sha = _sha(
        source_tokenizer_manifest_sha256, "source_tokenizer_manifest_sha256"
    )
    provisional = {"files": files}
    rows = _safe_rows(provisional, "v2 overlay")
    if set(rows) != set(OVERLAY_PATHS):
        raise ValueError("v2 overlay must contain exactly the approved small files")
    if any("hidden" in part.casefold() for path in rows for part in PurePosixPath(path).parts):
        raise ValueError("hidden data may not enter the v2 training overlay")

    config_sha = rows["config.json"]["sha256"]
    dataset_sha = rows["dataset/manifest.json"]["sha256"]
    prepared_sha = rows["prepared/train.jsonl"]["sha256"]
    preparation_sha = rows["prepared/preparation.manifest.json"]["sha256"]
    validation_sha = rows["prepared/prepared-validation.json"]["sha256"]
    prompt = config.get("production_contract")
    training = config.get("training")
    dataset_config = config.get("dataset")
    prompt_sha = prompt.get("prompt_contract_sha256") if isinstance(prompt, dict) else None
    if (
        config.get("schema_version") != "1.0"
        or config.get("experiment_id") != "bookforge-gemma4-e2b-lora-r16-v2"
        or not isinstance(prompt, dict)
        or _SHA256.fullmatch(str(prompt_sha)) is None
        or not isinstance(training, dict)
        or training.get("preparation_policy") != PREPARATION_POLICY
        or not isinstance(dataset_config, dict)
        or dataset_config.get("manifest_path") != "datasets/story-fidelity-v2/manifest.json"
    ):
        raise ValueError("v2 config identity or prompt contract changed")

    splits = dataset_manifest.get("splits")
    if (
        dataset_manifest.get("dataset_id") != "story-fidelity-v2"
        or not isinstance(splits, dict)
        or set(splits) != {"train", "development", "hidden"}
        or not isinstance(splits["train"], dict)
        or not isinstance(splits["development"], dict)
        or not isinstance(splits["hidden"], dict)
        or splits["train"].get("path") != "train.jsonl"
        or splits["train"].get("sha256") != rows["dataset/train.jsonl"]["sha256"]
        or splits["development"].get("path") != "development.jsonl"
        or splits["development"].get("sha256")
        != rows["dataset/development.jsonl"]["sha256"]
        or splits["hidden"].get("path") is not None
        or splits["hidden"].get("public") is not False
    ):
        raise ValueError("v2 dataset manifest does not bind only its public files")

    source_train_sha = rows["dataset/train.jsonl"]["sha256"]
    if (
        preparation_manifest.get("schema_version")
        != "bookforge-jax-training-preparation-v2"
        or preparation_manifest.get("policy") != PREPARATION_POLICY
        or preparation_manifest.get("source_train_sha256") != source_train_sha
        or preparation_manifest.get("prepared_sha256") != prepared_sha
        or preparation_manifest.get("prompt_contract_sha256") != prompt_sha
        or preparation_manifest.get("prepared_records") != PREPARED_RECORDS
        or preparation_manifest.get("assistant_turns_per_record") != 1
        or preparation_manifest.get("pair_adjacency_preserved") is not True
    ):
        raise ValueError("v2 preparation manifest is not bound to the staged population")
    if (
        prepared_validation.get("schema_version")
        != "bookforge-jax-prepared-validation-v1"
        or prepared_validation.get("status") != "passed"
        or prepared_validation.get("config_sha256") != config_sha
        or prepared_validation.get("prepared_sha256") != prepared_sha
        or prepared_validation.get("preparation_manifest_sha256") != preparation_sha
        or prepared_validation.get("tokenizer_manifest_sha256") != tokenizer_sha
        or prepared_validation.get("prompt_contract_sha256") != prompt_sha
        or prepared_validation.get("records") != PREPARED_RECORDS
        or prepared_validation.get("assistant_turns_per_record") != 1
    ):
        raise ValueError("prepared validation is not bound to the v2 inputs")

    ordered_rows = [rows[path] for path in sorted(rows)]
    bindings = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "prepared_sha256": prepared_sha,
        "preparation_manifest_sha256": preparation_sha,
        "prepared_validation_sha256": validation_sha,
        "prompt_contract_sha256": prompt_sha,
        "source_train_sha256": source_train_sha,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "policy": PREPARATION_POLICY,
        "records": PREPARED_RECORDS,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": OVERLAY_PRODUCER,
        "status": "complete",
        "target_run_id": target_run_id,
        "prefix": f"{OVERLAY_PREFIX}/{target_run_id}",
        "files": ordered_rows,
        "bindings": bindings,
        "privacy": {"public_records_only": True, "hidden_records_included": False},
    }


def build_full_training_manifest(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    source_input_manifest: dict[str, Any],
    roundtrip_completion_sha256: str,
    roundtrip_completion: dict[str, Any],
    base_receipt: dict[str, Any],
    base_manifest: dict[str, Any],
    overlay_manifest_sha256: str,
    overlay_manifest: dict[str, Any],
    target_run_id: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Combine v2 files with the cached tokenizer and base checkpoint."""

    source_run_id = _run_id(source_run_id, "source_run_id")
    target_run_id = _run_id(target_run_id, "target_run_id")
    if source_run_id == target_run_id:
        raise ValueError("source and target run IDs must differ")
    source_manifest_sha = _sha(
        source_input_manifest_sha256, "source_input_manifest_sha256"
    )
    _sha(roundtrip_completion_sha256, "roundtrip_completion_sha256")
    overlay_sha = _sha(overlay_manifest_sha256, "overlay_manifest_sha256")
    if manifest_sha256(source_input_manifest) != source_manifest_sha:
        raise ValueError("source input manifest checksum changed")
    if manifest_sha256(overlay_manifest) != overlay_sha:
        raise ValueError("v2 overlay manifest checksum changed")
    if (
        source_input_manifest.get("schema_version") != SCHEMA_VERSION
        or source_input_manifest.get("producer") != TARGET_PRODUCER
        or source_input_manifest.get("status") != "complete"
        or source_input_manifest.get("run_id") != source_run_id
    ):
        raise ValueError("source input manifest identity changed")
    if (
        roundtrip_completion.get("schema_version") != SCHEMA_VERSION
        or roundtrip_completion.get("status") != "succeeded"
        or roundtrip_completion.get("backend") != "modal-l4x2"
        or roundtrip_completion.get("run_id") != source_run_id
        or roundtrip_completion.get("input_manifest_sha256") != source_manifest_sha
    ):
        raise ValueError("roundtrip completion identity changed")
    if (
        overlay_manifest.get("schema_version") != SCHEMA_VERSION
        or overlay_manifest.get("producer") != OVERLAY_PRODUCER
        or overlay_manifest.get("status") != "complete"
        or overlay_manifest.get("target_run_id") != target_run_id
        or overlay_manifest.get("prefix") != f"{OVERLAY_PREFIX}/{target_run_id}"
        or overlay_manifest.get("privacy")
        != {"public_records_only": True, "hidden_records_included": False}
    ):
        raise ValueError("v2 overlay identity changed")

    source_rows = _safe_rows(source_input_manifest, "source input manifest")
    overlay_rows = _safe_rows(overlay_manifest, "v2 overlay manifest")
    overlay_bindings = overlay_manifest.get("bindings")
    if not isinstance(overlay_bindings, dict):
        raise ValueError("v2 overlay has no checksum bindings")
    if set(overlay_rows) != set(OVERLAY_PATHS):
        raise ValueError("v2 overlay file population changed")
    tokenizer_paths = tuple(
        sorted(
            path
            for path in source_rows
            if path == "tokenizer.manifest.json" or path.startswith("tokenizer/")
        )
    )
    if "tokenizer.manifest.json" not in tokenizer_paths or not any(
        path.startswith("tokenizer/") for path in tokenizer_paths
    ):
        raise ValueError("source input manifest has no complete tokenizer")
    if overlay_bindings.get("tokenizer_manifest_sha256") != source_rows[
        "tokenizer.manifest.json"
    ]["sha256"]:
        raise ValueError("v2 overlay is not bound to the cached tokenizer")

    release_rows = _safe_rows(roundtrip_completion, "roundtrip completion")
    base_rows = _safe_rows(base_manifest, "base Orbax manifest")
    relative_leaf = base_receipt.get("relative_path")
    if (
        base_receipt.get("schema_version") != SCHEMA_VERSION
        or base_receipt.get("role") != "base-maxtext"
        or base_receipt.get("expected_step") != 0
        or not isinstance(relative_leaf, str)
        or PurePosixPath(relative_leaf).is_absolute()
        or ".." in PurePosixPath(relative_leaf).parts
        or base_receipt.get("artifact_manifest") != base_manifest
    ):
        raise ValueError("base Orbax receipt identity changed")

    target_rows = [dict(row) for row in overlay_rows.values()]
    target_rows.extend(dict(source_rows[path]) for path in tokenizer_paths)
    for path, row in base_rows.items():
        release_path = f"base-orbax/{relative_leaf}/{path}"
        if release_rows.get(release_path) != {**row, "path": release_path}:
            raise ValueError("roundtrip completion does not bind every base Orbax byte")
        target_rows.append({**row, "path": f"checkpoint/{relative_leaf}/{path}"})

    receipt_release_path = "evidence/base-orbax.receipt.json"
    manifest_release_path = "base-orbax.manifest.json"
    receipt_row = release_rows.get(receipt_release_path)
    manifest_row = release_rows.get(manifest_release_path)
    if receipt_row is None or manifest_row is None:
        raise ValueError("roundtrip completion lacks base Orbax evidence")
    source_bindings = {
        "source_config_sha256": roundtrip_completion.get("config_sha256"),
        "source_dataset_manifest_sha256": roundtrip_completion.get(
            "dataset_manifest_sha256"
        ),
        "source_tokenizer_manifest_sha256": source_rows["tokenizer.manifest.json"][
            "sha256"
        ],
        "source_base_orbax_receipt_sha256": receipt_row["sha256"],
        "source_base_orbax_manifest_sha256": manifest_row["sha256"],
    }
    expected_completion_bindings = {
        "config_sha256": source_bindings["source_config_sha256"],
        "dataset_manifest_sha256": source_bindings[
            "source_dataset_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": source_bindings[
            "source_tokenizer_manifest_sha256"
        ],
        "base_orbax_receipt_sha256": source_bindings[
            "source_base_orbax_receipt_sha256"
        ],
        "base_orbax_manifest_sha256": source_bindings[
            "source_base_orbax_manifest_sha256"
        ],
    }
    if any(
        roundtrip_completion.get(name) != value
        for name, value in expected_completion_bindings.items()
    ):
        raise ValueError("roundtrip completion source bindings changed")
    target_rows.extend(
        (
            {**receipt_row, "path": "checkpoint.receipt.json"},
            {**manifest_row, "path": "checkpoint.manifest.json"},
        )
    )
    paths = [row["path"] for row in target_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("v2, tokenizer, and checkpoint inputs overlap")
    target_rows.sort(key=lambda row: row["path"])
    # Keep this provider-neutral manifest byte-for-byte compatible with the GCS
    # stager. Modal-specific provenance belongs in the separately approved clone
    # receipt, never in the runnable training population identity.
    prepared_training = {
        "policy": overlay_bindings.get("policy"),
        "preparation_manifest_sha256": overlay_bindings.get(
            "preparation_manifest_sha256"
        ),
        "prepared_sha256": overlay_bindings.get("prepared_sha256"),
        "prepared_validation_sha256": overlay_bindings.get(
            "prepared_validation_sha256"
        ),
        "prompt_contract_sha256": overlay_bindings.get("prompt_contract_sha256"),
        "records": overlay_bindings.get("records"),
        "source_train_sha256": overlay_bindings.get("source_train_sha256"),
        "tokenizer_manifest_sha256": overlay_bindings.get(
            "tokenizer_manifest_sha256"
        ),
    }
    target = {
        "schema_version": SCHEMA_VERSION,
        "producer": TARGET_PRODUCER,
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
        "prepared_training": prepared_training,
    }
    return target, tokenizer_paths, tuple(sorted(overlay_rows))


def overlay_approval_token(*, target_run_id: str, overlay_manifest_sha256: str) -> str:
    return (
        "APPROVE_MODAL_JAX_V2_OVERLAY_STAGE:"
        f"{target_run_id}:{overlay_manifest_sha256}"
    )


def clone_approval_token(
    *,
    source_run_id: str,
    source_input_manifest_sha256: str,
    roundtrip_completion_sha256: str,
    overlay_manifest_sha256: str,
    target_run_id: str,
    target_manifest_sha256: str,
) -> str:
    return (
        "APPROVE_MODAL_JAX_FULL_INPUT_V2_CLONE:"
        f"{source_run_id}:{source_input_manifest_sha256}:"
        f"{roundtrip_completion_sha256}:{overlay_manifest_sha256}:"
        f"{target_run_id}:{target_manifest_sha256}"
    )
