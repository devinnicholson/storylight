"""Build an immutable v3 recovery population from public inputs and a cached base."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"
OVERLAY_PRODUCER = "bookforge-modal-jax-v3-recovery-overlay-stager"
TARGET_PRODUCER = "bookforge-gcp-jax-input-stager"
OVERLAY_PREFIX = "full-input-v3-recovery"
EXPERIMENT_ID = "bookforge-gemma4-e2b-lora-r16-v3-canary"
RECOVERY_SCHEMA_VERSION = "bookforge-jax-v3-recovery-inputs-v1"
RECOVERY_PRODUCER = "bookforge-jax-v3-recovery-input-builder"
PREPARATION_POLICY = "public-balanced-counterfactual-recovery-v1"
CANARY_RECORDS = 40
RECOVERY_FILES = (
    "overfit-canary.source.jsonl",
    "overfit-canary.train.jsonl",
    "public-probe.source.jsonl",
    "public-probe.teacher.jsonl",
)
OVERLAY_PATHS = (
    "config.json",
    "dataset/development.jsonl",
    "dataset/manifest.json",
    "dataset/train.jsonl",
    "recovery/inputs.manifest.json",
    *(f"recovery/{path}" for path in RECOVERY_FILES),
)
PREPARED_PATH = "prepared/train.jsonl"

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


def _validate_public_dataset(
    dataset: dict[str, Any], rows: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
    splits = dataset.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "development", "hidden"}:
        raise ValueError("v3 dataset split population changed")
    train = splits.get("train")
    development = splits.get("development")
    hidden = splits.get("hidden")
    if (
        dataset.get("dataset_id") != "story-fidelity-v2"
        or not isinstance(train, dict)
        or not isinstance(development, dict)
        or not isinstance(hidden, dict)
        or train.get("path") != "train.jsonl"
        or train.get("public") is not True
        or train.get("sha256") != rows["dataset/train.jsonl"]["sha256"]
        or development.get("path") != "development.jsonl"
        or development.get("public") is not True
        or development.get("sha256") != rows["dataset/development.jsonl"]["sha256"]
        or hidden.get("path") is not None
        or hidden.get("public") is not False
    ):
        raise ValueError("v3 dataset manifest does not bind only public files")
    return (
        rows["dataset/manifest.json"]["sha256"],
        rows["dataset/train.jsonl"]["sha256"],
        rows["dataset/development.jsonl"]["sha256"],
    )


def _validate_recovery(
    recovery: dict[str, Any],
    *,
    rows: dict[str, dict[str, Any]],
    config: dict[str, Any],
    config_sha: str,
    dataset_sha: str,
    train_sha: str,
    development_sha: str,
) -> dict[str, Any]:
    recovery_rows = _safe_rows(recovery, "v3 recovery manifest")
    if set(recovery_rows) != set(RECOVERY_FILES):
        raise ValueError("v3 recovery file population changed")
    for relative, row in recovery_rows.items():
        if rows[f"recovery/{relative}"] != {**row, "path": f"recovery/{relative}"}:
            raise ValueError("v3 overlay does not bind every sealed recovery byte")

    configured = config.get("recovery")
    populations = recovery.get("populations")
    if not isinstance(configured, dict) or not isinstance(populations, dict):
        raise ValueError("v3 recovery configuration or populations are missing")
    canary = populations.get("overfit_canary")
    probe = populations.get("public_probe")
    expected_canary = configured.get("overfit_canary")
    expected_probe = configured.get("public_probe")
    expected_population_keys = {
        "split",
        "pairs",
        "records",
        "record_ids_sha256",
        "pair_ids_sha256",
        "source_sha256",
        "prepared_sha256",
    }
    population_values = (canary, probe, expected_canary, expected_probe)
    if not all(isinstance(value, dict) for value in population_values):
        raise ValueError("v3 recovery population binding is malformed")
    for actual, expected, label in (
        (canary, expected_canary, "canary"),
        (probe, expected_probe, "probe"),
    ):
        if any(actual.get(key) != expected.get(key) for key in expected_population_keys):
            raise ValueError(f"v3 {label} population differs from the config")

    dataset = recovery.get("dataset")
    disjoint = recovery.get("disjoint")
    if (
        recovery.get("schema_version") != RECOVERY_SCHEMA_VERSION
        or recovery.get("producer") != RECOVERY_PRODUCER
        or recovery.get("status") != "complete"
        or recovery.get("experiment_id") != EXPERIMENT_ID
        or recovery.get("config_sha256") != config_sha
        or not isinstance(dataset, dict)
        or dataset
        != {
            "dataset_id": "story-fidelity-v2",
            "manifest_sha256": dataset_sha,
            "train_sha256": train_sha,
            "development_sha256": development_sha,
        }
        or recovery.get("selection")
        != {
            "policy": PREPARATION_POLICY,
            "seed": configured.get("selection_seed"),
        }
        or recovery.get("lineage") != configured.get("previous_attempt")
        or recovery.get("learnability_acceptance") != configured.get("learnability_acceptance")
        or not isinstance(disjoint, dict)
        or set(disjoint) != set(configured.get("disjoint_fields", ()))
        or not all(value is True for value in disjoint.values())
        or recovery.get("privacy")
        != {"public_records_only": True, "hidden_records_included": False}
    ):
        raise ValueError("v3 recovery manifest binding changed")
    if (
        canary.get("purpose") != "overfit-canary"
        or canary.get("split") != "train"
        or canary.get("records") != CANARY_RECORDS
        or canary.get("source_sha256") != recovery_rows["overfit-canary.source.jsonl"]["sha256"]
        or canary.get("prepared_sha256") != recovery_rows["overfit-canary.train.jsonl"]["sha256"]
        or probe.get("purpose") != "public-probe"
        or probe.get("split") != "development"
        or probe.get("source_sha256") != recovery_rows["public-probe.source.jsonl"]["sha256"]
        or probe.get("prepared_sha256") != recovery_rows["public-probe.teacher.jsonl"]["sha256"]
    ):
        raise ValueError("v3 recovery files do not match their populations")
    return {"canary": dict(canary), "probe": dict(probe)}


def build_overlay_manifest(
    *,
    target_run_id: str,
    files: list[dict[str, Any]],
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    recovery_manifest: dict[str, Any],
    source_tokenizer_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and describe only the public host-originated v3 files."""

    target_run_id = _run_id(target_run_id, "target_run_id")
    tokenizer_sha = _sha(source_tokenizer_manifest_sha256, "source_tokenizer_manifest_sha256")
    rows = _safe_rows({"files": files}, "v3 overlay")
    if set(rows) != set(OVERLAY_PATHS):
        raise ValueError("v3 overlay must contain exactly the approved public files")
    if any("hidden" in part.casefold() for path in rows for part in PurePosixPath(path).parts):
        raise ValueError("hidden data may not enter the v3 recovery overlay")

    config_sha = rows["config.json"]["sha256"]
    prompt = config.get("production_contract")
    training = config.get("training")
    dataset_config = config.get("dataset")
    configured_recovery = config.get("recovery")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(prompt, dict)
        or _SHA256.fullmatch(str(prompt.get("prompt_contract_sha256"))) is None
        or not isinstance(training, dict)
        or training.get("preparation_policy") != PREPARATION_POLICY
        or not isinstance(dataset_config, dict)
        or dataset_config.get("manifest_path") != "datasets/story-fidelity-v2/manifest.json"
        or dataset_config.get("manifest_sha256") != rows["dataset/manifest.json"]["sha256"]
        or not isinstance(configured_recovery, dict)
        or configured_recovery.get("schema_version") != "bookforge-jax-v3-recovery-v1"
    ):
        raise ValueError("v3 config identity or recovery contract changed")
    dataset_sha, train_sha, development_sha = _validate_public_dataset(dataset_manifest, rows)
    populations = _validate_recovery(
        recovery_manifest,
        rows=rows,
        config=config,
        config_sha=config_sha,
        dataset_sha=dataset_sha,
        train_sha=train_sha,
        development_sha=development_sha,
    )
    bindings = {
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "recovery_manifest_sha256": rows["recovery/inputs.manifest.json"]["sha256"],
        "policy": PREPARATION_POLICY,
        "prepared_sha256": populations["canary"]["prepared_sha256"],
        "records": CANARY_RECORDS,
        "source_train_sha256": train_sha,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "overfit_canary": populations["canary"],
        "public_probe": populations["probe"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": OVERLAY_PRODUCER,
        "status": "complete",
        "target_run_id": target_run_id,
        "prefix": f"{OVERLAY_PREFIX}/{target_run_id}",
        "files": [rows[path] for path in sorted(rows)],
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
    """Combine v3 public files with the cached tokenizer and base checkpoint."""

    source_run_id = _run_id(source_run_id, "source_run_id")
    target_run_id = _run_id(target_run_id, "target_run_id")
    if source_run_id == target_run_id:
        raise ValueError("source and target run IDs must differ")
    source_sha = _sha(source_input_manifest_sha256, "source_input_manifest_sha256")
    _sha(roundtrip_completion_sha256, "roundtrip_completion_sha256")
    overlay_sha = _sha(overlay_manifest_sha256, "overlay_manifest_sha256")
    if manifest_sha256(source_input_manifest) != source_sha:
        raise ValueError("source input manifest checksum changed")
    if manifest_sha256(overlay_manifest) != overlay_sha:
        raise ValueError("v3 overlay manifest checksum changed")
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
        or roundtrip_completion.get("input_manifest_sha256") != source_sha
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
        raise ValueError("v3 overlay identity changed")

    source_rows = _safe_rows(source_input_manifest, "source input manifest")
    overlay_rows = _safe_rows(overlay_manifest, "v3 overlay manifest")
    bindings = overlay_manifest.get("bindings")
    if not isinstance(bindings, dict) or set(overlay_rows) != set(OVERLAY_PATHS):
        raise ValueError("v3 overlay population or bindings changed")
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
    if (
        bindings.get("tokenizer_manifest_sha256")
        != source_rows["tokenizer.manifest.json"]["sha256"]
    ):
        raise ValueError("v3 overlay is not bound to the cached tokenizer")

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
    canary_row = overlay_rows["recovery/overfit-canary.train.jsonl"]
    target_rows.append({**canary_row, "path": PREPARED_PATH})
    target_rows.extend(dict(source_rows[path]) for path in tokenizer_paths)
    for path, row in base_rows.items():
        release_path = f"base-orbax/{relative_leaf}/{path}"
        if release_rows.get(release_path) != {**row, "path": release_path}:
            raise ValueError("roundtrip completion does not bind every base Orbax byte")
        target_rows.append({**row, "path": f"checkpoint/{relative_leaf}/{path}"})

    receipt_row = release_rows.get("evidence/base-orbax.receipt.json")
    manifest_row = release_rows.get("base-orbax.manifest.json")
    if receipt_row is None or manifest_row is None:
        raise ValueError("roundtrip completion lacks base Orbax evidence")
    expected_completion = {
        "config_sha256": source_rows["config.json"]["sha256"],
        "dataset_manifest_sha256": source_rows["dataset/manifest.json"]["sha256"],
        "tokenizer_manifest_sha256": source_rows["tokenizer.manifest.json"]["sha256"],
        "base_orbax_receipt_sha256": receipt_row["sha256"],
        "base_orbax_manifest_sha256": manifest_row["sha256"],
    }
    if any(roundtrip_completion.get(key) != value for key, value in expected_completion.items()):
        raise ValueError("roundtrip completion source bindings changed")
    target_rows.extend(
        (
            {**receipt_row, "path": "checkpoint.receipt.json"},
            {**manifest_row, "path": "checkpoint.manifest.json"},
        )
    )
    paths = [row["path"] for row in target_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("v3, tokenizer, and checkpoint inputs overlap")
    target_rows.sort(key=lambda row: row["path"])
    prepared_training = {
        "policy": bindings.get("policy"),
        "records": bindings.get("records"),
        "prepared_sha256": bindings.get("prepared_sha256"),
        "source_train_sha256": bindings.get("source_train_sha256"),
        "tokenizer_manifest_sha256": bindings.get("tokenizer_manifest_sha256"),
        "recovery_manifest_sha256": bindings.get("recovery_manifest_sha256"),
        "overfit_canary": bindings.get("overfit_canary"),
        "public_probe": bindings.get("public_probe"),
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
    return f"APPROVE_MODAL_JAX_V3_RECOVERY_OVERLAY_STAGE:{target_run_id}:{overlay_manifest_sha256}"


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
        "APPROVE_MODAL_JAX_FULL_INPUT_V3_RECOVERY_CLONE:"
        f"{source_run_id}:{source_input_manifest_sha256}:"
        f"{roundtrip_completion_sha256}:{overlay_manifest_sha256}:"
        f"{target_run_id}:{target_manifest_sha256}"
    )
