"""Fail-closed verification for a staged v3 public recovery population."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .configuration import RECOVERY_EXPERIMENT_ID, ExperimentConfig
from .integrity import sha256_file
from .recovery_inputs import PRODUCER, SCHEMA_VERSION, SELECTION_POLICY

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECOVERY_FILES = {
    "overfit-canary.source.jsonl",
    "overfit-canary.train.jsonl",
    "public-probe.source.jsonl",
    "public-probe.teacher.jsonl",
}


class RecoveryStagingError(RuntimeError):
    """The staged recovery population differs from its immutable contract."""


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryStagingError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryStagingError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RecoveryStagingError(f"{label} must contain one object")
    return value


def _manifest_rows(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rows = document.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise RecoveryStagingError("recovery manifest has no file declarations")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("path"), str)
            or type(raw.get("bytes")) is not int
            or int(raw["bytes"]) < 0
            or not isinstance(raw.get("sha256"), str)
            or _SHA256.fullmatch(raw["sha256"]) is None
        ):
            raise RecoveryStagingError("recovery manifest file declaration is malformed")
        pure = PurePosixPath(raw["path"])
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != raw["path"]:
            raise RecoveryStagingError("recovery manifest contains an unsafe path")
        if raw["path"] in rows:
            raise RecoveryStagingError("recovery manifest contains a duplicate path")
        rows[raw["path"]] = dict(raw)
    return rows


def _verify_file(path: Path, row: Mapping[str, Any], label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RecoveryStagingError(f"{label} must be a regular file")
    if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
        raise RecoveryStagingError(f"{label} differs from its recovery manifest")


def _configured_population(actual: object, configured: object, *, label: str) -> dict[str, Any]:
    if not isinstance(actual, dict) or not isinstance(configured, Mapping):
        raise RecoveryStagingError(f"{label} population binding is malformed")
    keys = {
        "split",
        "pairs",
        "records",
        "record_ids_sha256",
        "pair_ids_sha256",
        "source_sha256",
        "prepared_sha256",
    }
    if any(actual.get(key) != configured.get(key) for key in keys):
        raise RecoveryStagingError(f"{label} population differs from the v3 config")
    return actual


def verify_recovery_training_input(
    input_directory: Path,
    *,
    experiment: ExperimentConfig,
    input_manifest: Mapping[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    tokenizer_manifest_sha256: str,
) -> dict[str, Any] | None:
    """Verify canary-only training and the untouched disjoint public probe."""

    if experiment.experiment_id != RECOVERY_EXPERIMENT_ID:
        return None
    recovery_config = experiment.recovery
    if config_sha256 != experiment.sha256:
        raise RecoveryStagingError("v3 config checksum changed after loading")

    input_rows = input_manifest.get("files")
    if not isinstance(input_rows, list):
        raise RecoveryStagingError("staged input manifest has no file population")
    fixed_paths = {
        "config.json",
        "dataset/manifest.json",
        "dataset/train.jsonl",
        "dataset/development.jsonl",
        "prepared/train.jsonl",
        "recovery/inputs.manifest.json",
        *(f"recovery/{path}" for path in _RECOVERY_FILES),
        "tokenizer.manifest.json",
        "checkpoint.manifest.json",
        "checkpoint.receipt.json",
    }
    for row in input_rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise RecoveryStagingError("staged input file declaration is malformed")
        pure = PurePosixPath(relative)
        if any("hidden" in part.casefold() for part in pure.parts):
            raise RecoveryStagingError("hidden bytes are forbidden from v3 recovery inputs")
        if (
            relative not in fixed_paths
            and not relative.startswith("tokenizer/")
            and not relative.startswith("checkpoint/")
        ):
            raise RecoveryStagingError("v3 staged input contains an unapproved file")

    recovery_root = input_directory / "recovery"
    recovery_manifest_path = recovery_root / "inputs.manifest.json"
    recovery_manifest = _json_object(recovery_manifest_path, "recovery manifest")
    recovery_manifest_sha = sha256_file(recovery_manifest_path)
    rows = _manifest_rows(recovery_manifest)
    if set(rows) != _RECOVERY_FILES:
        raise RecoveryStagingError("recovery file population changed")
    for relative, row in rows.items():
        _verify_file(recovery_root / relative, row, f"recovery file {relative}")

    dataset_manifest_path = input_directory / "dataset/manifest.json"
    dataset_manifest = _json_object(dataset_manifest_path, "dataset manifest")
    splits = dataset_manifest.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    development = splits.get("development") if isinstance(splits, dict) else None
    hidden = splits.get("hidden") if isinstance(splits, dict) else None
    train_path = input_directory / "dataset/train.jsonl"
    development_path = input_directory / "dataset/development.jsonl"
    if (
        dataset_manifest.get("dataset_id") != "story-fidelity-v2"
        or sha256_file(dataset_manifest_path) != experiment.dataset["manifest_sha256"]
        or not isinstance(train, dict)
        or train.get("path") != "train.jsonl"
        or train.get("public") is not True
        or train.get("sha256") != sha256_file(train_path)
        or not isinstance(development, dict)
        or development.get("path") != "development.jsonl"
        or development.get("public") is not True
        or development.get("sha256") != sha256_file(development_path)
        or not isinstance(hidden, dict)
        or hidden.get("path") is not None
        or hidden.get("public") is not False
    ):
        raise RecoveryStagingError("v3 dataset does not bind only public bytes")

    populations = recovery_manifest.get("populations")
    if not isinstance(populations, dict):
        raise RecoveryStagingError("recovery populations are missing")
    canary = _configured_population(
        populations.get("overfit_canary"),
        recovery_config.get("overfit_canary"),
        label="overfit canary",
    )
    probe = _configured_population(
        populations.get("public_probe"),
        recovery_config.get("public_probe"),
        label="public probe",
    )
    disjoint = recovery_manifest.get("disjoint")
    if (
        recovery_manifest.get("schema_version") != SCHEMA_VERSION
        or recovery_manifest.get("producer") != PRODUCER
        or recovery_manifest.get("status") != "complete"
        or recovery_manifest.get("experiment_id") != experiment.experiment_id
        or recovery_manifest.get("config_sha256") != config_sha256
        or recovery_manifest.get("dataset")
        != {
            "dataset_id": "story-fidelity-v2",
            "manifest_sha256": experiment.dataset["manifest_sha256"],
            "train_sha256": train.get("sha256"),
            "development_sha256": development.get("sha256"),
        }
        or recovery_manifest.get("selection")
        != {
            "policy": SELECTION_POLICY,
            "seed": recovery_config["selection_seed"],
        }
        or recovery_manifest.get("lineage") != recovery_config["previous_attempt"]
        or recovery_manifest.get("learnability_acceptance")
        != recovery_config["learnability_acceptance"]
        or not isinstance(disjoint, dict)
        or set(disjoint) != set(recovery_config["disjoint_fields"])
        or not all(value is True for value in disjoint.values())
        or recovery_manifest.get("privacy")
        != {"public_records_only": True, "hidden_records_included": False}
    ):
        raise RecoveryStagingError("sealed v3 recovery contract changed")

    canary_source = rows["overfit-canary.source.jsonl"]["sha256"]
    canary_teacher = rows["overfit-canary.train.jsonl"]["sha256"]
    probe_source = rows["public-probe.source.jsonl"]["sha256"]
    probe_teacher = rows["public-probe.teacher.jsonl"]["sha256"]
    if (
        canary.get("purpose") != "overfit-canary"
        or canary.get("split") != "train"
        or canary.get("source_sha256") != canary_source
        or canary.get("prepared_sha256") != canary_teacher
        or probe.get("purpose") != "public-probe"
        or probe.get("split") != "development"
        or probe.get("source_sha256") != probe_source
        or probe.get("prepared_sha256") != probe_teacher
        or len({canary_source, canary_teacher, probe_source, probe_teacher}) != 4
    ):
        raise RecoveryStagingError("canary/probe byte bindings changed")

    prepared = input_directory / "prepared/train.jsonl"
    if (
        prepared_sha256 != canary_teacher
        or sha256_file(prepared) != canary_teacher
        or prepared.read_bytes() != (recovery_root / "overfit-canary.train.jsonl").read_bytes()
        or prepared.read_bytes() == (recovery_root / "public-probe.teacher.jsonl").read_bytes()
    ):
        raise RecoveryStagingError("training input is not exactly the overfit canary")

    binding = {
        "policy": SELECTION_POLICY,
        "records": canary["records"],
        "prepared_sha256": canary_teacher,
        "source_train_sha256": train["sha256"],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "recovery_manifest_sha256": recovery_manifest_sha,
        "overfit_canary": canary,
        "public_probe": probe,
    }
    if input_manifest.get("prepared_training") != binding:
        raise RecoveryStagingError("staged v3 recovery binding changed")
    return binding
