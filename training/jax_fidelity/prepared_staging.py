"""Fail-closed verification for staged v2 prepared-training evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import ExperimentConfig
from .integrity import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PreparedStagingError(RuntimeError):
    """Staged prepared-training evidence differs from its immutable contract."""


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparedStagingError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparedStagingError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PreparedStagingError(f"{label} must contain one object")
    return value


def verify_v2_prepared_training_input(
    input_directory: Path,
    *,
    experiment: ExperimentConfig,
    input_manifest: Mapping[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    tokenizer_manifest_sha256: str,
) -> dict[str, Any] | None:
    """Verify the preparation manifest and token-budget validation for v2."""

    if not experiment.experiment_id.endswith("-v2"):
        return None
    production = experiment.production
    training = experiment.training

    preparation_path = input_directory / "prepared/preparation.manifest.json"
    validation_path = input_directory / "prepared/prepared-validation.json"
    preparation = _json_object(preparation_path, label="v2 preparation manifest")
    validation = _json_object(validation_path, label="v2 prepared-validation evidence")
    preparation_sha256 = sha256_file(preparation_path)
    validation_sha256 = sha256_file(validation_path)
    prompt_sha256 = production.get("prompt_contract_sha256")
    policy = training.get("preparation_policy")
    records = preparation.get("prepared_records")

    dataset_manifest = _json_object(
        input_directory / "dataset/manifest.json",
        label="v2 dataset manifest",
    )
    splits = dataset_manifest.get("splits")
    train_split = splits.get("train") if isinstance(splits, dict) else None
    source_train_sha256 = train_split.get("sha256") if isinstance(train_split, dict) else None
    if not isinstance(source_train_sha256, str) or _SHA256.fullmatch(source_train_sha256) is None:
        raise PreparedStagingError("v2 dataset manifest has no checksum-bound train split")
    source_train_path = input_directory / "dataset/train.jsonl"
    if (
        source_train_path.is_symlink()
        or not source_train_path.is_file()
        or sha256_file(source_train_path) != source_train_sha256
    ):
        raise PreparedStagingError("v2 source training bytes changed")

    if (
        preparation.get("schema_version") != "bookforge-jax-training-preparation-v2"
        or preparation.get("policy") != policy
        or type(records) is not int
        or records < 1
        or preparation.get("source_train_sha256") != source_train_sha256
        or preparation.get("prepared_sha256") != prepared_sha256
        or preparation.get("prompt_contract_sha256") != prompt_sha256
        or preparation.get("assistant_turns_per_record") != 1
        or preparation.get("pair_adjacency_preserved") is not True
    ):
        raise PreparedStagingError("v2 preparation manifest does not match the training contract")

    expected_validation = {
        "schema_version": "bookforge-jax-prepared-validation-v1",
        "status": "passed",
        "config_sha256": config_sha256,
        "prepared_sha256": prepared_sha256,
        "preparation_manifest_sha256": preparation_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "prompt_contract_sha256": prompt_sha256,
        "records": records,
        "assistant_turns_per_record": 1,
        "input_budget_tokens": production.get("input_budget_tokens"),
        "completion_budget_tokens": production.get("completion_budget_tokens"),
        "max_target_length": training.get("max_target_length"),
    }
    if any(validation.get(name) != value for name, value in expected_validation.items()):
        raise PreparedStagingError("v2 prepared-validation evidence does not match staged inputs")
    for name in (
        "maximum_prompt_tokens",
        "maximum_completion_tokens",
        "maximum_total_tokens",
    ):
        if type(validation.get(name)) is not int or int(validation[name]) < 1:
            raise PreparedStagingError("v2 prepared-validation token evidence is invalid")
    if (
        int(validation["maximum_prompt_tokens"]) > int(production["input_budget_tokens"])
        or int(validation["maximum_completion_tokens"])
        > int(production["completion_budget_tokens"])
        or int(validation["maximum_total_tokens"]) > int(training["max_target_length"])
    ):
        raise PreparedStagingError(
            "v2 prepared-validation evidence exceeds the approved token budgets"
        )

    binding = {
        "policy": policy,
        "records": records,
        "prepared_sha256": prepared_sha256,
        "preparation_manifest_sha256": preparation_sha256,
        "prepared_validation_sha256": validation_sha256,
        "prompt_contract_sha256": prompt_sha256,
        "source_train_sha256": source_train_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
    }
    if input_manifest.get("prepared_training") != binding:
        raise PreparedStagingError("staged input manifest v2 preparation binding changed")
    return binding
