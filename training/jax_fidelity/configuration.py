"""Strict configuration loading for the bounded Gemma 4 LoRA experiment."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "1.0"
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MAXTEXT_MODEL_NAME = "gemma4-e2b"
PROMPT_SOURCE = "src/bookforge/tensorrt_slot_client.py:_slot_messages"
SLOT_LABELS = ("SETTING", "ACTOR", "ACTION", "MAGIC")
EOS_TOKEN_IDS = (1, 106, 50)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class ConfigError(ValueError):
    """The immutable experiment contract is invalid or unexpectedly changed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ConfigError(f"{name} must be exactly {expected!r}")


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ConfigError(f"{name} must be an integer in 1..{maximum}")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment data plus its content hash."""

    path: Path
    sha256: str
    data: Mapping[str, Any]

    @property
    def experiment_id(self) -> str:
        return str(self.data["experiment_id"])

    @property
    def production(self) -> Mapping[str, Any]:
        return self.data["production_contract"]

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.data["dataset"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.data["training"]

    @property
    def conversion(self) -> Mapping[str, Any]:
        return self.data["conversion"]

    @property
    def versions(self) -> Mapping[str, Any]:
        return self.data["versions"]


def validate_config(document: Mapping[str, Any]) -> None:
    """Reject drift that could change architecture, masking, or export compatibility."""

    _exact(document.get("schema_version"), SCHEMA_VERSION, "schema_version")
    experiment_id = document.get("experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{4,80}", experiment_id
    ):
        raise ConfigError("experiment_id must be a stable lowercase slug")

    production = _mapping(document.get("production_contract"), "production_contract")
    _exact(production.get("model_id"), MODEL_ID, "production_contract.model_id")
    _exact(production.get("model_revision"), MODEL_REVISION, "production_contract.model_revision")
    _exact(
        production.get("maxtext_model_name"),
        MAXTEXT_MODEL_NAME,
        "production_contract.maxtext_model_name",
    )
    _exact(production.get("prompt_source"), PROMPT_SOURCE, "production_contract.prompt_source")
    _exact(production.get("slots"), list(SLOT_LABELS), "production_contract.slots")
    _exact(production.get("scan_layers"), False, "production_contract.scan_layers")
    _exact(production.get("use_multimodal"), False, "production_contract.use_multimodal")
    _exact(production.get("input_budget_tokens"), 512, "production_contract.input_budget_tokens")
    _exact(
        production.get("completion_budget_tokens"),
        64,
        "production_contract.completion_budget_tokens",
    )
    _exact(
        production.get("eos_token_ids"),
        list(EOS_TOKEN_IDS),
        "production_contract.eos_token_ids",
    )

    dataset = _mapping(document.get("dataset"), "dataset")
    manifest_path = dataset.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.endswith(".json"):
        raise ConfigError("dataset.manifest_path must be a JSON path")
    _exact(dataset.get("hash_algorithm"), "sha256", "dataset.hash_algorithm")
    split_records = _mapping(
        dataset.get("required_split_records"), "dataset.required_split_records"
    )
    _exact(split_records.get("train"), 4096, "dataset.required_split_records.train")
    _exact(split_records.get("development"), 512, "dataset.required_split_records.development")
    _exact(split_records.get("hidden"), 512, "dataset.required_split_records.hidden")

    training = _mapping(document.get("training"), "training")
    _exact(training.get("method"), "lora", "training.method")
    _exact(training.get("completion_only"), True, "training.completion_only")
    _exact(training.get("weight_quantization"), None, "training.weight_quantization")
    rank = training.get("rank")
    if rank not in (8, 16) or type(rank) is not int:
        raise ConfigError("training.rank must be 8 or 16")
    alpha = training.get("alpha")
    if type(alpha) not in (int, float) or float(alpha) <= 0:
        raise ConfigError("training.alpha must be positive")
    _positive_int(training.get("steps"), "training.steps", maximum=10_000)
    _positive_int(training.get("smoke_steps"), "training.smoke_steps", maximum=10)
    _positive_int(
        training.get("per_device_batch_size"),
        "training.per_device_batch_size",
        maximum=8,
    )
    _exact(training.get("max_target_length"), 576, "training.max_target_length")
    learning_rate = training.get("learning_rate")
    if type(learning_rate) not in (int, float) or not 0 < float(learning_rate) <= 1e-3:
        raise ConfigError("training.learning_rate must be in (0, 1e-3]")
    if type(training.get("seed")) is not int:
        raise ConfigError("training.seed must be an integer")
    _exact(training.get("dtype"), "bfloat16", "training.dtype")
    _exact(training.get("weight_dtype"), "bfloat16", "training.weight_dtype")

    conversion = _mapping(document.get("conversion"), "conversion")
    _exact(conversion.get("scan_layers"), False, "conversion.scan_layers")
    _exact(conversion.get("use_multimodal"), False, "conversion.use_multimodal")
    _exact(conversion.get("max_kl_divergence"), 0.03, "conversion.max_kl_divergence")
    _exact(conversion.get("save_dtype"), "bfloat16", "conversion.save_dtype")
    _exact(conversion.get("eos_token_ids"), list(EOS_TOKEN_IDS), "conversion.eos_token_ids")

    versions = _mapping(document.get("versions"), "versions")
    _exact(versions.get("python"), "3.12.11", "versions.python")
    _exact(versions.get("jax"), "0.11.0", "versions.jax")
    _exact(versions.get("jaxlib"), "0.11.0", "versions.jaxlib")
    _exact(versions.get("flax"), "0.12.8", "versions.flax")
    _exact(versions.get("optax"), "0.2.8", "versions.optax")
    _exact(versions.get("orbax_checkpoint"), "0.12.2", "versions.orbax_checkpoint")
    _exact(versions.get("maxtext_release"), "0.2.4", "versions.maxtext_release")
    maxtext_revision = versions.get("maxtext_revision")
    if not isinstance(maxtext_revision, str) or not _GIT_REVISION.fullmatch(maxtext_revision):
        raise ConfigError("versions.maxtext_revision must be a 40-character Git revision")
    container = versions.get("container_image")
    pinned_container = (
        isinstance(container, str)
        and "@" in container
        and _SHA256.fullmatch(container.rsplit("@", 1)[1]) is not None
    )
    if not pinned_container:
        raise ConfigError("versions.container_image must contain an immutable sha256 digest")


def load_config(path: Path | str) -> ExperimentConfig:
    """Read and validate a JSON configuration without importing JAX."""

    from .integrity import sha256_file

    config_path = Path(path)
    try:
        document = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"could not load configuration: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError("configuration root must be an object")
    validate_config(document)
    locked_versions = json.loads(
        Path(__file__).with_name("versions.json").read_text(encoding="utf-8")
    )
    if document["versions"] != locked_versions:
        raise ConfigError("configuration versions differ from training/jax_fidelity/versions.json")
    return ExperimentConfig(
        path=config_path.resolve(),
        sha256=sha256_file(config_path),
        data=_deep_freeze(document),
    )
