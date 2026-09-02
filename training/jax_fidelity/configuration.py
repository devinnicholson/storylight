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
RECOVERY_EXPERIMENT_ID = "bookforge-gemma4-e2b-lora-r16-v3-canary"
RECOVERY_SCHEMA_VERSION = "bookforge-jax-v3-recovery-v1"
RECOVERY_SELECTION_POLICY = "public-balanced-counterfactual-recovery-v1"
RECOVERY_CATEGORY_COUNT = 20
# Pinned Gemma4 E2B: 15 * (Q/K/V/O) + 20 * (Q/O) KV-sharing
# attention targets + 35 * (wi_0/wi_1/wo) MLP targets.
GEMMA4_E2B_LORA_PAIR_COUNT = 205
MAXTEXT_NATIVE_LORA_PATCH_SHA256 = (
    "83fffa58387f879e87fc98bd6bf41390575b538e6513629fc5989ae3dcd163d2"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BARE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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


def _bare_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _BARE_SHA256.fullmatch(value) is None:
        raise ConfigError(f"{name} must be a lowercase SHA-256")
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

    @property
    def recovery(self) -> Mapping[str, Any]:
        recovery = self.data.get("recovery")
        if not isinstance(recovery, Mapping):
            raise ConfigError("configuration does not define a recovery experiment")
        return recovery


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
    if experiment_id.endswith("-v2") or experiment_id == RECOVERY_EXPERIMENT_ID:
        from .formatting import prompt_contract_sha256

        prompt_sha = production.get("prompt_contract_sha256")
        if (
            not isinstance(prompt_sha, str)
            or _BARE_SHA256.fullmatch(prompt_sha) is None
            or prompt_sha != prompt_contract_sha256()
        ):
            raise ConfigError(
                "production_contract.prompt_contract_sha256 must match the deployed prompt"
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
    if experiment_id.endswith("-v2"):
        _exact(
            training.get("preparation_policy"),
            "balanced-counterfactual-pairs-v1",
            "training.preparation_policy",
        )
        _exact(training.get("packing"), False, "training.packing")
        _exact(training.get("num_epoch"), 4, "training.num_epoch")
        _exact(
            training.get("enable_data_shuffling"),
            False,
            "training.enable_data_shuffling",
        )
        _exact(training.get("enable_dropout"), False, "training.enable_dropout")
        _exact(
            training.get("gradient_accumulation_steps"),
            1,
            "training.gradient_accumulation_steps",
        )
        _exact(
            training.get("gradient_clipping_threshold"),
            1.0,
            "training.gradient_clipping_threshold",
        )
        _exact(training.get("lr_schedule_type"), "cosine", "training.lr_schedule_type")
        _exact(
            training.get("warmup_steps_fraction"),
            0.05,
            "training.warmup_steps_fraction",
        )
        _exact(
            training.get("learning_rate_final_fraction"),
            0.1,
            "training.learning_rate_final_fraction",
        )
        _exact(training.get("adam_weight_decay"), 0.0, "training.adam_weight_decay")
    elif experiment_id == RECOVERY_EXPERIMENT_ID:
        _exact(
            dataset.get("manifest_path"),
            "datasets/story-fidelity-v2/manifest.json",
            "dataset.manifest_path",
        )
        _bare_sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")
        _exact(
            training.get("preparation_policy"),
            RECOVERY_SELECTION_POLICY,
            "training.preparation_policy",
        )
        _exact(training.get("steps"), 100, "training.steps")
        _exact(training.get("smoke_steps"), 1, "training.smoke_steps")
        _exact(training.get("rank"), 16, "training.rank")
        _exact(training.get("alpha"), 32.0, "training.alpha")
        _exact(
            training.get("expected_lora_pair_count"),
            GEMMA4_E2B_LORA_PAIR_COUNT,
            "training.expected_lora_pair_count",
        )
        _exact(
            training.get("approved_maxtext_patch_sha256"),
            MAXTEXT_NATIVE_LORA_PATCH_SHA256,
            "training.approved_maxtext_patch_sha256",
        )
        _exact(training.get("learning_rate"), 0.0001, "training.learning_rate")
        _exact(training.get("packing"), False, "training.packing")
        _exact(training.get("num_epoch"), 5, "training.num_epoch")
        _exact(
            training.get("enable_data_shuffling"),
            False,
            "training.enable_data_shuffling",
        )
        _exact(training.get("enable_dropout"), False, "training.enable_dropout")
        _exact(
            training.get("gradient_accumulation_steps"),
            1,
            "training.gradient_accumulation_steps",
        )
        _exact(
            training.get("gradient_clipping_threshold"),
            1.0,
            "training.gradient_clipping_threshold",
        )
        _exact(training.get("lr_schedule_type"), "cosine", "training.lr_schedule_type")
        _exact(
            training.get("warmup_steps_fraction"),
            0.0,
            "training.warmup_steps_fraction",
        )
        _exact(
            training.get("learning_rate_final_fraction"),
            1.0,
            "training.learning_rate_final_fraction",
        )
        _exact(training.get("adam_weight_decay"), 0.0, "training.adam_weight_decay")

        recovery = _mapping(document.get("recovery"), "recovery")
        expected_recovery_fields = {
            "schema_version",
            "selection_policy",
            "selection_seed",
            "overfit_canary",
            "public_probe",
            "disjoint_fields",
            "previous_attempt",
            "execution",
            "learnability_acceptance",
        }
        if set(recovery) != expected_recovery_fields:
            raise ConfigError("recovery must contain exactly the immutable v3 fields")
        _exact(
            recovery.get("schema_version"),
            RECOVERY_SCHEMA_VERSION,
            "recovery.schema_version",
        )
        _exact(
            recovery.get("selection_policy"),
            RECOVERY_SELECTION_POLICY,
            "recovery.selection_policy",
        )
        if type(recovery.get("selection_seed")) is not int:
            raise ConfigError("recovery.selection_seed must be an integer")
        _exact(
            recovery.get("disjoint_fields"),
            ["family_id", "pair_id", "passage_sha256", "record_id", "template_family"],
            "recovery.disjoint_fields",
        )
        learnability = _mapping(
            recovery.get("learnability_acceptance"),
            "recovery.learnability_acceptance",
        )
        if set(learnability) != {
            "schema_version",
            "nonzero_gradient_epsilon",
            "minimum_nonzero_gradient_fraction",
            "rolling_loss_window_steps",
            "minimum_rolling_loss_relative_reduction",
            "minimum_parameter_norm_relative_change",
        }:
            raise ConfigError(
                "recovery.learnability_acceptance has unexpected fields"
            )
        for name, expected in (
            ("schema_version", "bookforge-jax-v3-learnability-acceptance-v1"),
            ("nonzero_gradient_epsilon", 1e-12),
            ("minimum_nonzero_gradient_fraction", 0.9),
            ("rolling_loss_window_steps", 20),
            ("minimum_rolling_loss_relative_reduction", 0.1),
            ("minimum_parameter_norm_relative_change", 1e-6),
        ):
            _exact(
                learnability.get(name),
                expected,
                f"recovery.learnability_acceptance.{name}",
            )

        population_fields = {
            "split",
            "pairs_per_category",
            "pairs",
            "records",
            "record_ids_sha256",
            "pair_ids_sha256",
            "source_sha256",
            "prepared_sha256",
        }
        for name, split, pairs_per_category in (
            ("overfit_canary", "train", 1),
            ("public_probe", "development", 2),
        ):
            population = _mapping(recovery.get(name), f"recovery.{name}")
            if set(population) != population_fields:
                raise ConfigError(f"recovery.{name} has unexpected fields")
            _exact(population.get("split"), split, f"recovery.{name}.split")
            _exact(
                population.get("pairs_per_category"),
                pairs_per_category,
                f"recovery.{name}.pairs_per_category",
            )
            _exact(
                population.get("pairs"),
                RECOVERY_CATEGORY_COUNT * pairs_per_category,
                f"recovery.{name}.pairs",
            )
            _exact(
                population.get("records"),
                RECOVERY_CATEGORY_COUNT * 2 * pairs_per_category,
                f"recovery.{name}.records",
            )
            for field in (
                "record_ids_sha256",
                "pair_ids_sha256",
                "source_sha256",
                "prepared_sha256",
            ):
                _bare_sha256(population.get(field), f"recovery.{name}.{field}")

        previous = _mapping(recovery.get("previous_attempt"), "recovery.previous_attempt")
        if set(previous) != {
            "experiment_id",
            "config_sha256",
            "training_run_id",
            "training_completion_sha256",
            "development_rejection_sha256",
            "training_events_sha256",
        }:
            raise ConfigError("recovery.previous_attempt has unexpected fields")
        _exact(
            previous.get("experiment_id"),
            "bookforge-gemma4-e2b-lora-r16-v2",
            "recovery.previous_attempt.experiment_id",
        )
        training_run_id = previous.get("training_run_id")
        if not isinstance(training_run_id, str) or re.fullmatch(
            r"lora-train-[0-9a-f]{20}", training_run_id
        ) is None:
            raise ConfigError("recovery.previous_attempt.training_run_id is invalid")
        for field in (
            "config_sha256",
            "training_completion_sha256",
            "development_rejection_sha256",
            "training_events_sha256",
        ):
            _bare_sha256(previous.get(field), f"recovery.previous_attempt.{field}")

        execution = _mapping(recovery.get("execution"), "recovery.execution")
        if set(execution) != {
            "diagnostic_only",
            "expected_accelerators",
            "expected_global_batch_size",
            "full_development_evaluation_authorized",
            "merge_authorized",
        }:
            raise ConfigError("recovery.execution has unexpected fields")
        _exact(execution.get("diagnostic_only"), True, "recovery.execution.diagnostic_only")
        _exact(
            execution.get("expected_accelerators"),
            2,
            "recovery.execution.expected_accelerators",
        )
        _exact(
            execution.get("expected_global_batch_size"),
            2,
            "recovery.execution.expected_global_batch_size",
        )
        _exact(
            execution.get("full_development_evaluation_authorized"),
            False,
            "recovery.execution.full_development_evaluation_authorized",
        )
        _exact(
            execution.get("merge_authorized"),
            False,
            "recovery.execution.merge_authorized",
        )

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
