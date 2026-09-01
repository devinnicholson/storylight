"""Reproducible, fail-closed packaging for Bookforge's offline JAX lab."""

from .configuration import ConfigError, ExperimentConfig, load_config
from .formatting import (
    SLOT_LABELS,
    completion_only_example,
    format_training_record,
    production_messages,
    validate_slot_target,
)
from .integrity import DatasetIntegrityError, sha256_file, validate_dataset_manifest
from .manifests import ManifestError, complete_run, start_run

__all__ = [
    "ConfigError",
    "DatasetIntegrityError",
    "ExperimentConfig",
    "ManifestError",
    "SLOT_LABELS",
    "complete_run",
    "completion_only_example",
    "format_training_record",
    "load_config",
    "production_messages",
    "sha256_file",
    "start_run",
    "validate_dataset_manifest",
    "validate_slot_target",
]
