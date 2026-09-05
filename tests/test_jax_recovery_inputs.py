# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bookforge.fidelity_schema import DatasetSplit
from training.jax_fidelity.integrity import canonical_json_bytes, sha256_file
from training.jax_fidelity.recovery_inputs import (
    RecoveryInputError,
    _assert_disjoint,
    _load_split,
    _select_population,
    build_recovery_inputs,
)

CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"
REJECTED_CONFIG = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
DATASET_MANIFEST = ROOT / "datasets/story-fidelity-v2/manifest.json"
TRAIN = ROOT / "datasets/story-fidelity-v2/train.jsonl"


def _write_json(path: Path, document: object) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _lineage_fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    training_run_id = "lora-train-2cd8c181b1ba1183c17d"
    rejected_config = tmp_path / "config-v2.json"
    rejected_config.write_bytes(REJECTED_CONFIG.read_bytes())
    training = tmp_path / "training-completion.json"
    _write_json(
        training,
        {"schema_version": "1.0", "status": "succeeded", "run_id": training_run_id},
    )
    rejection = tmp_path / "development-rejection.json"
    _write_json(
        rejection,
        {
            "schema_version": "bookforge-jax-development-rejection-v1",
            "status": "rejected",
            "training_run_id": training_run_id,
            "config_sha256": sha256_file(rejected_config),
            "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST),
            "checkpoint_bytes_downloaded": False,
        },
    )
    events = tmp_path / "training.tfevents"
    events.write_bytes(b"sealed zero-gradient training event evidence")

    config_document = json.loads(CONFIG.read_text())
    previous = config_document["recovery"]["previous_attempt"]
    previous.update(
        {
            "config_sha256": sha256_file(rejected_config),
            "training_completion_sha256": sha256_file(training),
            "development_rejection_sha256": sha256_file(rejection),
            "training_events_sha256": sha256_file(events),
        }
    )
    config = tmp_path / "config-v3.json"
    _write_json(config, config_document)
    return {
        "config": config,
        "rejected_config": rejected_config,
        "training": training,
        "rejection": rejection,
        "events": events,
    }


def _build(tmp_path: Path, *, destination: str) -> tuple[dict, Path, dict[str, Path]]:
    lineage = _lineage_fixture(tmp_path)
    output = tmp_path / destination
    result = build_recovery_inputs(
        config_path=lineage["config"],
        dataset_manifest_path=DATASET_MANIFEST,
        rejected_config_path=lineage["rejected_config"],
        rejected_training_completion_path=lineage["training"],
        rejected_development_rejection_path=lineage["rejection"],
        rejected_training_events_path=lineage["events"],
        output_directory=output,
    )
    return result, output, lineage


def test_builder_is_byte_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    first, first_root, lineage = _build(tmp_path, destination="first")
    second_root = tmp_path / "second"
    second = build_recovery_inputs(
        config_path=lineage["config"],
        dataset_manifest_path=DATASET_MANIFEST,
        rejected_config_path=lineage["rejected_config"],
        rejected_training_completion_path=lineage["training"],
        rejected_development_rejection_path=lineage["rejection"],
        rejected_training_events_path=lineage["events"],
        output_directory=second_root,
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert {path.name: path.read_bytes() for path in sorted(first_root.iterdir())} == {
        path.name: path.read_bytes() for path in sorted(second_root.iterdir())
    }
    with pytest.raises(FileExistsError):
        build_recovery_inputs(
            config_path=lineage["config"],
            dataset_manifest_path=DATASET_MANIFEST,
            rejected_config_path=lineage["rejected_config"],
            rejected_training_completion_path=lineage["training"],
            rejected_development_rejection_path=lineage["rejection"],
            rejected_training_events_path=lineage["events"],
            output_directory=first_root,
        )


def test_builder_rejects_lineage_or_selection_drift(tmp_path: Path) -> None:
    lineage = _lineage_fixture(tmp_path)
    lineage["events"].write_bytes(b"changed")
    with pytest.raises(RecoveryInputError, match="training events"):
        build_recovery_inputs(
            config_path=lineage["config"],
            dataset_manifest_path=DATASET_MANIFEST,
            rejected_config_path=lineage["rejected_config"],
            rejected_training_completion_path=lineage["training"],
            rejected_development_rejection_path=lineage["rejection"],
            rejected_training_events_path=lineage["events"],
            output_directory=tmp_path / "bad-lineage",
        )

    lineage = _lineage_fixture(tmp_path / "selection")
    drifted = json.loads(lineage["config"].read_text())
    drifted["recovery"]["selection_seed"] += 1
    _write_json(lineage["config"], drifted)
    with pytest.raises(RecoveryInputError, match="configured population"):
        build_recovery_inputs(
            config_path=lineage["config"],
            dataset_manifest_path=DATASET_MANIFEST,
            rejected_config_path=lineage["rejected_config"],
            rejected_training_completion_path=lineage["training"],
            rejected_development_rejection_path=lineage["rejection"],
            rejected_training_events_path=lineage["events"],
            output_directory=tmp_path / "bad-selection",
        )


def test_disjointness_gate_rejects_any_reused_public_records() -> None:
    train = _load_split(TRAIN, split=DatasetSplit.TRAIN)
    canary = _select_population(
        train,
        seed=20260902,
        purpose="overfit-canary",
        pairs_per_category=1,
    )

    with pytest.raises(RecoveryInputError, match="overlap"):
        _assert_disjoint(canary, deepcopy(canary))
