# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bookforge.fidelity_dataset import CATEGORIES
from bookforge.fidelity_schema import DatasetSplit
from training.jax_fidelity.configuration import ConfigError, load_config, validate_config
from training.jax_fidelity.integrity import canonical_json_bytes, sha256_file
from training.jax_fidelity.recovery_inputs import (
    EXPERIMENT_ID,
    PRODUCER,
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


def test_v3_config_is_a_fail_closed_diagnostic_contract() -> None:
    config = load_config(CONFIG)

    assert config.experiment_id == EXPERIMENT_ID
    assert config.training["steps"] == 100
    assert config.training["learning_rate"] == 0.0001
    assert config.training["preparation_policy"] == config.recovery["selection_policy"]
    assert config.recovery["overfit_canary"]["records"] == 40
    assert config.recovery["public_probe"]["records"] == 80
    assert config.recovery["execution"] == {
        "diagnostic_only": True,
        "expected_accelerators": 2,
        "expected_global_batch_size": 2,
        "full_development_evaluation_authorized": False,
        "merge_authorized": False,
    }

    for path, value in (
        (("training", "learning_rate"), 0.0002),
        (("recovery", "overfit_canary", "records"), 42),
        (("recovery", "execution", "merge_authorized"), True),
        (
            (
                "recovery",
                "learnability_acceptance",
                "minimum_nonzero_gradient_fraction",
            ),
            0.01,
        ),
    ):
        drifted = json.loads(CONFIG.read_text())
        target = drifted
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        with pytest.raises(ConfigError):
            validate_config(drifted)


def test_builder_pins_balanced_public_populations_and_writes_manifest_last(
    tmp_path: Path,
) -> None:
    result, output, _ = _build(tmp_path, destination="inputs")
    manifest = json.loads((output / "inputs.manifest.json").read_text())

    assert manifest["producer"] == PRODUCER
    assert manifest["status"] == "complete"
    assert result["manifest_sha256"] == sha256_file(output / "inputs.manifest.json")
    assert manifest["privacy"] == {
        "public_records_only": True,
        "hidden_records_included": False,
    }
    assert (
        manifest["learnability_acceptance"]
        == json.loads((output.parent / "config-v3.json").read_text())["recovery"][
            "learnability_acceptance"
        ]
    )
    canary = manifest["populations"]["overfit_canary"]
    probe = manifest["populations"]["public_probe"]
    assert (canary["records"], canary["pairs"]) == (40, 20)
    assert (probe["records"], probe["pairs"]) == (80, 40)
    assert set(canary["category_pair_counts"]) == set(CATEGORIES)
    assert set(canary["category_pair_counts"].values()) == {1}
    assert set(probe["category_pair_counts"].values()) == {2}
    assert manifest["disjoint"] == {
        "family_id": True,
        "pair_id": True,
        "passage_sha256": True,
        "record_id": True,
        "template_family": True,
    }
    assert {row["path"] for row in manifest["files"]} == {
        "overfit-canary.source.jsonl",
        "overfit-canary.train.jsonl",
        "public-probe.source.jsonl",
        "public-probe.teacher.jsonl",
    }

    source_rows = [
        json.loads(line)
        for line in (output / "overfit-canary.source.jsonl").read_text().splitlines()
    ]
    prepared_rows = [
        json.loads(line)
        for line in (output / "overfit-canary.train.jsonl").read_text().splitlines()
    ]
    assert [row["record_id"] for row in prepared_rows] == [row["record_id"] for row in source_rows]
    assert all(
        source_rows[index]["pair_id"] == source_rows[index + 1]["pair_id"]
        and [source_rows[index]["pair_variant"], source_rows[index + 1]["pair_variant"]]
        == ["a", "b"]
        for index in range(0, len(source_rows), 2)
    )
    assert all(
        [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        for row in prepared_rows
    )


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
