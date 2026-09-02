"""Build immutable public inputs for the v3 LoRA learnability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bookforge.fidelity_dataset import CATEGORIES
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord, PairVariant

from .configuration import ExperimentConfig, load_config
from .formatting import format_training_record
from .integrity import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
    validate_dataset_manifest,
)
from .prepare import _pair_content, _safe_target

SCHEMA_VERSION = "bookforge-jax-v3-recovery-inputs-v1"
PRODUCER = "bookforge-jax-v3-recovery-input-builder"
SELECTION_POLICY = "public-balanced-counterfactual-recovery-v1"
EXPERIMENT_ID = "bookforge-gemma4-e2b-lora-r16-v3-canary"

_OUTPUT_PATHS = (
    "overfit-canary.source.jsonl",
    "overfit-canary.train.jsonl",
    "public-probe.source.jsonl",
    "public-probe.teacher.jsonl",
)


class RecoveryInputError(ValueError):
    """The recovery population or its rejected-run lineage is invalid."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RecoveryInputError(f"{label} must be a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryInputError(f"could not read {label}: {error}") from error
    if not isinstance(document, dict):
        raise RecoveryInputError(f"{label} must contain one JSON object")
    return document


def _approved_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise RecoveryInputError(f"{label} differs from its configured SHA-256")
    return _load_json(path, label=label)


def _load_split(path: Path, *, split: DatasetSplit) -> list[FidelityRecord]:
    records: list[FidelityRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise RecoveryInputError(f"{split.value} record {line_number} is blank")
        try:
            record = FidelityRecord.model_validate_json(line)
        except ValueError as error:
            raise RecoveryInputError(
                f"could not validate {split.value} record {line_number}: {error}"
            ) from error
        if record.split is not split:
            raise RecoveryInputError(
                f"{record.record_id} declares {record.split.value}, expected {split.value}"
            )
        if len(record.categories) != 1:
            raise RecoveryInputError(f"{record.record_id} must have exactly one category")
        _safe_target(record)
        records.append(record)
    return records


def _complete_distinct_pairs(
    records: Sequence[FidelityRecord],
) -> dict[str, list[tuple[bytes, list[FidelityRecord]]]]:
    grouped: dict[str, list[FidelityRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pair_id].append(record)

    distinct: dict[str, dict[bytes, list[FidelityRecord]]] = defaultdict(dict)
    content_category: dict[bytes, str] = {}
    for pair_id, pair in sorted(grouped.items()):
        variants = {record.pair_variant for record in pair}
        categories = {record.categories[0] for record in pair}
        families = {record.family_id for record in pair}
        dimensions = {record.counterfactual_dimension for record in pair}
        if (
            len(pair) != 2
            or variants != {PairVariant.A, PairVariant.B}
            or len(categories) != 1
            or len(families) != 1
            or len(dimensions) != 1
        ):
            raise RecoveryInputError(f"counterfactual pair {pair_id} is incomplete or inconsistent")
        category = next(iter(categories))
        if category not in CATEGORIES or next(iter(dimensions)) != category:
            raise RecoveryInputError(f"counterfactual pair {pair_id} has an invalid category")
        content = _pair_content(pair)
        previous_category = content_category.setdefault(content, category)
        if previous_category != category:
            raise RecoveryInputError("identical pair content appears under multiple categories")
        distinct[category].setdefault(content, pair)

    if set(distinct) != set(CATEGORIES):
        raise RecoveryInputError("public split does not cover every fidelity category")
    return {
        category: sorted(values.items(), key=lambda item: item[0])
        for category, values in distinct.items()
    }


def _selection_key(*, seed: int, purpose: str, category: str, content: bytes) -> bytes:
    prefix = f"{SELECTION_POLICY}:{seed}:{purpose}:{category}:".encode()
    return hashlib.sha256(prefix + content).digest()


def _select_population(
    records: Sequence[FidelityRecord],
    *,
    seed: int,
    purpose: str,
    pairs_per_category: int,
) -> list[FidelityRecord]:
    distinct = _complete_distinct_pairs(records)
    selected: list[FidelityRecord] = []
    for category in CATEGORIES:
        ranked = sorted(
            distinct[category],
            key=lambda item: _selection_key(
                seed=seed,
                purpose=purpose,
                category=category,
                content=item[0],
            ),
        )
        if len(ranked) < pairs_per_category:
            raise RecoveryInputError(
                f"{category} exposes {len(ranked)} distinct pairs; "
                f"{pairs_per_category} are required"
            )
        for _, pair in ranked[:pairs_per_category]:
            selected.extend(sorted(pair, key=lambda record: record.pair_variant.value))
    return selected


def _source_bytes(records: Sequence[FidelityRecord]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json", by_alias=True)) for record in records
    )


def _prepared_bytes(records: Sequence[FidelityRecord]) -> bytes:
    rows = []
    for record in records:
        rows.append(
            format_training_record(
                {
                    "record_id": record.record_id,
                    "passage": record.passage,
                    "target": _safe_target(record),
                }
            )
        )
    return b"".join(canonical_json_bytes(row) for row in rows)


def _population_binding(
    records: Sequence[FidelityRecord],
    *,
    split: str,
    purpose: str,
    source_sha256: str,
    prepared_sha256: str,
) -> dict[str, Any]:
    category_pair_ids: dict[str, set[str]] = {category: set() for category in CATEGORIES}
    for record in records:
        category_pair_ids[record.categories[0]].add(record.pair_id)
    pair_ids = sorted({record.pair_id for record in records})
    return {
        "purpose": purpose,
        "split": split,
        "records": len(records),
        "pairs": len(pair_ids),
        "category_pair_counts": {
            category: len(category_pair_ids[category]) for category in CATEGORIES
        },
        "record_ids_sha256": canonical_sha256(sorted(record.record_id for record in records)),
        "pair_ids_sha256": canonical_sha256(pair_ids),
        "family_ids_sha256": canonical_sha256(sorted({record.family_id for record in records})),
        "passage_hashes_sha256": canonical_sha256(
            sorted(record.passage_sha256 for record in records)
        ),
        "template_families_sha256": canonical_sha256(
            sorted({record.template_family for record in records})
        ),
        "source_sha256": source_sha256,
        "prepared_sha256": prepared_sha256,
        "pair_adjacency_preserved": True,
    }


def _assert_expected_population(
    binding: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    required = {
        "split",
        "pairs_per_category",
        "pairs",
        "records",
        "record_ids_sha256",
        "pair_ids_sha256",
        "source_sha256",
        "prepared_sha256",
    }
    if set(expected) != required:
        raise RecoveryInputError(f"{label} configuration has unexpected fields")
    pairs_per_category = expected["pairs_per_category"]
    checks = {
        "split": binding["split"] == expected["split"],
        "pairs": binding["pairs"] == expected["pairs"],
        "records": binding["records"] == expected["records"],
        "record_ids_sha256": binding["record_ids_sha256"] == expected["record_ids_sha256"],
        "pair_ids_sha256": binding["pair_ids_sha256"] == expected["pair_ids_sha256"],
        "source_sha256": binding["source_sha256"] == expected["source_sha256"],
        "prepared_sha256": binding["prepared_sha256"] == expected["prepared_sha256"],
        "category_balance": type(pairs_per_category) is int
        and set(binding["category_pair_counts"].values()) == {pairs_per_category},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RecoveryInputError(f"{label} differs from configured population: {failed}")


def _assert_disjoint(
    canary: Sequence[FidelityRecord], probe: Sequence[FidelityRecord]
) -> dict[str, bool]:
    fields = {
        "record_id": lambda record: record.record_id,
        "pair_id": lambda record: record.pair_id,
        "family_id": lambda record: record.family_id,
        "passage_sha256": lambda record: record.passage_sha256,
        "template_family": lambda record: record.template_family,
    }
    result = {
        name: {reader(record) for record in canary}.isdisjoint(reader(record) for record in probe)
        for name, reader in fields.items()
    }
    if not all(result.values()):
        failed = sorted(name for name, passed in result.items() if not passed)
        raise RecoveryInputError(f"canary and public probe overlap: {failed}")
    return result


def _validate_lineage(
    config: ExperimentConfig,
    *,
    rejected_config_path: Path,
    rejected_training_completion_path: Path,
    rejected_development_rejection_path: Path,
    rejected_training_events_path: Path,
) -> dict[str, str]:
    recovery = config.recovery
    previous = recovery["previous_attempt"]
    rejected_config_sha = sha256_file(rejected_config_path)
    if rejected_config_sha != previous["config_sha256"]:
        raise RecoveryInputError("rejected config differs from configured SHA-256")
    rejected_config = load_config(rejected_config_path)
    if rejected_config.experiment_id != previous["experiment_id"]:
        raise RecoveryInputError("rejected config experiment identity changed")

    training = _approved_json(
        rejected_training_completion_path,
        previous["training_completion_sha256"],
        label="rejected training completion",
    )
    rejection = _approved_json(
        rejected_development_rejection_path,
        previous["development_rejection_sha256"],
        label="development rejection",
    )
    if (
        training.get("schema_version") != "1.0"
        or training.get("status") != "succeeded"
        or training.get("run_id") != previous["training_run_id"]
    ):
        raise RecoveryInputError("rejected training completion identity changed")
    if (
        rejection.get("schema_version") != "bookforge-jax-development-rejection-v1"
        or rejection.get("status") != "rejected"
        or rejection.get("training_run_id") != previous["training_run_id"]
        or rejection.get("config_sha256") != previous["config_sha256"]
        or rejection.get("dataset_manifest_sha256") != config.dataset["manifest_sha256"]
        or rejection.get("checkpoint_bytes_downloaded") is not False
    ):
        raise RecoveryInputError("development rejection identity or disposition changed")
    if (
        not rejected_training_events_path.is_file()
        or rejected_training_events_path.is_symlink()
        or sha256_file(rejected_training_events_path) != previous["training_events_sha256"]
    ):
        raise RecoveryInputError("rejected training events differ from configured SHA-256")
    return {
        "experiment_id": previous["experiment_id"],
        "config_sha256": previous["config_sha256"],
        "training_run_id": previous["training_run_id"],
        "training_completion_sha256": previous["training_completion_sha256"],
        "development_rejection_sha256": previous["development_rejection_sha256"],
        "training_events_sha256": previous["training_events_sha256"],
    }


def _write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_recovery_inputs(
    *,
    config_path: Path,
    dataset_manifest_path: Path,
    rejected_config_path: Path,
    rejected_training_completion_path: Path,
    rejected_development_rejection_path: Path,
    rejected_training_events_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build four immutable public JSONL files and write their manifest last."""

    config = load_config(config_path)
    if config.experiment_id != EXPERIMENT_ID:
        raise RecoveryInputError("recovery input builder requires the v3 canary config")
    recovery = config.recovery
    if recovery["selection_policy"] != SELECTION_POLICY:
        raise RecoveryInputError("recovery selection policy changed")
    lineage = _validate_lineage(
        config,
        rejected_config_path=rejected_config_path,
        rejected_training_completion_path=rejected_training_completion_path,
        rejected_development_rejection_path=rejected_development_rejection_path,
        rejected_training_events_path=rejected_training_events_path,
    )

    dataset = validate_dataset_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=config.dataset["manifest_sha256"],
        required_split_records=config.dataset["required_split_records"],
    )
    manifest = _load_json(dataset_manifest_path, label="dataset manifest")
    splits = manifest.get("splits")
    if (
        manifest.get("dataset_id") != "story-fidelity-v2"
        or not isinstance(splits, dict)
        or not isinstance(splits.get("train"), dict)
        or not isinstance(splits.get("development"), dict)
        or not isinstance(splits.get("hidden"), dict)
        or splits["train"].get("public") is not True
        or splits["development"].get("public") is not True
        or splits["hidden"].get("public") is not False
        or splits["hidden"].get("path") is not None
    ):
        raise RecoveryInputError("recovery dataset must expose only public train/development bytes")
    by_name = {split.name: split for split in dataset.splits}
    if set(by_name) != {"train", "development"}:
        raise RecoveryInputError("recovery builder received an unexpected accessible split")

    seed = recovery["selection_seed"]
    canary_expected = recovery["overfit_canary"]
    probe_expected = recovery["public_probe"]
    train_records = _load_split(by_name["train"].path, split=DatasetSplit.TRAIN)
    development_records = _load_split(by_name["development"].path, split=DatasetSplit.DEVELOPMENT)
    canary = _select_population(
        train_records,
        seed=seed,
        purpose="overfit-canary",
        pairs_per_category=canary_expected["pairs_per_category"],
    )
    probe = _select_population(
        development_records,
        seed=seed,
        purpose="public-probe",
        pairs_per_category=probe_expected["pairs_per_category"],
    )
    disjoint = _assert_disjoint(canary, probe)

    payloads = {
        _OUTPUT_PATHS[0]: _source_bytes(canary),
        _OUTPUT_PATHS[1]: _prepared_bytes(canary),
        _OUTPUT_PATHS[2]: _source_bytes(probe),
        _OUTPUT_PATHS[3]: _prepared_bytes(probe),
    }
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    canary_binding = _population_binding(
        canary,
        split="train",
        purpose="overfit-canary",
        source_sha256=hashes[_OUTPUT_PATHS[0]],
        prepared_sha256=hashes[_OUTPUT_PATHS[1]],
    )
    probe_binding = _population_binding(
        probe,
        split="development",
        purpose="public-probe",
        source_sha256=hashes[_OUTPUT_PATHS[2]],
        prepared_sha256=hashes[_OUTPUT_PATHS[3]],
    )
    _assert_expected_population(canary_binding, canary_expected, label="overfit canary")
    _assert_expected_population(probe_binding, probe_expected, label="public probe")
    if set(disjoint) != set(recovery["disjoint_fields"]):
        raise RecoveryInputError("configured disjointness fields changed")

    file_rows = [
        {"path": name, "bytes": len(payloads[name]), "sha256": hashes[name]}
        for name in sorted(payloads)
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "dataset": {
            "dataset_id": "story-fidelity-v2",
            "manifest_sha256": dataset.manifest_sha256,
            "train_sha256": by_name["train"].sha256,
            "development_sha256": by_name["development"].sha256,
        },
        "selection": {"policy": SELECTION_POLICY, "seed": seed},
        "learnability_acceptance": dict(recovery["learnability_acceptance"]),
        "lineage": lineage,
        "populations": {
            "overfit_canary": canary_binding,
            "public_probe": probe_binding,
        },
        "disjoint": disjoint,
        "privacy": {"public_records_only": True, "hidden_records_included": False},
        "files": file_rows,
    }

    destination = output_directory.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"recovery output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    for name in sorted(payloads):
        _write_once(destination / name, payloads[name])
    _write_once(destination / "inputs.manifest.json", canonical_json_bytes(document))
    return {**document, "manifest_sha256": sha256_file(destination / "inputs.manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--rejected-config", type=Path, required=True)
    parser.add_argument("--rejected-training-completion", type=Path, required=True)
    parser.add_argument("--rejected-development-rejection", type=Path, required=True)
    parser.add_argument("--rejected-training-events", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = build_recovery_inputs(
        config_path=args.config,
        dataset_manifest_path=args.dataset_manifest,
        rejected_config_path=args.rejected_config,
        rejected_training_completion_path=args.rejected_training_completion,
        rejected_development_rejection_path=args.rejected_development_rejection,
        rejected_training_events_path=args.rejected_training_events,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
