"""Materialize exact production chat records from a verified fidelity split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from bookforge.fidelity_dataset import CATEGORIES
from bookforge.fidelity_schema import FidelityRecord, PairVariant

from .formatting import format_training_record, prompt_contract_sha256, validate_slot_target
from .integrity import canonical_json_bytes, sha256_file, validate_dataset_manifest

PAIR_DEDUPLICATION_POLICY = "balanced-counterfactual-pairs-v1"
PAIR_DEDUPLICATION_SEED = 20260902
PAIRS_PER_CATEGORY = 8


def prepare_training_jsonl(source: Path | str, destination: Path | str) -> dict[str, Any]:
    source_path = Path(source)
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    count = 0
    try:
        with (
            source_path.open("r", encoding="utf-8") as input_stream,
            os.fdopen(descriptor, "wb") as output_stream,
        ):
            for line_number, line in enumerate(input_stream, start=1):
                try:
                    record = json.loads(line)
                    prepared = format_training_record(record)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"could not prepare {source_path}:{line_number}: {error}"
                    ) from error
                output_stream.write(canonical_json_bytes(prepared))
                count += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if count == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("training split contained no records")
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "records": count,
    }


def _pair_content(records: list[FidelityRecord]) -> bytes:
    ordered = sorted(records, key=lambda record: record.pair_variant.value)
    return canonical_json_bytes(
        {
            "category": ordered[0].categories[0],
            "records": [
                {
                    "variant": record.pair_variant.value,
                    "passage": record.passage,
                    "target": record.target.model_dump(by_alias=True),
                }
                for record in ordered
            ],
        }
    )


def _safe_target(record: FidelityRecord) -> str:
    target = validate_slot_target(
        "\n".join(
            f"{label}: {getattr(record.target, label.casefold())}"
            for label in ("SETTING", "ACTOR", "ACTION", "MAGIC")
        )
    )
    normalized = target.casefold()
    leaked = [
        term
        for term in (*record.privacy_terms, *record.forbidden_terms)
        if term.casefold() in normalized
    ]
    if leaked:
        raise ValueError(
            f"training target {record.record_id} contains excluded source terms: {leaked}"
        )
    return target


def prepare_pair_deduplicated_training_jsonl(
    source: Path | str,
    destination: Path | str,
    manifest_destination: Path | str,
    *,
    seed: int = PAIR_DEDUPLICATION_SEED,
) -> dict[str, Any]:
    """Select one balanced representative of each unique counterfactual pair."""

    source_path = Path(source)
    output_path = Path(destination)
    manifest_path = Path(manifest_destination)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"prepared training output already exists: {output_path}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"preparation manifest already exists: {manifest_path}")

    records = [
        FidelityRecord.model_validate_json(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[FidelityRecord]] = defaultdict(list)
    for record in records:
        if len(record.categories) != 1:
            raise ValueError(f"training record {record.record_id} must have one category")
        _safe_target(record)
        grouped[record.pair_id].append(record)

    representatives: dict[bytes, list[FidelityRecord]] = {}
    content_categories: dict[bytes, str] = {}
    for pair_id, pair in sorted(grouped.items()):
        variants = {record.pair_variant for record in pair}
        categories = {record.categories[0] for record in pair}
        if variants != {PairVariant.A, PairVariant.B} or len(pair) != 2 or len(categories) != 1:
            raise ValueError(f"training pair {pair_id} is incomplete or crosses categories")
        content = _pair_content(pair)
        category = next(iter(categories))
        previous_category = content_categories.setdefault(content, category)
        if previous_category != category:
            raise ValueError("identical pair content appears under multiple categories")
        representatives.setdefault(content, pair)

    selected = sorted(
        representatives.items(),
        key=lambda item: hashlib.sha256(str(seed).encode() + b":" + item[0]).digest(),
    )
    category_pair_counts = {category: 0 for category in CATEGORIES}
    for content, _ in selected:
        category_pair_counts[content_categories[content]] += 1
    if set(content_categories.values()) != set(CATEGORIES) or any(
        count != PAIRS_PER_CATEGORY for count in category_pair_counts.values()
    ):
        raise ValueError(
            "pair deduplication did not produce the exact balanced category population"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    prepared_records = 0
    try:
        with os.fdopen(descriptor, "wb") as output_stream:
            for _, pair in selected:
                for record in sorted(pair, key=lambda item: item.pair_variant.value):
                    prepared = format_training_record(
                        {
                            "record_id": record.record_id,
                            "passage": record.passage,
                            "target": _safe_target(record),
                        }
                    )
                    roles = [message["role"] for message in prepared["messages"]]
                    if roles != ["system", "user", "assistant"]:
                        raise ValueError(
                            "prepared training records must have one supervised assistant turn"
                        )
                    output_stream.write(canonical_json_bytes(prepared))
                    prepared_records += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())

        expected_records = len(CATEGORIES) * PAIRS_PER_CATEGORY * 2
        if prepared_records != expected_records:
            raise ValueError(
                f"pair deduplication produced {prepared_records}, expected {expected_records}"
            )
        document = {
            "schema_version": "bookforge-jax-training-preparation-v2",
            "policy": PAIR_DEDUPLICATION_POLICY,
            "seed": seed,
            "source_train_sha256": sha256_file(source_path),
            "source_records": len(records),
            "source_pairs": len(grouped),
            "distinct_pairs": len(selected),
            "prepared_records": prepared_records,
            "category_pair_counts": category_pair_counts,
            "prompt_contract_sha256": prompt_contract_sha256(),
            "assistant_turns_per_record": 1,
            "pair_adjacency_preserved": True,
            "prepared_sha256": sha256_file(output_path),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_descriptor = os.open(
            manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(manifest_descriptor, "wb") as manifest_stream:
            manifest_stream.write(canonical_json_bytes(document))
            manifest_stream.flush()
            os.fsync(manifest_stream.fileno())
    except Exception:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return {**document, "manifest_sha256": sha256_file(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-dedupe", action="store_true")
    parser.add_argument("--preparation-manifest", type=Path)
    args = parser.parse_args()
    dataset = validate_dataset_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
    )
    train = next((split_ for split_ in dataset.splits if split_.name == "train"), None)
    if train is None:
        raise SystemExit("verified dataset does not expose a train split")
    if args.pair_dedupe:
        if args.preparation_manifest is None:
            parser.error("--pair-dedupe requires --preparation-manifest")
        result = prepare_pair_deduplicated_training_jsonl(
            train.path,
            args.output,
            args.preparation_manifest,
        )
    else:
        if args.preparation_manifest is not None:
            parser.error("--preparation-manifest requires --pair-dedupe")
        result = prepare_training_jsonl(train.path, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
