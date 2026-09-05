import base64
import hashlib
from collections import defaultdict

import pytest

from bookforge.fidelity_dataset import (
    CATEGORIES,
    HIDDEN_KEY_PREFIX,
    SPLIT_COUNTS,
    VOCABULARY,
    DatasetSplit,
    generate_split,
    records_jsonl,
    validate_safe_record,
    validate_split_isolation,
)
from bookforge.fidelity_schema import PairVariant


def _hidden_key(label: str) -> str:
    material = hashlib.sha384(label.encode()).digest()
    return HIDDEN_KEY_PREFIX + base64.urlsafe_b64encode(material).decode()


def test_public_splits_are_exact_deterministic_and_cover_categories() -> None:
    for split in (DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT):
        first = list(generate_split(split))
        second = list(generate_split(split))

        assert len(first) == SPLIT_COUNTS[split]
        assert records_jsonl(first) == records_jsonl(second)
        assert {category for record in first for category in record.categories} == set(CATEGORIES)
        assert all(len(record.target.as_wire().split()) <= 64 for record in first)
        assert all(len(record.passage.split()) <= 512 for record in first)


def test_counterfactual_pairs_and_families_are_split_isolated() -> None:
    records = [
        *generate_split(DatasetSplit.TRAIN),
        *generate_split(DatasetSplit.DEVELOPMENT),
    ]
    validate_split_isolation(records)
    pairs: dict[str, set[PairVariant]] = defaultdict(set)
    for record in records:
        pairs[record.pair_id].add(record.pair_variant)

    assert all(variants == set(PairVariant) for variants in pairs.values())
    assert len({record.family_id for record in records}) == 144
    assert len(pairs) == 2304


def test_hidden_requires_custodian_key_and_uses_held_out_vocabulary() -> None:
    with pytest.raises(ValueError, match="private hidden_key"):
        list(generate_split(DatasetSplit.HIDDEN))

    first = list(generate_split(DatasetSplit.HIDDEN, hidden_key=_hidden_key("first")))
    second = list(generate_split(DatasetSplit.HIDDEN, hidden_key=_hidden_key("first")))
    different = list(generate_split(DatasetSplit.HIDDEN, hidden_key=_hidden_key("second")))

    assert len(first) == SPLIT_COUNTS[DatasetSplit.HIDDEN]
    assert records_jsonl(first) == records_jsonl(second)
    assert records_jsonl(first) != records_jsonl(different)
    assert max(record.provenance.seed for record in first) > 1_000_000_000
    public_words = set(VOCABULARY[DatasetSplit.TRAIN].actors) | set(
        VOCABULARY[DatasetSplit.DEVELOPMENT].actors
    )
    hidden_words = set(VOCABULARY[DatasetSplit.HIDDEN].actors)
    assert public_words.isdisjoint(hidden_words)
    assert all(record.template_family.startswith("hidden-") for record in first)


def test_hidden_fields_are_independently_keyed_beyond_old_eight_example_support() -> None:
    outputs = set()
    for index in range(24):
        first = next(generate_split(DatasetSplit.HIDDEN, hidden_key=_hidden_key(str(index))))
        outputs.add(
            (
                first.target.setting,
                first.target.actor,
                first.target.action,
                first.target.magic,
                first.counterfactual_dimension,
            )
        )

    assert len(outputs) > 8


def test_reserved_content_is_controlled_and_targets_never_echo_it() -> None:
    development = list(generate_split(DatasetSplit.DEVELOPMENT))
    contacts = [record for record in development if "reserved_contact_data" in record.categories]
    injections = [record for record in development if "prompt_injection" in record.categories]

    assert contacts and injections
    assert all(record.privacy_terms for record in (*contacts, *injections))
    for record in (*contacts, *injections):
        rendered = record.target.as_wire().casefold()
        assert all(term.casefold() not in rendered for term in record.privacy_terms)
        validate_safe_record(record)
