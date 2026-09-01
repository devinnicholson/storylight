import hashlib

import pytest
from pydantic import ValidationError

from bookforge.fidelity_schema import (
    DatasetSplit,
    ExpectationKind,
    FidelityExpectation,
    FidelityProvenance,
    FidelityRecord,
    PairVariant,
    SlotName,
    TargetSlots,
)


def _record() -> FidelityRecord:
    passage = "In the quiet station, a copper fox raises a blue lantern into warm rain."
    target = TargetSlots(
        SETTING="quiet station",
        ACTOR="copper fox",
        ACTION="raises a blue lantern",
        MAGIC="warm rain",
    )
    expectations = tuple(
        FidelityExpectation(
            kind=ExpectationKind.SLOT,
            label=f"slot-{slot.value.lower()}",
            slot=slot,
            alternatives=(getattr(target, slot.value.lower()),),
        )
        for slot in SlotName
    )
    return FidelityRecord(
        record_id="train-f000-p00-a",
        split=DatasetSplit.TRAIN,
        family_id="train-f000",
        pair_id="train-f000-p00",
        pair_variant=PairVariant.A,
        counterfactual_dimension="attributes",
        template_family="train-contrast-attributes-v1",
        categories=("attributes",),
        passage=passage,
        passage_sha256=hashlib.sha256(passage.encode()).hexdigest(),
        target=target,
        expectations=expectations,
        allowed_concepts=("quiet station", "copper fox", "blue lantern", "warm rain"),
        provenance=FidelityProvenance(seed=42),
    )


def test_target_serializes_as_exact_four_line_contract() -> None:
    target = _record().target

    assert target.as_wire().splitlines() == [
        "SETTING: quiet station",
        "ACTOR: copper fox",
        "ACTION: raises a blue lantern",
        "MAGIC: warm rain",
    ]
    assert set(target.model_dump(by_alias=True)) == {"SETTING", "ACTOR", "ACTION", "MAGIC"}


def test_record_hash_and_json_are_deterministic() -> None:
    first = _record()
    second = FidelityRecord.model_validate_json(first.canonical_json())

    assert first == second
    assert first.sha256() == second.sha256()
    assert first.canonical_json().endswith("}")


def test_rejects_wrong_passage_hash_and_private_target_echo() -> None:
    payload = _record().model_dump(mode="json", by_alias=True)
    payload["passage_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="passage_sha256"):
        FidelityRecord.model_validate(payload)

    payload = _record().model_dump(mode="json", by_alias=True)
    payload["privacy_terms"] = ["copper fox"]
    with pytest.raises(ValidationError, match="private terms"):
        FidelityRecord.model_validate(payload)


def test_expectation_validation_requires_kind_specific_fields() -> None:
    with pytest.raises(ValidationError, match="require count"):
        FidelityExpectation(
            kind=ExpectationKind.COUNT,
            label="lantern-count",
            slot=SlotName.ACTION,
            alternatives=("two lanterns",),
        )

    with pytest.raises(ValidationError, match="subject/predicate/object"):
        FidelityExpectation(
            kind=ExpectationKind.RELATION,
            label="above-bridge",
            slot=SlotName.ACTION,
            alternatives=("above the bridge",),
        )


def test_target_rejects_extra_slot_and_multiline_values() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TargetSlots(
            SETTING="station",
            ACTOR="fox",
            ACTION="waits",
            MAGIC="rain",
            CAMERA="wide",
        )
    with pytest.raises(ValidationError, match="one line"):
        TargetSlots(
            SETTING="station\nplatform",
            ACTOR="fox",
            ACTION="waits",
            MAGIC="rain",
        )
