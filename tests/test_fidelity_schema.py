import hashlib

import pytest
from pydantic import ValidationError

from storylight.fidelity_schema import (
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


def test_rejects_wrong_passage_hash_and_private_target_echo() -> None:
    payload = _record().model_dump(mode="json", by_alias=True)
    payload["passage_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="passage_sha256"):
        FidelityRecord.model_validate(payload)

    payload = _record().model_dump(mode="json", by_alias=True)
    payload["privacy_terms"] = ["copper fox"]
    with pytest.raises(ValidationError, match="private terms"):
        FidelityRecord.model_validate(payload)
