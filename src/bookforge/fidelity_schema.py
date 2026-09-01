"""Versioned contracts for the Story Fidelity Lab dataset."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "1.0"
GENERATOR_ID = "bookforge-story-fidelity"
GENERATOR_REVISION = "synthetic-v1"

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$"),
]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    HIDDEN = "hidden"


class SlotName(StrEnum):
    SETTING = "SETTING"
    ACTOR = "ACTOR"
    ACTION = "ACTION"
    MAGIC = "MAGIC"


class PairVariant(StrEnum):
    A = "a"
    B = "b"


class ExpectationKind(StrEnum):
    SLOT = "slot"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    COUNT = "count"
    ROLE = "role"
    TRANSFORMATION = "transformation"
    ORDER = "order"
    ALLOWED_CONCEPT = "allowed_concept"
    PRIVACY = "privacy"


class TargetSlots(BaseModel):
    """The exact four-line contract consumed by Bookforge's planner adapter."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    setting: BoundedText = Field(alias="SETTING")
    actor: BoundedText = Field(alias="ACTOR")
    action: BoundedText = Field(alias="ACTION")
    magic: BoundedText = Field(alias="MAGIC")

    @model_validator(mode="after")
    def validate_lines(self) -> TargetSlots:
        for name in ("setting", "actor", "action", "magic"):
            value = getattr(self, name)
            if "\n" in value or "\r" in value:
                raise ValueError(f"{name} must fit on one line")
        return self

    def as_wire(self) -> str:
        values = self.model_dump(by_alias=True)
        return "\n".join(f"{slot}: {values[slot]}" for slot in SlotName)


class FidelityExpectation(BaseModel):
    """One machine-checkable fact expected in a planner slot."""

    model_config = ConfigDict(extra="forbid")

    kind: ExpectationKind
    label: Identifier
    slot: SlotName
    alternatives: Annotated[tuple[BoundedText, ...], Field(min_length=1, max_length=8)]
    subject: BoundedText | None = None
    predicate: BoundedText | None = None
    object: BoundedText | None = None
    count: Annotated[int, Field(ge=0, le=20)] | None = None
    required: bool = True

    @model_validator(mode="after")
    def validate_kind_fields(self) -> FidelityExpectation:
        if self.kind == ExpectationKind.COUNT and self.count is None:
            raise ValueError("count expectations require count")
        if self.kind in {ExpectationKind.RELATION, ExpectationKind.ROLE} and (
            not self.subject or not self.predicate or not self.object
        ):
            raise ValueError(f"{self.kind.value} expectations require subject/predicate/object")
        if self.kind == ExpectationKind.TRANSFORMATION and (not self.subject or not self.object):
            raise ValueError("transformation expectations require source and result")
        return self


class FidelityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generator: Literal[GENERATOR_ID] = GENERATOR_ID
    revision: Literal[GENERATOR_REVISION] = GENERATOR_REVISION
    origin: Literal["synthetic"] = "synthetic"
    source_policy: Literal["original-templates-only"] = "original-templates-only"
    license: Literal["CC0-1.0"] = "CC0-1.0"
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]


class FidelityRecord(BaseModel):
    """One original story, its exact target, and deterministic evaluation facts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    record_id: Identifier
    split: DatasetSplit
    family_id: Identifier
    pair_id: Identifier
    pair_variant: PairVariant
    counterfactual_dimension: Identifier
    template_family: Identifier
    categories: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=6)]
    passage: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=20, max_length=1800)
    ]
    passage_sha256: Digest
    target: TargetSlots
    expectations: Annotated[tuple[FidelityExpectation, ...], Field(min_length=4, max_length=12)]
    allowed_concepts: Annotated[tuple[BoundedText, ...], Field(min_length=4, max_length=32)]
    forbidden_terms: Annotated[tuple[BoundedText, ...], Field(max_length=16)] = ()
    privacy_terms: Annotated[tuple[BoundedText, ...], Field(max_length=8)] = ()
    provenance: FidelityProvenance

    @model_validator(mode="after")
    def validate_record(self) -> FidelityRecord:
        passage_digest = hashlib.sha256(self.passage.encode("utf-8")).hexdigest()
        if passage_digest != self.passage_sha256:
            raise ValueError("passage_sha256 does not match passage")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be unique")
        if len(set(self.allowed_concepts)) != len(self.allowed_concepts):
            raise ValueError("allowed_concepts must be unique")
        rendered_target = self.target.as_wire().casefold()
        for term in (*self.forbidden_terms, *self.privacy_terms):
            if term.casefold() in rendered_target:
                raise ValueError("forbidden and private terms may not appear in target slots")
        expectation_slots = {expectation.slot for expectation in self.expectations}
        if expectation_slots != set(SlotName):
            raise ValueError("expectations must cover every target slot")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def passage_sha256(passage: str) -> str:
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()
