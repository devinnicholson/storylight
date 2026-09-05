"""Compact, source-grounded facts for faithful live-scene rendering.

The source passage is consumed only by the local validation methods in this
module.  The wire representation and renderer prompt contain bounded visual
facts rather than the passage itself.
"""

from __future__ import annotations

import math
import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StringConstraints, field_validator, model_validator

from bookforge.domain import FrozenStrictModel
from bookforge.privacy_policy import (
    COLOR_WORDS,
    SENSITIVE_CONTENT_PATTERN,
    contains_distinctive_source_phrase,
    contains_token_sequence,
    printed_source_payload_candidates,
    privacy_tokens,
    proper_name_candidates,
)
from bookforge.semantic_text import normalize_semantic_phrase

SCENE_FACTS_VERSION = "2.0"
WIRE_HEADER = "V2"
SUPPORTED_WIRE_TOKEN_BUDGETS = frozenset({64, 96, 128})

Reference = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,23}$"),
]


def _validate_wire_phrase(value: str) -> str:
    if value == "-" or any(character in value for character in "|,\r\n"):
        raise ValueError("fact phrases may not contain wire delimiters")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("fact phrases may not contain control characters")
    return value


FactPhrase = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
    AfterValidator(_validate_wire_phrase),
]
FactAtom = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=28),
    AfterValidator(_validate_wire_phrase),
]


class SceneRelationKind(StrEnum):
    ABOVE = "above"
    ATTACHED_TO = "attached_to"
    BEHIND = "behind"
    BELOW = "below"
    BESIDE = "beside"
    BETWEEN = "between"
    CARRIES = "carries"
    CONTAINS = "contains"
    HOLDS = "holds"
    IN_FRONT_OF = "in_front_of"
    INSIDE = "inside"
    LEFT_OF = "left_of"
    LOOKS_AT = "looks_at"
    NEXT_TO = "next_to"
    ON = "on"
    OUTSIDE = "outside"
    OWNS = "owns"
    RIGHT_OF = "right_of"
    TOUCHES = "touches"
    UNDER = "under"
    WEARS = "wears"


class SceneMotionDirection(StrEnum):
    FALLS = "falls"
    RISES = "rises"


class SceneSalienceLayer(StrEnum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"


class SceneNegativeKind(StrEnum):
    ACTION = "action"
    ADDITIONAL_OBJECT = "additional_object"
    ADDITIONAL_SUBJECT = "additional_subject"
    ATTRIBUTE = "attribute"
    EFFECT = "effect"


_OPPOSING_RELATIONS = {
    SceneRelationKind.ABOVE: SceneRelationKind.BELOW,
    SceneRelationKind.BELOW: SceneRelationKind.ABOVE,
    SceneRelationKind.BEHIND: SceneRelationKind.IN_FRONT_OF,
    SceneRelationKind.IN_FRONT_OF: SceneRelationKind.BEHIND,
    SceneRelationKind.INSIDE: SceneRelationKind.OUTSIDE,
    SceneRelationKind.OUTSIDE: SceneRelationKind.INSIDE,
    SceneRelationKind.LEFT_OF: SceneRelationKind.RIGHT_OF,
    SceneRelationKind.RIGHT_OF: SceneRelationKind.LEFT_OF,
}
_DIRECTIONAL_RELATIONS = frozenset(_OPPOSING_RELATIONS)
_CONTRADICTORY_STATE_GROUPS = (
    frozenset({"awake", "asleep"}),
    frozenset({"closed", "open"}),
    frozenset({"dark", "lit"}),
    frozenset({"dry", "wet"}),
    frozenset({"empty", "full"}),
)


class SceneSettingFact(FrozenStrictModel):
    label: FactPhrase
    attributes: Annotated[tuple[FactAtom, ...], Field(max_length=3)] = ()

    @field_validator("attributes")
    @classmethod
    def unique_attributes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(values, "setting attributes")
        return values


class SceneSubjectFact(FrozenStrictModel):
    ref: Reference
    label: FactPhrase
    count: Annotated[int | None, Field(ge=1, le=12)] = None
    color: FactAtom | None = None
    attributes: Annotated[tuple[FactAtom, ...], Field(max_length=4)] = ()
    actions: Annotated[tuple[FactPhrase, ...], Field(max_length=3)] = ()

    @field_validator("attributes", "actions")
    @classmethod
    def unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(values, "subject facts")
        return values


class SceneObjectFact(FrozenStrictModel):
    ref: Reference
    label: FactPhrase
    count: Annotated[int | None, Field(ge=1, le=12)] = None
    color: FactAtom | None = None
    states: Annotated[tuple[FactAtom, ...], Field(max_length=3)] = ()
    attributes: Annotated[tuple[FactAtom, ...], Field(max_length=4)] = ()

    @field_validator("states", "attributes")
    @classmethod
    def unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(values, "object facts")
        return values


class SceneRelationshipFact(FrozenStrictModel):
    source: Reference
    relation: SceneRelationKind
    target: Reference
    secondary_target: Reference | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> SceneRelationshipFact:
        if self.source == self.target:
            raise ValueError("a relationship cannot target itself")
        if self.relation is SceneRelationKind.BETWEEN:
            if self.secondary_target is None:
                raise ValueError("between relationships require secondary_target")
            if self.secondary_target in {self.source, self.target}:
                raise ValueError("between relationships require three distinct references")
        elif self.secondary_target is not None:
            raise ValueError("secondary_target is valid only for between relationships")
        return self


class SceneNegativeFact(FrozenStrictModel):
    kind: SceneNegativeKind
    value: FactPhrase
    target: Reference | None = None

    @model_validator(mode="after")
    def validate_target(self) -> SceneNegativeFact:
        target_required = self.kind in {
            SceneNegativeKind.ACTION,
            SceneNegativeKind.ATTRIBUTE,
        }
        if target_required != (self.target is not None):
            requirement = "require" if target_required else "must not include"
            raise ValueError(f"{self.kind.value} negatives {requirement} a target")
        return self


class SceneTransformationFact(FrozenStrictModel):
    source: Reference
    result_label: FactPhrase
    result_color: FactAtom | None = None
    result_attributes: Annotated[tuple[FactAtom, ...], Field(max_length=3)] = ()
    result_count: Annotated[int | None, Field(strict=True, ge=1, le=12)] = None

    @field_validator("result_attributes")
    @classmethod
    def unique_attributes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(values, "transformation attributes")
        return values


class SceneMotionFact(FrozenStrictModel):
    source: Reference
    direction: SceneMotionDirection | None = None
    destination: Reference | None = None

    @model_validator(mode="after")
    def validate_motion(self) -> SceneMotionFact:
        if self.direction is None and self.destination is None:
            raise ValueError("motion requires a direction or destination")
        if self.source == self.destination:
            raise ValueError("motion destination must differ from its source")
        return self


class SceneSalienceFact(FrozenStrictModel):
    source: Reference
    layer: SceneSalienceLayer


class SceneEventFact(FrozenStrictModel):
    ref: Reference
    source: Reference
    action: FactAtom
    object: Reference | None = None

    @model_validator(mode="after")
    def validate_event(self) -> SceneEventFact:
        if self.source == self.object:
            raise ValueError("an event object must differ from its source")
        return self


class SceneTemporalOrderFact(FrozenStrictModel):
    before: Reference
    after: Reference

    @model_validator(mode="after")
    def validate_order(self) -> SceneTemporalOrderFact:
        if self.before == self.after:
            raise ValueError("temporal order requires two distinct events")
        return self


class SceneFactsV2(FrozenStrictModel):
    """A bounded visual fact graph produced and validated on the edge."""

    version: Literal[SCENE_FACTS_VERSION] = SCENE_FACTS_VERSION
    setting: SceneSettingFact
    subjects: Annotated[tuple[SceneSubjectFact, ...], Field(max_length=4)] = ()
    objects: Annotated[tuple[SceneObjectFact, ...], Field(max_length=6)] = ()
    relationships: Annotated[tuple[SceneRelationshipFact, ...], Field(max_length=8)] = ()
    motions: Annotated[tuple[SceneMotionFact, ...], Field(max_length=4)] = ()
    salience: Annotated[tuple[SceneSalienceFact, ...], Field(max_length=4)] = ()
    events: Annotated[tuple[SceneEventFact, ...], Field(max_length=6)] = ()
    temporal_order: Annotated[tuple[SceneTemporalOrderFact, ...], Field(max_length=6)] = ()
    negatives: Annotated[tuple[SceneNegativeFact, ...], Field(max_length=6)] = ()
    transformation: SceneTransformationFact | None = None

    @model_validator(mode="after")
    def validate_graph(self) -> SceneFactsV2:
        entities = (*self.subjects, *self.objects)
        if not entities:
            raise ValueError("scene facts require at least one subject or object")
        refs = [entity.ref for entity in entities]
        if len(refs) != len(set(refs)):
            raise ValueError("scene entity references must be unique")
        labels: dict[tuple[str, ...], list[str | None]] = {}
        for entity in entities:
            labels.setdefault(_normalized_phrase(entity.label), []).append(entity.color)
        for colors in labels.values():
            if len(colors) > 1 and (
                any(color is None for color in colors)
                or any(len(_normalized_phrase(color or "")) != 1 for color in colors)
                or len({_normalized_phrase(color or "") for color in colors}) != len(colors)
            ):
                raise ValueError(
                    "repeated entity labels require distinct colors with one word each "
                    "or one entity with an explicit count"
                )

        known_refs = set(refs)
        subject_refs = {subject.ref for subject in self.subjects}
        for relationship in self.relationships:
            relation_refs = {
                relationship.source,
                relationship.target,
                relationship.secondary_target,
            } - {None}
            if not relation_refs.issubset(known_refs):
                raise ValueError("relationships may reference only declared scene entities")
            if (
                relationship.relation in _SUBJECT_OWNERSHIP_RELATIONS
                and relationship.source not in subject_refs
            ):
                raise ValueError("action and ownership relationships must start at a subject")
        relation_keys = [
            (
                relation.source,
                relation.relation,
                relation.target,
                relation.secondary_target,
            )
            for relation in self.relationships
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("scene relationships must be unique")
        directional_edges = {
            (relation.source, relation.target, relation.relation)
            for relation in self.relationships
            if relation.relation in _DIRECTIONAL_RELATIONS
        }
        for source, target, relation in directional_edges:
            if (source, target, _OPPOSING_RELATIONS[relation]) in directional_edges or (
                target,
                source,
                relation,
            ) in directional_edges:
                raise ValueError("scene relationships cannot contradict one another")

        for item in self.objects:
            normalized_states = {
                token
                for state in item.states
                for token in privacy_tokens(state)
            }
            if any(group.issubset(normalized_states) for group in _CONTRADICTORY_STATE_GROUPS):
                raise ValueError("one scene object cannot have contradictory states")

        motion_keys = []
        for motion in self.motions:
            motion_refs = {motion.source, motion.destination} - {None}
            if not motion_refs.issubset(known_refs):
                raise ValueError("motions may reference only declared scene entities")
            motion_keys.append((motion.source, motion.direction, motion.destination))
        if len(motion_keys) != len(set(motion_keys)):
            raise ValueError("scene motions must be unique")
        motion_directions: dict[str, set[SceneMotionDirection]] = {}
        for motion in self.motions:
            if motion.direction is not None:
                motion_directions.setdefault(motion.source, set()).add(motion.direction)
        if any(len(directions) > 1 for directions in motion_directions.values()):
            raise ValueError("one scene entity cannot move in contradictory directions")

        salience_keys = []
        for salience in self.salience:
            if salience.source not in known_refs:
                raise ValueError("salience may reference only declared scene entities")
            salience_keys.append((salience.source, salience.layer))
        if len(salience_keys) != len(set(salience_keys)):
            raise ValueError("scene salience facts must be unique")
        salience_layers: dict[str, set[SceneSalienceLayer]] = {}
        for fact in self.salience:
            salience_layers.setdefault(fact.source, set()).add(fact.layer)
        if any(len(layers) > 1 for layers in salience_layers.values()):
            raise ValueError("one scene entity cannot occupy contradictory salience layers")

        event_refs = [event.ref for event in self.events]
        if len(event_refs) != len(set(event_refs)):
            raise ValueError("scene event references must be unique")
        for event in self.events:
            if event.source not in known_refs or (
                event.object is not None and event.object not in known_refs
            ):
                raise ValueError("events may reference only declared scene entities")
        known_events = set(event_refs)
        order_keys = []
        for order in self.temporal_order:
            if not {order.before, order.after}.issubset(known_events):
                raise ValueError("temporal order may reference only declared scene events")
            order_keys.append((order.before, order.after))
        if len(order_keys) != len(set(order_keys)):
            raise ValueError("scene temporal-order facts must be unique")
        _validate_acyclic_order(order_keys)

        for negative in self.negatives:
            if negative.target is not None and negative.target not in known_refs:
                raise ValueError("negatives may reference only declared scene entities")
        negative_keys = [
            (negative.kind, negative.target, _normalized_phrase(negative.value))
            for negative in self.negatives
        ]
        if len(negative_keys) != len(set(negative_keys)):
            raise ValueError("scene negatives must be unique")
        self._validate_no_positive_negative_contradictions(entities)

        if self.transformation is not None and self.transformation.source not in known_refs:
            raise ValueError("a transformation must start at a declared scene entity")
        return self

    def _validate_no_positive_negative_contradictions(
        self,
        entities: tuple[SceneSubjectFact | SceneObjectFact, ...],
    ) -> None:
        by_ref = {entity.ref: entity for entity in entities}
        actions_by_ref: dict[str, list[str]] = {
            subject.ref: list(subject.actions) for subject in self.subjects
        }
        for event in self.events:
            actions_by_ref.setdefault(event.source, []).append(event.action)
            if event.object is not None:
                actions_by_ref[event.source].append(
                    f"{event.action} {by_ref[event.object].label}"
                )
        for relation in self.relationships:
            actions_by_ref.setdefault(relation.source, []).append(
                f"{relation.relation.value.replace('_', ' ')} {by_ref[relation.target].label}"
            )

        for negative in self.negatives:
            positive_values: tuple[str, ...]
            if negative.kind is SceneNegativeKind.ACTION and negative.target is not None:
                positive_values = tuple(actions_by_ref.get(negative.target, ()))
            elif negative.kind is SceneNegativeKind.ATTRIBUTE and negative.target is not None:
                entity = by_ref[negative.target]
                states = entity.states if isinstance(entity, SceneObjectFact) else ()
                values = tuple(
                    value
                    for value in (entity.color, *states, *entity.attributes)
                    if value is not None
                )
                positive_values = (*values, *(f"{value} {entity.label}" for value in values))
            elif negative.kind is SceneNegativeKind.ADDITIONAL_SUBJECT:
                positive_values = tuple(subject.label for subject in self.subjects)
            elif negative.kind is SceneNegativeKind.ADDITIONAL_OBJECT:
                positive_values = tuple(item.label for item in self.objects)
            elif negative.kind is SceneNegativeKind.EFFECT and self.transformation is not None:
                positive_values = (self.transformation.result_label,)
            else:
                positive_values = ()
            if any(_phrases_conflict(negative.value, value) for value in positive_values):
                raise ValueError("scene negatives cannot contradict positive facts")

    def to_wire(self, *, token_budget: Literal[64, 96, 128] | None = None) -> str:
        """Serialize to the bounded, order-preserving line protocol used for decoding.

        ``token_budget`` uses a deterministic byte-based estimate, not a
        model-specific tokenizer.  Runtime experiments should still apply the
        selected model's hard output-token limit.
        """

        lines = [WIRE_HEADER, _wire_line("G", self.setting.label, self.setting.attributes)]
        lines.extend(
            _wire_line(
                "S",
                subject.ref,
                subject.count,
                subject.label,
                subject.color,
                subject.attributes,
                subject.actions,
            )
            for subject in self.subjects
        )
        lines.extend(
            _wire_line(
                "O",
                item.ref,
                item.count,
                item.label,
                item.color,
                item.states,
                item.attributes,
            )
            for item in self.objects
        )
        lines.extend(
            _wire_line(
                "R",
                relationship.source,
                relationship.relation.value,
                relationship.target,
                relationship.secondary_target,
            )
            for relationship in self.relationships
        )
        lines.extend(
            _wire_line("M", motion.source, motion.direction, motion.destination)
            for motion in self.motions
        )
        lines.extend(_wire_line("L", salience.source, salience.layer) for salience in self.salience)
        lines.extend(
            _wire_line("E", event.ref, event.source, event.action, event.object)
            for event in self.events
        )
        lines.extend(_wire_line("Q", order.before, order.after) for order in self.temporal_order)
        lines.extend(
            _wire_line("N", negative.kind.value, negative.target, negative.value)
            for negative in self.negatives
        )
        if self.transformation is not None:
            lines.append(
                _wire_line(
                    "T",
                    self.transformation.source,
                    self.transformation.result_label,
                    self.transformation.result_color,
                    self.transformation.result_attributes,
                    *(
                        (self.transformation.result_count,)
                        if self.transformation.result_count is not None
                        else ()
                    ),
                )
            )
        wire = "\n".join(lines)
        if token_budget is not None:
            if token_budget not in SUPPORTED_WIRE_TOKEN_BUDGETS:
                raise ValueError("token_budget must be one of 64, 96, or 128")
            if estimate_wire_tokens(wire) > token_budget:
                raise SceneFactsWireBudgetError(token_budget)
        return wire

    @classmethod
    def from_wire(cls, wire: str) -> SceneFactsV2:
        return parse_scene_facts_wire(wire)

    def validate_source_grounding(self, *, source_text: str) -> None:
        validate_scene_facts_grounding(self, source_text=source_text)

    def to_renderer_prompt(
        self,
        *,
        source_text: str,
        visual_style: str = "luminous storybook illustration",
    ) -> str:
        return compile_scene_facts_prompt(
            self,
            source_text=source_text,
            visual_style=visual_style,
        )


class SceneFactsGroundingError(ValueError):
    """Raised without source or fact values when a fact is not source-grounded."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        self.paths = paths
        super().__init__("scene facts were not source-grounded: " + ", ".join(paths))


class SceneFactsPrivacyError(ValueError):
    """Raised when semantic facts are unsafe to send to a remote renderer."""


class SceneFactsWireBudgetError(ValueError):
    def __init__(self, token_budget: int) -> None:
        self.token_budget = token_budget
        super().__init__(f"scene facts exceed the approximate {token_budget}-token wire budget")


def estimate_wire_tokens(wire: str) -> int:
    """Return a deterministic tokenizer-independent planning estimate."""

    return math.ceil(len(wire.encode("utf-8")) / 4)


def parse_scene_facts_wire(wire: str) -> SceneFactsV2:
    if len(wire.encode("utf-8")) > 4_096:
        raise ValueError("scene facts wire payload is too large")
    if "\r" in wire or not wire or wire != wire.strip():
        raise ValueError("scene facts wire payload must be trimmed LF-delimited text")
    lines = wire.split("\n")
    if not lines or lines[0] != WIRE_HEADER:
        raise ValueError("scene facts wire payload has an unknown header")
    if len(lines) < 3 or any(not line for line in lines):
        raise ValueError("scene facts wire payload is incomplete")

    setting: SceneSettingFact | None = None
    subjects: list[SceneSubjectFact] = []
    objects: list[SceneObjectFact] = []
    relationships: list[SceneRelationshipFact] = []
    motions: list[SceneMotionFact] = []
    salience: list[SceneSalienceFact] = []
    events: list[SceneEventFact] = []
    temporal_order: list[SceneTemporalOrderFact] = []
    negatives: list[SceneNegativeFact] = []
    transformation: SceneTransformationFact | None = None
    rank = -1
    ranks = {
        "G": 0,
        "S": 1,
        "O": 2,
        "R": 3,
        "M": 4,
        "L": 5,
        "E": 6,
        "Q": 7,
        "N": 8,
        "T": 9,
    }

    for line in lines[1:]:
        parts = line.split("|")
        tag = parts[0]
        if tag not in ranks or ranks[tag] < rank:
            raise ValueError("scene facts wire fields are unknown or out of order")
        rank = ranks[tag]
        if tag == "G":
            _require_width(parts, 3, tag)
            if setting is not None:
                raise ValueError("scene facts wire contains more than one setting")
            setting = SceneSettingFact(label=parts[1], attributes=_wire_list(parts[2]))
        elif tag == "S":
            _require_width(parts, 7, tag)
            subjects.append(
                SceneSubjectFact(
                    ref=parts[1],
                    count=_wire_count(parts[2]),
                    label=parts[3],
                    color=_wire_optional(parts[4]),
                    attributes=_wire_list(parts[5]),
                    actions=_wire_list(parts[6]),
                )
            )
        elif tag == "O":
            _require_width(parts, 7, tag)
            objects.append(
                SceneObjectFact(
                    ref=parts[1],
                    count=_wire_count(parts[2]),
                    label=parts[3],
                    color=_wire_optional(parts[4]),
                    states=_wire_list(parts[5]),
                    attributes=_wire_list(parts[6]),
                )
            )
        elif tag == "R":
            _require_width(parts, 5, tag)
            relationships.append(
                SceneRelationshipFact(
                    source=parts[1],
                    relation=parts[2],
                    target=parts[3],
                    secondary_target=_wire_optional(parts[4]),
                )
            )
        elif tag == "M":
            _require_width(parts, 4, tag)
            motions.append(
                SceneMotionFact(
                    source=parts[1],
                    direction=_wire_optional(parts[2]),
                    destination=_wire_optional(parts[3]),
                )
            )
        elif tag == "L":
            _require_width(parts, 3, tag)
            salience.append(SceneSalienceFact(source=parts[1], layer=parts[2]))
        elif tag == "E":
            _require_width(parts, 5, tag)
            events.append(
                SceneEventFact(
                    ref=parts[1],
                    source=parts[2],
                    action=parts[3],
                    object=_wire_optional(parts[4]),
                )
            )
        elif tag == "Q":
            _require_width(parts, 3, tag)
            temporal_order.append(SceneTemporalOrderFact(before=parts[1], after=parts[2]))
        elif tag == "N":
            _require_width(parts, 4, tag)
            negatives.append(
                SceneNegativeFact(
                    kind=parts[1],
                    target=_wire_optional(parts[2]),
                    value=parts[3],
                )
            )
        else:
            _require_width(parts, 6 if len(parts) == 6 else 5, tag)
            result_count = _wire_count(parts[5]) if len(parts) == 6 else None
            if len(parts) == 6 and result_count is None:
                raise ValueError("scene facts transformation count must be explicit")
            if transformation is not None:
                raise ValueError("scene facts wire contains more than one transformation")
            transformation = SceneTransformationFact(
                source=parts[1],
                result_label=parts[2],
                result_color=_wire_optional(parts[3]),
                result_attributes=_wire_list(parts[4]),
                result_count=result_count,
            )

    if setting is None:
        raise ValueError("scene facts wire requires exactly one setting")
    return SceneFactsV2(
        setting=setting,
        subjects=tuple(subjects),
        objects=tuple(objects),
        relationships=tuple(relationships),
        motions=tuple(motions),
        salience=tuple(salience),
        events=tuple(events),
        temporal_order=tuple(temporal_order),
        negatives=tuple(negatives),
        transformation=transformation,
    )


def validate_scene_facts_grounding(facts: SceneFactsV2, *, source_text: str) -> None:
    """Fail closed unless every outbound semantic fact has local source evidence."""

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be nonempty")
    if len(source_text) > 4_000:
        raise ValueError("source_text exceeds the local grounding limit")

    _validate_facts_privacy(facts, source_text=source_text)
    sentences = _source_sentences(source_text)
    action_sentences = sentences + _explicit_carried_sentences(facts, source_text)
    issues: list[str] = []
    _check_phrase(facts.setting.label, sentences, "setting.label", issues)
    for index, attribute in enumerate(facts.setting.attributes):
        _check_near_phrase(
            attribute,
            facts.setting.label,
            sentences,
            f"setting.attributes[{index}]",
            issues,
        )

    entity_by_ref, entity_labels = _grounding_entities(facts, sentences)
    subject_aliases = {
        _normalized_phrase(label)
        for subject in facts.subjects
        for label in (
            subject.label,
            *(f"{color} {subject.label}" for color in COLOR_WORDS),
            entity_by_ref[subject.ref].label,
        )
    }
    subject_labels = tuple(
        label for label in entity_labels if _normalized_phrase(label) in subject_aliases
    )
    ambiguous_labels = tuple(
        entity.label
        for entity in (*facts.subjects, *facts.objects)
        if entity_by_ref[entity.ref].label != entity.label
    )
    for kind, entities in (("subjects", facts.subjects), ("objects", facts.objects)):
        for index, entity in enumerate(entities):
            path = f"{kind}[{index}]"
            label = entity_by_ref[entity.ref].label
            entity_sentences = _sentences_with_phrase(sentences, label)
            if not entity_sentences:
                issues.append(f"{path}.label")
                continue
            if entity.count is not None and not _count_near_label(
                entity.count,
                label,
                entity_sentences,
                entity_labels=entity_labels,
            ):
                issues.append(f"{path}.count")
            if entity.color is not None and label == entity.label and not _descriptor_grounded(
                entity.color,
                entity.label,
                entity_sentences,
                entity_labels=entity_labels,
            ):
                issues.append(f"{path}.color")
            for attribute_index, attribute in enumerate(entity.attributes):
                if not _descriptor_grounded(
                    attribute,
                    label,
                    entity_sentences,
                    entity_labels=entity_labels,
                ):
                    issues.append(f"{path}.attributes[{attribute_index}]")
            if isinstance(entity, SceneSubjectFact):
                for action_index, action in enumerate(entity.actions):
                    unqualified_object = any(
                        _contains_phrase(_normalized_phrase(action), ambiguous)
                        and not any(
                            _contains_phrase(_normalized_phrase(action), reference)
                            for reference in entity_labels
                            if reference != ambiguous
                            and _contains_phrase(_normalized_phrase(reference), ambiguous)
                        )
                        for ambiguous in ambiguous_labels
                    )
                    if unqualified_object or not _action_grounded(
                        action,
                        label,
                        _sentences_with_phrase(action_sentences, label),
                        entity_labels=entity_labels,
                    ):
                        issues.append(f"{path}.actions[{action_index}]")
            else:
                for state_index, state in enumerate(entity.states):
                    if not _descriptor_grounded(
                        state,
                        label,
                        entity_sentences,
                        entity_labels=entity_labels,
                    ):
                        issues.append(f"{path}.states[{state_index}]")

    for index, relationship in enumerate(facts.relationships):
        source = entity_by_ref[relationship.source].label
        target = entity_by_ref[relationship.target].label
        secondary = (
            entity_by_ref[relationship.secondary_target].label
            if relationship.secondary_target is not None
            else None
        )
        if not _relationship_grounded(
            relationship.relation,
            source,
            target,
            secondary,
            action_sentences,
            entity_labels=entity_labels,
        ):
            issues.append(f"relationships[{index}]")

    for index, motion in enumerate(facts.motions):
        source = entity_by_ref[motion.source].label
        destination = (
            entity_by_ref[motion.destination].label if motion.destination is not None else None
        )
        if not _motion_grounded(
            motion,
            source_label=source,
            destination_label=destination,
            sentences=sentences,
            entity_labels=entity_labels,
            subject_labels=subject_labels,
        ):
            issues.append(f"motions[{index}]")

    for index, salience in enumerate(facts.salience):
        source = entity_by_ref[salience.source].label
        if not _salience_grounded(
            salience.layer.value,
            source,
            _sentences_with_phrase(sentences, source),
            entity_labels=entity_labels,
        ):
            issues.append(f"salience[{index}]")

    events_by_ref = {event.ref: event for event in facts.events}
    for index, event in enumerate(facts.events):
        if not _event_grounded(
            event,
            prior_events=facts.events,
            entities=entity_by_ref,
            sentences=sentences,
            subject_labels=subject_labels,
        ):
            issues.append(f"events[{index}]")
    for index, order in enumerate(facts.temporal_order):
        if not _temporal_order_grounded(
            events_by_ref[order.before],
            events_by_ref[order.after],
            entities=entity_by_ref,
            sentences=sentences,
            subject_labels=subject_labels,
        ):
            issues.append(f"temporal_order[{index}]")

    for index, negative in enumerate(facts.negatives):
        target_label = entity_by_ref[negative.target].label if negative.target else None
        if not _negative_grounded(
            negative,
            target_label=target_label,
            sentences=sentences,
            entity_labels=entity_labels,
        ):
            issues.append(f"negatives[{index}]")

    if facts.transformation is not None:
        source = entity_by_ref[facts.transformation.source].label
        if not _transformation_grounded(
            facts.transformation,
            source,
            sentences,
            entity_labels=entity_labels,
            related_labels=tuple(
                entity_by_ref[ref].label
                for relation in facts.relationships
                if relation.source == facts.transformation.source
                for ref in (relation.target, relation.secondary_target)
                if ref is not None
            ),
        ):
            issues.append("transformation")
    if issues:
        raise SceneFactsGroundingError(tuple(issues))


def compile_scene_facts_prompt(
    facts: SceneFactsV2,
    *,
    source_text: str,
    visual_style: str = "luminous storybook illustration",
) -> str:
    """Compile a grounded graph without including or quoting ``source_text``."""

    style = " ".join(visual_style.split())
    if not style or len(style) > 120 or any(character in style for character in "\r\n"):
        raise ValueError("visual_style must be a nonempty single line of at most 120 characters")
    _validate_style_privacy(style, source_text=source_text)
    validate_scene_facts_grounding(facts, source_text=source_text)

    entities, _ = _grounding_entities(facts, _source_sentences(source_text))
    clauses = [
        f"Style: {style}",
        f"Setting: {_descriptor(facts.setting.label, facts.setting.attributes)}",
    ]
    for subject in facts.subjects:
        descriptor = _entity_descriptor(
            subject.label,
            count=subject.count,
            color=subject.color,
            modifiers=subject.attributes,
        )
        action = f"; action: {' and '.join(subject.actions)}" if subject.actions else ""
        clauses.append(f"Subject: {descriptor}{action}")
    for item in facts.objects:
        modifiers = (*item.states, *item.attributes)
        descriptor = _entity_descriptor(
            item.label,
            count=item.count,
            color=item.color,
            modifiers=modifiers,
        )
        clauses.append(f"Object: {descriptor}")
    if facts.relationships:
        rendered_relations = [
            _render_relationship(relationship, entities=entities)
            for relationship in facts.relationships
        ]
        clauses.append("Relations: " + "; ".join(rendered_relations))
    if facts.motions:
        clauses.append(
            "Motion: "
            + "; ".join(_render_motion(motion, entities=entities) for motion in facts.motions)
        )
    if facts.salience:
        clauses.append(
            "Composition: "
            + "; ".join(
                f"{entities[salience.source].label} in the {salience.layer.value}"
                for salience in facts.salience
            )
        )
    if facts.events:
        clauses.append(
            "Events: "
            + "; ".join(_render_event(event, entities=entities) for event in facts.events)
        )
    if facts.temporal_order:
        events = {event.ref: event for event in facts.events}
        clauses.append(
            "Order: "
            + "; ".join(
                f"{_render_event(events[order.before], entities=entities)} before "
                f"{_render_event(events[order.after], entities=entities)}"
                for order in facts.temporal_order
            )
        )
    if facts.negatives:
        clauses.append(
            "Exclude: "
            + "; ".join(
                _render_negative(negative, entities=entities) for negative in facts.negatives
            )
        )
    if facts.transformation is not None:
        result = _entity_descriptor(
            facts.transformation.result_label,
            count=facts.transformation.result_count,
            color=facts.transformation.result_color,
            modifiers=facts.transformation.result_attributes,
        )
        source_label = entities[facts.transformation.source].label
        clauses.append(f"Transformation: {source_label} becomes {result}")
    clauses.append("One continuous full-bleed scene; no captions, labels, signs, or readable text")
    return ". ".join(clauses) + "."


_SUBJECT_OWNERSHIP_RELATIONS = frozenset(
    {
        SceneRelationKind.CARRIES,
        SceneRelationKind.HOLDS,
        SceneRelationKind.LOOKS_AT,
        SceneRelationKind.OWNS,
        SceneRelationKind.WEARS,
    }
)
_SYMMETRIC_RELATIONS = frozenset(
    {
        SceneRelationKind.ATTACHED_TO,
        SceneRelationKind.BESIDE,
        SceneRelationKind.NEXT_TO,
        SceneRelationKind.TOUCHES,
    }
)
_RELATION_MARKERS: dict[SceneRelationKind, tuple[tuple[str, ...], ...]] = {
    SceneRelationKind.ABOVE: (("above",), ("over",)),
    SceneRelationKind.ATTACHED_TO: (("attach", "to"), ("tie", "to")),
    SceneRelationKind.BEHIND: (("behind",),),
    SceneRelationKind.BELOW: (("below",), ("beneath",), ("under",)),
    SceneRelationKind.BESIDE: (("beside",), ("alongside",)),
    SceneRelationKind.BETWEEN: (("between",),),
    SceneRelationKind.CARRIES: (("carry",),),
    SceneRelationKind.CONTAINS: (("contain",), ("inside",)),
    SceneRelationKind.HOLDS: (("hold",),),
    SceneRelationKind.IN_FRONT_OF: (("in", "front", "of"),),
    SceneRelationKind.INSIDE: (("inside",), ("within",)),
    SceneRelationKind.LEFT_OF: (("left", "of"),),
    SceneRelationKind.LOOKS_AT: (("look", "at"), ("watch",)),
    SceneRelationKind.NEXT_TO: (("next", "to"), ("beside",)),
    SceneRelationKind.ON: (("on",), ("atop",)),
    SceneRelationKind.OUTSIDE: (("outside",),),
    SceneRelationKind.OWNS: (("own",), ("belong", "to")),
    SceneRelationKind.RIGHT_OF: (("right", "of"),),
    SceneRelationKind.TOUCHES: (("touch",),),
    SceneRelationKind.UNDER: (("under",), ("beneath",), ("below",)),
    SceneRelationKind.WEARS: (("wear",),),
}
_RELATION_RENDER = {
    SceneRelationKind.ATTACHED_TO: "attached to",
    SceneRelationKind.CARRIES: "carries",
    SceneRelationKind.CONTAINS: "contains",
    SceneRelationKind.HOLDS: "holds",
    SceneRelationKind.IN_FRONT_OF: "in front of",
    SceneRelationKind.LEFT_OF: "left of",
    SceneRelationKind.LOOKS_AT: "looks at",
    SceneRelationKind.NEXT_TO: "next to",
    SceneRelationKind.OWNS: "owns",
    SceneRelationKind.OUTSIDE: "outside",
    SceneRelationKind.RIGHT_OF: "right of",
    SceneRelationKind.TOUCHES: "touches",
    SceneRelationKind.WEARS: "wears",
}
_COUNT_WORDS = {
    1: ("a", "an", "one", "single", "exactly one"),
    2: ("two", "pair", "exactly two"),
    3: ("three", "trio", "exactly three"),
    4: ("four", "exactly four"),
    5: ("five", "exactly five"),
    6: ("six", "exactly six"),
    7: ("seven", "exactly seven"),
    8: ("eight", "exactly eight"),
    9: ("nine", "exactly nine"),
    10: ("ten", "exactly ten"),
    11: ("eleven", "exactly eleven"),
    12: ("twelve", "exactly twelve"),
}
_TRANSFORMATION_MARKERS = (
    ("become",),
    ("turn", "into"),
    ("transform", "into"),
    ("change", "into"),
)
_NEGATION_MARKERS = frozenset({"no", "not", "nothing", "without", "never"})
_ACTION_BINDING_BOUNDARIES = frozenset(
    {
        "after",
        "although",
        "as",
        "because",
        "before",
        "but",
        "hear",
        "if",
        "notice",
        "observe",
        "once",
        "see",
        "since",
        "then",
        "though",
        "until",
        "watch",
        "when",
        "whereas",
        "while",
    }
)
_DESCRIPTOR_BINDING_BOUNDARIES = frozenset(
    {
        "above",
        "behind",
        "below",
        "beside",
        "but",
        "contain",
        "hold",
        "inside",
        "near",
        "outside",
        "stand",
        "under",
        "watch",
        "while",
        "with",
    }
)
_EMAIL = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
_URL = r"(?:https?://|www\.)\S+"
_PHONE = r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)"
_PII = re.compile(rf"(?:{_EMAIL})|(?:{_URL})|(?:{_PHONE})", re.IGNORECASE)
_INJECTION = re.compile(
    r"(?:\b(?:system|assistant|developer|user)\s*:|"
    r"\bignore (?:all |the )?(?:previous|prior) instructions?\b|"
    r"\b(?:system prompt|developer message)\b|"
    r"\breveal (?:the |your )?(?:prompt|instructions?)\b|"
    r"\b(?:draw|show|write|print|display|render)\b[^\r\n.!?]{0,64}"
    r"\b(?:password|account details?|on screen)\b)",
    re.IGNORECASE,
)
_PRINTED_TEXT = re.compile(
    r"\b(?:(?:sign|placard|page|scrap|screen|label)\s+"
    r"(?:reads?|says?|saying|printed|showing)|"
    r"(?:reads?|reading)\s+(?:text|words|a?\s*sign|placard|page|scrap|screen|label))\b",
    re.IGNORECASE,
)
_NAMED = re.compile(
    r"\b(?:named|called|mr|mrs|ms|miss|dr|professor)\.?\s+([^\W\d_][\w'’-]*)",
    re.IGNORECASE,
)


def _require_unique(values: tuple[str, ...], label: str) -> None:
    normalized = [_normalized_phrase(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")


def _validate_acyclic_order(edges: list[tuple[str, str]]) -> None:
    successors: dict[str, set[str]] = {}
    for before, after in edges:
        successors.setdefault(before, set()).add(after)

    def visit(event: str, visiting: set[str], visited: set[str]) -> None:
        if event in visiting:
            raise ValueError("scene temporal order must be acyclic")
        if event in visited:
            return
        visiting.add(event)
        for successor in successors.get(event, ()):
            visit(successor, visiting, visited)
        visiting.remove(event)
        visited.add(event)

    visited: set[str] = set()
    for event in successors:
        visit(event, set(), visited)


def _wire_line(tag: str, *values: object) -> str:
    return "|".join((tag, *(_wire_value(value) for value in values)))


def _wire_value(value: object) -> str:
    if value is None or value == ():
        return "-"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def _require_width(parts: list[str], width: int, tag: str) -> None:
    if len(parts) != width or any(not value for value in parts):
        raise ValueError(f"scene facts wire {tag} field has an invalid shape")


def _wire_optional(value: str) -> str | None:
    return None if value == "-" else value


def _wire_list(value: str) -> tuple[str, ...]:
    if value == "-":
        return ()
    values = tuple(value.split(","))
    if any(not item for item in values):
        raise ValueError("scene facts wire list contains an empty value")
    return values


def _wire_count(value: str) -> int | None:
    if value == "-":
        return None
    if not value.isascii() or not value.isdigit():
        raise ValueError("scene facts wire count must be an ASCII integer")
    return int(value)


def _normalized_phrase(value: str) -> tuple[str, ...]:
    return normalize_semantic_phrase(value)


def _source_sentences(source_text: str) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tokens
        for sentence in _asserted_units(source_text)
        if (tokens := _normalized_phrase(sentence))
    )


def _grounding_entities(
    facts: SceneFactsV2, sentences: tuple[tuple[str, ...], ...]
) -> tuple[dict[str, SceneSubjectFact | SceneObjectFact], tuple[str, ...]]:
    entities = (*facts.subjects, *facts.objects)
    labels = tuple(entity.label for entity in entities)
    references = {}
    binding_labels = set()
    color_words = {
        " ".join(_normalized_phrase(color))
        for color in COLOR_WORDS | {item.color for item in entities if item.color}
    }
    for entity in entities:
        colors = {
            color
            for color in color_words
            if _descriptor_grounded(color, entity.label, sentences, entity_labels=labels)
        }
        repeated = sum(
            _normalized_phrase(label) == _normalized_phrase(entity.label) for label in labels
        ) > 1
        if len(colors) > 1 or repeated:
            qualified = {f"{color} {entity.label}" for color in colors}
            if " ".join(_normalized_phrase(entity.color or "")) not in colors or not all(
                any(_contains_phrase(sentence, label) for sentence in sentences)
                for label in qualified
            ):
                raise SceneFactsGroundingError(("entities.identity",))
            label = f"{entity.color} {entity.label}"
            references[entity.ref] = entity.model_copy(update={"label": label})
            binding_labels.update(qualified)
        else:
            references[entity.ref] = entity
            binding_labels.add(entity.label)
    return references, tuple(sorted(binding_labels))


def _asserted_units(source_text: str) -> tuple[str, ...]:
    without_quotes = re.sub(
        r'''"[^"]*(?:"|$)|“[^”]*(?:”|$)|(?<!\w)['‘](?:[^'’]|['’](?=\w))*(?:['’](?!\w)|$)''',
        " ",
        source_text,
    )
    return tuple(_asserted_clause(unit) for unit in re.split(r"[.!?;]+", without_quotes))


def _asserted_clause(sentence: str) -> str:
    if re.search(
        r"\b(?:if|unless|whether|would|could|might|may|will|must|should|can|perhaps|"
        r"allegedly|reportedly|supposedly|hypothetically|in a dream)\b",
        sentence,
        re.IGNORECASE,
    ):
        return ""
    report = re.search(
        r"\b(?:says?|said|tells?|told|claims?|claimed|reports?|reported|promises?|promised|"
        r"denies|deny|denied|imagines?|imagined|believes?|believed|thinks?|thought|"
        r"supposes?|supposed|hears?|heard|dreams?|dreamed)\b",
        sentence,
        re.IGNORECASE,
    )
    if report is None:
        return sentence
    # A comma alone may introduce speech or trailing attribution, not a true clause.
    independent = re.search(r",\s*(?:and|but|while)\b", sentence[: report.start()], re.IGNORECASE)
    return sentence[: independent.start()] if independent else ""


def _explicit_carried_sentences(
    facts: SceneFactsV2, source_text: str
) -> tuple[tuple[str, ...], ...]:
    """Prove a complete passive clause before reordering it for action checks."""
    result = []
    setting = re.escape(facts.setting.label.casefold())
    for sentence in _asserted_units(source_text.casefold()):
        clause = re.sub(
            rf"^(?:in|at|inside)\s+(?:(?:a|an|the)\s+)?{setting},\s*",
            "",
            sentence.strip(),
        )
        match = re.fullmatch(
            r"([a-z0-9 -]+) (?:is|are|was|were) carried by ([a-z0-9 -]+)", clause
        )
        if match is None:
            continue
        patient, actor = match.groups()
        if any(_passive_noun_matches(actor, entity) for entity in facts.subjects) and any(
            _passive_noun_matches(patient, entity) for entity in facts.objects
        ):
            result.append(_normalized_phrase(f"{actor} carries {patient}"))
    return tuple(result)


def _passive_noun_matches(phrase: str, entity: SceneSubjectFact | SceneObjectFact) -> bool:
    tokens = _normalized_phrase(phrase)
    label = _normalized_phrase(entity.label)
    if not label or tokens[-len(label) :] != label:
        return False
    modifiers = {"a", "an", "the"}
    for value in (entity.color, *entity.attributes):
        if value:
            modifiers.update(_normalized_phrase(value))
    if entity.count is not None:
        modifiers.add(str(entity.count))
        for value in _COUNT_WORDS[entity.count]:
            modifiers.update(_normalized_phrase(value))
    return all(token in modifiers for token in tokens[: -len(label)])


def _contains_phrase(tokens: tuple[str, ...], phrase: str | tuple[str, ...]) -> bool:
    needle = _normalized_phrase(phrase) if isinstance(phrase, str) else phrase
    width = len(needle)
    return width > 0 and any(
        tokens[index : index + width] == needle for index in range(len(tokens) - width + 1)
    )


def _phrases_conflict(left: str, right: str) -> bool:
    left_tokens = _normalized_phrase(left)
    right_tokens = _normalized_phrase(right)
    return _contains_phrase(left_tokens, right_tokens) or _contains_phrase(
        right_tokens,
        left_tokens,
    )


def _sentences_with_phrase(
    sentences: tuple[tuple[str, ...], ...], phrase: str
) -> tuple[tuple[str, ...], ...]:
    return tuple(sentence for sentence in sentences if _contains_phrase(sentence, phrase))


def _check_phrase(
    phrase: str,
    sentences: tuple[tuple[str, ...], ...],
    path: str,
    issues: list[str],
) -> None:
    if not any(_contains_phrase(sentence, phrase) for sentence in sentences):
        issues.append(path)


def _check_near_phrase(
    detail: str,
    label: str,
    sentences: tuple[tuple[str, ...], ...],
    path: str,
    issues: list[str],
) -> None:
    if not _phrase_near_label(detail, label, _sentences_with_phrase(sentences, label)):
        issues.append(path)


def _phrase_positions(tokens: tuple[str, ...], phrase: str) -> tuple[tuple[int, int], ...]:
    needle = _normalized_phrase(phrase)
    width = len(needle)
    return tuple(
        (index, index + width)
        for index in range(len(tokens) - width + 1)
        if tokens[index : index + width] == needle
    )


def _phrase_near_label(
    detail: str,
    label: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    radius: int = 7,
) -> bool:
    for sentence in sentences:
        label_positions = _phrase_positions(sentence, label)
        detail_positions = _phrase_positions(sentence, detail)
        for label_start, label_end in label_positions:
            if any(
                detail_start >= max(0, label_start - radius)
                and detail_end <= min(len(sentence), label_end + radius)
                for detail_start, detail_end in detail_positions
            ):
                return True
    return False


def _count_near_label(
    count: int,
    label: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
) -> bool:
    candidates = (*_COUNT_WORDS[count], str(count))
    return any(
        _descriptor_grounded(
            candidate,
            label,
            sentences,
            entity_labels=entity_labels,
        )
        for candidate in candidates
    )


def _descriptor_grounded(
    detail: str,
    label: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        for label_start, label_end in _phrase_positions(sentence, label):
            for detail_start, detail_end in _phrase_positions(sentence, detail):
                if (
                    label_start - 4 <= detail_start < label_start
                    and detail_end <= label_start
                    and not _position_negated(sentence, detail_start)
                    and not _DESCRIPTOR_BINDING_BOUNDARIES.intersection(
                        sentence[detail_end:label_start]
                    )
                    and not _has_intervening_entity(
                        sentence,
                        start=detail_end,
                        end=label_start,
                        excluded=(label,),
                        entity_labels=entity_labels,
                    )
                ):
                    return True
            if (
                label_end < len(sentence)
                and sentence[label_end] in {"be", "look", "appear", "remain"}
                and any(
                    label_end < detail_start <= label_end + 3
                    and not _position_negated(sentence, detail_start)
                    and not _DESCRIPTOR_BINDING_BOUNDARIES.intersection(
                        sentence[label_end:detail_start]
                    )
                    for detail_start, _ in _phrase_positions(sentence, detail)
                )
            ):
                return True
    return False


def _action_grounded(
    action: str,
    source: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
) -> bool:
    action_tokens = tuple(
        token for token in _normalized_phrase(action) if token not in {"a", "an", "the"}
    )
    if not action_tokens:
        return False
    for sentence in sentences:
        action_positions = _token_positions(sentence, (action_tokens[0],))
        for _, source_end in _phrase_positions(sentence, source):
            for action_start, _ in action_positions:
                if source_end > action_start or _position_negated(sentence, action_start):
                    continue
                if _has_binding_boundary(sentence[source_end:action_start]) or {
                    "and",
                    "or",
                }.intersection(sentence[source_end:action_start]):
                    continue
                if _has_intervening_entity(
                    sentence,
                    start=source_end,
                    end=action_start,
                    excluded=(source,),
                    entity_labels=entity_labels,
                ):
                    continue
                window = sentence[action_start : action_start + len(action_tokens) + 12]
                cursor = 1
                for token in action_tokens[1:]:
                    next_position = next(
                        (index for index in range(cursor, len(window)) if window[index] == token),
                        None,
                    )
                    if next_position is None:
                        break
                    gap = window[cursor:next_position]
                    if _has_action_binding_boundary(gap):
                        break
                    cursor = next_position + 1
                else:
                    return True
    return False


def _has_action_binding_boundary(tokens: tuple[str, ...]) -> bool:
    return bool(
        _has_binding_boundary(tokens)
        or _DESCRIPTOR_BINDING_BOUNDARIES.intersection(tokens)
        or _NEGATION_MARKERS.intersection(tokens)
        or {"and", "or", "neither", "nor"}.intersection(tokens)
        or any(
            _token_positions(tokens, marker)
            for markers in _RELATION_MARKERS.values()
            for marker in markers
        )
        or any(_token_positions(tokens, marker) for marker in _TRANSFORMATION_MARKERS)
    )


def _salience_grounded(
    layer: str,
    source: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        for _, source_end in _phrase_positions(sentence, source):
            for layer_start, _ in _phrase_positions(sentence, layer):
                if not source_end <= layer_start <= source_end + 4:
                    continue
                if _position_negated(sentence, layer_start):
                    continue
                if not _has_intervening_entity(
                    sentence,
                    start=source_end,
                    end=layer_start,
                    excluded=(source,),
                    entity_labels=entity_labels,
                ):
                    return True
    return False


def _relationship_grounded(
    relation: SceneRelationKind,
    source: str,
    target: str,
    secondary: str | None,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        source_positions = _phrase_positions(sentence, source)
        target_positions = _phrase_positions(sentence, target)
        if not source_positions or not target_positions:
            continue
        secondary_positions = _phrase_positions(sentence, secondary) if secondary else ()
        if secondary is not None and not secondary_positions:
            continue
        for marker in _RELATION_MARKERS[relation]:
            marker_positions = _token_positions(sentence, marker)
            for source_start, source_end in source_positions:
                for target_start, target_end in target_positions:
                    for marker_start, marker_end in marker_positions:
                        if _position_negated(sentence, marker_start):
                            continue
                        forward = source_end <= marker_start and marker_end <= target_start
                        reverse = target_end <= marker_start and marker_end <= source_start
                        if relation in _SYMMETRIC_RELATIONS:
                            if forward and not (
                                _has_binding_boundary(sentence[source_end:marker_start])
                                or _has_intervening_entity(
                                    sentence,
                                    start=source_end,
                                    end=marker_start,
                                    excluded=(source, target),
                                    entity_labels=entity_labels,
                                )
                                or _has_intervening_entity(
                                    sentence,
                                    start=marker_end,
                                    end=target_start,
                                    excluded=(source, target),
                                    entity_labels=entity_labels,
                                )
                            ):
                                return True
                            if reverse and not (
                                _has_binding_boundary(sentence[marker_end:source_start])
                                or _has_intervening_entity(
                                    sentence,
                                    start=target_end,
                                    end=marker_start,
                                    excluded=(source, target),
                                    entity_labels=entity_labels,
                                )
                                or _has_intervening_entity(
                                    sentence,
                                    start=marker_end,
                                    end=source_start,
                                    excluded=(source, target),
                                    entity_labels=entity_labels,
                                )
                            ):
                                return True
                            continue
                        if relation is SceneRelationKind.CONTAINS:
                            if (
                                marker == ("contain",)
                                and forward
                                and not _has_binding_boundary(sentence[source_end:marker_start])
                            ):
                                return True
                            if (
                                marker == ("inside",)
                                and reverse
                                and not _has_binding_boundary(sentence[marker_end:source_start])
                            ):
                                return True
                            continue
                        if relation is SceneRelationKind.OWNS and marker == ("belong", "to"):
                            if reverse:
                                return True
                            continue
                        if not forward:
                            continue
                        if _has_binding_boundary(sentence[source_end:marker_start]):
                            continue
                        if _has_intervening_entity(
                            sentence,
                            start=source_end,
                            end=marker_start,
                            excluded=(source, target, *((secondary,) if secondary else ())),
                            entity_labels=entity_labels,
                        ):
                            continue
                        if relation is not SceneRelationKind.BETWEEN:
                            return True
                        if any(target_end <= start for start, _ in secondary_positions):
                            return True
    return False


def _has_intervening_entity(
    sentence: tuple[str, ...],
    *,
    start: int,
    end: int,
    excluded: tuple[str, ...],
    entity_labels: tuple[str, ...],
) -> bool:
    excluded_normalized = {_normalized_phrase(label) for label in excluded}
    return any(
        normalized not in excluded_normalized
        and any(
            start <= entity_start and entity_end <= end for entity_start, entity_end in positions
        )
        for label in entity_labels
        if (normalized := _normalized_phrase(label))
        for positions in (_phrase_positions(sentence, label),)
    )


def _has_binding_boundary(tokens: tuple[str, ...]) -> bool:
    if _ACTION_BINDING_BOUNDARIES.intersection(tokens):
        return True
    determiners = {"a", "an", "each", "every", "one", "the", "this", "that"}
    return any(
        token == "and"
        and "between" not in tokens[:index]
        and index + 2 < len(tokens)
        and tokens[index + 1] in determiners
        for index, token in enumerate(tokens)
    )


def _motion_grounded(
    motion: SceneMotionFact,
    *,
    source_label: str,
    destination_label: str | None,
    sentences: tuple[tuple[str, ...], ...],
    entity_labels: tuple[str, ...],
    subject_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        source_positions = _phrase_positions(sentence, source_label)
        if not source_positions:
            continue
        direction_grounded = motion.direction is None
        if motion.direction is not None:
            direction_positions = tuple(
                position
                for form in (motion.direction.value, motion.direction.value.removesuffix("s"))
                for position in _phrase_positions(sentence, form)
            )
            direction_grounded = any(
                source_end <= direction_start
                and not _position_negated(sentence, direction_start)
                and not _has_binding_boundary(sentence[source_end:direction_start])
                and not _has_intervening_entity(
                    sentence,
                    start=source_end,
                    end=direction_start,
                    excluded=(source_label,),
                    entity_labels=entity_labels,
                )
                for _, source_end in source_positions
                for direction_start, _ in direction_positions
            )
        if not direction_grounded:
            continue
        if destination_label is None:
            return True
        destination_positions = _phrase_positions(sentence, destination_label)
        toward_positions = _token_positions(sentence, ("toward",))
        if any(
            source_end <= toward_start
            and toward_end <= destination_start
            and not _position_negated(sentence, toward_start)
            and not _has_binding_boundary(sentence[source_end:toward_start])
            and not _has_binding_boundary(sentence[toward_end:destination_start])
            and not _has_intervening_entity(
                sentence,
                start=source_end,
                end=toward_start,
                excluded=(source_label,),
                entity_labels=subject_labels,
            )
            and not _has_intervening_entity(
                sentence,
                start=toward_end,
                end=destination_start,
                excluded=(destination_label,),
                entity_labels=entity_labels,
            )
            for _, source_end in source_positions
            for toward_start, toward_end in toward_positions
            for destination_start, _ in destination_positions
        ):
            return True
    return False


def _event_grounded(
    event: SceneEventFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
    sentences: tuple[tuple[str, ...], ...],
    subject_labels: tuple[str, ...],
    prior_events: tuple[SceneEventFact, ...] = (),
) -> bool:
    return any(
        _event_spans(
            event,
            sentence,
            entities=entities,
            subject_labels=subject_labels,
            prior_events=prior_events,
        )
        for sentence in sentences
    )


def _event_spans(
    event: SceneEventFact,
    sentence: tuple[str, ...],
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
    subject_labels: tuple[str, ...],
    prior_events: tuple[SceneEventFact, ...] = (),
) -> tuple[tuple[int, int], ...]:
    """Locate actions only where their actor and optional object bind together."""

    source_label = entities[event.source].label
    object_label = entities[event.object].label if event.object is not None else None
    bound_actions = tuple(
        (action_start, action_end)
        for action_start, action_end in _phrase_positions(sentence, event.action)
        if any(
            source_end <= action_start
            and not _position_negated(sentence, action_start)
            and not _has_binding_boundary(sentence[source_end:action_start])
            and not {"and", "or"}.intersection(sentence[source_end:action_start])
            and not _has_intervening_entity(
                sentence,
                start=source_end,
                end=action_start,
                excluded=(source_label,),
                entity_labels=subject_labels,
            )
            for _, source_end in _phrase_positions(sentence, source_label)
        )
    )
    coordinated_actions = []
    for action_start, action_end in _phrase_positions(sentence, event.action):
        for connector in (("and", "then"), ("and", "afterward"), ("and", "only", "afterward")):
            connector_start = action_start - len(connector)
            if connector_start < 0 or sentence[connector_start:action_start] != connector:
                continue
            if any(
                prior.source == event.source
                and prior.ref != event.ref
                and prior.object is not None
                and any(
                    end == connector_start
                    for _, end in _event_spans(
                        prior,
                        sentence[:connector_start],
                        entities=entities,
                        subject_labels=subject_labels,
                    )
                )
                for prior in prior_events
            ):
                coordinated_actions.append((action_start, action_end))
    bound_actions += tuple(coordinated_actions)
    if object_label is None:
        return bound_actions
    return tuple(
        (action_start, object_end)
        for action_start, action_end in bound_actions
        for object_start, object_end in _phrase_positions(sentence, object_label)
        if action_end <= object_start
        and not _position_negated(sentence, object_start)
        and not _has_action_binding_boundary(sentence[action_end:object_start])
        and not _has_intervening_entity(
            sentence,
            start=action_end,
            end=object_start,
            excluded=(object_label,),
            entity_labels=tuple(
                dict.fromkeys((*subject_labels, *(entity.label for entity in entities.values())))
            ),
        )
    )


def _temporal_order_grounded(
    before: SceneEventFact,
    after: SceneEventFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
    sentences: tuple[tuple[str, ...], ...],
    subject_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        before_actions = _event_spans(
            before,
            sentence,
            entities=entities,
            subject_labels=subject_labels,
            prior_events=(before, after),
        )
        after_actions = _event_spans(
            after,
            sentence,
            entities=entities,
            subject_labels=subject_labels,
            prior_events=(before, after),
        )
        if any(
            before_end <= after_start
            and not {"before", "after"}.intersection(sentence[:before_start])
            and (
                {"before", "then", "afterward"}.intersection(
                    sentence[before_end:after_start]
                )
                or "first" in sentence[max(0, before_start - 2) : before_start + 1]
            )
            for before_start, before_end in before_actions
            for after_start, _ in after_actions
        ):
            return True
    return False


def _position_negated(tokens: tuple[str, ...], position: int) -> bool:
    return bool(_NEGATION_MARKERS.intersection(tokens[max(0, position - 3) : position]))


def _token_positions(
    tokens: tuple[str, ...],
    phrase: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    width = len(phrase)
    return tuple(
        (index, index + width)
        for index in range(len(tokens) - width + 1)
        if tokens[index : index + width] == phrase
    )


def _negative_grounded(
    negative: SceneNegativeFact,
    *,
    target_label: str | None,
    sentences: tuple[tuple[str, ...], ...],
    entity_labels: tuple[str, ...],
) -> bool:
    for sentence in sentences:
        if not _contains_phrase(sentence, negative.value):
            continue
        if target_label is None and _NEGATION_MARKERS.intersection(sentence):
            return True
        if target_label is None:
            continue
        for _, target_end in _phrase_positions(sentence, target_label):
            for value_start, _ in _phrase_positions(sentence, negative.value):
                if target_end > value_start:
                    continue
                if not _NEGATION_MARKERS.intersection(sentence[target_end:value_start]):
                    continue
                if _has_binding_boundary(sentence[target_end:value_start]):
                    continue
                if not _has_intervening_entity(
                    sentence,
                    start=target_end,
                    end=value_start,
                    excluded=(target_label,),
                    entity_labels=entity_labels,
                ):
                    return True
    return False


def _transformation_grounded(
    transformation: SceneTransformationFact,
    source_label: str,
    sentences: tuple[tuple[str, ...], ...],
    *,
    entity_labels: tuple[str, ...],
    related_labels: tuple[str, ...],
) -> bool:
    details = (
        transformation.result_label,
        *((transformation.result_color,) if transformation.result_color else ()),
        *transformation.result_attributes,
    )
    for sentence in sentences:
        if not _contains_phrase(sentence, source_label):
            continue
        for marker in _TRANSFORMATION_MARKERS:
            for _, source_end in _phrase_positions(sentence, source_label):
                for marker_start, marker_end in _token_positions(sentence, marker):
                    if (
                        source_end > marker_start
                        or _position_negated(sentence, marker_start)
                        or _has_binding_boundary(sentence[source_end:marker_start])
                    ):
                        continue
                    if not all(
                        any(
                            marker_end <= detail_start <= marker_end + 12
                            and not _position_negated(sentence, detail_start)
                            for detail_start, _ in _phrase_positions(sentence, detail)
                        )
                        for detail in details
                    ):
                        continue
                    if transformation.result_count is not None and not any(
                        marker_end <= result_start <= marker_end + 12
                        and not _has_binding_boundary(sentence[marker_end:result_start])
                        and not {"and", "or"}.intersection(sentence[marker_end:result_start])
                        and _count_near_label(
                            transformation.result_count,
                            transformation.result_label,
                            (sentence[marker_end:result_end],),
                            entity_labels=entity_labels,
                        )
                        for result_start, result_end in _phrase_positions(
                            sentence, transformation.result_label
                        )
                    ):
                        continue
                    if not _has_intervening_entity(
                        sentence,
                        start=source_end,
                        end=marker_start,
                        excluded=(source_label, *related_labels),
                        entity_labels=entity_labels,
                    ):
                        return True
    return False


def _fact_values(facts: SceneFactsV2) -> tuple[str, ...]:
    values = [facts.setting.label, *facts.setting.attributes]
    for subject in facts.subjects:
        values.extend((subject.ref, subject.label, *(subject.attributes), *(subject.actions)))
        if subject.color:
            values.append(subject.color)
    for item in facts.objects:
        values.extend((item.ref, item.label, *item.states, *item.attributes))
        if item.color:
            values.append(item.color)
    for motion in facts.motions:
        if motion.direction is not None:
            values.append(motion.direction.value)
    values.extend(salience.layer.value for salience in facts.salience)
    values.extend(event.action for event in facts.events)
    values.extend(negative.value for negative in facts.negatives)
    if facts.transformation is not None:
        values.extend(
            (
                facts.transformation.result_label,
                *facts.transformation.result_attributes,
            )
        )
        if facts.transformation.result_color:
            values.append(facts.transformation.result_color)
    return tuple(values)


def _validate_facts_privacy(facts: SceneFactsV2, *, source_text: str) -> None:
    values = _fact_values(facts)
    for value in values:
        normalized_value = unicodedata.normalize("NFKC", value)
        if _PII.search(normalized_value):
            raise SceneFactsPrivacyError("scene facts contain possible contact data")
        if _INJECTION.search(normalized_value):
            raise SceneFactsPrivacyError("scene facts contain instruction-like content")
        if _PRINTED_TEXT.search(normalized_value):
            raise SceneFactsPrivacyError("scene facts contain printed-text content")
        if len(_normalized_phrase(value)) > 8:
            raise SceneFactsPrivacyError("scene facts contain an overlong semantic phrase")
    printed_payloads = printed_source_payload_candidates(source_text)
    if any(
        contains_token_sequence(privacy_tokens(value), payload)
        for value in values
        for payload in printed_payloads
    ):
        raise SceneFactsPrivacyError("scene facts contain a printed source payload")

    named_tokens = {
        *(privacy_tokens(match.group(1)) for match in _NAMED.finditer(source_text)),
        *proper_name_candidates(source_text),
    }
    if any(
        named and contains_token_sequence(privacy_tokens(value), named)
        for named in named_tokens
        for value in values
    ):
        raise SceneFactsPrivacyError("scene facts contain a source proper-name candidate")
    if any(
        SENSITIVE_CONTENT_PATTERN.search(unicodedata.normalize("NFKC", value)) for value in values
    ):
        raise SceneFactsPrivacyError("scene facts contain protected sensitive content")


def _validate_style_privacy(style: str, *, source_text: str) -> None:
    normalized_style = unicodedata.normalize("NFKC", style)
    if _PII.search(normalized_style):
        raise SceneFactsPrivacyError("visual style contains possible contact data")
    if _INJECTION.search(normalized_style):
        raise SceneFactsPrivacyError("visual style contains instruction-like content")
    if _PRINTED_TEXT.search(normalized_style):
        raise SceneFactsPrivacyError("visual style contains printed-text content")

    style_tokens = privacy_tokens(style)
    source_tokens = privacy_tokens(source_text)
    if (
        (source_tokens and contains_token_sequence(style_tokens, source_tokens))
        or contains_distinctive_source_phrase(style_tokens, source_tokens)
        or any(
            contains_token_sequence(style_tokens, candidate)
            for candidate in proper_name_candidates(source_text)
        )
    ):
        raise SceneFactsPrivacyError("visual style overlaps protected source content")


def _descriptor(label: str, modifiers: tuple[str, ...]) -> str:
    return " ".join((*modifiers, label)) if modifiers else label


def _entity_descriptor(
    label: str,
    *,
    count: int | None,
    color: str | None,
    modifiers: tuple[str, ...],
) -> str:
    count_text = "" if count is None else ("exactly one" if count == 1 else f"exactly {count}")
    return " ".join(
        value for value in (count_text, *((color,) if color else ()), *modifiers, label) if value
    )


def _render_relationship(
    relationship: SceneRelationshipFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
) -> str:
    predicate = _RELATION_RENDER.get(
        relationship.relation,
        relationship.relation.value.replace("_", " "),
    )
    rendered = (
        f"{entities[relationship.source].label} {predicate} {entities[relationship.target].label}"
    )
    if relationship.secondary_target is not None:
        rendered += f" and {entities[relationship.secondary_target].label}"
    return rendered


def _render_motion(
    motion: SceneMotionFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
) -> str:
    rendered = entities[motion.source].label
    if motion.direction is not None:
        rendered += f" {motion.direction.value}"
    if motion.destination is not None:
        rendered += f" travels toward {entities[motion.destination].label}"
    return rendered


def _render_event(
    event: SceneEventFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
) -> str:
    rendered = f"{entities[event.source].label} {event.action}"
    if event.object is not None:
        rendered += f" {entities[event.object].label}"
    return rendered


def _render_negative(
    negative: SceneNegativeFact,
    *,
    entities: dict[str, SceneSubjectFact | SceneObjectFact],
) -> str:
    if negative.target is None:
        return f"no {negative.value}"
    target = entities[negative.target].label
    if negative.kind is SceneNegativeKind.ACTION:
        return f"{target} does not {negative.value}"
    return f"{target} is not {negative.value}"
