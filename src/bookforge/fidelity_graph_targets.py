"""Deterministic SceneFactsV2 targets for the public Story Fidelity corpus.

This adapter deliberately derives targets from the existing, typed fidelity
contract.  It is not a second parser for arbitrary stories.  Unsupported or
ambiguous corpus semantics produce an explicit ineligible result so they can
never become invented training labels.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from bookforge.domain import FrozenStrictModel
from bookforge.fidelity_dataset import selected_objects
from bookforge.fidelity_schema import (
    DatasetSplit,
    ExpectationKind,
    FidelityExpectation,
    FidelityRecord,
    SlotName,
)
from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsGroundingError,
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneFactsWireBudgetError,
    SceneMotionFact,
    SceneNegativeFact,
    SceneNegativeKind,
    SceneObjectFact,
    SceneRelationKind,
    SceneRelationshipFact,
    SceneSalienceFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
    SceneTransformationFact,
)
from bookforge.semantic_text import normalize_semantic_phrase

GRAPH_TARGET_SCHEMA_VERSION = "1.0"


class GraphTargetRefusal(StrEnum):
    """Stable, value-free reasons that a record cannot become a target."""

    NON_PUBLIC_SPLIT = "non_public_split"
    AMBIGUOUS_CONTRACT = "ambiguous_contract"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    CONSTRUCTION_REJECTED = "construction_rejected"
    GROUNDING_REJECTED = "grounding_rejected"
    PRIVACY_REJECTED = "privacy_rejected"
    FORBIDDEN_TERM_LEAK = "forbidden_term_leak"
    WIRE_BUDGET_EXCEEDED = "wire_budget_exceeded"


class FidelityGraphTarget(FrozenStrictModel):
    """Eligibility plus an optional graph, without source or secret content."""

    schema_version: Literal[GRAPH_TARGET_SCHEMA_VERSION] = GRAPH_TARGET_SCHEMA_VERSION
    record_id: str
    split: DatasetSplit
    categories: tuple[str, ...]
    eligible: bool
    facts: SceneFactsV2 | None = None
    refusal: GraphTargetRefusal | None = None
    detail_codes: tuple[str, ...] = ()
    token_budget: Literal[64, 96, 128]

    @model_validator(mode="after")
    def validate_outcome(self) -> FidelityGraphTarget:
        if self.eligible != (self.facts is not None):
            raise ValueError("eligible graph targets must contain facts")
        if self.eligible == (self.refusal is not None):
            raise ValueError("ineligible graph targets must contain a refusal")
        return self


class FidelityGraphCategoryCoverage(FrozenStrictModel):
    category: str
    total: int = Field(ge=0)
    eligible: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    refusal_counts: Mapping[str, int]


class FidelityGraphCoverage(FrozenStrictModel):
    total: int = Field(ge=0)
    eligible: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    by_category: tuple[FidelityGraphCategoryCoverage, ...]


class _Refusal(ValueError):
    def __init__(self, reason: GraphTargetRefusal, *detail_codes: str) -> None:
        self.reason = reason
        self.detail_codes = tuple(detail_codes)
        super().__init__(reason.value)


_SUPPORTED_RELATIONS = {
    "above": SceneRelationKind.ABOVE,
    "beneath": SceneRelationKind.BELOW,
    "below": SceneRelationKind.BELOW,
    "inside": SceneRelationKind.INSIDE,
    "outside": SceneRelationKind.OUTSIDE,
}
_COUNT_ACTION = re.compile(r"\bwatches\s+(?:two|five)\s+(?P<label>.+?)\s+circle\b", re.IGNORECASE)
_FIRST_EVENT_ACTION = re.compile(r"\bfirst\s+rings\s+the\s+(?P<label>.+)\Z", re.IGNORECASE)


def derive_fidelity_graph_target(
    record: FidelityRecord,
    *,
    token_budget: Literal[64, 96, 128] = 128,
) -> FidelityGraphTarget:
    """Derive one public target, returning a refusal instead of guessing.

    The split guard is intentionally the first operation.  Hidden records are
    not inspected, parsed, grounded, or serialized by this adapter.
    """

    if record.split not in {DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT}:
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=GraphTargetRefusal.NON_PUBLIC_SPLIT,
        )

    try:
        facts = _derive_public_facts(record)
        _validate_public_grounding(record, facts)
        _reject_private_or_forbidden_values(record, facts)
        facts.to_wire(token_budget=token_budget)
    except _Refusal as refusal:
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=refusal.reason,
            detail_codes=refusal.detail_codes,
        )
    except SceneFactsPrivacyError:
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=GraphTargetRefusal.PRIVACY_REJECTED,
        )
    except SceneFactsGroundingError as error:
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=GraphTargetRefusal.GROUNDING_REJECTED,
            detail_codes=error.paths,
        )
    except SceneFactsWireBudgetError:
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=GraphTargetRefusal.WIRE_BUDGET_EXCEEDED,
        )
    except (TypeError, ValueError):
        # Pydantic errors can contain rejected values.  Keep the public result
        # diagnostic useful without copying any source text or private token.
        return _ineligible(
            record,
            token_budget=token_budget,
            reason=GraphTargetRefusal.CONSTRUCTION_REJECTED,
        )

    return FidelityGraphTarget(
        record_id=record.record_id,
        split=record.split,
        categories=record.categories,
        eligible=True,
        facts=facts,
        token_budget=token_budget,
    )


def summarize_fidelity_graph_coverage(
    results: Iterable[FidelityGraphTarget],
) -> FidelityGraphCoverage:
    """Summarize deterministic eligibility without reading source passages."""

    materialized = tuple(results)
    totals: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    refusals: dict[str, Counter[str]] = {}
    for result in materialized:
        for category in result.categories:
            totals[category] += 1
            if result.eligible:
                eligible[category] += 1
            elif result.refusal is not None:
                refusals.setdefault(category, Counter())[result.refusal.value] += 1

    rows = tuple(
        FidelityGraphCategoryCoverage(
            category=category,
            total=totals[category],
            eligible=eligible[category],
            coverage=eligible[category] / totals[category],
            refusal_counts=dict(sorted(refusals.get(category, Counter()).items())),
        )
        for category in sorted(totals)
    )
    accepted = sum(result.eligible for result in materialized)
    return FidelityGraphCoverage(
        total=len(materialized),
        eligible=accepted,
        coverage=accepted / len(materialized) if materialized else 0.0,
        by_category=rows,
    )


def _derive_public_facts(record: FidelityRecord) -> SceneFactsV2:
    category = _single_category(record)
    specialized = _specialized_expectation(record)
    _validate_slot_expectations(record)

    actor_label, actor_attributes = _actor_identity(record, specialized)
    if category in {"destination", "salience", "temporal_order"}:
        actions = ()
    elif category == "containment_relations":
        actions = ("watches",)
    else:
        actions = (record.target.action,)
    subject = SceneSubjectFact(
        ref="s1",
        label=actor_label,
        attributes=actor_attributes,
        actions=actions,
    )
    objects: list[SceneObjectFact] = []
    relationships: list[SceneRelationshipFact] = []
    motions: list[SceneMotionFact] = []
    salience: list[SceneSalienceFact] = []
    events: list[SceneEventFact] = []
    temporal_order: list[SceneTemporalOrderFact] = []
    negatives: list[SceneNegativeFact] = []
    transformation: SceneTransformationFact | None = None

    primary_object = _primary_object(record, specialized, category=category)
    if category == "destination":
        if primary_object is None:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "followed_object")
        subject = subject.model_copy(update={"actions": (f"follows the {primary_object}",)})
    if primary_object is not None:
        count = specialized.count if category == "counts" else None
        color = specialized.object if category == "attributes" else None
        objects.append(
            SceneObjectFact(
                ref="o1",
                label=primary_object,
                count=count,
                color=color,
            )
        )

    if specialized.kind is ExpectationKind.RELATION and category not in {
        "destination",
        "reversed_motion",
    }:
        relation = _SUPPORTED_RELATIONS.get(specialized.predicate or "")
        if relation is None:
            raise _Refusal(
                GraphTargetRefusal.UNSUPPORTED_SEMANTICS,
                "relation_kind",
            )
        source_ref = _entity_ref(
            specialized.subject,
            subject=subject,
            objects=objects,
        )
        target_ref = _ensure_object(
            specialized.object,
            objects=objects,
        )
        if source_ref == target_ref:
            raise _Refusal(
                GraphTargetRefusal.AMBIGUOUS_CONTRACT,
                "relation_self_reference",
            )
        if category == "spatial_relations":
            if primary_object is None:
                raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "held_object")
            held_ref = _ensure_object(primary_object, objects=objects)
            relationships.extend(
                (
                    SceneRelationshipFact(
                        source="s1",
                        relation=SceneRelationKind.HOLDS,
                        target=held_ref,
                    ),
                    SceneRelationshipFact(
                        source=held_ref,
                        relation=relation,
                        target=target_ref,
                    ),
                )
            )
        else:
            relationships.append(
                SceneRelationshipFact(
                    source=source_ref,
                    relation=relation,
                    target=target_ref,
                )
            )

    if category == "transformation":
        source_ref = _entity_ref(
            specialized.subject,
            subject=subject,
            objects=objects,
        )
        if not specialized.object:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "transformation_result")
        transformation = SceneTransformationFact(
            source=source_ref,
            result_label=specialized.object,
        )
    else:
        magic_label = specialized.subject if category == "reversed_motion" else record.target.magic
        _ensure_object(magic_label, objects=objects)

    if category == "destination":
        destination_ref = _ensure_object(specialized.object, objects=objects)
        motions.append(SceneMotionFact(source="s1", destination=destination_ref))
    elif category == "reversed_motion":
        result_ref = _entity_ref(specialized.subject, subject=subject, objects=objects)
        motions.append(SceneMotionFact(source=result_ref, direction=specialized.object))
    elif category == "salience":
        salience.append(SceneSalienceFact(source="s1", layer=specialized.object))
    elif category == "temporal_order":
        if primary_object is None:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "first_event_object")
        first_object_ref = _ensure_object(primary_object, objects=objects)
        second_object_ref = _ensure_object(specialized.object, objects=objects)
        events.extend(
            (
                SceneEventFact(ref="e1", source="s1", action="rings", object=first_object_ref),
                SceneEventFact(ref="e2", source="s1", action="raises", object=second_object_ref),
            )
        )
        temporal_order.append(SceneTemporalOrderFact(before="e1", after="e2"))

    if category == "negation":
        rejected = _single_forbidden(record)
        negatives.append(
            SceneNegativeFact(
                kind=SceneNegativeKind.ACTION,
                target="s1",
                value=rejected,
            )
        )
    elif category == "hallucination":
        negatives.append(
            SceneNegativeFact(
                kind=SceneNegativeKind.ADDITIONAL_SUBJECT,
                value=_single_forbidden(record),
            )
        )

    return SceneFactsV2(
        setting=SceneSettingFact(label=record.target.setting),
        subjects=(subject,),
        objects=tuple(objects),
        relationships=tuple(relationships),
        motions=tuple(motions),
        salience=tuple(salience),
        events=tuple(events),
        temporal_order=tuple(temporal_order),
        negatives=tuple(negatives),
        transformation=transformation,
    )


def _validate_public_grounding(record: FidelityRecord, facts: SceneFactsV2) -> None:
    """Validate facts, resolving only four exact synthetic pronoun templates.

    SceneFactsV2 correctly refuses to infer pronoun antecedents from arbitrary
    prose.  These public corpus records provide typed antecedents, so this
    adapter can prove them by reconstructing the generator's complete sentence.
    No partial match or general pronoun heuristic is accepted.
    """

    try:
        facts.validate_source_grounding(source_text=record.passage)
        return
    except SceneFactsGroundingError as error:
        failed_paths = error.paths

    category = record.categories[0]
    expected_paths = {
        "coreference": ("subjects[0].actions[0]",),
        "negation": ("subjects[0].actions[0]",),
        "passive_voice": ("subjects[0].actions[0]",),
        "prompt_injection": ("subjects[0].actions[0]",),
        "transformation": ("transformation",),
    }
    if failed_paths != expected_paths.get(category):
        raise _Refusal(GraphTargetRefusal.GROUNDING_REJECTED, *failed_paths)

    specialized = _specialized_expectation(record)
    proven = {
        "coreference": _prove_coreference,
        "negation": _prove_negation,
        "passive_voice": _prove_passive_voice,
        "prompt_injection": _prove_prompt_injection,
        "transformation": _prove_transformation,
    }[category](record, specialized)
    if not proven:
        raise _Refusal(GraphTargetRefusal.GROUNDING_REJECTED, *failed_paths)


def _prove_coreference(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> bool:
    if not all((specialized.subject, specialized.predicate, specialized.object)):
        return False
    if specialized.subject != record.target.actor:
        return False
    if record.target.action != f"{specialized.predicate} the {specialized.object}":
        return False
    pattern = re.compile(
        rf"In the {re.escape(record.target.setting)}, the "
        rf"{re.escape(specialized.subject)} places the "
        rf"(?P<first>[^.\r\n]{{1,64}}) beside the {re.escape(specialized.object)}\. "
        rf"The {re.escape(specialized.subject)} {re.escape(specialized.predicate)} "
        rf"the latter, and it becomes {re.escape(record.target.magic)}\."
    )
    match = pattern.fullmatch(record.passage)
    return bool(
        match
        and not _phrases_match(match.group("first"), specialized.object)
        and not _phrases_match(match.group("first"), specialized.subject)
    )


def _prove_negation(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> bool:
    if not all((specialized.subject, specialized.predicate, specialized.object)):
        return False
    if specialized.subject != record.target.actor:
        return False
    if record.target.action != f"{specialized.predicate} the {specialized.object}":
        return False
    try:
        rejected_action = _single_forbidden(record)
    except _Refusal:
        return False
    expected = (
        f"In the {record.target.setting}, the {specialized.subject} does not "
        f"{rejected_action}. Instead, it {specialized.predicate} the object, causing "
        f"{record.target.magic}."
    )
    return record.passage == expected


def _prove_passive_voice(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> bool:
    if not all((specialized.subject, specialized.predicate, specialized.object)):
        return False
    if specialized.subject != record.target.actor:
        return False
    if record.target.action != f"{specialized.predicate} the {specialized.object}":
        return False
    pattern = re.compile(
        rf"In the {re.escape(record.target.setting)}, the "
        rf"{re.escape(specialized.object)} is carried by the "
        rf"{re.escape(specialized.subject)}; the (?P<observer>[^;.\r\n]{{1,64}}) "
        rf"merely watches\. {re.escape(record.target.magic.capitalize())} follows\."
    )
    match = pattern.fullmatch(record.passage)
    return bool(
        match
        and not _phrases_match(match.group("observer"), specialized.subject)
        and not _phrases_match(match.group("observer"), specialized.object)
    )


def _prove_prompt_injection(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> bool:
    if specialized.kind is not ExpectationKind.PRIVACY or len(record.privacy_terms) != 1:
        return False
    objects = [
        concept
        for concept in record.allowed_concepts
        if record.target.action == f"raises the {concept}"
    ]
    if len(objects) != 1:
        return False
    expected = (
        f"In the {record.target.setting}, the {record.target.actor} finds a scrap printed "
        f"with “{record.privacy_terms[0]}”. It folds the scrap away, raises the "
        f"{objects[0]}, and releases {record.target.magic}."
    )
    return record.passage == expected


def _prove_transformation(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> bool:
    if not specialized.subject or not specialized.object:
        return False
    if specialized.object != record.target.magic:
        return False
    if record.target.action != f"opens the {specialized.subject}":
        return False
    try:
        rejected_result = _single_forbidden(record)
    except _Refusal:
        return False
    expected = (
        f"In the {record.target.setting}, the {record.target.actor} opens the "
        f"{specialized.subject}. The object transforms completely into "
        f"{specialized.object}, not {rejected_result}."
    )
    return record.passage == expected


def _single_category(record: FidelityRecord) -> str:
    if len(record.categories) != 1 or record.counterfactual_dimension != record.categories[0]:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "category")
    return record.categories[0]


def _specialized_expectation(record: FidelityRecord) -> FidelityExpectation:
    specialized = tuple(
        expectation
        for expectation in record.expectations
        if expectation.kind is not ExpectationKind.SLOT
    )
    if len(specialized) != 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "specialized_expectation")
    return specialized[0]


def _validate_slot_expectations(record: FidelityRecord) -> None:
    slots = {
        expectation.slot: expectation
        for expectation in record.expectations
        if expectation.kind is ExpectationKind.SLOT
    }
    if set(slots) != set(SlotName):
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "slot_expectations")
    for slot in SlotName:
        expected = getattr(record.target, slot.value.casefold())
        if slots[slot].alternatives != (expected,):
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "slot_target_mismatch")


def _actor_identity(
    record: FidelityRecord,
    specialized: FidelityExpectation,
) -> tuple[str, tuple[str, ...]]:
    if record.categories[0] == "scale":
        if not specialized.subject or not specialized.object:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "scale_attribute")
        return specialized.subject, (specialized.object,)
    return record.target.actor, ()


def _primary_object(
    record: FidelityRecord,
    specialized: FidelityExpectation,
    *,
    category: str,
) -> str | None:
    if category == "counts":
        match = _COUNT_ACTION.search(record.target.action)
        if match is None:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "counted_object")
        label = match.group("label")
        if (
            not specialized.subject
            or selected_objects(specialized.subject).casefold() != label.casefold()
        ):
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "count_binding")
        return label
    if category == "attributes":
        if not specialized.subject:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "attribute_subject")
        return specialized.subject
    if category == "containment_relations":
        if not specialized.subject:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "contained_object")
        return specialized.subject
    if category == "transformation":
        if not specialized.subject:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "transformation_source")
        return specialized.subject
    if category == "destination":
        return _single_allowed_entity(record, excluded=(specialized.object,))
    if category == "temporal_order":
        match = _FIRST_EVENT_ACTION.fullmatch(record.target.action)
        if match is None:
            raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "first_event_object")
        return match.group("label")
    if category == "salience":
        return None

    candidates = []
    action = _normalize(record.target.action)
    excluded = {
        _normalize(record.target.action),
        _normalize(record.target.actor),
        _normalize(record.target.setting),
        _normalize(record.target.magic),
    }
    for concept in record.allowed_concepts:
        normalized = _normalize(concept)
        if specialized.kind is ExpectationKind.RELATION and (
            (specialized.object and _phrases_match(concept, specialized.object))
            or (specialized.predicate and _phrases_match(concept, specialized.predicate))
        ):
            continue
        if normalized and normalized not in excluded and _contains_tokens(action, normalized):
            candidates.append(concept)
    candidates = _most_specific(candidates)
    if len(candidates) > 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "primary_object")
    return candidates[0] if candidates else None


def _single_allowed_entity(
    record: FidelityRecord,
    *,
    excluded: tuple[str | None, ...] = (),
) -> str:
    excluded_values = {
        _normalize(value)
        for value in (
            record.target.setting,
            record.target.actor,
            record.target.action,
            record.target.magic,
            *excluded,
        )
        if value
    }
    candidates = [
        concept for concept in record.allowed_concepts if _normalize(concept) not in excluded_values
    ]
    if len(candidates) != 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "supporting_entity")
    return candidates[0]


def _most_specific(values: list[str]) -> list[str]:
    normalized = [(value, _normalize(value)) for value in dict.fromkeys(values)]
    return [
        value
        for value, tokens in normalized
        if not any(
            value != other and _contains_tokens(other_tokens, tokens)
            for other, other_tokens in normalized
        )
    ]


def _entity_ref(
    label: str | None,
    *,
    subject: SceneSubjectFact,
    objects: list[SceneObjectFact],
) -> str:
    if not label:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "relation_subject")
    matches = [entity.ref for entity in (subject, *objects) if _phrases_match(entity.label, label)]
    if len(matches) != 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "relation_subject_binding")
    return matches[0]


def _ensure_object(label: str | None, *, objects: list[SceneObjectFact]) -> str:
    if not label:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "object_label")
    matches = [item for item in objects if _phrases_match(item.label, label)]
    if len(matches) > 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "object_binding")
    if matches:
        return matches[0].ref
    ref = f"o{len(objects) + 1}"
    objects.append(SceneObjectFact(ref=ref, label=label))
    return ref


def _single_forbidden(record: FidelityRecord) -> str:
    if len(record.forbidden_terms) != 1:
        raise _Refusal(GraphTargetRefusal.AMBIGUOUS_CONTRACT, "negative_fact")
    return record.forbidden_terms[0]


def _reject_private_or_forbidden_values(
    record: FidelityRecord,
    facts: SceneFactsV2,
) -> None:
    semantic_text = " ".join(_semantic_values(facts))
    for value in (*record.privacy_terms, *record.forbidden_terms):
        if _contains_tokens(_normalize(semantic_text), _normalize(value)):
            raise _Refusal(GraphTargetRefusal.FORBIDDEN_TERM_LEAK)


def _semantic_values(facts: SceneFactsV2) -> tuple[str, ...]:
    values = [facts.setting.label, *facts.setting.attributes]
    for subject in facts.subjects:
        values.extend((subject.label, *subject.attributes, *subject.actions))
        if subject.color:
            values.append(subject.color)
    for item in facts.objects:
        values.extend((item.label, *item.states, *item.attributes))
        if item.color:
            values.append(item.color)
    for motion in facts.motions:
        if motion.direction is not None:
            values.append(motion.direction.value)
    values.extend(salience.layer.value for salience in facts.salience)
    values.extend(event.action for event in facts.events)
    if facts.transformation is not None:
        values.extend((facts.transformation.result_label, *facts.transformation.result_attributes))
        if facts.transformation.result_color:
            values.append(facts.transformation.result_color)
    return tuple(values)


def _phrases_match(left: str, right: str) -> bool:
    left_tokens = _normalize(left)
    right_tokens = _normalize(right)
    return _contains_tokens(left_tokens, right_tokens) or _contains_tokens(
        right_tokens, left_tokens
    )


def _normalize(value: str) -> tuple[str, ...]:
    return normalize_semantic_phrase(value)


def _contains_tokens(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    return width > 0 and any(
        haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)
    )


def _ineligible(
    record: FidelityRecord,
    *,
    token_budget: Literal[64, 96, 128],
    reason: GraphTargetRefusal,
    detail_codes: tuple[str, ...] = (),
) -> FidelityGraphTarget:
    return FidelityGraphTarget(
        record_id=record.record_id,
        split=record.split,
        categories=record.categories if reason is not GraphTargetRefusal.NON_PUBLIC_SPLIT else (),
        eligible=False,
        refusal=reason,
        detail_codes=detail_codes,
        token_budget=token_budget,
    )
