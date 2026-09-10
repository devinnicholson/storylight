"""Source-grounded, authored display steps for bounded temporal graphs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from storylight.domain import StoryPack
from storylight.live_scene_planner import (
    LiveSceneGraphPlan,
    LiveScenePlacedLayerPlan,
    validate_live_scene_plan_privacy,
)
from storylight.scene_facts import (
    SceneFactsV2,
    SceneObjectFact,
    SceneRelationKind,
    SceneSubjectFact,
)
from storylight.semantic_text import normalize_semantic_phrase

_DYNAMIC_RELATIONS = {
    SceneRelationKind.CARRIES,
    SceneRelationKind.HOLDS,
    SceneRelationKind.LOOKS_AT,
    SceneRelationKind.TOUCHES,
    SceneRelationKind.WEARS,
}


def _digest(value: object) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, separators=(",", ":"))
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _mentions(value: str, label: str) -> bool:
    tokens, phrase = normalize_semantic_phrase(value), normalize_semantic_phrase(label)
    return any(tokens[index : index + len(phrase)] == phrase for index in range(len(tokens)))


def _validate_page_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", value):
        raise ValueError("invalid source page identifier")


@dataclass(frozen=True)
class DisplaySourcePage:
    page_id: str
    source_text: str
    seed: int
    facts: SceneFactsV2
    phase_selection: Mapping[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        _validate_page_id(self.page_id)
        if (
            not isinstance(self.source_text, str)
            or not self.source_text.strip()
            or len(self.source_text) > 4_000
        ):
            raise ValueError("display source requires one to 4000 characters")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("display seed must be an unsigned 32-bit integer")
        if not isinstance(self.facts, SceneFactsV2):
            raise ValueError("display source requires typed facts")


@dataclass(frozen=True)
class DisplayStep:
    source_page_id: str
    phase: str
    ordinal: int
    source_sha256: str
    parent_graph_sha256: str
    facts: SceneFactsV2
    event_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_page_id(self.source_page_id)
        ordinals = {"still": 0, "before": 0, "after": 1, "first": 0, "then": 1}
        if (
            not isinstance(self.phase, str)
            or self.phase not in ordinals
            or type(self.ordinal) is not int
            or self.ordinal != ordinals[self.phase]
        ):
            raise ValueError("invalid display phase or ordinal")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (self.source_sha256, self.parent_graph_sha256)
        ):
            raise ValueError("invalid display parent digest")
        if not isinstance(self.facts, SceneFactsV2):
            raise ValueError("display step requires typed facts")
        if self.facts.transformation is not None or self.facts.temporal_order:
            raise ValueError("display phase cannot contain an unsplit temporal graph")
        if self.phase in {"first", "then"}:
            if (
                not isinstance(self.event_ref, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", self.event_ref)
                or tuple(event.ref for event in self.facts.events) != (self.event_ref,)
            ):
                raise ValueError("display event reference differs from phase facts")
        elif self.event_ref is not None:
            raise ValueError("unordered phase cannot carry an ordered event reference")

    @property
    def step_id(self) -> str:
        return f"{self.source_page_id}-{self.phase}"

    def manifest_entry(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "source_page_id": self.source_page_id,
            "phase": self.phase,
            "ordinal": self.ordinal,
            "source_sha256": self.source_sha256,
            "parent_graph_sha256": self.parent_graph_sha256,
            "graph_sha256": _digest(self.facts.model_dump(mode="json")),
            "event_ref": self.event_ref,
        }


def derive_display_steps(
    facts: SceneFactsV2,
    *,
    source_text: str,
    source_page_id: str,
    phase_selection: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DisplayStep, ...]:
    """Select dynamic fact IDs explicitly; never infer an event's postconditions.

    Each phase names action:<subject-ref>:<index>, motion:<index>, relation:<index>,
    or event:<ref>. Every unscoped dynamic fact must appear in at least one phase.
    Ordered events and their equivalent relationships are assigned automatically.
    """
    _validate_page_id(source_page_id)
    facts = SceneFactsV2.model_validate(facts.model_dump())
    facts.validate_source_grounding(source_text=source_text)
    if facts.transformation is None and not facts.temporal_order:
        if phase_selection is not None:
            raise ValueError("static display graph does not accept phase selection")
        return (DisplayStep(
            source_page_id=source_page_id, phase="still", ordinal=0,
            source_sha256=_digest(source_text),
            parent_graph_sha256=_digest(facts.model_dump(mode="json")), facts=facts,
        ),)
    entities = {item.ref: item for item in (*facts.subjects, *facts.objects)}
    transformation = facts.transformation
    ordered = {}
    if transformation is not None:
        if facts.temporal_order:
            raise ValueError("combined transformation and ordered events are unsupported")
        if any(
            entity.ref != transformation.source
            and normalize_semantic_phrase(entity.label)
            == normalize_semantic_phrase(transformation.result_label)
            for entity in entities.values()
        ):
            raise ValueError("transformation result overlaps an existing entity identity")
        phases = ("before", "after")
    else:
        if len(facts.events) != 2 or len(facts.temporal_order) != 1:
            raise ValueError("display order requires exactly two explicitly ordered events")
        phases = ("first", "then")
        order = facts.temporal_order[0]
        ordered = dict(zip(phases, (order.before, order.after), strict=True))
    dynamic: dict[str, set[str]] = {}
    automatic = {phase: set() for phase in phases}
    for subject in facts.subjects:
        for index, action in enumerate(subject.actions):
            equivalent = False
            for event in facts.events:
                if event.ref not in ordered.values() or event.source != subject.ref:
                    continue
                suffixes = {()}
                if event.object is not None:
                    target = entities[event.object]
                    label = normalize_semantic_phrase(target.label)
                    suffixes = set()
                    if sum(
                        normalize_semantic_phrase(entity.label) == label
                        for entity in entities.values()
                    ) == 1:
                        suffixes.add(label)
                    if target.color:
                        suffixes.add(normalize_semantic_phrase(target.color) + label)
                if normalize_semantic_phrase(action) in {
                    normalize_semantic_phrase(event.action) + suffix for suffix in suffixes
                }:
                    equivalent = True
                    break
            if not equivalent:
                dynamic[f"action:{subject.ref}:{index}"] = {
                    subject.ref,
                    *(ref for ref, entity in entities.items() if _mentions(action, entity.label)),
                }
    for index, motion in enumerate(facts.motions):
        dynamic[f"motion:{index}"] = {ref for ref in (motion.source, motion.destination) if ref}
    for index, relation in enumerate(facts.relationships):
        if relation.relation not in _DYNAMIC_RELATIONS:
            continue
        key = f"relation:{index}"
        matching = {
            phase
            for phase, event_ref in ordered.items()
            for event in facts.events
            if event.ref == event_ref
            and event.source == relation.source
            and event.object == relation.target
            and normalize_semantic_phrase(event.action)
            == normalize_semantic_phrase(relation.relation.value)
        }
        if matching:
            for phase in matching:
                automatic[phase].add(key)
        else:
            dynamic[key] = {
                ref for ref in (relation.source, relation.target, relation.secondary_target) if ref
            }
    for event in facts.events:
        if event.ref not in ordered.values():
            dynamic[f"event:{event.ref}"] = {ref for ref in (event.source, event.object) if ref}

    if phase_selection is None:
        if dynamic:
            raise ValueError("unscoped dynamic facts require explicit phase selection")
        selected = {phase: set() for phase in phases}
    else:
        if set(phase_selection) != set(phases):
            raise ValueError("phase selection must cover exactly the display phases")
        selected = {}
        for phase in phases:
            values = phase_selection[phase]
            if isinstance(values, str) or any(not isinstance(value, str) for value in values):
                raise ValueError("phase selection requires fact identifiers")
            selected[phase] = set(values)
            if len(selected[phase]) != len(values) or not selected[phase].issubset(dynamic):
                raise ValueError("duplicate or unknown phase fact identifier")
        if set().union(*selected.values()) != set(dynamic):
            raise ValueError("phase selection leaves dynamic facts unassigned")

    steps = []
    for ordinal, phase in enumerate(phases):
        removed = (
            {transformation.source} if transformation is not None and phase == "after" else set()
        )
        if any(dynamic[key] & removed for key in selected[phase]):
            raise ValueError("selected phase fact references the consumed transformation source")
        kept = selected[phase] | automatic[phase]
        subjects = tuple(
            subject.model_copy(
                update={
                    "actions": tuple(
                        action
                        for index, action in enumerate(subject.actions)
                        if f"action:{subject.ref}:{index}" in kept
                    )
                }
            )
            for subject in facts.subjects
            if subject.ref not in removed
        )
        objects = tuple(item for item in facts.objects if item.ref not in removed)
        if removed:
            result_ref = "result"
            while result_ref in entities:
                result_ref += "_"
            objects += (
                SceneObjectFact(
                    ref=result_ref,
                    label=transformation.result_label,
                    count=transformation.result_count,
                    color=transformation.result_color,
                    attributes=transformation.result_attributes,
                ),
            )
        relationships = tuple(
            relation
            for index, relation in enumerate(facts.relationships)
            if not {relation.source, relation.target, relation.secondary_target} & removed
            and (
                relation.relation not in _DYNAMIC_RELATIONS or f"relation:{index}" in kept
            )
        )
        phase_facts = SceneFactsV2(
            setting=facts.setting,
            subjects=subjects,
            objects=objects,
            relationships=relationships,
            motions=tuple(
                motion for index, motion in enumerate(facts.motions) if f"motion:{index}" in kept
            ),
            salience=tuple(item for item in facts.salience if item.source not in removed),
            events=tuple(
                event
                for event in facts.events
                if event.ref == ordered.get(phase) or f"event:{event.ref}" in kept
            ),
            negatives=tuple(
                item
                for item in facts.negatives
                if item.target not in removed
                and not any(_mentions(item.value, entities[ref].label) for ref in removed)
            ),
        )
        phase_facts.validate_source_grounding(source_text=source_text)
        steps.append(
            DisplayStep(
                source_page_id=source_page_id,
                phase=phase,
                ordinal=ordinal,
                source_sha256=_digest(source_text),
                parent_graph_sha256=_digest(facts.model_dump(mode="json")),
                facts=phase_facts,
                event_ref=ordered.get(phase),
            )
        )
    return steps[0], steps[1]


def _descriptor(entity: SceneSubjectFact | SceneObjectFact) -> str:
    fields = [entity.label]
    if entity.count is not None:
        fields.append(f"count {entity.count}")
    if entity.color:
        fields.append(f"color {entity.color}")
    fields.extend(f"attribute {value}" for value in entity.attributes)
    if isinstance(entity, SceneObjectFact):
        fields.extend(f"state {value}" for value in entity.states)
    return "; ".join(fields)


def build_display_plan(
    step: DisplayStep, *, source_text: str, visual_style: str
) -> LiveSceneGraphPlan:
    if step.source_sha256 != _digest(source_text):
        raise ValueError("display step source differs from its parent")
    if not visual_style.strip() or len(visual_style.strip()) > 120:
        raise ValueError("display style must fit the graph renderer contract")
    facts = SceneFactsV2.model_validate(step.facts.model_dump())
    facts.validate_source_grounding(source_text=source_text)
    entities = (*facts.subjects, *facts.objects)
    if len(entities) < 2:
        raise ValueError("display plan requires two distinct retained entities")
    focus, accent = entities[0], entities[-1]
    plan = LiveSceneGraphPlan(
        scene_summary=f"Scene in {facts.setting.label}",
        art_direction=visual_style,
        camera_motion="locked",
        background_prompt=facts.setting.label,
        focus_label=focus.label,
        focus=LiveScenePlacedLayerPlan(
            kind="character" if isinstance(focus, SceneSubjectFact) else "prop",
            prompt=_descriptor(focus),
            anchor=(0.38, 0.56, 0.38, 0.62),
            depth=5,
            motion="parallax",
        ),
        accent=LiveScenePlacedLayerPlan(
            kind="character" if isinstance(accent, SceneSubjectFact) else "prop",
            prompt=_descriptor(accent),
            anchor=(0.72, 0.45, 0.3, 0.4),
            depth=9,
            motion="parallax",
        ),
        scene_facts=facts,
    )
    validate_live_scene_plan_privacy(plan, source_text=source_text)
    page = plan.to_page(
        source_text=source_text, visual_style=visual_style, seed=0, page_id=step.step_id
    )
    if page.scene_spec.master_prompt != facts.to_renderer_prompt(
        source_text=source_text, visual_style=visual_style
    ):
        raise ValueError("display step renderer differs from its graph")
    return plan


def display_manifest(steps: Sequence[DisplayStep]) -> dict[str, object]:
    identifiers = [step.step_id for step in steps]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("display step identifiers must be unique")
    if not steps:
        raise ValueError("display manifest requires at least one display step")
    index = 0
    source_ids = set()
    while index < len(steps):
        first = steps[index]
        if first.source_page_id in source_ids:
            raise ValueError("display source page identifiers must be unique")
        source_ids.add(first.source_page_id)
        if first.phase == "still":
            index += 1
            continue
        if index + 1 == len(steps):
            raise ValueError("display manifest requires complete ordered step pairs")
        first, second = steps[index : index + 2]
        if (
            first.source_page_id != second.source_page_id
            or (first.ordinal, second.ordinal) != (0, 1)
            or (first.phase, second.phase) not in {("before", "after"), ("first", "then")}
            or first.source_sha256 != second.source_sha256
            or first.parent_graph_sha256 != second.parent_graph_sha256
        ):
            raise ValueError("display step order or parent binding differs")
        index += 2
    return {
        "schema_version": 1,
        "kind": "authored-display-steps",
        "steps": [step.manifest_entry() for step in steps],
        "semantic_accuracy_assessed": False,
        "visual_fidelity_assessed": False,
    }


def build_display_story_pack(
    source_pages: Sequence[DisplaySourcePage], *, story_id: str, title: str, visual_style: str
) -> tuple[StoryPack, dict[str, object]]:
    """Build a local plan from supplied graphs; no media or model accuracy is implied."""
    if not 1 <= len(source_pages) <= 12:
        raise ValueError("display story requires one to twelve source pages")
    if any(not isinstance(page, DisplaySourcePage) for page in source_pages):
        raise ValueError("display story requires typed source pages")
    source_by_id = {page.page_id: page for page in source_pages}
    if len(source_by_id) != len(source_pages):
        raise ValueError("display source page identifiers must be unique")
    steps = tuple(
        step
        for source in source_pages
        for step in derive_display_steps(
            source.facts, source_text=source.source_text, source_page_id=source.page_id,
            phase_selection=source.phase_selection,
        )
    )
    if len(steps) > 12:
        raise ValueError("display story exceeds twelve display pages")
    manifest = display_manifest(steps)
    pages = []
    for step in steps:
        source = source_by_id[step.source_page_id]
        plan = build_display_plan(
            step, source_text=source.source_text, visual_style=visual_style
        )
        page = plan.to_page(
            source_text=source.source_text, visual_style=visual_style,
            seed=source.seed, page_id=step.step_id,
        )
        if page.scene_spec.master_prompt != step.facts.to_renderer_prompt(
            source_text=source.source_text, visual_style=visual_style
        ):
            raise ValueError("display page renderer differs from its graph")
        pages.append(page)
    pack = StoryPack(
        schema_version="2.0", story_id=story_id, title=title, reading_level=2,
        visual_style=visual_style, compiler_model="local-authored-scene-facts-v2",
        compiler_contract_revision="authored-display-steps-v1", planning_scope="scene",
        pages=pages, assets=[],
    )
    return pack, {**manifest, "state": "planned", "assets_generated": False}
