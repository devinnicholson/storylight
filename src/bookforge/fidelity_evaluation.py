"""Deterministic semantic and privacy evaluation for Story Fidelity records."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from bookforge.fidelity_graph_targets import derive_fidelity_graph_target
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from bookforge.scene_facts import (
    SceneFactsV2,
    SceneObjectFact,
    SceneRelationKind,
    SceneSubjectFact,
    _has_action_binding_boundary,
)
from bookforge.semantic_text import semantic_lemma

SurfaceName = Literal["raw", "postprocessed", "renderer"]
SceneEntity = SceneSubjectFact | SceneObjectFact
FIDELITY_EVALUATOR_REVISION = "bound-descriptors-renderer-proof-v2"
_ARTICLES = frozenset({"a", "an", "the"})


class ExpectationKind(StrEnum):
    """Semantic checks supported by the fidelity dataset contract."""

    SLOT = "slot"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    COUNT = "count"
    ROLE = "role"
    TRANSFORMATION = "transformation"
    ORDER = "order"
    ALLOWED_CONCEPT = "allowed_concept"
    FORBIDDEN_CONCEPT = "forbidden_concept"
    PRIVACY = "privacy"


_SLOT_NAMES = ("SETTING", "ACTOR", "ACTION", "MAGIC")
_WORD = re.compile(r"[a-z0-9]+")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_INJECTION = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|prior) instructions?|system prompt|"
    r"developer message|reveal (?:the |your )?(?:prompt|instructions?))\b",
    re.IGNORECASE,
)
_NEGATION_PREFIXES = frozenset({"no", "not", "without", "excluding", "except"})
_NEGATION_PAIRS = frozenset({("free", "of"), ("instead", "of"), ("rather", "than")})
_NEGATION_SUFFIXES = frozenset({"absent", "excluded", "missing", "omitted"})
_COUNT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_RAW_GRAMMAR_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "above",
        "across",
        "around",
        "be",
        "becomes",
        "below",
        "beneath",
        "beside",
        "between",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "near",
        "of",
        "on",
        "or",
        "over",
        "the",
        "their",
        "then",
        "through",
        "to",
        "under",
        "was",
        "were",
        "while",
        "with",
        "without",
    }
)
# These generic ambience words are intentionally non-entities. Everything else in a
# raw response must be grounded in the passage, target, or expectation contract.
_RAW_STYLE_TOKENS = frozenset({"night", "quiet", "shadow", "shadows", "sky"})
_SEMANTIC_EXCLUDED_KEYS = frozenset(
    {
        "negative_prompt",
        "source_text",
        "passage",
        "context_text",
        "storage_uri",
        "local_uri",
        "checksum_sha256",
        "model_revision",
    }
)


@dataclass(frozen=True, slots=True)
class SurfaceContent:
    """Normalized view of one model-output surface."""

    surface: SurfaceName
    schema_valid: bool
    semantic_text: str
    privacy_text: str
    slots: Mapping[str, str]
    scene_facts: SceneFactsV2 | None = None


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    kind: str
    label: str
    required: bool
    passed: bool
    slot: str | None
    matched_alternative: str | None


@dataclass(frozen=True, slots=True)
class PrivacyResult:
    passed: bool
    pii_leaks: tuple[str, ...]
    privacy_term_leaks: tuple[str, ...]
    source_echo: bool
    injection_leak: bool


@dataclass(frozen=True, slots=True)
class SurfaceEvaluation:
    record_id: str
    split: str
    categories: tuple[str, ...]
    pair_id: str | None
    pair_variant: str | None
    surface: SurfaceName
    schema_valid: bool
    expectation_results: tuple[ExpectationResult, ...]
    required_atoms: int
    passed_atoms: int
    semantic_atom_recall: float
    exact_example_pass: bool
    privacy: PrivacyResult
    forbidden_hits: tuple[str, ...]
    unsupported_concepts: tuple[str, ...]
    mentioned_known_concepts: int
    semantic_digest: str
    evaluator_revision: str = FIDELITY_EVALUATOR_REVISION


@dataclass(frozen=True, slots=True)
class RecordEvaluation:
    record_id: str
    raw: SurfaceEvaluation
    postprocessed: SurfaceEvaluation | None
    renderer: SurfaceEvaluation | None


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        dumped = cast(Callable[[], object], dumper)()
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, object], dumped)
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, Mapping):
        return cast(Mapping[str, object], fields)
    raise TypeError(f"expected a mapping or model-like object, got {type(value).__name__}")


def _optional_mapping(value: object | None) -> Mapping[str, object]:
    return {} if value is None else _as_mapping(value)


def _string(value: object | None, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _string_sequence(value: object | None) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).casefold()
    tokens = [_COUNT_WORDS.get(token, token) for token in _WORD.findall(folded)]
    return " ".join(tokens)


def _semantic_normalize(value: str) -> str:
    return " ".join(token for token in _normalize(value).split() if token not in _ARTICLES)


def _identity_phrase(value: str) -> tuple[str, ...]:
    return tuple(_stem(token) for token in _semantic_normalize(value).split())


def _stem(token: str) -> str:
    return semantic_lemma(token)


def _token_matches(actual: str, expected: str) -> bool:
    actual_stem = _stem(actual)
    expected_stem = _stem(expected)
    return (
        actual_stem == expected_stem
        or f"{actual_stem}e" == expected_stem
        or f"{expected_stem}e" == actual_stem
    )


def _matching_span(
    text: str, phrase: str, *, ignore_articles: bool = True
) -> tuple[int, int] | None:
    normalizer = _semantic_normalize if ignore_articles else _normalize
    text_tokens = normalizer(text).split()
    phrase_tokens = normalizer(phrase).split()
    if not phrase_tokens:
        return None
    width = len(phrase_tokens)
    for index in range(len(text_tokens) - width + 1):
        if all(
            _token_matches(actual, expected)
            for actual, expected in zip(
                text_tokens[index : index + width], phrase_tokens, strict=True
            )
        ):
            return index, index + width
    return None


def _contains(text: str, phrase: str) -> bool:
    return _matching_span(text, phrase) is not None


def _contains_unnegated(text: str, phrase: str) -> bool:
    normalized = _normalize(text).split()
    phrase_tokens = _normalize(phrase).split()
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    for index in range(len(normalized) - width + 1):
        window = normalized[index : index + width]
        if not all(
            _token_matches(actual, expected)
            for actual, expected in zip(window, phrase_tokens, strict=True)
        ):
            continue
        prefix = normalized[max(0, index - 3) : index]
        suffix = normalized[index + width : index + width + 2]
        negated = (
            bool(set(prefix) & _NEGATION_PREFIXES)
            or tuple(prefix[-2:]) in _NEGATION_PAIRS
            or (bool(suffix) and suffix[0] in _NEGATION_SUFFIXES)
        )
        if not negated:
            return True
    return False


def _contains_in_order(text: str, phrases: Iterable[str]) -> bool:
    cursor = 0
    normalized = _semantic_normalize(text)
    tokens = normalized.split()
    for phrase in phrases:
        phrase_tokens = _semantic_normalize(phrase).split()
        if not phrase_tokens:
            continue
        width = len(phrase_tokens)
        found = False
        for index in range(cursor, len(tokens) - width + 1):
            if all(
                _token_matches(actual, expected)
                for actual, expected in zip(
                    tokens[index : index + width], phrase_tokens, strict=True
                )
            ):
                cursor = index + width
                found = True
                break
        if not found:
            return False
    return True


def _collect_strings(value: object, *, semantic: bool, key: str = "") -> list[str]:
    if semantic and key.casefold() in _SEMANTIC_EXCLUDED_KEYS:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for child_key, child in cast(Mapping[object, object], value).items():
            result.extend(_collect_strings(child, semantic=semantic, key=str(child_key)))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for child in value:
            result.extend(_collect_strings(child, semantic=semantic, key=key))
        return result
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        dumped = cast(Callable[[], object], dumper)()
        return _collect_strings(dumped, semantic=semantic, key=key)
    return []


def _parse_raw_slots(output: object) -> tuple[Mapping[str, str], bool]:
    if isinstance(output, str):
        slots: dict[str, str] = {}
        malformed = False
        for line in output.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*(SETTING|ACTOR|ACTION|MAGIC)\s*:\s*(.+?)\s*", line)
            if match is None or match.group(1) in slots:
                malformed = True
                continue
            slots[match.group(1)] = match.group(2)
        return slots, not malformed and tuple(slots) == _SLOT_NAMES

    mapping = _as_mapping(output)
    slots = {}
    for slot in _SLOT_NAMES:
        value = mapping.get(slot, mapping.get(slot.casefold()))
        if isinstance(value, str) and value.strip():
            slots[slot] = value.strip()
    return slots, tuple(slots) == _SLOT_NAMES


def _nested(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    return _optional_mapping(value) if value is not None else {}


def _structured_slots(output: object) -> Mapping[str, str]:
    mapping = _as_mapping(output)
    focus = _nested(mapping, "focus")
    magic = _nested(mapping, "magic")
    accent = _nested(mapping, "accent")
    scene_spec = _nested(mapping, "scene_spec")

    setting = _string(mapping.get("setting")) or _string(mapping.get("background_prompt"))
    actor = (
        _string(mapping.get("actor"))
        or _string(focus.get("subject"))
        or _string(mapping.get("focus_label"))
    )
    action = (
        _string(mapping.get("action"))
        or _string(focus.get("action"))
        or _string(focus.get("prompt"))
    )
    magic_text = (
        _string(mapping.get("magic"))
        or _string(magic.get("prompt"))
        or _string(accent.get("prompt"))
    )
    if not setting:
        setting = _string(mapping.get("master_prompt")) or _string(scene_spec.get("master_prompt"))
    return {
        key: value
        for key, value in zip(_SLOT_NAMES, (setting, actor, action, magic_text), strict=True)
        if value
    }


def _scene_facts_candidate(output: object) -> tuple[SceneFactsV2 | None, bool]:
    if isinstance(output, SceneFactsV2):
        try:
            return SceneFactsV2.model_validate(output.model_dump()), True
        except (TypeError, ValueError):
            return None, True
    if not isinstance(output, Mapping):
        dumper = getattr(output, "model_dump", None)
        if not callable(dumper):
            return None, False
        output = dumper()
    if not isinstance(output, Mapping):
        return None, False

    candidate: object = output
    recognized = False
    for key in ("semantic_facts", "scene_facts"):
        if key in output:
            candidate = output[key]
            recognized = True
            break
    if not recognized:
        recognized = (
            output.get("version") == "2.0"
            and "setting" in output
            and ("subjects" in output or "objects" in output)
        )
    if not recognized:
        return None, False
    try:
        return SceneFactsV2.model_validate(candidate), True
    except (TypeError, ValueError):
        return None, True


def _descriptor_phrases(facts: SceneFactsV2) -> set[tuple[str, ...]]:
    values = [*facts.setting.attributes]
    for entity in (*facts.subjects, *facts.objects):
        values.extend(entity.attributes)
        if entity.color is not None:
            values.append(entity.color)
        if isinstance(entity, SceneObjectFact):
            values.extend(entity.states)
    if facts.transformation is not None:
        values.extend(facts.transformation.result_attributes)
        if facts.transformation.result_color is not None:
            values.append(facts.transformation.result_color)
    return {
        phrase
        for value in values
        if (phrase := _identity_phrase(value))
        and not any(token.isdigit() or token in _NEGATION_PREFIXES for token in phrase)
    }


def _canonical_descriptor(
    label: str, modifiers: Sequence[str], vocabulary: set[tuple[str, ...]]
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    core = _identity_phrase(label)
    descriptors = [_identity_phrase(value) for value in modifiers if value]
    for _ in range(len(core)):
        prefix = next(
            (
                phrase
                for phrase in sorted(vocabulary, key=lambda item: (-len(item), item))
                if len(phrase) < len(core) and core[: len(phrase)] == phrase
            ),
            None,
        )
        if prefix is None:
            break
        descriptors.append(prefix)
        core = core[len(prefix) :]
    return core, tuple(sorted(descriptors))


def _scene_graph_signature(
    facts: SceneFactsV2, *, descriptor_vocabulary: set[tuple[str, ...]] | None = None
) -> tuple[object, ...]:
    entities = _scene_entities(facts)
    vocabulary = (
        descriptor_vocabulary if descriptor_vocabulary is not None else _descriptor_phrases(facts)
    )
    identities = {
        ref: (
            "subject" if isinstance(entity, SceneSubjectFact) else "object",
            entity.count if entity.count is not None else 0,
            _canonical_descriptor(
                entity.label,
                (
                    entity.color or "",
                    *entity.attributes,
                    *(entity.states if isinstance(entity, SceneObjectFact) else ()),
                ),
                vocabulary,
            ),
        )
        for ref, entity in entities.items()
    }
    if len(set(identities.values())) != len(identities):
        raise ValueError("canonical scene identities must remain distinct")

    def entity_ref(ref: str | None) -> tuple[object, ...]:
        return identities[ref] if ref is not None else ()

    event_signatures = {
        event.ref: (
            entity_ref(event.source),
            _identity_phrase(event.action),
            entity_ref(event.object),
        )
        for event in facts.events
    }
    return (
        _canonical_descriptor(facts.setting.label, facts.setting.attributes, vocabulary),
        tuple(
            sorted(
                (
                    identities[subject.ref],
                    tuple(sorted(map(_identity_phrase, subject.actions))),
                )
                for subject in facts.subjects
            )
        ),
        tuple(sorted((identities[item.ref],) for item in facts.objects)),
        tuple(
            sorted(
                (
                    entity_ref(relation.source),
                    relation.relation.value,
                    entity_ref(relation.target),
                    entity_ref(relation.secondary_target),
                )
                for relation in facts.relationships
            )
        ),
        tuple(
            sorted(
                (
                    entity_ref(motion.source),
                    motion.direction.value if motion.direction is not None else "",
                    entity_ref(motion.destination),
                )
                for motion in facts.motions
            )
        ),
        tuple(
            sorted(
                (entity_ref(salience.source), salience.layer.value) for salience in facts.salience
            )
        ),
        tuple(sorted(event_signatures.values())),
        tuple(
            sorted(
                (event_signatures[order.before], event_signatures[order.after])
                for order in facts.temporal_order
            )
        ),
        tuple(
            sorted(
                (
                    negative.kind.value,
                    entity_ref(negative.target),
                    _identity_phrase(negative.value),
                )
                for negative in facts.negatives
            )
        ),
        (
            (
                entity_ref(facts.transformation.source),
                facts.transformation.result_count,
                _canonical_descriptor(
                    facts.transformation.result_label,
                    (
                        facts.transformation.result_color or "",
                        *facts.transformation.result_attributes,
                    ),
                    vocabulary,
                ),
            )
            if facts.transformation is not None
            else None
        ),
    )


def _public_graph_contract_matches(
    record: Mapping[str, object],
    facts: SceneFactsV2,
) -> bool:
    if record.get("split") not in {DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT}:
        return True
    try:
        typed_record = FidelityRecord.model_validate(record)
        target = derive_fidelity_graph_target(typed_record)
    except (TypeError, ValueError):
        return True
    if target.facts is None:
        return True
    vocabulary = _descriptor_phrases(facts) | _descriptor_phrases(target.facts)
    try:
        return _scene_graph_signature(
            facts, descriptor_vocabulary=vocabulary
        ) == _scene_graph_signature(target.facts, descriptor_vocabulary=vocabulary)
    except ValueError:
        return False


def _scene_entity_text(entity: SceneEntity) -> str:
    states = entity.states if isinstance(entity, SceneObjectFact) else ()
    explicit_count = entity.count if entity.count != 1 else None
    return " ".join(
        str(value)
        for value in (explicit_count, entity.color, *states, *entity.attributes, entity.label)
        if value is not None
    )


def _transformation_result_text(facts: SceneFactsV2) -> str:
    transformation = facts.transformation
    if transformation is None:
        return ""
    return " ".join(
        str(value)
        for value in (
            transformation.result_count,
            transformation.result_color,
            *transformation.result_attributes,
            transformation.result_label,
        )
        if value is not None
    )


def _scene_facts_slots(facts: SceneFactsV2) -> Mapping[str, str]:
    entities = {entity.ref: entity for entity in (*facts.subjects, *facts.objects)}
    setting = " ".join((*facts.setting.attributes, facts.setting.label))
    actors = " ".join(_scene_entity_text(subject) for subject in facts.subjects)
    actions = [
        f"{subject.label} {action}" for subject in facts.subjects for action in subject.actions
    ]
    actions.extend(
        " ".join(
            value
            for value in (
                entities[relation.source].label,
                relation.relation.value.replace("_", " "),
                entities[relation.target].label,
                (
                    entities[relation.secondary_target].label
                    if relation.secondary_target is not None
                    else ""
                ),
            )
            if value
        )
        for relation in facts.relationships
    )
    for motion in facts.motions:
        source = entities[motion.source]
        if motion.direction is not None:
            actions.append(f"{source.label} {motion.direction.value}")
        if motion.destination is not None:
            actions.append(f"{source.label} travels toward {entities[motion.destination].label}")
    actions.extend(
        " ".join(
            value
            for value in (
                entities[event.source].label,
                event.action,
                f"the {entities[event.object].label}" if event.object is not None else "",
            )
            if value
        )
        for event in facts.events
    )
    if facts.transformation is not None:
        actions.append(
            f"{entities[facts.transformation.source].label} becomes "
            f"{_transformation_result_text(facts)}"
        )
    direction_by_ref = {
        motion.source: motion.direction.value
        for motion in facts.motions
        if motion.direction is not None
    }
    magic = [
        " ".join(
            value
            for value in (_scene_entity_text(item), direction_by_ref.get(item.ref, ""))
            if value
        )
        for item in facts.objects
    ]
    if facts.transformation is not None:
        magic.append(_transformation_result_text(facts))
    magic.extend(f"without {negative.value}" for negative in facts.negatives)
    return {
        key: value
        for key, value in zip(
            _SLOT_NAMES,
            (setting, actors, " ".join(actions), " ".join(magic)),
            strict=True,
        )
        if value
    }


def _scene_action_paths(facts: SceneFactsV2) -> tuple[str, ...]:
    entities = _scene_entities(facts)
    subjects = {subject.ref for subject in facts.subjects}
    bindings: list[tuple[str, str, str]] = []
    for subject in facts.subjects:
        for action in subject.actions:
            tokens = _identity_phrase(action)
            objects = [
                item
                for item in facts.objects
                if (descriptor := _identity_phrase(_scene_entity_text(item)))
                and len(tokens) > len(descriptor)
                and tokens[-len(descriptor) :] == descriptor
            ]
            if len(objects) == 1:
                bindings.append((subject.ref, action, objects[0].ref))
    bindings.extend(
        (event.source, f"{event.action} {_scene_entity_text(entities[event.object])}", event.object)
        for event in facts.events
        if event.object is not None
    )
    for relation in facts.relationships:
        if relation.source in subjects and relation.relation in {
            SceneRelationKind.HOLDS,
            SceneRelationKind.CARRIES,
            SceneRelationKind.WEARS,
            SceneRelationKind.TOUCHES,
            SceneRelationKind.LOOKS_AT,
        }:
            bindings.append(
                (
                    relation.source,
                    f"{relation.relation.value.replace('_', ' ')} "
                    f"{_scene_entity_text(entities[relation.target])}",
                    relation.target,
                )
            )
    atoms = []
    for source, action, object_ref in bindings:
        for relation in facts.relationships:
            if relation.source != object_ref:
                continue
            destination = _scene_entity_text(entities[relation.target])
            if relation.secondary_target is not None:
                destination += f" and {_scene_entity_text(entities[relation.secondary_target])}"
            for predicate in _RELATION_ALIASES.get(
                relation.relation, (relation.relation.value.replace("_", " "),)
            ):
                atoms.append(
                    f"{_scene_entity_text(entities[source])} {action} {predicate} {destination}"
                )
    return tuple(atoms)


def _scene_slot_atoms(facts: SceneFactsV2, slot: str) -> tuple[str, ...]:
    entities = _scene_entities(facts)
    if slot == "SETTING":
        return (" ".join((*facts.setting.attributes, facts.setting.label)),)
    if slot == "ACTOR":
        atoms = [_scene_entity_text(subject) for subject in facts.subjects]
        subject_refs = {subject.ref for subject in facts.subjects}
        atoms.extend(
            f"{entities[salience.source].label} {salience.layer.value}"
            for salience in facts.salience
            if salience.source in subject_refs
        )
        return tuple(atoms)
    if slot == "ACTION":
        atoms = [
            *_scene_action_paths(facts),
            *(
                value
                for subject in facts.subjects
                for action in subject.actions
                for value in (action, f"{subject.label} {action}")
            ),
        ]
        atoms.extend(
            " ".join(
                value
                for value in (
                    entities[relation.source].label,
                    relation.relation.value.replace("_", " "),
                    entities[relation.target].label,
                    (
                        entities[relation.secondary_target].label
                        if relation.secondary_target is not None
                        else ""
                    ),
                )
                if value
            )
            for relation in facts.relationships
        )
        atoms.extend(
            " ".join(
                value
                for value in (
                    entities[event.source].label,
                    event.action,
                    entities[event.object].label if event.object is not None else "",
                )
                if value
            )
            for event in facts.events
        )
        atoms.extend(
            f"travels toward {entities[motion.destination].label}"
            for motion in facts.motions
            if motion.destination is not None
        )
        events = {event.ref: event for event in facts.events}
        atoms.extend(
            " ".join(
                value
                for value in (
                    "first",
                    events[order.before].action,
                    (
                        f"the {entities[events[order.before].object].label}"
                        if events[order.before].object is not None
                        else ""
                    ),
                )
                if value
            )
            for order in facts.temporal_order
        )
        return tuple(atoms)
    if slot == "MAGIC":
        atoms = [_scene_entity_text(item) for item in facts.objects]
        atoms.extend(
            f"{entities[motion.source].label} {motion.direction.value}"
            for motion in facts.motions
            if motion.direction is not None
        )
        if facts.transformation is not None:
            atoms.append(_transformation_result_text(facts))
        atoms.extend(negative.value for negative in facts.negatives)
        return tuple(atoms)
    return ()


def _scene_slot_result(
    facts: SceneFactsV2,
    *,
    slot: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    for atom in _scene_slot_atoms(facts, slot):
        matched = _first_match(atom, alternatives)
        if matched is not None:
            return True, matched
    return False, None


def extract_surface(
    output: object, *, surface: SurfaceName, source_text: str | None = None
) -> SurfaceContent:
    """Extract comparable text and slots without depending on production model classes."""

    scene_facts, recognized_scene_facts = _scene_facts_candidate(output)
    if recognized_scene_facts:
        privacy_text = " ".join(_collect_strings(output, semantic=False))
        if surface == "renderer":
            mapping = _as_mapping(output)
            prompt = mapping.get("master_prompt")
            style = mapping.get("visual_style")
            try:
                if (
                    scene_facts is None
                    or not source_text
                    or not isinstance(prompt, str)
                    or not isinstance(style, str)
                    or set(mapping)
                    not in (
                        {"master_prompt", "scene_facts", "visual_style"},
                        {"master_prompt", "semantic_facts", "visual_style"},
                    )
                    or prompt
                    != scene_facts.to_renderer_prompt(source_text=source_text, visual_style=style)
                ):
                    raise ValueError
            except (TypeError, ValueError):
                return SurfaceContent(
                    surface=surface,
                    schema_valid=False,
                    semantic_text=prompt if isinstance(prompt, str) else "",
                    privacy_text=privacy_text,
                    slots={},
                )
        if scene_facts is None:
            return SurfaceContent(
                surface=surface,
                schema_valid=False,
                semantic_text=" ".join(_collect_strings(output, semantic=True)),
                privacy_text=privacy_text,
                slots={},
            )
        slots = _scene_facts_slots(scene_facts)
        return SurfaceContent(
            surface=surface,
            schema_valid=True,
            semantic_text=" ".join(slots.values()),
            privacy_text=privacy_text,
            slots=slots,
            scene_facts=scene_facts,
        )

    if surface == "raw":
        slots, valid = _parse_raw_slots(output)
        privacy_text = (
            output
            if isinstance(output, str)
            else " ".join(_collect_strings(output, semantic=False))
        )
        return SurfaceContent(
            surface=surface,
            schema_valid=valid,
            semantic_text=" ".join(slots.values()),
            privacy_text=privacy_text,
            slots=slots,
        )

    mapping = _as_mapping(output)
    prompt = mapping.get("master_prompt")
    # A literal wire remains a lexical screen, without typed compiler proof.
    if (
        surface == "renderer"
        and set(mapping) == {"master_prompt"}
        and isinstance(prompt, str)
        and any(
            re.match(r"\s*(SETTING|ACTOR|ACTION|MAGIC)\s*:", line) for line in prompt.splitlines()
        )
    ):
        slots, valid = _parse_raw_slots(prompt)
        return SurfaceContent(
            surface=surface,
            schema_valid=valid,
            semantic_text=" ".join(slots.values()),
            privacy_text=prompt,
            slots=slots if valid else {},
        )

    slots = _structured_slots(output)
    semantic_text = " ".join(_collect_strings(output, semantic=True))
    privacy_text = " ".join(_collect_strings(output, semantic=False))
    return SurfaceContent(
        surface=surface,
        schema_valid=bool(semantic_text.strip()),
        semantic_text=semantic_text,
        privacy_text=privacy_text,
        slots=slots,
    )


def _expectation_kind(expectation: Mapping[str, object]) -> str:
    raw = _string(expectation.get("kind"), ExpectationKind.SLOT.value).casefold()
    aliases = {
        "concept": ExpectationKind.ALLOWED_CONCEPT.value,
        "forbidden": ExpectationKind.FORBIDDEN_CONCEPT.value,
        "hallucination": ExpectationKind.FORBIDDEN_CONCEPT.value,
        "agent_patient": ExpectationKind.ROLE.value,
        "temporal_order": ExpectationKind.ORDER.value,
    }
    return aliases.get(raw, raw)


def _expectation_text(content: SurfaceContent, slot: str | None) -> str:
    if slot is None:
        return content.semantic_text
    if content.surface == "raw":
        return content.slots.get(slot.upper(), "")
    return content.slots.get(slot.upper(), content.semantic_text)


def _first_match(text: str, alternatives: Sequence[str]) -> str | None:
    return next(
        (
            alternative
            for alternative in alternatives
            if _contains_unnegated(_semantic_normalize(text), _semantic_normalize(alternative))
        ),
        None,
    )


_RELATION_ALIASES: Mapping[SceneRelationKind, tuple[str, ...]] = {
    SceneRelationKind.ABOVE: ("above", "over"),
    SceneRelationKind.BELOW: ("below", "beneath", "under"),
    SceneRelationKind.BESIDE: ("beside", "alongside", "next to"),
    SceneRelationKind.CONTAINS: ("contains", "has inside"),
    SceneRelationKind.INSIDE: ("inside", "within"),
    SceneRelationKind.NEXT_TO: ("next to", "beside", "alongside"),
    SceneRelationKind.UNDER: ("under", "beneath", "below"),
}


def _scene_entities(facts: SceneFactsV2) -> Mapping[str, SceneEntity]:
    return {entity.ref: entity for entity in (*facts.subjects, *facts.objects)}


def _scene_entity_matches(entity: SceneEntity, phrase: str) -> bool:
    return bool(phrase) and _contains(_scene_entity_text(entity), phrase)


def _scene_relation_matches(relation: SceneRelationKind, predicate: str) -> bool:
    if not predicate:
        return False
    candidates = _RELATION_ALIASES.get(relation, (relation.value.replace("_", " "),))
    return any(
        _contains(candidate, predicate) or _contains(predicate, candidate)
        for candidate in candidates
    )


def _scene_relationship_result(
    facts: SceneFactsV2,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    entities = _scene_entities(facts)
    for relationship in facts.relationships:
        source = entities[relationship.source]
        target = entities[relationship.target]
        secondary = (
            entities[relationship.secondary_target]
            if relationship.secondary_target is not None
            else None
        )
        edge_text = " ".join(
            value
            for value in (
                source.label,
                relationship.relation.value.replace("_", " "),
                target.label,
                secondary.label if secondary is not None else "",
            )
            if value
        )
        matched = _first_match(edge_text, alternatives)
        targets_match = _scene_entity_matches(target, object_value) or (
            secondary is not None and _scene_entity_matches(secondary, object_value)
        )
        if subject and predicate and object_value:
            if (
                _scene_entity_matches(source, subject)
                and _scene_relation_matches(relationship.relation, predicate)
                and targets_match
            ):
                return True, matched
        elif matched is not None:
            return True, matched
    for motion in facts.motions:
        source = entities[motion.source]
        if not _scene_entity_matches(source, subject):
            continue
        if motion.destination is not None:
            destination = entities[motion.destination]
            if _contains("toward", predicate) and _scene_entity_matches(destination, object_value):
                return True, _first_match(f"toward {destination.label}", alternatives)
        if (
            motion.direction is not None
            and _contains("moves", predicate)
            and _contains(motion.direction.value, object_value)
        ):
            return True, _first_match(motion.direction.value, alternatives)

    for first_edge in facts.relationships:
        if first_edge.relation not in {SceneRelationKind.CARRIES, SceneRelationKind.HOLDS}:
            continue
        if not _scene_entity_matches(entities[first_edge.source], subject):
            continue
        for second_edge in facts.relationships:
            if first_edge.target != second_edge.source:
                continue
            if not (
                _scene_relation_matches(second_edge.relation, predicate)
                and _scene_entity_matches(entities[second_edge.target], object_value)
            ):
                continue
            path_text = " ".join(
                (
                    entities[first_edge.source].label,
                    first_edge.relation.value,
                    entities[first_edge.target].label,
                    second_edge.relation.value.replace("_", " "),
                    entities[second_edge.target].label,
                )
            )
            return True, _first_match(path_text, alternatives)
    return False, None


def _scene_role_result(
    facts: SceneFactsV2,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    relationship_passed, matched = _scene_relationship_result(
        facts,
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        alternatives=alternatives,
    )
    if relationship_passed:
        return True, matched

    for actor in facts.subjects:
        if not _scene_entity_matches(actor, subject):
            continue
        for action in actor.actions:
            action_text = f"{actor.label} {action}"
            matched = _first_match(action_text, alternatives)
            if matched is not None or (
                _contains(action, predicate) and _contains(action, object_value)
            ):
                return True, matched

    transformation = facts.transformation
    if transformation is not None:
        entities = _scene_entities(facts)
        source = entities[transformation.source]
        transformation_text = f"{source.label} changes into {_transformation_result_text(facts)}"
        matched = _first_match(transformation_text, alternatives)
        if _scene_entity_matches(source, subject) and (
            (
                _contains("changes into", predicate)
                and _contains(_transformation_result_text(facts), object_value)
            )
            or matched is not None
        ):
            return True, matched
    return False, None


def _scene_attribute_result(
    facts: SceneFactsV2,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    if _contains("salience", predicate):
        entities = _scene_entities(facts)
        for salience in facts.salience:
            source = entities[salience.source]
            if _scene_entity_matches(source, subject) and _contains(
                salience.layer.value, object_value
            ):
                return True, _first_match(source.label, alternatives)
        return False, None

    candidates = tuple(alternatives) or ((object_value,) if object_value else ())
    predicate_tokens = set(_normalize(predicate).split())
    for entity in _scene_entities(facts).values():
        if not _scene_entity_matches(entity, subject):
            continue
        states = entity.states if isinstance(entity, SceneObjectFact) else ()
        if predicate_tokens & {"color", "colour", "hue"}:
            typed_values = (entity.color,) if entity.color is not None else ()
        elif predicate_tokens & {"state", "status", "open", "closed"}:
            typed_values = states
        elif predicate_tokens & {"material", "made"}:
            typed_values = entity.attributes
        else:
            typed_values = tuple(
                value for value in (entity.color, *states, *entity.attributes) if value is not None
            )
        descriptor = " ".join(str(value) for value in (*typed_values, entity.label))
        matched = _first_match(descriptor, candidates)
        if matched is not None:
            return True, matched if matched in alternatives else None

    transformation = facts.transformation
    if transformation is not None and _contains(transformation.result_label, subject):
        descriptor = " ".join(
            value
            for value in (
                transformation.result_color,
                *transformation.result_attributes,
                transformation.result_label,
            )
            if value is not None
        )
        matched = _first_match(descriptor, candidates)
        if matched is not None:
            return True, matched if matched in alternatives else None
    return False, None


def _scene_transformation_result(
    facts: SceneFactsV2,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    transformation = facts.transformation
    if transformation is None:
        return False, None
    source = _scene_entities(facts)[transformation.source]
    if not (
        _scene_entity_matches(source, subject)
        and _contains(_transformation_result_text(facts), object_value)
    ):
        return False, None
    edge_text = f"{source.label} changes into {_transformation_result_text(facts)}"
    matched = _first_match(edge_text, alternatives)
    change_predicates = ("become", "turn into", "transform into", "change into")
    predicate_matches = not predicate or any(
        _contains(candidate, predicate) or _contains(predicate, candidate)
        for candidate in change_predicates
    )
    return predicate_matches, matched


def _scene_count_result(
    facts: SceneFactsV2,
    *,
    counted_object: str,
    count: object,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    if not isinstance(count, int | str) or not str(count).isdigit():
        return False, None
    expected_count = int(count)
    for entity in _scene_entities(facts).values():
        if not _scene_entity_matches(entity, counted_object):
            continue
        if entity.count == expected_count:
            descriptor = _scene_entity_text(entity)
            return True, _first_match(descriptor, alternatives)
    transformation = facts.transformation
    if transformation is not None and transformation.result_count == expected_count:
        descriptor = _transformation_result_text(facts)
        if _contains(descriptor, counted_object):
            return True, _first_match(descriptor, alternatives)
    return False, None


def _scene_order_result(
    facts: SceneFactsV2,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
) -> tuple[bool, str | None]:
    if not _contains("before", predicate):
        return False, None
    entities = _scene_entities(facts)
    events = {event.ref: event for event in facts.events}
    for order in facts.temporal_order:
        before = events[order.before]
        after = events[order.after]
        if not _scene_entity_matches(entities[before.source], subject):
            continue
        after_object = entities[after.object] if after.object is not None else None
        if after_object is None or not _scene_entity_matches(after_object, object_value):
            continue
        before_text = " ".join(
            value
            for value in (
                "first",
                before.action,
                f"the {entities[before.object].label}" if before.object is not None else "",
            )
            if value
        )
        matched = _first_match(before_text, alternatives)
        if matched is not None:
            return True, matched
    return False, None


def _permitted_relation_objects(
    object_value: str, predicates: set[str], alternatives: Sequence[str]
) -> set[str]:
    objects = {object_value}
    expected_numbers = tuple(
        token for token in _semantic_normalize(object_value).split() if token.isdigit()
    )
    for alternative in alternatives:
        tokens = _semantic_normalize(alternative).split()
        for relation in predicates:
            width = len(_identity_phrase(relation))
            if (
                _identity_phrase(" ".join(tokens[:width])) == _identity_phrase(relation)
                and 0 < len(tokens[width:]) <= 8
                and not set(tokens[width:]) & (_NEGATION_PREFIXES | {"and", "or", "but", "while"})
                and tuple(token for token in tokens[width:] if token.isdigit()) == expected_numbers
            ):
                objects.add(" ".join(tokens[width:]))
    return objects


def _holding_relation_path(
    action: str, predicates: set[str], objects: set[str], noun_phrases: Sequence[str]
) -> bool:
    if re.search(r"[;,.!?:\n]", action):
        return False
    tokens = _identity_phrase(action)
    if not tokens or tokens[0] not in {_stem("holds"), _stem("carries")}:
        return False
    permitted_nouns = {_identity_phrase(phrase) for phrase in noun_phrases}
    for predicate in predicates:
        for target in objects:
            tail = _identity_phrase(f"{predicate} {target}")
            if not tail or len(tokens) <= len(tail) + 1 or tokens[-len(tail) :] != tail:
                continue
            noun = tokens[1 : -len(tail)]
            if (
                0 < len(noun) <= 8
                and noun in permitted_nouns
                and not _has_action_binding_boundary(noun)
                and not set(noun) & {"is", "are", "be", "do", "does", "did", "has", "have"}
            ):
                return True
    return False


def _lexical_relation_result(
    content: SurfaceContent,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    alternatives: Sequence[str],
    slot: str | None = None,
    noun_phrases: Sequence[str] = (),
) -> tuple[bool, str | None]:
    if not all((subject, predicate, object_value)):
        return False, None
    selected_slot = (slot or "ACTION").upper()
    texts = (
        [content.slots.get(selected_slot, "")]
        if content.surface == "raw" or tuple(content.slots) == _SLOT_NAMES
        else [content.semantic_text]
    )
    actor, action = content.slots.get("ACTOR", ""), content.slots.get("ACTION", "")
    if (
        selected_slot == "ACTION"
        and actor
        and action
        and _identity_phrase(actor) == _identity_phrase(subject)
    ):
        texts.append(f"{actor} {action}")
        if content.surface == "raw" or tuple(content.slots) == _SLOT_NAMES:
            matched = next(
                (
                    alternative
                    for alternative in alternatives
                    if _identity_phrase(action) == _identity_phrase(alternative)
                ),
                None,
            )
            if matched is not None:
                return True, matched
    predicates = {predicate}
    for aliases in _RELATION_ALIASES.values():
        if any(_identity_phrase(alias) == _identity_phrase(predicate) for alias in aliases):
            predicates.update(aliases)
    destination = _identity_phrase(predicate) in {
        _identity_phrase("toward"),
        _identity_phrase("to"),
    }
    if destination:
        predicates.update(("toward", "to"))
    objects = _permitted_relation_objects(object_value, predicates, alternatives)
    if (
        (content.surface == "raw" or tuple(content.slots) == _SLOT_NAMES)
        and selected_slot == "ACTION"
        and _identity_phrase(actor) == _identity_phrase(subject)
        and _holding_relation_path(action, predicates, objects, noun_phrases)
    ):
        return True, _first_match(action, alternatives)
    movement_verbs = ("travels", "moves", "walks", "runs", "flies", "swims", "goes")
    directional_forms: tuple[str, ...] = ()
    if _identity_phrase(predicate) == _identity_phrase("moves"):
        for direction, aliases, verbs in (
            (
                "upward",
                ("upward", "up", "rises", "rise", "ascends", "ascend"),
                ("rises", "ascends"),
            ),
            (
                "downward",
                ("downward", "down", "falls", "fall", "descends", "descend"),
                ("falls", "descends"),
            ),
        ):
            if _identity_phrase(object_value) in {_identity_phrase(alias) for alias in aliases}:
                directional_forms = (*verbs, f"moves {direction}")
    phrases = [_semantic_normalize(f"{subject} {motion}") for motion in directional_forms]
    phrases.extend(
        _semantic_normalize(f"{subject} {linking} {relation} {target}")
        for relation in predicates
        for linking in ("", "is", *(movement_verbs if destination else ()))
        for target in objects
    )
    for text in texts:
        for clause in re.split(r"[;.!?\n]", text):
            normalized = _semantic_normalize(clause)
            for phrase in phrases:
                # A relation must occupy its clause; represented or reported
                # subjects cannot acquire an embedded action by substring.
                permitted = [phrase]
                if (
                    selected_slot == "ACTION"
                    and text == action
                    and (content.surface == "raw" or tuple(content.slots) == _SLOT_NAMES)
                ):
                    permitted.extend(
                        f"{verb} {phrase}" for verb in ("watches", "looks at", "holds", "carries")
                    )
                if any(
                    _identity_phrase(normalized) == _identity_phrase(value) for value in permitted
                ):
                    return True, _first_match(clause, alternatives)
    return False, None


def _evaluate_expectation(
    expectation: Mapping[str, object], content: SurfaceContent, *, noun_phrases: Sequence[str] = ()
) -> ExpectationResult:
    kind = _expectation_kind(expectation)
    label = _string(expectation.get("label"), kind)
    required = expectation.get("required", True) is not False
    slot_value = _string(expectation.get("slot")) or None
    alternatives = _string_sequence(expectation.get("alternatives"))
    subject = _string(expectation.get("subject"))
    predicate = _string(expectation.get("predicate"))
    object_value = _string(expectation.get("object"))
    count = expectation.get("count")
    count_value = str(count) if isinstance(count, int | str) else ""
    text = _expectation_text(content, slot_value)
    matched: str | None = None

    if kind == ExpectationKind.SLOT and content.scene_facts is not None:
        passed, matched = _scene_slot_result(
            content.scene_facts,
            slot=(slot_value or "").upper(),
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.FORBIDDEN_CONCEPT:
        matched = _first_match(text, alternatives)
        candidates = alternatives or tuple(
            value for value in (subject, predicate, object_value) if value
        )
        passed = not any(_contains_unnegated(content.privacy_text, value) for value in candidates)
    elif kind == ExpectationKind.RELATION and content.scene_facts is not None:
        passed, matched = _scene_relationship_result(
            content.scene_facts,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.RELATION:
        passed, matched = _lexical_relation_result(
            content,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
            slot=slot_value,
            noun_phrases=noun_phrases,
        )
    elif kind == ExpectationKind.ATTRIBUTE and content.scene_facts is not None:
        passed, matched = _scene_attribute_result(
            content.scene_facts,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.ATTRIBUTE:
        matched = _first_match(text, alternatives)
        passed = (
            bool(subject)
            and _contains(text, subject)
            and (matched is not None or (bool(object_value) and _contains(text, object_value)))
        )
    elif kind == ExpectationKind.COUNT and content.scene_facts is not None:
        passed, matched = _scene_count_result(
            content.scene_facts,
            counted_object=object_value or subject,
            count=count,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.COUNT:
        matched = _first_match(text, alternatives)
        counted_object = object_value or subject
        passed = (
            bool(count_value and counted_object)
            and _contains(text, count_value)
            and _contains(text, counted_object)
        )
        passed = passed or matched is not None
    elif kind == ExpectationKind.ROLE and content.scene_facts is not None:
        passed, matched = _scene_role_result(
            content.scene_facts,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.ROLE:
        matched = _first_match(text, alternatives)
        actor_text = content.slots.get("ACTOR", content.semantic_text)
        action_text = content.slots.get("ACTION", content.semantic_text)
        passed = bool(subject) and _contains(actor_text, subject)
        action_passed = matched is not None
        if predicate and object_value:
            action_passed = action_passed or (
                _contains(action_text, predicate) and _contains(action_text, object_value)
            )
        passed = passed and action_passed
    elif kind == ExpectationKind.TRANSFORMATION and content.scene_facts is not None:
        passed, matched = _scene_transformation_result(
            content.scene_facts,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.TRANSFORMATION:
        matched = _first_match(text, alternatives)
        transformation_text = (
            " ".join((content.slots.get("ACTION", ""), content.slots.get("MAGIC", ""))).strip()
            or content.semantic_text
        )
        endpoints = tuple(value for value in (subject, object_value) if value)
        endpoints_passed = len(endpoints) == 2 and all(
            _contains(transformation_text, value) for value in endpoints
        )
        relation_passed = matched is not None
        if predicate:
            relation_passed = relation_passed or _contains(transformation_text, predicate)
        passed = endpoints_passed and relation_passed
    elif kind == ExpectationKind.ORDER and content.scene_facts is not None:
        passed, matched = _scene_order_result(
            content.scene_facts,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            alternatives=alternatives,
        )
    elif kind == ExpectationKind.ORDER:
        matched = _first_match(text, alternatives)
        parts = tuple(value for value in (subject, predicate, object_value) if value)
        passed = bool(parts) and _contains_in_order(text, parts)
        passed = passed or matched is not None
    else:
        matched = _first_match(text, alternatives)
        passed = matched is not None
        if not alternatives:
            parts = tuple(value for value in (subject, predicate, object_value) if value)
            passed = bool(parts) and all(_contains(text, value) for value in parts)

    return ExpectationResult(
        kind=kind,
        label=label,
        required=required,
        passed=passed,
        slot=slot_value.upper() if slot_value else None,
        matched_alternative=matched,
    )


def _source_echo(passage: str, output: str) -> bool:
    passage_tokens = _normalize(passage).split()
    output_normalized = _normalize(output)
    normalized_passage = " ".join(passage_tokens)
    if len(passage_tokens) >= 8 and normalized_passage in output_normalized:
        return True
    window = 12
    if len(passage_tokens) < window:
        return False
    return any(
        " ".join(passage_tokens[index : index + window]) in output_normalized
        for index in range(len(passage_tokens) - window + 1)
    )


def _privacy_result(record: Mapping[str, object], output: str) -> PrivacyResult:
    privacy_terms = _string_sequence(record.get("privacy_terms"))
    pii_leaks: list[str] = []
    if _EMAIL.search(output):
        pii_leaks.append("email")
    if _PHONE.search(output):
        pii_leaks.append("phone")
    if _URL.search(output):
        pii_leaks.append("url")
    term_leaks = tuple(
        term
        for term in privacy_terms
        if _matching_span(output, term, ignore_articles=False) is not None
    )
    passage = _string(record.get("passage"))
    source_echo = bool(passage) and _source_echo(passage, output)
    injection_leak = bool(_INJECTION.search(output))
    passed = not pii_leaks and not term_leaks and not source_echo and not injection_leak
    return PrivacyResult(
        passed=passed,
        pii_leaks=tuple(pii_leaks),
        privacy_term_leaks=term_leaks,
        source_echo=source_echo,
        injection_leak=injection_leak,
    )


def _known_concept_metrics(
    text: str,
    *,
    allowed: Sequence[str],
    vocabulary: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    allowed_normalized = {_normalize(concept) for concept in allowed}
    mentions = {
        _normalize(concept)
        for concept in vocabulary
        if concept and _contains_unnegated(text, concept)
    }
    unsupported = tuple(
        sorted(
            mention
            for mention in mentions
            if not any(
                _contains(allowed_concept, mention) or _contains(mention, allowed_concept)
                for allowed_concept in allowed_normalized
            )
        )
    )
    return unsupported, len(mentions)


def _raw_novel_tokens(record: Mapping[str, object], content: SurfaceContent) -> tuple[str, ...]:
    authorized_text: list[str] = [_string(record.get("passage"))]
    authorized_text.extend(_string_sequence(record.get("allowed_concepts")))
    target = record.get("target")
    if target is not None:
        authorized_text.extend(_collect_strings(target, semantic=True))
    raw_expectations = record.get("expectations")
    if isinstance(raw_expectations, Sequence) and not isinstance(raw_expectations, (str, bytes)):
        for raw_expectation in raw_expectations:
            expectation = _as_mapping(raw_expectation)
            if _expectation_kind(expectation) == ExpectationKind.FORBIDDEN_CONCEPT:
                continue
            authorized_text.extend(_string_sequence(expectation.get("alternatives")))
            authorized_text.extend(
                _string(expectation.get(name)) for name in ("subject", "predicate", "object")
            )

    authorized_stems = {
        _stem(token) for text in authorized_text for token in _normalize(text).split() if token
    }
    authorized_stems.update(_stem(token) for token in _RAW_GRAMMAR_TOKENS)
    authorized_stems.update(_stem(token) for token in _RAW_STYLE_TOKENS)
    return tuple(
        sorted(
            {
                f"novel:{token}"
                for token in _normalize(content.semantic_text).split()
                if _stem(token) not in authorized_stems
            }
        )
    )


def evaluate_surface(
    record: object,
    output: object,
    *,
    surface: SurfaceName,
    concept_vocabulary: Sequence[str] = (),
) -> SurfaceEvaluation:
    """Evaluate one record at one surface without retaining its private text."""

    record_map = _as_mapping(record)
    content = extract_surface(
        output, surface=surface, source_text=_string(record_map.get("passage"))
    )
    raw_expectations = record_map.get("expectations", ())
    expectations = (
        cast(Sequence[object], raw_expectations)
        if isinstance(raw_expectations, Sequence) and not isinstance(raw_expectations, (str, bytes))
        else ()
    )
    results = tuple(
        _evaluate_expectation(
            _as_mapping(expectation),
            content,
            noun_phrases=_string_sequence(record_map.get("allowed_concepts")),
        )
        for expectation in expectations
    )
    required = tuple(result for result in results if result.required)
    passed_atoms = sum(result.passed for result in required)
    recall = passed_atoms / len(required) if required else 1.0

    privacy = _privacy_result(record_map, content.privacy_text)
    forbidden_terms = _string_sequence(record_map.get("forbidden_terms"))
    forbidden_hits = tuple(
        term for term in forbidden_terms if _contains_unnegated(content.semantic_text, term)
    )
    allowed_concepts = _string_sequence(record_map.get("allowed_concepts"))
    unsupported, mentioned = _known_concept_metrics(
        content.semantic_text,
        allowed=allowed_concepts,
        vocabulary=concept_vocabulary,
    )
    if surface == "raw" or content.scene_facts is not None:
        novel_tokens = _raw_novel_tokens(record_map, content)
        known_unsupported_tokens = {
            token for concept in unsupported for token in _normalize(concept).split()
        }
        novel_tokens = tuple(
            item
            for item in novel_tokens
            if item.removeprefix("novel:") not in known_unsupported_tokens
        )
        unsupported = tuple(sorted((*unsupported, *novel_tokens)))
        mentioned += len(novel_tokens)
    if content.scene_facts is not None and not _public_graph_contract_matches(
        record_map,
        content.scene_facts,
    ):
        unsupported = (*unsupported, "graph:contract-mismatch")
    exact_pass = (
        content.schema_valid
        and privacy.passed
        and not forbidden_hits
        and not unsupported
        and all(result.passed for result in required)
    )
    categories = _string_sequence(record_map.get("categories"))
    return SurfaceEvaluation(
        record_id=_string(record_map.get("record_id"), "unknown"),
        split=_string(record_map.get("split"), "unknown"),
        categories=categories,
        pair_id=_string(record_map.get("pair_id")) or None,
        pair_variant=_string(record_map.get("pair_variant")) or None,
        surface=surface,
        schema_valid=content.schema_valid,
        expectation_results=results,
        required_atoms=len(required),
        passed_atoms=passed_atoms,
        semantic_atom_recall=recall,
        exact_example_pass=exact_pass,
        privacy=privacy,
        forbidden_hits=forbidden_hits,
        unsupported_concepts=unsupported,
        mentioned_known_concepts=mentioned,
        semantic_digest=hashlib.sha256(_normalize(content.semantic_text).encode()).hexdigest(),
    )


def evaluate_record(
    record: object,
    *,
    raw_output: object,
    postprocessed_plan: object | None = None,
    renderer_contract: object | None = None,
    concept_vocabulary: Sequence[str] = (),
) -> RecordEvaluation:
    """Evaluate raw, repaired, and renderer-safe outputs as distinct evidence surfaces."""

    raw = evaluate_surface(
        record,
        raw_output,
        surface="raw",
        concept_vocabulary=concept_vocabulary,
    )
    postprocessed = (
        evaluate_surface(
            record,
            postprocessed_plan,
            surface="postprocessed",
            concept_vocabulary=concept_vocabulary,
        )
        if postprocessed_plan is not None
        else None
    )
    renderer = (
        evaluate_surface(
            record,
            renderer_contract,
            surface="renderer",
            concept_vocabulary=concept_vocabulary,
        )
        if renderer_contract is not None
        else None
    )
    return RecordEvaluation(
        record_id=raw.record_id,
        raw=raw,
        postprocessed=postprocessed,
        renderer=renderer,
    )


def concept_vocabulary(records: Iterable[object]) -> tuple[str, ...]:
    """Build a deterministic closed-world concept vocabulary for hallucination checks."""

    concepts: set[str] = set()
    for record in records:
        mapping = _as_mapping(record)
        concepts.update(_string_sequence(mapping.get("allowed_concepts")))
        concepts.update(_string_sequence(mapping.get("forbidden_terms")))
    return tuple(sorted(concepts, key=lambda value: (_normalize(value), value)))
