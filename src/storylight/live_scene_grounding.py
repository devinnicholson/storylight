"""Conservative source bindings for the small planner's visual selections.

This is a rejection gate, not a replacement planner. It recognizes bounded noun
phrases and actions using the existing scene-fact bindings. Unrecognized prose
is refused rather than treated as evidence for an invented scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from storylight.privacy_policy import COLOR_WORDS, COUNT_WORDS, VISIBLE_VERBS
from storylight.scene_facts import (
    _action_grounded,
    _asserted_units,
    _count_near_label,
    _descriptor_grounded,
    _phrase_positions,
    _source_sentences,
)
from storylight.semantic_text import normalize_semantic_phrase


class LiveSceneGroundingError(ValueError):
    """The selected visual facts could not be bound to the source."""


# These are equivalences, not a vocabulary of permitted subjects. In particular,
# dog does not license a puppy or a particular breed, nor box a likely ASR fox.
_ALIASES = {
    "leap": "jump",
    "leaps": "jumps",
    "leaped": "jumped",
    "leapt": "jumped",
    "leaping": "jumping",
    "fast": "quick",
    "swift": "quick",
    "sluggish": "lazy",
    "gray": "grey",
    "beneath": "under",
    "underneath": "under",
    "alongside": "beside",
    "canine": "dog",
    "canines": "dogs",
    "rising": "rise",
}
_MODIFIERS = COLOR_WORDS | {
    "quick",
    "lazy",
    "small",
    "little",
    "large",
    "big",
    "tiny",
    "young",
    "old",
    "striped",
    "spotted",
    "folded",
    "paper",
    "wooden",
    "stone",
    "glowing",
    "luminous",
    "open",
    "closed",
    "silver",
    "transparent",
    "enormous",
}
_VERBS = {normalize_semantic_phrase(word)[0] for word in VISIBLE_VERBS} | {
    "jump",
    "throw",
    "rest",
    "sit",
    "stand",
    "sleep",
    "become",
    "turn",
    "be",
    "appear",
    "bloom",
    "fly",
    "fall",
    "emerge",
    "transform",
    "cross",
    "chase",
    "light",
}
_NEUTRAL = {
    "background",
    "neutral background",
    "plain background",
    "neutral backdrop",
    "plain backdrop",
    "unspecified setting",
    "none",
    "no additional objects",
    "no magic",
}
_ARTICLES = {"a", "an", "the"}


def _canonical(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"\b(?:don't|doesn't|didn't)\b", "does not", text)
    return re.sub(r"\b[a-z]+\b", lambda match: _ALIASES.get(match[0], match[0]), text)


@dataclass(frozen=True)
class _Entity:
    label: str
    modifiers: tuple[str, ...]
    count: int | None


def _entity(text: str) -> _Entity:
    words = re.findall(r"[a-z]+|\d+", text)
    if words and words[0] in _ARTICLES:
        words.pop(0)
    count = None
    if words and (words[0] in COUNT_WORDS or words[0].isdigit()):
        count = int(COUNT_WORDS.get(words[0], words[0]))
        words.pop(0)
        if not 1 <= count <= 12:
            raise LiveSceneGroundingError("unsupported visual count")
    modifiers = []
    while len(words) > 1 and words[0] in _MODIFIERS:
        modifiers.append(words.pop(0))
    if not words or any(word in {"not", "no", "never", "without"} for word in words):
        raise LiveSceneGroundingError("unsupported visual entity")
    return _Entity(" ".join(words), tuple(modifiers), count)


def _parts(text: str) -> tuple[_Entity, str]:
    words = text.split()
    for index, word in enumerate(words):
        prefix = [item for item in words[:index] if item not in _ARTICLES]
        has_noun = any(item not in _MODIFIERS and item not in COUNT_WORDS for item in prefix)
        if (
            index
            and has_noun
            and normalize_semantic_phrase(word)
            and (normalize_semantic_phrase(word)[0] in _VERBS)
        ):
            return _entity(" ".join(words[:index])), " ".join(words[index:])
    return _entity(text), ""


def validate_wire_grounding(wire_plan: object, *, source_text: str) -> None:
    """Reject unsupported raw wire selections before sanitation or paid rendering.

    The object contract is the existing background_prompt/focus.subject,
    focus.action/magic.prompt wire schema; no planner or client import is needed.
    Errors contain field names only, never passage or model-output values.
    """
    graph = getattr(wire_plan, "scene_facts", None)
    if graph is not None:
        try:
            graph.validate_source_grounding(source_text=source_text)
        except ValueError as error:
            raise LiveSceneGroundingError("ungrounded scene graph") from error
        return
    source = _canonical(source_text)
    # Clause boundaries keep attributes, quantities, and negation attached to
    # their own actor rather than another occurrence of the same noun.
    clauses = tuple(
        part.strip()
        for unit in _asserted_units(source)
        for part in re.split(r"\s+(?:and|but|while)\s+|[,;]", unit)
        if part.strip()
    )
    sentences = _source_sentences(". ".join(clauses))
    focus = _entity(_canonical(wire_plan.focus.subject))
    action = _canonical(wire_plan.focus.action).strip()
    if not action or action in _NEUTRAL:
        raise LiveSceneGroundingError("missing focus action")
    selections = [("focus", focus, action)]
    rendered = [f"{_canonical(wire_plan.focus.subject)} {action}"]
    for field, text in (
        ("background", wire_plan.background_prompt),
        ("supporting", wire_plan.magic.prompt),
    ):
        text = _canonical(text).strip(" .")
        if text in _NEUTRAL:
            continue
        for phrase in re.split(r"\s+and\s+|[,;]", text):
            if not phrase.strip():
                continue
            tokens = normalize_semantic_phrase(phrase)
            if tokens and tokens[0] in {"no", "without"}:
                if tokens not in sentences:
                    raise LiveSceneGroundingError("ungrounded negative constraint")
                rendered.append(phrase.strip())
                continue
            entity, predicate = _parts(phrase.strip())
            selections.append((field, entity, predicate))
            rendered.append(phrase.strip())
    labels = tuple(entity.label for _, entity, _ in selections)
    for field, entity, predicate in selections:
        matched = False
        for sentence in sentences:
            # No positive selection may be licensed by a negated clause, even
            # when an adverb separates the negation from the verb.
            if {"no", "not", "never", "without"}.intersection(sentence):
                continue
            if not _phrase_positions(sentence, entity.label):
                continue
            local = (sentence,)
            if any(
                not _descriptor_grounded(modifier, entity.label, local, entity_labels=labels)
                for modifier in entity.modifiers
            ):
                continue
            if entity.count is not None and not _count_near_label(
                entity.count, entity.label, local, entity_labels=labels
            ):
                continue
            colors = {
                color
                for color in COLOR_WORDS
                if _descriptor_grounded(color, entity.label, local, entity_labels=labels)
            }
            if colors and not colors.issubset(entity.modifiers):
                continue
            if predicate and not _action_grounded(
                predicate, entity.label, local, entity_labels=labels
            ):
                continue
            matched = True
            break
        if not matched:
            raise LiveSceneGroundingError(f"ungrounded {field} facts")
    # For the bounded, explicit-subject clauses we can parse, check the reverse
    # binding too: the model may not drop an actor, quantity, or action target.
    output_sentences = _source_sentences(". ".join(rendered))
    for clause in clauses:
        source_tokens = normalize_semantic_phrase(clause)
        if {"no", "not", "never", "without"}.intersection(source_tokens):
            if source_tokens not in output_sentences:
                raise LiveSceneGroundingError("missing source negative constraint")
            if source_tokens[0] in {"no", "without"}:
                absent = _entity(" ".join(clause.split()[1:]))
                for sentence in output_sentences:
                    if {"no", "not", "never", "without"}.intersection(sentence):
                        continue
                    if _phrase_positions(sentence, absent.label) and all(
                        _descriptor_grounded(
                            modifier, absent.label, (sentence,), entity_labels=labels
                        )
                        for modifier in absent.modifiers
                    ):
                        raise LiveSceneGroundingError("contradictory source constraint")
            continue
        entity, predicate = _parts(clause)
        for sentence in output_sentences:
            local = (sentence,)
            if not _phrase_positions(sentence, entity.label):
                continue
            if any(
                not _descriptor_grounded(modifier, entity.label, local, entity_labels=labels)
                for modifier in entity.modifiers
            ):
                continue
            if entity.count is not None and not _count_near_label(
                entity.count, entity.label, local, entity_labels=labels
            ):
                continue
            if not predicate or _action_grounded(
                predicate, entity.label, local, entity_labels=labels
            ):
                break
        else:
            raise LiveSceneGroundingError("missing source actor, attribute, or action")


def prepare_grounded_wire(wire_plan: object, *, source_text: str) -> object:
    """Keep validated facts while separating ordinary source wording for privacy.

    The legacy sanitizer can delete colors, quantities, and relation words. This
    path only removes articles; remaining privacy conflicts are rejected by the
    planner's existing privacy gate, never repaired by deleting a visual fact.
    """
    validate_wire_grounding(wire_plan, source_text=source_text)
    if getattr(wire_plan, "scene_facts", None) is not None:
        return wire_plan

    def without_articles(value: str) -> str:
        return " ".join(re.sub(r"\b(?:a|an|the)\b", "", value, flags=re.IGNORECASE).split())

    result = wire_plan.model_copy(
        update={
            "background_prompt": without_articles(wire_plan.background_prompt),
            "focus": wire_plan.focus.model_copy(
                update={
                    "subject": without_articles(wire_plan.focus.subject),
                    "action": without_articles(wire_plan.focus.action),
                }
            ),
            "magic": wire_plan.magic.model_copy(
                update={"prompt": without_articles(wire_plan.magic.prompt)}
            ),
        }
    )
    validate_wire_grounding(result, source_text=source_text)
    return result
