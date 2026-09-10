"""Bounded local grammar for the optional four-slot scene graph adapter.

Only explicit noun phrases and predicates are supported. The slots select the
focal actor/action; source clauses supply their bound details. This is not a
general English parser, and unsupported bindings return a value-free refusal.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal

from storylight.privacy_policy import COLOR_WORDS, COUNT_WORDS, VISIBLE_VERBS
from storylight.scene_facts import (
    SceneEventFact,
    SceneFactsGroundingError,
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneMotionFact,
    SceneNegativeFact,
    SceneObjectFact,
    SceneRelationshipFact,
    SceneSalienceFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
    SceneTransformationFact,
    compile_scene_facts_prompt,
)
from storylight.semantic_text import normalize_semantic_phrase


class LiveSceneFactsRefusal(StrEnum):
    INVALID_INPUT = "invalid_input"
    INVALID_HYBRID = "invalid_hybrid"
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    UNGROUNDED = "ungrounded"
    PRIVACY = "privacy"
    INVALID_GRAPH = "invalid_graph"


@dataclass(frozen=True, slots=True)
class LiveSceneFactsResult:
    facts: SceneFactsV2 | None = None
    refusal: LiveSceneFactsRefusal | None = None

    def __post_init__(self) -> None:
        if (self.facts is None) == (self.refusal is None):
            raise ValueError("exactly one live scene result field is required")
        if self.facts is not None and not isinstance(self.facts, SceneFactsV2):
            raise TypeError("facts must be a scene graph")
        if self.refusal is not None and not isinstance(self.refusal, LiveSceneFactsRefusal):
            raise TypeError("refusal must be a live scene refusal code")


class _Refuse(Exception):
    def __init__(self, code: LiveSceneFactsRefusal) -> None:
        self.code = code
        super().__init__(code.value)


_STATES = frozenset(
    {"awake", "asleep", "closed", "open", "dark", "lit", "dry", "wet", "empty", "full"}
)
_ATTRIBUTES = frozenset(
    {
        "big",
        "bright",
        "ceramic",
        "enormous",
        "folded",
        "glowing",
        "large",
        "little",
        "luminous",
        "old",
        "round",
        "small",
        "tiny",
        "transparent",
        "wooden",
        "young",
    }
)
_COUNTS = {**{word: int(value) for word, value in COUNT_WORDS.items()}, "eleven": 11, "twelve": 12}
_RELATIONS = {
    "above": "above",
    "over": "above",
    "below": "below",
    "beneath": "under",
    "under": "under",
    "behind": "behind",
    "beside": "beside",
    "inside": "inside",
    "within": "inside",
    "outside": "outside",
    "on": "on",
    "atop": "on",
    "in front of": "in_front_of",
    "left of": "left_of",
    "right of": "right_of",
    "next to": "next_to",
    "attached to": "attached_to",
    "between": "between",
}
_OWNERSHIP = {
    "hold": "holds",
    "carry": "carries",
    "wear": "wears",
    "own": "owns",
    "contain": "contains",
    "touch": "touches",
}
_RESULT_MOTION_VERBS = frozenset({"fly", "flies"})
_RESULT_VERBS = frozenset(
    {
        *_RESULT_MOTION_VERBS,
        "appear",
        "appears",
        "appeared",
        "emerge",
        "emerges",
        "emerged",
        "form",
        "forms",
        "formed",
        "bloom",
        "blooms",
        "bloomed",
        "rise",
        "rises",
        "fall",
        "falls",
    }
)
_VERBS = frozenset(
    {
        *VISIBLE_VERBS,
        *_RESULT_VERBS,
        "raise",
        "raises",
        "raised",
        "lower",
        "lowers",
        "lowered",
        "close",
        "closes",
        "closed",
        "stop",
        "stops",
        "stopped",
        "stand",
        "stands",
        "fall",
        "falls",
        "fell",
        "rise",
        "rises",
        "rose",
        "wear",
        "wears",
        "wore",
        "own",
        "owns",
        "contain",
        "contains",
        "touch",
        "touches",
        "tap",
        "taps",
        "drop",
        "drops",
        "held",
        "carried",
        "ran",
        "swam",
        "opened",
        "lifted",
        "turned",
        "turns",
        "become",
        "becomes",
        "became",
        "transform",
        "transforms",
        "change",
        "changes",
        "is",
        "are",
        "was",
        "were",
        "belongs",
        "belong",
        "looks",
        "look",
    }
)
_PREDICATE = re.compile(r"\b(?:" + "|".join(sorted(_VERBS, key=len, reverse=True)) + r")\b")
_RELATION = re.compile(
    r"\b(?:" + "|".join(sorted(_RELATIONS, key=len, reverse=True)) + r"|toward|towards)\b"
)
_BOUNDARY = re.compile(
    r"\b(?:and|or|but|as|while|when|although|because|if|unless|after|before|then|"
    r"named|called|who|which|that)\b"
)
_SCOPED_SOURCE = re.compile(
    r"\b(?:if|unless|would|could|might|neither|either|imagine|imagines|imagined|in a dream|"
    r"allegedly|hypothetically|reportedly|supposedly)\b"
)
_NEUTRAL_ADVERBS = frozenset(
    {"carefully", "deliberately", "gently", "quietly", "slowly", "steadily"}
)


def _key(value: str) -> tuple[str, ...]:
    return normalize_semantic_phrase(value)


@dataclass(frozen=True)
class _Noun:
    label: str
    count: int | None = None
    color: str | None = None
    states: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    singular_article: bool = field(default=False, compare=False)


def _noun(value: str) -> _Noun:
    value = value.strip().strip(".")
    if not value or _BOUNDARY.search(value) or not re.fullmatch(r"[a-z0-9 -]+", value):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    words = value.split()
    singular_article = words[0] in {"a", "an"}
    if words[0] in {"the", "a", "an"}:
        words.pop(0)
    if words and words[0] == "exactly":
        words.pop(0)
    count = None
    if words and (words[0] in _COUNTS or words[0].isdigit()):
        count = _COUNTS.get(words[0], int(words[0]) if words[0].isdigit() else 0)
        words.pop(0)
        if not 1 <= count <= 12:
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    colors, states, attributes = [], [], []
    while len(words) > 1:
        word = words[0]
        if word in COLOR_WORDS:
            colors.append(word)
        elif word in _STATES:
            states.append(word)
        elif word in _ATTRIBUTES:
            attributes.append(word)
        else:
            break
        words.pop(0)
    if not words or len(words) > 5 or len(colors) > 1:
        raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
    if any(
        word in _VERBS
        or word in {"not", "no", "never", "without", "it", "he", "she", "they", "another"}
        for word in words
    ):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    return _Noun(
        " ".join(words),
        count,
        colors[0] if colors else None,
        tuple(states),
        tuple(attributes),
        singular_article=singular_article and _key(words[-1]) == (words[-1],),
    )


@dataclass(frozen=True)
class _Clause:
    subject: _Noun
    verb: str
    object: _Noun | None = None
    relation: str | None = None
    anchor: _Noun | None = None
    secondary: _Noun | None = None
    negative: bool = False
    state: str | None = None
    layer: str | None = None
    transform: bool = False
    passive: bool = field(default=False, compare=False)


def _clause(value: str) -> _Clause:
    passive = re.fullmatch(r"(.+?) (?:is|are|was|were) carried by (.+)", value)
    if passive:
        return _Clause(_noun(passive[2]), "carries", _noun(passive[1]), passive=True)
    match = _PREDICATE.search(value)
    # Hybrid clauses may use a bare relation rather than a copula.
    if match is None:
        match = _RELATION.search(value)
        if match is None:
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
        value = value[: match.start()] + "is " + value[match.start() :]
        match = _PREDICATE.search(value)
    assert match is not None
    head = value[: match.start()].strip()
    while head and head.rsplit(" ", 1)[-1] in _NEUTRAL_ADVERBS:
        head = head.rsplit(" ", 1)[0] if " " in head else ""
    negative = bool(re.search(r"\b(?:does not|do not|did not|never)$", head))
    head = re.sub(r"\s+(?:does not|do not|did not|never)$", "", head)
    subject = _noun(head)
    verb = match.group()
    tail = value[match.end() :].strip()
    if tail.startswith("not "):
        negative, tail = True, tail[4:]
    if _key(verb) == ("be",) and tail in _STATES | COLOR_WORDS:
        return _Clause(subject, verb, negative=negative, state=tail)
    if _key(verb) == ("be",) and tail in {"in the foreground", "in the background"}:
        return _Clause(subject, verb, layer=tail.rsplit(" ", 1)[1], negative=negative)
    if _key(verb) == ("become",) or (
        _key(verb) in {("turn",), ("transform",), ("change",)} and tail.startswith("into ")
    ):
        return _Clause(
            subject, verb, _noun(tail.removeprefix("into ")), negative=negative, transform=True
        )
    if _key(verb) == ("belong",) and tail.startswith("to "):
        return _Clause(_noun(tail[3:]), "owns", subject, negative=negative)
    relation_match = _RELATION.search(tail)
    if _key(verb) == ("stand",) and tail and (
        relation_match is None
        or relation_match.start() != 0
        or relation_match.group() in {"toward", "towards"}
    ):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    if relation_match:
        relation = relation_match.group()
        before = tail[: relation_match.start()].strip()
        after = tail[relation_match.end() :].strip()
        secondary = None
        if relation == "between":
            parts = after.split(" and ")
            if len(parts) != 2:
                raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
            after, second = parts
            secondary = _noun(second)
        return _Clause(
            subject,
            verb,
            _noun(before) if before else None,
            relation,
            _noun(after),
            secondary,
            negative,
        )
    if _key(verb) == ("be",):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    return _Clause(subject, verb, _noun(tail) if tail else None, negative=negative)


def _source_clauses(source: str) -> tuple[_Clause, ...]:
    clauses = []
    # Coordination with an explicit subject is allowed; pronouns and inherited
    # subjects are deliberately unsupported to avoid coreference guesses.
    parts = re.split(r"[.!?;,]|\b(?:then|before)\b", source)
    parts = [
        clause
        for part in parts
        for clause in (
            [part] if " between " in part else re.split(r"\band(?=\s+(?:a|an|the)\s)", part)
        )
    ]
    expanded = []
    for part in parts:
        simultaneous = re.split(r"\b(?:as|while)\b", part)
        if len(simultaneous) == 2:
            try:
                _clause(simultaneous[0].strip())
            except _Refuse:
                pass
            else:
                expanded.extend(simultaneous)
                continue
        expanded.append(part)
    for part in expanded:
        part = re.sub(r"^(?:first\s+)", "", part.strip())
        try:
            clause = _clause(part)
            clauses.append(clause)
            if clause.object and clause.relation and clause.anchor:
                clauses.append(
                    _Clause(
                        clause.object,
                        "is",
                        relation=clause.relation,
                        anchor=clause.anchor,
                        secondary=clause.secondary,
                        negative=clause.negative,
                    )
                )
        except _Refuse:
            continue
    return tuple(clauses)


def _compatible(requested: _Noun, actual: _Noun) -> bool:
    return (
        _key(requested.label) == _key(actual.label)
        and (
            requested.count is None
            or requested.count == actual.count
            or (requested.count == 1 and actual.count is None and actual.singular_article)
        )
        and (requested.color is None or requested.color == actual.color)
        and set(requested.states).issubset(actual.states)
        and set(requested.attributes).issubset(actual.attributes)
    )


def _matches(requested: _Clause, actual: _Clause) -> bool:
    return (
        _compatible(requested.subject, actual.subject)
        and _key(requested.verb) == _key(actual.verb)
        and requested.negative == actual.negative
        and requested.transform == actual.transform
        and (
            requested.object is None
            or (actual.object is not None and _compatible(requested.object, actual.object))
        )
        and (
            requested.relation is None
            or (
                requested.relation == actual.relation
                and requested.anchor is not None
                and actual.anchor is not None
                and _compatible(requested.anchor, actual.anchor)
                and (
                    requested.secondary is None
                    or (
                        actual.secondary is not None
                        and _compatible(requested.secondary, actual.secondary)
                    )
                )
            )
        )
        and (requested.state is None or requested.state == actual.state)
        and (requested.layer is None or requested.layer == actual.layer)
    )


def _result_matches(requested: _Clause, actual: _Clause) -> bool:
    if _key(requested.verb) == ("be",) and requested.relation is not None:
        requested = _Clause(
            requested.subject,
            actual.verb,
            relation=requested.relation,
            anchor=requested.anchor,
            secondary=requested.secondary,
            negative=requested.negative,
        )
    return _matches(requested, actual)


def _action_linked_results(
    source: str,
    selected: list[_Clause],
    *,
    setting: str,
) -> tuple[_Noun, ...]:
    results = []
    for sentence in re.split(r"[.!?;]", source):
        match = re.fullmatch(r"(.+),\s*(?:calling forth|causing)\s+(.+)", sentence.strip())
        if match is None:
            continue
        action_text = re.sub(
            rf"^(?:in|at|inside)\s+(?:a\s+|an\s+|the\s+)?{re.escape(setting)},\s*",
            "",
            match[1],
            count=1,
        )
        try:
            action = _clause(action_text)
            result = _noun(match[2])
        except _Refuse:
            continue
        if not action.negative and action in selected:
            results.append(result)
    return tuple(results)


def _complete_source_clauses(
    source: str, setting: str
) -> tuple[tuple[_Clause, ...], tuple[tuple[_Clause, _Clause], ...]]:
    source = re.sub(
        rf"^(?:in|at|inside)\s+(?:(?:a|an|the)\s+)?{re.escape(setting)},\s*",
        "",
        source.strip(),
        count=1,
    )
    clauses, ordered = [], []
    for sentence in re.split(r"[.!?;]", source):
        if not sentence.strip():
            continue
        parts = re.split(r"\b(?:then|before)\b", sentence)
        if len(parts) > 2 or (len(parts) == 1 and re.search(r"\bfirst\b", sentence)):
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
        group = []
        for part in parts:
            part = re.sub(r"^first\s+", "", part.strip())
            if _PREDICATE.search(part) is None:
                raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
            clause = _clause(part)
            if (
                clause.verb in _RESULT_VERBS
                and clause.object is None
                and clause.verb not in (
                    _RESULT_MOTION_VERBS
                    | {"appear", "appears", "appeared", "rise", "rises", "fall", "falls"}
                )
            ) or any(
                noun is not None
                and noun.label.split()[0] in {"to", "from", "for", "by", "with", "about", "in"}
                for noun in (clause.subject, clause.object, clause.anchor, clause.secondary)
            ):
                raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
            group.append(clause)
        clauses.extend(group)
        if len(group) == 2:
            ordered.append((group[0], group[1]))
    if not clauses:
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    return tuple(clauses), tuple(ordered)


def _build(
    slots: dict[str, str], source: str, *, scope: Literal["focal", "scene"] = "focal"
) -> SceneFactsV2:
    if _SCOPED_SOURCE.search(source):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    actor = _noun(slots["ACTOR"])
    clauses = _source_clauses(source)
    scene_clauses, scene_order = (
        _complete_source_clauses(source, slots["SETTING"]) if scope == "scene" else ((), ())
    )
    source_colors: dict[tuple[str, ...], set[str]] = {}
    for clause in clauses:
        for noun in (clause.subject, clause.object, clause.anchor, clause.secondary):
            if noun and noun.color:
                source_colors.setdefault(_key(noun.label), set()).add(noun.color)

    def identity(noun: _Noun) -> tuple[tuple[str, ...], str | None] | None:
        label = _key(noun.label)
        colors = source_colors.get(label, set())
        if noun.color is None and len(colors) > 1:
            return None
        return label, noun.color or next(iter(colors), None)

    def resolved(noun: _Noun | None) -> _Noun | None:
        if noun is None:
            return None
        key = identity(noun)
        return replace(noun, color=key[1]) if key is not None else noun

    def matches(request: _Clause, actual: _Clause) -> bool:
        if any(
            noun is not None and identity(noun) is None
            for clause in (request, actual)
            for noun in (clause.subject, clause.object, clause.anchor, clause.secondary)
        ):
            return False
        return _matches(
            replace(
                request,
                subject=resolved(request.subject),
                object=resolved(request.object),
                anchor=resolved(request.anchor),
                secondary=resolved(request.secondary),
            ),
            replace(
                actual,
                subject=resolved(actual.subject),
                object=resolved(actual.object),
                anchor=resolved(actual.anchor),
                secondary=resolved(actual.secondary),
            ),
        )

    requested = []
    ordered_indices = []
    for group in re.split(r"[;,]", slots["ACTION"]):
        values = re.split(r"\b(?:then|before)\b", group)
        if re.search(r"\b(?:then|before|first)\b", group):
            if len(values) != 2:
                raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
            ordered_indices.append((len(requested), len(requested) + 1))
        for value in values:
            value = re.sub(r"^first\s+", "", value.strip())
            fragment = re.fullmatch(r"carried by (.+)", value)
            if fragment:
                agent = _noun(fragment[1])
                passive_candidates = [
                    clause
                    for clause in clauses
                    if clause.passive
                    and _key(clause.verb) == ("carry",)
                    and identity(actor) == identity(agent) == identity(clause.subject)
                ]
                if len(passive_candidates) != 1:
                    raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
                if not (
                    _compatible(actor, passive_candidates[0].subject)
                    and _compatible(agent, passive_candidates[0].subject)
                ):
                    raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
                requested.append(passive_candidates[0])
                continue
            match = _PREDICATE.match(value)
            if match or value.startswith(("does not ", "do not ", "never ")):
                value = f"{slots['ACTOR']} {value}"
            requested.append(_clause(value))
    actor_key = identity(actor)
    if actor_key is None or not any(identity(c.subject) == actor_key for c in requested):
        raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
    selected = []
    request_sources = []
    for request in requested:
        candidates = [candidate for candidate in clauses if matches(request, candidate)]
        if not candidates:
            raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
        if len(set(candidates)) != 1:
            raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
        request_sources.append(candidates[0])
        if candidates[0] not in selected:
            selected.append(candidates[0])

    if scene_clauses:
        present = {
            identity(noun)
            for clause in scene_clauses
            if not clause.negative and not clause.transform
            for noun in (clause.subject, clause.object, clause.anchor, clause.secondary)
            if noun is not None
        }
        for clause in scene_clauses:
            if clause.negative and any(
                noun is not None and identity(noun) not in present
                for noun in (clause.subject, clause.object, clause.anchor, clause.secondary)
            ):
                raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
            if not clause.transform and clause not in selected:
                selected.append(clause)

    nouns: dict[tuple[tuple[str, ...], str | None], _Noun] = {}
    subject_keys = {actor_key}

    def add(noun: _Noun) -> None:
        key = identity(noun)
        if key is None:
            raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
        noun = replace(noun, color=key[1])
        prior = nouns.get(key)
        if prior is not None:
            if (
                noun.count is not None and prior.count is not None and noun.count != prior.count
            ) or (noun.color is not None and prior.color is not None and noun.color != prior.color):
                raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
            noun = _Noun(
                prior.label,
                noun.count or prior.count,
                noun.color or prior.color,
                tuple(dict.fromkeys((*prior.states, *noun.states))),
                tuple(dict.fromkeys((*prior.attributes, *noun.attributes))),
                singular_article=noun.singular_article or prior.singular_article,
            )
        nouns[key] = noun

    for clause in selected:
        if (scope == "scene" and _key(clause.subject.label) == _key(actor.label)) or (
            _key(clause.verb) != ("be",)
            and not clause.state
            and not clause.layer
            and not (scope == "scene" and clause.negative)
            and not (
                scope == "scene"
                and clause.verb in _RESULT_VERBS - _RESULT_MOTION_VERBS
                and clause.object is None
                and clause not in request_sources
            )
        ):
            subject_keys.add(identity(clause.subject))
        for noun in (clause.subject, clause.object, clause.anchor, clause.secondary):
            if noun:
                add(noun)

    if scope == "scene":
        object_keys = {
            identity(noun)
            for clause in selected
            for noun in (clause.object, clause.anchor, clause.secondary)
            if noun is not None
        } | {
            identity(clause.subject)
            for clause in selected
            if clause.verb in _RESULT_VERBS and clause.object is None
        }
        if any(
            _key(clause.verb) == ("be",)
            and identity(clause.subject) not in subject_keys | object_keys
            for clause in selected
        ):
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)

    if not _compatible(actor, nouns[actor_key]):
        raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)

    magic_value = slots["MAGIC"]
    cause = re.fullmatch(r"(.+?) causes (.+)", magic_value)
    if cause:
        cause_subject = _noun(cause[1])
        cause_known = nouns.get(identity(cause_subject))
        if cause_known is None or not _compatible(cause_subject, cause_known):
            raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
        magic_value = cause[2]
    try:
        magic_clause = _clause(magic_value)
    except _Refuse:
        magic_clause = None
    transformation = None
    if magic_clause is not None:
        if magic_clause.transform and magic_clause.object is not None:
            magic = magic_clause.object
        elif (magic_clause.verb in _RESULT_VERBS or magic_clause.relation is not None) and (
            magic_clause.object is None and not magic_clause.negative
        ):
            magic = magic_clause.subject
        else:
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    else:
        magic = _noun(magic_value)
    transforms = [
        c
        for c in clauses
        if c.transform
        and not c.negative
        and c.object
        and identity(c.subject) in nouns
        and _compatible(c.subject, nouns[identity(c.subject)])
        and _compatible(magic, c.object)
    ]
    if len(transforms) > 1:
        raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
    if magic_clause and magic_clause.transform:
        known = nouns.get(identity(magic_clause.subject))
        if (
            known is None
            or not _compatible(magic_clause.subject, known)
            or not any(
                identity(magic_clause.subject) == identity(c.subject) for c in transforms
            )
        ):
            raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
    if transforms and (magic_clause is None or magic_clause.transform):
        transformation = transforms[0]
    else:
        # Match an affirmative result subject, not a phrase embedded in a
        # report, comparison, or imagined event.
        results = [
            clause
            for clause in clauses
            if clause.verb in _RESULT_VERBS
            and not clause.negative
            and clause.object is None
            and _compatible(magic, clause.subject)
            and (magic_clause is None or _result_matches(magic_clause, clause))
        ]
        linked = [
            result
            for result in _action_linked_results(source, selected, setting=slots["SETTING"])
            if magic_clause is None and _compatible(magic, result)
        ]
        if not results and not linked:
            raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
        for result in results:
            if result.verb in _RESULT_MOTION_VERBS:
                subject_keys.add(identity(result.subject))
            for noun in (result.subject, result.anchor, result.secondary):
                if noun:
                    add(noun)
            if result not in selected:
                selected.append(result)
        for result in linked:
            add(result)

    if any(clause.transform and clause != transformation for clause in scene_clauses):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)

    # Recover explicit attributes/edges of selected entities. New source-only
    # actors or unrelated objects are never pulled into the focal graph.
    for clause in clauses:
        if clause in selected or clause.transform or identity(clause.subject) not in nouns:
            continue
        targets = (clause.object, clause.anchor, clause.secondary)
        if any(noun and identity(noun) not in nouns for noun in targets):
            continue
        if (
            clause.state
            or clause.layer
            or clause.relation
            or clause.negative
            or _key(clause.verb)[0] in _OWNERSHIP
            or _key(clause.verb)[0] in {"rise", "ris", "fall"}
        ):
            selected.append(clause)
            add(clause.subject)

    refs = {key: f"n{index}" for index, key in enumerate(nouns)}
    modifiers = "|".join(sorted(COLOR_WORDS | _STATES | _ATTRIBUTES))
    introducers = "|".join(("a", "an", "another", *_COUNTS))
    for noun in nouns.values():
        introductions = re.findall(
            rf"\b(?:{introducers})\s+(?:(?:{modifiers})\s+){{0,5}}{re.escape(noun.label)}\b",
            source,
        )
        if sum(_noun(value).color in {None, noun.color} for value in introductions) > 1:
            raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
    relationships, motions, salience, negatives, events = [], [], [], [], []
    actions: dict[str, list[str]] = {}
    ordered_sources = [
        (request_sources[before], request_sources[after]) for before, after in ordered_indices
    ]
    ordered_sources.extend(pair for pair in scene_order if pair not in ordered_sources)
    event_by_clause: dict[_Clause, SceneEventFact] = {}

    def object_label(noun: _Noun | None) -> str | None:
        if noun is None:
            return None
        value = nouns[identity(noun)]
        if value.color is not None:
            return f"{value.color} {value.label}"
        return value.label

    for clause in selected:
        ref = refs[identity(clause.subject)]
        obj = refs[identity(clause.object)] if clause.object else None
        verb = _key(clause.verb)[0]
        if clause.negative:
            value = clause.state or " ".join(
                v for v in (clause.verb, object_label(clause.object)) if v
            )
            negatives.append(
                SceneNegativeFact(
                    kind="attribute" if clause.state else "action", target=ref, value=value
                )
            )
            continue
        if clause.state:
            key = identity(clause.subject)
            old = nouns[key]
            if clause.state in COLOR_WORDS:
                if old.color is not None and old.color != clause.state:
                    raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
                nouns[key] = replace(old, color=clause.state)
            else:
                nouns[key] = replace(old, states=tuple(dict.fromkeys((*old.states, clause.state))))
        if clause.layer:
            salience.append(SceneSalienceFact(source=ref, layer=clause.layer))
        if verb in _OWNERSHIP and obj:
            relationships.append(
                SceneRelationshipFact(source=ref, relation=_OWNERSHIP[verb], target=obj)
            )
        if clause.relation and clause.anchor:
            anchor = refs[identity(clause.anchor)]
            if clause.relation in {"toward", "towards"}:
                motions.append(SceneMotionFact(source=ref, destination=anchor))
            else:
                relationships.append(
                    SceneRelationshipFact(
                        source=obj or ref,
                        relation=_RELATIONS[clause.relation],
                        target=anchor,
                        secondary_target=refs[identity(clause.secondary)]
                        if clause.secondary
                        else None,
                    )
                )
        if verb in {"rise", "ris", "fall"}:
            motions.append(
                SceneMotionFact(source=ref, direction="falls" if verb == "fall" else "rises")
            )
        requested_action = any(matches(request, clause) for request in requested) or (
            clause in scene_clauses
            and not (clause.verb in _RESULT_VERBS and clause.object is None)
        )
        if (
            verb != "be"
            and not clause.layer
            and not clause.state
            and any(clause in pair for pair in ordered_sources)
        ):
            event = SceneEventFact(
                ref=f"e{len(events)}", source=ref, action=clause.verb, object=obj
            )
            events.append(event)
            event_by_clause[clause] = event
        elif (
            verb != "be"
            and (requested_action or clause.verb in _RESULT_MOTION_VERBS)
            and identity(clause.subject) in subject_keys
        ):
            action = " ".join(
                v for v in (clause.verb, object_label(clause.object)) if v
            )
            actions.setdefault(ref, []).append(action)
    orders = []
    for before, after in ordered_sources:
        if before not in event_by_clause or after not in event_by_clause:
            raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
        orders.append(
            SceneTemporalOrderFact(
                before=event_by_clause[before].ref, after=event_by_clause[after].ref
            )
        )
    subjects, objects = [], []
    for key, noun in sorted(nouns.items(), key=lambda item: item[0] != actor_key):
        common = dict(ref=refs[key], label=noun.label, count=noun.count, color=noun.color)
        if key in subject_keys:
            subjects.append(
                SceneSubjectFact(
                    **common,
                    attributes=(*noun.states, *noun.attributes),
                    actions=tuple(actions.get(refs[key], ())),
                )
            )
        else:
            objects.append(
                SceneObjectFact(**common, states=noun.states, attributes=noun.attributes)
            )
    transformed = None
    if transformation and transformation.object:
        result = transformation.object
        transformed = SceneTransformationFact(
            source=refs[identity(transformation.subject)],
            result_label=result.label,
            result_count=result.count,
            result_color=result.color,
            result_attributes=(*result.states, *result.attributes),
        )
    return SceneFactsV2(
        setting=SceneSettingFact(label=slots["SETTING"]),
        subjects=tuple(subjects),
        objects=tuple(objects),
        relationships=tuple(dict.fromkeys(relationships)),
        motions=tuple(dict.fromkeys(motions)),
        salience=tuple(dict.fromkeys(salience)),
        events=tuple(events),
        temporal_order=tuple(orders),
        negatives=tuple(dict.fromkeys(negatives)),
        transformation=transformed,
    )


def adapt_live_scene_facts(
    slots: Mapping[str, str],
    *,
    source_text: str,
    scope: Literal["focal", "scene"] = "focal",
) -> LiveSceneFactsResult:
    """Return a locally grounded graph or a stable refusal containing no values.

    Input is limited to four nonempty values of at most 512 characters each and
    a source of at most 4,000 characters. No source, exception details, or raw
    slot values are logged or retained in a refusal.

    Scene scope includes every supported source clause and refuses incomplete
    coverage. Both scopes require the original slot selections to be grounded.
    """
    if (
        scope not in ("focal", "scene")
        or not isinstance(slots, Mapping)
        or set(slots) != {"SETTING", "ACTOR", "ACTION", "MAGIC"}
        or not isinstance(source_text, str)
        or not source_text.strip()
        or len(source_text) > 4_000
        or any(
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 512
            or "\n" in value
            or "\r" in value
            for value in slots.values()
        )
    ):
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_INPUT)
    # Lazy import preserves the existing planner/client dependency direction.
    from storylight.tensorrt_slot_client import _validated_hybrid_slots

    if any(
        "|" in clause
        and (len(clause.split("|")) != 3 or any(not part.strip() for part in clause.split("|")))
        for label in ("ACTION", "MAGIC")
        for clause in re.split(r"[;,]", slots[label])
    ):
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_HYBRID)
    try:
        expanded = _validated_hybrid_slots(dict(slots))
    except ValueError:
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_HYBRID)
    try:
        normalized = {
            key: unicodedata.normalize("NFKC", value).casefold().strip()
            for key, value in expanded.items()
        }
        normalized["SETTING"] = re.sub(r"^(?:a|an|the)\s+", "", normalized["SETTING"], count=1)
        source = unicodedata.normalize("NFKC", source_text).casefold()
        facts = _build(normalized, source, scope=scope)
        compile_scene_facts_prompt(facts, source_text=source_text)
        return LiveSceneFactsResult(facts=facts)
    except _Refuse as error:
        return LiveSceneFactsResult(refusal=error.code)
    except SceneFactsPrivacyError:
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.PRIVACY)
    except SceneFactsGroundingError:
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.UNGROUNDED)
    except ValueError:
        return LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_GRAPH)
