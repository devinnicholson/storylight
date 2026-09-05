"""Bounded local grammar for the optional four-slot scene graph adapter.

Only explicit noun phrases and predicates are supported. The slots select the
focal actor/action; source clauses supply their bound details. This is not a
general English parser, and unsupported bindings return a value-free refusal.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from bookforge.privacy_policy import COLOR_WORDS, COUNT_WORDS, VISIBLE_VERBS
from bookforge.scene_facts import (
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
from bookforge.semantic_text import normalize_semantic_phrase


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


def _noun(value: str) -> _Noun:
    value = value.strip().strip(".")
    if not value or _BOUNDARY.search(value) or not re.fullmatch(r"[a-z0-9 -]+", value):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    words = value.split()
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
        " ".join(words), count, colors[0] if colors else None, tuple(states), tuple(attributes)
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


def _clause(value: str) -> _Clause:
    passive = re.fullmatch(r"(.+?) (?:is|are|was|were) carried by (.+)", value)
    if passive:
        return _Clause(_noun(passive[2]), "carries", _noun(passive[1]))
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
        and (requested.count is None or requested.count == actual.count)
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


def _build(slots: dict[str, str], source: str) -> SceneFactsV2:
    if _SCOPED_SOURCE.search(source):
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    actor = _noun(slots["ACTOR"])
    clauses = _source_clauses(source)
    requested = []
    for value in re.split(r"[;,]|\b(?:then|before)\b", slots["ACTION"]):
        value = re.sub(r"^first\s+", "", value.strip())
        match = _PREDICATE.match(value)
        if match or value.startswith(("does not ", "do not ", "never ")):
            value = f"{slots['ACTOR']} {value}"
        requested.append(_clause(value))
    if not requested or not any(_key(c.subject.label) == _key(actor.label) for c in requested):
        raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
    selected = []
    for request in requested:
        matches = [candidate for candidate in clauses if _matches(request, candidate)]
        if not matches:
            raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)
        if len(set(matches)) != 1:
            raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
        if matches[0] not in selected:
            selected.append(matches[0])

    nouns: dict[tuple[str, ...], _Noun] = {}
    subject_keys = {_key(actor.label)}

    def add(noun: _Noun) -> None:
        key = _key(noun.label)
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
            )
        nouns[key] = noun

    for clause in selected:
        if _key(clause.verb)[0] in _OWNERSHIP:
            subject_keys.add(_key(clause.subject.label))
        for noun in (clause.subject, clause.object, clause.anchor, clause.secondary):
            if noun:
                add(noun)

    if not _compatible(actor, nouns[_key(actor.label)]):
        raise _Refuse(LiveSceneFactsRefusal.UNGROUNDED)

    magic_value = slots["MAGIC"]
    cause = re.fullmatch(r"(.+?) causes (.+)", magic_value)
    if cause:
        cause_subject = _noun(cause[1])
        cause_known = nouns.get(_key(cause_subject.label))
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
        and _key(c.subject.label) in nouns
        and _compatible(c.subject, nouns[_key(c.subject.label)])
        and _compatible(magic, c.object)
    ]
    if len(transforms) > 1:
        raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
    if magic_clause and magic_clause.transform:
        known = nouns.get(_key(magic_clause.subject.label))
        if (
            known is None
            or not _compatible(magic_clause.subject, known)
            or not any(
                _key(magic_clause.subject.label) == _key(c.subject.label) for c in transforms
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
                subject_keys.add(_key(result.subject.label))
            for noun in (result.subject, result.anchor, result.secondary):
                if noun:
                    add(noun)
            if result not in selected:
                selected.append(result)
        for result in linked:
            add(result)

    # Recover explicit attributes/edges of selected entities. New source-only
    # actors or unrelated objects are never pulled into the focal graph.
    for clause in clauses:
        if clause in selected or clause.transform or _key(clause.subject.label) not in nouns:
            continue
        targets = (clause.object, clause.anchor, clause.secondary)
        if any(noun and _key(noun.label) not in nouns for noun in targets):
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
        if len(introductions) > 1:
            raise _Refuse(LiveSceneFactsRefusal.AMBIGUOUS_BINDING)
    relationships, motions, salience, negatives, events = [], [], [], [], []
    actions: dict[str, list[str]] = {}
    temporal = bool(re.search(r"\b(?:before|then|first)\b", slots["ACTION"]))
    requested_events = []
    for clause in selected:
        ref = refs[_key(clause.subject.label)]
        obj = refs[_key(clause.object.label)] if clause.object else None
        verb = _key(clause.verb)[0]
        if clause.negative:
            value = clause.state or " ".join(
                v for v in (clause.verb, clause.object.label if clause.object else None) if v
            )
            negatives.append(
                SceneNegativeFact(
                    kind="attribute" if clause.state else "action", target=ref, value=value
                )
            )
            continue
        if clause.state:
            old = nouns[_key(clause.subject.label)]
            if clause.state in COLOR_WORDS:
                add(_Noun(old.label, color=clause.state))
            else:
                add(_Noun(old.label, states=(clause.state,)))
        if clause.layer:
            salience.append(SceneSalienceFact(source=ref, layer=clause.layer))
        if verb in _OWNERSHIP and obj:
            relationships.append(
                SceneRelationshipFact(source=ref, relation=_OWNERSHIP[verb], target=obj)
            )
        if clause.relation and clause.anchor:
            anchor = refs[_key(clause.anchor.label)]
            if clause.relation in {"toward", "towards"}:
                motions.append(SceneMotionFact(source=ref, destination=anchor))
            else:
                relationships.append(
                    SceneRelationshipFact(
                        source=obj or ref,
                        relation=_RELATIONS[clause.relation],
                        target=anchor,
                        secondary_target=refs[_key(clause.secondary.label)]
                        if clause.secondary
                        else None,
                    )
                )
        if verb in {"rise", "ris", "fall"}:
            motions.append(
                SceneMotionFact(source=ref, direction="falls" if verb == "fall" else "rises")
            )
        requested_action = any(_matches(request, clause) for request in requested)
        if verb != "be" and not clause.layer and not clause.state and requested_action and temporal:
            event = SceneEventFact(
                ref=f"e{len(events)}", source=ref, action=clause.verb, object=obj
            )
            events.append(event)
            requested_events.append(event)
        elif (
            verb != "be"
            and (requested_action or clause.verb in _RESULT_MOTION_VERBS)
            and _key(clause.subject.label) in subject_keys
        ):
            action = " ".join(
                v for v in (clause.verb, clause.object.label if clause.object else None) if v
            )
            actions.setdefault(ref, []).append(action)
    orders = []
    if temporal and len(requested_events) != 2:
        raise _Refuse(LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX)
    if temporal and len(requested_events) == 2:
        orders.append(
            SceneTemporalOrderFact(before=requested_events[0].ref, after=requested_events[1].ref)
        )
    subjects, objects = [], []
    for key, noun in nouns.items():
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
            source=refs[_key(transformation.subject.label)],
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


def adapt_live_scene_facts(slots: Mapping[str, str], *, source_text: str) -> LiveSceneFactsResult:
    """Return a locally grounded graph or a stable refusal containing no values.

    Input is limited to four nonempty values of at most 512 characters each and
    a source of at most 4,000 characters. No source, exception details, or raw
    slot values are logged or retained in a refusal.
    """
    if (
        not isinstance(slots, Mapping)
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
    from bookforge.tensorrt_slot_client import _validated_hybrid_slots

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
        source = unicodedata.normalize("NFKC", source_text).casefold()
        facts = _build(normalized, source)
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
