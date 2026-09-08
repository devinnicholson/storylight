"""Explicit deterministic compilation of short, reviewed visual descriptions.

Only one or two complete subject/action clauses and an optional simple absence
are accepted. Unsupported grammar is a refusal, never a guessed scene.
"""

from __future__ import annotations

import re
from dataclasses import replace
from time import perf_counter

from bookforge.domain import ModelMetrics
from bookforge.live_scene_facts import _Clause, _Noun, _noun, _Refuse
from bookforge.live_scene_grounding import _entity
from bookforge.live_scene_planner import (
    LiveSceneGraphWirePlan,
    LiveScenePlannerError,
    LiveScenePlanningResult,
    LiveSceneWireFocus,
    LiveSceneWireMagic,
    validate_live_scene_plan_privacy,
)
from bookforge.privacy_policy import COLOR_WORDS
from bookforge.scene_facts import (
    SceneFactsV2,
    SceneNegativeFact,
    SceneObjectFact,
    SceneRelationshipFact,
    SceneSettingFact,
    SceneSubjectFact,
)
from bookforge.semantic_text import normalize_semantic_phrase

REVISION = "bounded-description-v1"
_RELATIONS = {
    "over": "above",
    "above": "above",
    "under": "under",
    "beside": "beside",
    "behind": "behind",
    "inside": "inside",
    "on": "on",
    "below": "below",
}
_VERBS = (
    "jump",
    "jumps",
    "jumped",
    "jumping",
    "stand",
    "stands",
    "standing",
    "sit",
    "sits",
    "sitting",
    "run",
    "runs",
    "running",
    "ran",
    "sleep",
    "sleeps",
    "sleeping",
    "carry",
    "carries",
    "carried",
    "carrying",
    "hold",
    "holds",
    "held",
    "holding",
    "chase",
    "chases",
    "chasing",
)
_PREDICATE = re.compile(
    r"^(?P<subject>.+?) (?:(?:is|are|was|were) )?"
    r"(?P<negative>(?:does not |do not |did not |not |never ))?"
    r"(?P<verb>" + "|".join(sorted(_VERBS, key=len, reverse=True)) + r")(?: (?P<tail>.+))?$"
)
_RELATION = re.compile(r"\b(" + "|".join(_RELATIONS) + r")\b")


def _visual_noun(text: str) -> _Noun:
    parsed = _entity(text)
    # Reuse the strict noun grammar for boundaries, pronouns, and phrase limits.
    noun = _noun(parsed.label)
    colors = tuple(value for value in parsed.modifiers if value in COLOR_WORDS)
    if len(colors) > 1:
        raise ValueError("ambiguous color")
    count = parsed.count
    if count is None and text.split()[0] in {"a", "an"}:
        count = 1
    return replace(
        noun,
        count=count,
        color=colors[0] if colors else None,
        attributes=tuple(value for value in parsed.modifiers if value not in COLOR_WORDS),
        singular_article=text.split()[0] in {"a", "an"},
    )


def _parse(text: str) -> _Clause:
    match = _PREDICATE.fullmatch(text)
    if match is None:
        raise ValueError("unsupported clause")
    subject = _visual_noun(match["subject"])
    tail = match["tail"] or ""
    relation = _RELATION.search(tail)
    obj = anchor = None
    if relation:
        if tail[: relation.start()].strip():
            obj = _visual_noun(tail[: relation.start()].strip())
        anchor = _visual_noun(tail[relation.end() :].strip())
    elif tail:
        obj = _visual_noun(tail)
    if obj is not None and anchor is not None:
        raise ValueError("ambiguous spatial attachment")
    verb = normalize_semantic_phrase(match["verb"])[0]
    if verb in {"stand", "sit", "sleep"} and obj is not None:
        raise ValueError("unsupported action target")
    if verb in {"carry", "hold", "chase", "chas"} and obj is None:
        raise ValueError("missing action target")
    return _Clause(
        subject,
        match["verb"],
        obj,
        relation[0] if relation else None,
        anchor,
        negative=bool(match["negative"]),
    )


def _facts(text: str) -> SceneFactsV2:
    if not isinstance(text, str) or not 1 <= len(text) <= 500:
        raise ValueError("description bound")
    lowered = text.lower().strip()
    if not re.fullmatch(r"[a-z0-9 .!?]+", lowered):
        raise ValueError("unsupported characters")
    parts = [part.strip() for part in re.split(r"[.!?]+", lowered) if part.strip()]
    absent = []
    clauses = []
    predicates = {}
    for part in parts:
        if part.startswith("no "):
            absent.append(_visual_noun(part[3:]))
        else:
            clause = _parse(part)
            clauses.append(clause)
            match = _PREDICATE.fullmatch(part)
            predicates[clause] = " ".join(
                value for value in (match["verb"], match["tail"]) if value
            )
    if not 1 <= len(clauses) <= 2 or len(absent) > 1 or not any(not c.negative for c in clauses):
        raise ValueError("unsupported clause count")

    nouns: dict[tuple, _Noun] = {}
    subject_keys = set()

    def key(noun):
        label = normalize_semantic_phrase(noun.label)
        matches = [
            identity
            for identity in nouns
            if identity[0] == label and (noun.color is None or identity[1] == noun.color)
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous entity")
        return matches[0] if matches else (label, noun.color)

    def add(noun):
        identity = key(noun)
        prior = nouns.get(identity)
        if prior is not None:
            if set(noun.attributes) - set(prior.attributes):
                raise ValueError("ambiguous changed attributes")
            if noun.singular_article or noun.count is not None:
                raise ValueError("ambiguous repeated introduction")
            if prior.count is not None and noun.count is not None and prior.count != noun.count:
                raise ValueError("conflicting count")
            noun = replace(
                prior,
                count=prior.count or noun.count,
                attributes=tuple(dict.fromkeys((*prior.attributes, *noun.attributes))),
            )
        nouns[identity] = noun
        return identity

    for clause in clauses:
        if clause.negative:
            continue
        subject_keys.add(add(clause.subject))
        for noun in (clause.object, clause.anchor):
            if noun:
                add(noun)
    refs = {identity: f"n{index}" for index, identity in enumerate(nouns)}
    actions = {identity: [] for identity in subject_keys}
    relationships, negatives = [], []
    for clause in clauses:
        identity = key(clause.subject)
        if identity not in subject_keys:
            raise ValueError("negative actor not established")
        if clause.negative:
            known = nouns[identity]
            if (
                clause.subject.singular_article
                or set(clause.subject.attributes) - set(known.attributes)
                or (clause.subject.count is not None and clause.subject.count != known.count)
            ):
                raise ValueError("ambiguous negative actor")
            # Keep the whole negative predicate; dropping its target would
            # broaden 'does not jump over dog' to 'does not jump'.
            value = predicates[clause]
            for positive in clauses:
                if positive.negative or key(positive.subject) != identity:
                    continue
                if normalize_semantic_phrase(positive.verb) != normalize_semantic_phrase(
                    clause.verb
                ):
                    continue
                if clause.relation is not None and clause.relation != positive.relation:
                    continue
                if all(
                    negative is None or (actual is not None and key(negative) == key(actual))
                    for negative, actual in (
                        (clause.object, positive.object),
                        (clause.anchor, positive.anchor),
                    )
                ):
                    raise ValueError("contradictory action")
            negatives.append(SceneNegativeFact(kind="action", target=refs[identity], value=value))
            continue
        actions[identity].append(predicates[clause])
        if clause.relation:
            relationships.append(
                SceneRelationshipFact(
                    source=refs[key(clause.object)] if clause.object else refs[identity],
                    relation=_RELATIONS[clause.relation],
                    target=refs[key(clause.anchor)],
                )
            )
    for noun in absent:
        if any(normalize_semantic_phrase(noun.label) == identity[0] for identity in nouns):
            raise ValueError("contradictory absence")
        if noun.count is not None or noun.color or noun.attributes:
            raise ValueError("unsupported qualified absence")
        negatives.append(SceneNegativeFact(kind="additional_object", value=noun.label))
    subjects, objects = [], []
    for identity, noun in nouns.items():
        fields = dict(
            ref=refs[identity],
            label=noun.label,
            color=noun.color,
            count=noun.count,
            attributes=noun.attributes,
        )
        if identity in subject_keys:
            subjects.append(SceneSubjectFact(**fields, actions=tuple(actions[identity])))
        else:
            objects.append(SceneObjectFact(**fields))
    return SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=tuple(subjects),
        objects=tuple(objects),
        relationships=tuple(relationships),
        negatives=tuple(negatives),
    )


def plan_bounded_description(text: str, visual_style: str, seed: int) -> LiveScenePlanningResult:
    """Compile supported reviewed text without a learned model or a fallback."""
    started = perf_counter()
    try:
        if not isinstance(visual_style, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("invalid render inputs")
        facts = _facts(text)
        facts.validate_source_grounding(source_text=text)
        first = facts.subjects[0]
        wire = LiveSceneGraphWirePlan(
            background_prompt="unspecified",
            focus=LiveSceneWireFocus(
                kind="character", subject=first.label, action=first.actions[0].split()[0]
            ),
            magic=LiveSceneWireMagic(kind="effect", prompt="none"),
            scene_facts=facts,
        )
        plan = wire.to_live_scene_plan(context_text=text)
        validate_live_scene_plan_privacy(plan, source_text=text)
        # Validate the actual style and final graph renderer, not just extraction.
        expected = facts.to_renderer_prompt(source_text=text, visual_style=visual_style)
        page = plan.to_page(source_text=text, visual_style=visual_style, seed=seed)
        if page.scene_spec.master_prompt != expected:
            raise ValueError("graph render contract changed")
    except (ValueError, _Refuse, LiveScenePlannerError) as error:
        raise LiveScenePlannerError(
            "Reviewed description is unsupported; use one or two explicit subject-action sentences"
        ) from error
    elapsed = (perf_counter() - started) * 1000
    return LiveScenePlanningResult(
        plan=plan,
        metrics=ModelMetrics(backend="deterministic", model=REVISION, total_ms=elapsed),
        model_revision=REVISION,
        wall_ms=elapsed,
    )
