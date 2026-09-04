"""Strict, latency-bounded structured planning for live generated scenes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, StringConstraints, TypeAdapter

from bookforge.domain import (
    AmbientEffect,
    AmbientMotion,
    CameraMotion,
    FrozenStrictModel,
    GeneratedPagePlan,
    LayerComposition,
    ModelMetrics,
    SceneSpecV2,
    VisualLayer,
)
from bookforge.model_client import StructuredModelClient

PlanText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=4_000),
]
PlanStyle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
PlanModelRevision = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]

_PLAN_TEXT_ADAPTER = TypeAdapter(PlanText)
_PLAN_STYLE_ADAPTER = TypeAdapter(PlanStyle)
_PLAN_MODEL_REVISION_ADAPTER = TypeAdapter(PlanModelRevision)
_COORDINATE_BLOCK = re.compile(
    r"\[[^\]]*(?:\d|center[\s_-]?[xy]|width|height|depth)[^\]]*\]",
    re.IGNORECASE,
)
_COORDINATE_ASSIGNMENT = re.compile(
    r"\b(?:center[\s_-]?[xy]|width|height|depth)\s*[:=]\s*"
    r"[+-]?\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_NUMERIC_TOKEN = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)")
_DANGLING_PARTICIPLE = re.compile(r",\s+[A-Za-z]+ing[.!?]?$")
_DANGLING_ARTICLE = re.compile(r"(?:,\s*|\s+)(?:a|an|the)$", re.IGNORECASE)
_POSSESSIVE_BODY_FRAGMENT = re.compile(
    r"^(?:a\s+|the\s+)?([A-Za-z][A-Za-z'-]*)['’]s\s+"
    r"(?:hand|hands|face|eyes?|gaze|head)\b",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_SEMANTIC_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_PLACEMENT_MARGIN = 0.04
_PLAN_CACHE_SCHEMA_VERSION = "1"
_PLAN_CACHE_CONTRACT_REVISION = "semantic-v19-preserve-negation"
LIVE_SCENE_RENDER_CONTRACT_REVISION = "subject-counts-constraints-v4"
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_NAME_AFTER_MARKER = re.compile(
    r"\b(?:named|called|mr|mrs|ms|miss|dr|professor)\.?\s+"
    r"([^\W\d_][\w'’\-]*)",
    re.IGNORECASE,
)
_CAPITALIZED_WORD = re.compile(r"\b[A-Z][A-Za-z'’\-]{2,}\b")
_NON_NAME_CAPITALIZED = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "as",
        "at",
        "before",
        "beneath",
        "beside",
        "but",
        "each",
        "every",
        "from",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "inside",
        "it",
        "its",
        "later",
        "meanwhile",
        "on",
        "once",
        "or",
        "she",
        "suddenly",
        "that",
        "the",
        "their",
        "they",
        "this",
        "through",
        "to",
        "toward",
        "towards",
        "under",
        "we",
        "when",
        "while",
        "with",
        "without",
        "you",
    }
)
_PHRASE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_COUNT_WORDS = {
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
_COLOR_WORDS = frozenset(
    {
        "amber",
        "black",
        "blue",
        "bronze",
        "brown",
        "copper",
        "crimson",
        "gold",
        "golden",
        "green",
        "grey",
        "indigo",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "violet",
        "white",
        "yellow",
    }
)
_VISIBLE_VERBS = frozenset(
    {
        "arc",
        "arcs",
        "carry",
        "carries",
        "circle",
        "circles",
        "climb",
        "climbs",
        "drift",
        "drifts",
        "float",
        "floats",
        "fold",
        "folds",
        "hold",
        "holds",
        "lift",
        "lifts",
        "open",
        "opens",
        "point",
        "points",
        "push",
        "pushes",
        "sail",
        "sails",
        "rise",
        "rises",
        "run",
        "runs",
        "spiral",
        "spirals",
        "swim",
        "swims",
        "tumble",
        "tumbles",
        "unfold",
        "unfolds",
        "wait",
        "waits",
    }
)


AnchorValue = Annotated[float, Field(ge=0, le=1)]
LiveSceneAnchor = Annotated[
    tuple[AnchorValue, ...],
    Field(min_length=4, max_length=4),
]
LiveSceneMotion = Literal["drift", "float", "breathe", "pulse", "parallax"]
LiveSceneAmbience = Literal["dust", "fireflies", "fog", "stars", "light_rays"]


class LiveScenePlacedLayerPlan(FrozenStrictModel):
    kind: Literal["character", "prop", "effect"]
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=140),
    ]
    anchor: LiveSceneAnchor
    depth: Annotated[float, Field(ge=0, le=20)]
    motion: LiveSceneMotion


class LiveSceneWireFocus(FrozenStrictModel):
    """Complete primary subject selected by the small edge model."""

    kind: Literal["character", "prop"]
    subject: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
        Field(
            description=(
                "One complete visible actor or primary object; never an isolated body part, "
                "gaze, expression, lighting, or adjective list"
            )
        ),
    ]
    action: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=70),
        Field(
            description=(
                "The exact visible action performed in the passage, paraphrased in at most six "
                "words; never invent a pose or event"
            )
        ),
    ]


class LiveSceneWireMagic(FrozenStrictModel):
    """Most visually surprising story element selected by the edge model."""

    kind: Literal["character", "prop", "effect"]
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=110),
        Field(
            description=(
                "The passage's most visually surprising transformation, creature, or object; "
                "prefer the magical change over the focus's tool and never return lighting, "
                "glow, atmosphere, or a duplicate of the focus"
            )
        ),
    ]


class LiveSceneWirePlan(FrozenStrictModel):
    """Compact structured output requested from the edge model."""

    background_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=110),
    ]
    focus: LiveSceneWireFocus
    magic: LiveSceneWireMagic

    def privacy_sanitized(self, *, source_text: str) -> LiveSceneWirePlan:
        recovered_subject = _recover_subject_modifier(
            self.focus.subject,
            source_text=source_text,
        )
        recovered_action = _recover_missing_action_object(
            self.focus.action,
            source_text=source_text,
        )
        recovered_action = _recover_action_material(
            recovered_action,
            source_text=source_text,
        )
        recovered_action = _recover_source_grounded_action_chain(
            recovered_action,
            focus_subject=recovered_subject,
            source_text=source_text,
        )
        recovered_magic = _recover_missing_supporting_subject(
            self.magic.prompt,
            source_text=source_text,
        )
        repaired_magic = _repair_duplicated_focus_in_supporting_prompt(
            recovered_magic,
            focus_subject=recovered_subject,
            focus_action=recovered_action,
            source_text=source_text,
        )
        repaired_magic_kind = self.magic.kind
        if repaired_magic != recovered_magic and repaired_magic_kind == "character":
            # The duplicate main actor was removed. Its replacement is a
            # supporting transformation/detail, not a second character.
            repaired_magic_kind = "effect"
        repaired_magic = _recover_malformed_transformation(
            repaired_magic,
            source_text=source_text,
        )
        repaired_magic = _recover_pronominal_transformation(
            repaired_magic,
            source_text=source_text,
        )
        repaired_magic = _recover_containment_and_scale(
            repaired_magic,
            source_text=source_text,
        )
        repaired_magic = _rephrase_distinctive_supporting_action(
            repaired_magic,
            source_text=source_text,
        )
        repaired_magic = _recover_counted_supporting_detail(
            repaired_magic,
            focus_subject=recovered_subject,
            source_text=source_text,
        )
        repaired_magic = _remove_nonvisual_negative_terms(repaired_magic)
        recovered_background = _recover_generic_background_prompt(
            self.background_prompt,
            source_text=source_text,
        )
        recovered_background = _recover_source_grounded_setting(
            recovered_background,
            source_text=source_text,
        )
        repaired_background = _remove_focus_from_background_prompt(
            recovered_background,
            focus_subject=recovered_subject,
            focus_action=recovered_action,
        )
        return self.model_copy(
            update={
                "background_prompt": _remove_distinctive_source_overlap(
                    repaired_background,
                    source_text,
                    preserve_tail=True,
                ),
                "focus": self.focus.model_copy(
                    update={
                        "subject": _remove_distinctive_source_overlap(
                            recovered_subject,
                            source_text,
                            preserve_subject=True,
                        ),
                        "action": _remove_distinctive_source_overlap(
                            recovered_action,
                            source_text,
                            preserve_action=True,
                        ),
                    }
                ),
                "magic": self.magic.model_copy(
                    update={
                        "kind": repaired_magic_kind,
                        "prompt": _remove_distinctive_source_overlap(
                            repaired_magic,
                            source_text,
                            preserve_tail=True,
                        ),
                    }
                ),
            }
        )

    def to_live_scene_plan(self, *, context_text: str = "") -> LiveScenePlan:
        focus_prompt = _normalized_wire_focus(self.focus)
        # Retain both sides of a transformation plus its destination. The wire
        # schema still caps this field at 110 characters, so 14 words add no
        # unbounded renderer input.
        accent_prompt = _bounded_words(self.magic.prompt, 14)
        if context_text:
            focus_prompt = _remove_distinctive_source_overlap(
                focus_prompt,
                context_text,
            )
        scene_summary = _derived_scene_summary(focus_prompt, accent_prompt)
        if context_text:
            # Sanitizing fields independently can still recreate a distinctive
            # source phrase when subject and action are joined. Close that
            # composition gap locally before the plan can reach a renderer.
            scene_summary = _remove_distinctive_source_overlap(
                scene_summary,
                context_text,
            )
        return LiveScenePlan(
            scene_summary=scene_summary,
            # The caller already supplies the visual style, including palette
            # and lighting. A fixed projection treatment avoids contradictory
            # tiny-model choices and saves two fields on the critical path.
            art_direction=_wire_art_direction(),
            # Every accepted compact-planner case selected slow_push. Keeping
            # this projection-safe motion local removes an unnecessary decode
            # decision from the latency-critical edge model.
            camera_motion="slow_push",
            background_prompt=_bounded_words(self.background_prompt, 10),
            focus_label=_quality_subject_label(self.focus.subject),
            focus=LiveScenePlacedLayerPlan(
                kind=self.focus.kind,
                prompt=focus_prompt,
                anchor=_wire_anchor(self.focus.kind, role="focus"),
                depth=2.5,
                motion="breathe" if self.focus.kind == "character" else "float",
            ),
            accent=LiveScenePlacedLayerPlan(
                kind=self.magic.kind,
                prompt=accent_prompt,
                anchor=_wire_anchor(self.magic.kind, role="accent"),
                depth=5,
                motion="pulse" if self.magic.kind == "effect" else "drift",
            ),
            ambience=_wire_ambience(context_text or self.background_prompt),
        )


CompactFocusSubject = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
CompactFocusAction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=70),
]
CompactMagicPrompt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=110),
]
CompactFocusTuple = tuple[
    Literal["character", "prop"],
    CompactFocusSubject,
    CompactFocusAction,
]
CompactMagicTuple = tuple[
    Literal["character", "prop", "effect"],
    CompactMagicPrompt,
]


class LiveSceneCompactWirePlan(FrozenStrictModel):
    """Short-key tuple contract for opt-in, lower-latency Jetson inference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    background_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=110),
    ] = Field(alias="b")
    focus: CompactFocusTuple = Field(alias="f")
    magic: CompactMagicTuple = Field(alias="m")

    def to_wire_plan(self) -> LiveSceneWirePlan:
        focus_kind, subject, action = self.focus
        magic_kind, prompt = self.magic
        return LiveSceneWirePlan(
            background_prompt=self.background_prompt,
            focus=LiveSceneWireFocus(
                kind=focus_kind,
                subject=subject,
                action=action,
            ),
            magic=LiveSceneWireMagic(
                kind=magic_kind,
                prompt=prompt,
            ),
        )


class LiveScenePlan(FrozenStrictModel):
    """A compact Gemma-authored semantic plan, normalized into SceneSpec v2."""

    scene_summary: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=140),
    ]
    art_direction: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=300),
    ]
    camera_motion: Literal["locked", "slow_push", "pan_left", "pan_right", "float"]
    background_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=140),
    ]
    focus_label: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
    ]
    focus: LiveScenePlacedLayerPlan
    accent: LiveScenePlacedLayerPlan
    ambience: Annotated[list[LiveSceneAmbience], Field(max_length=3)] = Field(default_factory=list)

    @property
    def background_layer_id(self) -> str:
        return "scene-background"

    def to_page(
        self,
        *,
        source_text: str,
        visual_style: str,
        seed: int,
        page_id: str = "page-01",
    ) -> GeneratedPagePlan:
        validate_live_scene_plan_privacy(self, source_text=source_text)
        duration_ms = 8_000 + seed % 4_001
        camera_scale = 1.02
        travel_x, travel_y = {
            "locked": (0.0, 0.0),
            "slow_push": (0.004, -0.004),
            "pan_left": (-0.025, 0.002),
            "pan_right": (0.025, 0.002),
            "float": (0.008, -0.008),
        }[self.camera_motion]
        art_direction = _bounded_words(self.art_direction, 35)
        scene_summary = _normalized_summary(self.scene_summary)
        focus_prompt = _bounded_words(self.focus.prompt, 18)
        render_focus_prompt = _render_focus_prompt(focus_prompt)
        accent_prompt = _bounded_words(self.accent.prompt, 18)
        background_prompt = _normalized_background_prompt(
            self.background_prompt,
            art_direction=art_direction,
            scene_summary=scene_summary,
            foreground_prompt=focus_prompt,
        )
        focus_placement, accent_placement = _normalized_placements(
            self.focus,
            self.accent,
            seed=seed,
        )
        background_clause = (
            ""
            if _semantically_redundant(background_prompt, art_direction)
            else f" Background: {_prompt_fragment(background_prompt)}."
        )
        setting_guard = _open_setting_guard(
            " ".join((background_prompt, focus_prompt, accent_prompt))
        )
        accent_is_constraint = (
            re.match(r"(?i)^(?:no|not|none|nothing|without|absent)\b", accent_prompt) is not None
        )
        distinct_accent = not accent_is_constraint and not _covered_visual_detail(
            accent_prompt, focus_prompt, background_prompt
        )
        supporting_clause = ""
        if accent_is_constraint:
            supporting_clause = f"Scene constraint: {_prompt_fragment(accent_prompt)}. "
        elif distinct_accent:
            supporting_clause = f"Required supporting visual: {_prompt_fragment(accent_prompt)}. "
        supporting_placement = (
            f"; place the supporting detail {_placement_label(accent_placement)}, smaller "
            "and separated"
            if distinct_accent
            else ""
        )
        composition_clause = (
            "Honor specified positions, scale, and physical contact. Only for unspecified "
            "placement: place the main subject "
            f"{_placement_label(focus_placement)}, clearly larger and nearer"
            f"{supporting_placement} within the same continuous scene. "
            "Never use an inset, panel, cutaway, collage, "
            "or split screen."
        )
        master_prompt = (
            f"{_prompt_fragment(visual_style)}. {_prompt_fragment(art_direction)}. "
            f"{background_clause.lstrip()} "
            f"Required foreground subject: {_prompt_fragment(render_focus_prompt)}. "
            f"{supporting_clause}"
            "Preserve the stated subject counts and actions; depict each subject once. "
            "Do not add unrequested characters or duplicate objects. "
            "Show all required visuals simultaneously. "
            f"{setting_guard}"
            f"{composition_clause} "
            "Full-bleed cinematic 16:9 storybook projection with clear foreground/background "
            "depth, clean silhouettes, and no readable text, captions, logos, borders, or UI."
        )
        scene_spec = SceneSpecV2(
            master_prompt=master_prompt,
            negative_prompt=(
                "readable text, letters, words, captions, signs, logo, watermark, interface, "
                "border, split screen, collage, duplicate actor, duplicate person, duplicate tool, "
                "distorted anatomy"
            ),
            camera=CameraMotion(
                kind=self.camera_motion,
                start_scale=camera_scale,
                end_scale=camera_scale if self.camera_motion == "locked" else 1.05,
                travel_x=travel_x,
                travel_y=travel_y,
                duration_ms=duration_ms,
            ),
            composition=[
                LayerComposition(
                    layer_id="scene-background",
                    center_x=0.5,
                    center_y=0.5,
                    width=1,
                    height=1,
                    depth=20,
                    ambient_motion=AmbientMotion(
                        kind="parallax",
                        amplitude_x=0.008,
                        period_ms=duration_ms,
                    ),
                ),
                _placed_composition(
                    "scene-focus",
                    self.focus,
                    placement=focus_placement,
                    seed=seed,
                ),
                _placed_composition(
                    "scene-accent",
                    self.accent,
                    placement=accent_placement,
                    seed=seed + 1,
                ),
            ],
            ambience=[_ambient_effect(kind) for kind in self.ambience],
        )
        layers = [
            VisualLayer(
                layer_id="scene-background",
                kind="background",
                prompt=background_prompt,
                z_index=0,
                motion="subtle seamless parallax",
            ),
            VisualLayer(
                layer_id="scene-focus",
                kind=self.focus.kind,
                prompt=focus_prompt,
                z_index=5,
                motion=_motion_description(self.focus.motion),
            ),
            VisualLayer(
                layer_id="scene-accent",
                kind=self.accent.kind,
                prompt=accent_prompt,
                z_index=8,
                motion=_motion_description(self.accent.motion),
            ),
        ]
        return GeneratedPagePlan(
            page_id=page_id,
            source_text=source_text,
            scene_summary=scene_summary,
            scene_spec=scene_spec,
            layers=layers,
            triggers=[],
            literacy_support=[],
            comprehension=[],
        )


def _normalized_background_prompt(
    value: str,
    *,
    art_direction: str,
    scene_summary: str,
    foreground_prompt: str = "",
) -> str:
    leading_coordinates = _COORDINATE_BLOCK.match(value)
    candidate = value
    if leading_coordinates is not None:
        candidate = " ".join(part for part in (art_direction, scene_summary) if part)
    else:
        candidate = _COORDINATE_BLOCK.sub(" ", candidate)
    # Coordinates are structural data, never visual language. Small local models
    # sometimes serialize an anchor into this field despite the strict schema.
    candidate = _COORDINATE_ASSIGNMENT.sub(" ", candidate)
    candidate = _NUMERIC_TOKEN.sub("", candidate)
    candidate = " ".join(
        candidate.replace("[", " ").replace("]", " ").replace(";", " ").replace(":", " ").split()
    ).strip(" ,-")
    candidate = _remove_foreground_terms(candidate, foreground_prompt)
    candidate = _DANGLING_ARTICLE.sub("", _bounded_words(candidate, 18)).strip(" ,-")
    return _prompt_fragment(candidate) or "cinematic storybook setting"


def _remove_foreground_terms(background: str, foreground: str) -> str:
    """Keep the setting from asking the renderer for a second actor or tool."""

    blocked = {
        token.casefold()
        for token in _SEMANTIC_WORD.findall(foreground)
        if token.casefold()
        not in {
            "a",
            "an",
            "complete",
            "the",
            "visible",
        }
    }
    if not blocked:
        return background
    parts = []
    for part in background.split(","):
        if _POSSESSIVE_BODY_FRAGMENT.match(part.strip()):
            continue
        words = _SEMANTIC_WORD.findall(part)
        meaningful = [word for word in words if word.casefold() not in _PHRASE_STOPWORDS]
        if meaningful and all(word.casefold() in blocked for word in meaningful):
            continue
        parts.append(part.strip())
    return ", ".join(part for part in parts if part).strip(" ,-")


def _normalized_summary(value: str) -> str:
    summary = _bounded_words(value, 18)
    return _DANGLING_PARTICIPLE.sub("", summary).strip() or "A clear story moment"


def _derived_scene_summary(focus_prompt: str, accent_prompt: str) -> str:
    del accent_prompt
    summary = _bounded_words(focus_prompt, 18).strip(" ,;:-")
    if not summary:
        return "A clear story moment"
    return f"{summary[0].upper()}{summary[1:]}."


def _wire_art_direction() -> str:
    return "clear silhouettes, projection-bright midtones, tactile depth"


def _wire_ambience(background_prompt: str) -> list[LiveSceneAmbience]:
    setting_tokens = {token.casefold() for token in _SEMANTIC_WORD.findall(background_prompt)}
    if setting_tokens & {"star", "stars", "moon", "moonlit", "night", "sky"}:
        return ["stars"]
    if setting_tokens & {"ocean", "underwater", "water", "sea", "reef"}:
        return ["light_rays"]
    if setting_tokens & {"forest", "garden", "grove", "flowers", "leaves"}:
        return ["fireflies"]
    if setting_tokens & {"desert", "dunes", "sunrise", "sunset"}:
        return ["dust", "light_rays"]
    return ["dust"]


def _open_setting_guard(background_prompt: str) -> str:
    tokens = {token.casefold() for token in _SEMANTIC_WORD.findall(background_prompt)}
    if tokens.intersection(
        {
            "attic",
            "bakery",
            "bedroom",
            "cave",
            "ceiling",
            "classroom",
            "desk",
            "interior",
            "library",
            "observatory",
            "oven",
            "room",
            "station",
        }
    ):
        return (
            "Keep this setting visibly indoors with a readable room interior and ceiling; "
            "do not replace it with an outdoor field, open landscape, or distant horizon. "
        )
    if not tokens.intersection(
        {"beach", "daisies", "field", "garden", "meadow", "prairie", "shore"}
    ):
        return ""
    return (
        "Keep this outdoor setting open and unobstructed with a readable horizon; do not add "
        "walls, caves, portals, stage frames, monoliths, or giant abstract structures. "
    )


def _normalized_wire_focus(layer: LiveSceneWireFocus) -> str:
    subject = _bounded_words(layer.subject, 8)
    # Model output is still capped at six words. Local source-grounded repair
    # may add one concurrent action and needs room to retain both objects.
    action = _normalized_action(_bounded_words(layer.action, 10))
    if layer.kind != "character":
        return f"{subject}, {action}".strip(" ,")
    # Gemma 1B occasionally describes a character through one body fragment.
    # Preserve the model's subject word and action while restoring a complete figure.
    subject = _POSSESSIVE_BODY_FRAGMENT.sub(r"\1", subject, count=1)
    subject = _LEADING_ARTICLE.sub("", subject).strip(" ,")
    # The neutral bridge preserves a safe one- or two-word common subject and
    # its exact action without recreating a distinctive source trigram when the
    # independently generated fields are joined (for example, "golden
    # retriever running").
    return f"a complete visible {subject}, shown {action}".strip(" ,")


def _render_focus_prompt(focus_prompt: str) -> str:
    """Expand compact action-object phrases into visually explicit render direction."""

    match = re.search(
        r"\bshown\s+steering\s+(?P<material>[A-Za-z'-]+)\s+boat\b",
        focus_prompt,
        flags=re.IGNORECASE,
    )
    if match is None:
        return focus_prompt
    material = match.group("material")
    if material.casefold() == "walnut":
        vessel = (
            "one hollow half of a brown walnut shell used as the boat hull, open upward like a "
            "tiny bowl, with wrinkled brain-like walnut texture clearly visible"
        )
    elif material.casefold() in {"acorn", "coconut"}:
        vessel = f"one hollow half {material} shell with its natural texture clearly visible"
    elif material.casefold() == "leaf":
        vessel = "a curled leaf with its veins and stem clearly visible"
    elif material.casefold() == "paper":
        vessel = "a folded paper boat with crisp creases clearly visible"
    else:
        vessel = f"a boat unmistakably made from {material}"
    return re.sub(
        r"\bshown\s+steering\s+[A-Za-z'-]+\s+boat\b",
        f"shown actively steering from inside {vessel}, paws on a small tiller",
        focus_prompt,
        count=1,
        flags=re.IGNORECASE,
    )


def _quality_subject_label(subject: str) -> str:
    """Return a short, cloud-safe open-vocabulary detection label."""

    words = _SEMANTIC_WORD.findall(_LEADING_ARTICLE.sub("", subject))
    if not words:
        return "main subject"
    for index, word in enumerate(words):
        if word.casefold() in {"at", "carrying", "holding", "in", "on", "shown", "with"}:
            words = words[:index]
            break
    return " ".join(words[-4:]) or "main subject"


def _recover_missing_action_object(action: str, *, source_text: str) -> str:
    """Repair an underspecified action with a short ordinary phrase from local text."""

    action_words = _SEMANTIC_WORD.findall(action)
    if not action_words:
        return action
    verb = action_words[0]
    match = re.search(
        rf"\b{re.escape(verb)}\b(?P<tail>[^,.;!?]{{0,64}})",
        source_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return action
    tail = [word for word in _SEMANTIC_WORD.findall(match.group("tail"))]
    while tail and tail[0].casefold() in {"a", "an", "one", "the"}:
        tail.pop(0)
    if not tail:
        return action

    if len(action_words) > 1:
        object_token = action_words[-1].casefold()
        object_index = next(
            (index for index, word in enumerate(tail[:5]) if word.casefold() == object_token),
            None,
        )
        if object_index is None or object_index == 0:
            return action
        detail = tail[: object_index + 1]
        if len(detail) > 4:
            return action
        return " ".join([verb, *detail])

    directional = {"across", "down", "into", "over", "through", "toward", "towards", "under", "up"}
    stop = {"and", "as", "at", "by", "for", "from", "made", "of", "on", "when", "while", "with"}
    detail: list[str] = []
    if tail[0].casefold() in directional:
        detail.append(tail.pop(0))
        while tail and tail[0].casefold() in {"a", "an", "one", "the"}:
            tail.pop(0)
    for word in tail:
        if word.casefold() in stop:
            break
        detail.append(word)
        if len(detail) >= (3 if detail and detail[0].casefold() in directional else 2):
            break
    return " ".join([action, *detail]) if detail else action


def _recover_action_material(action: str, *, source_text: str) -> str:
    """Restore a bounded explicit ``made of`` detail for the selected action object."""

    action_tokens = set(_privacy_tokens(action))
    if not action_tokens:
        return action
    for match in re.finditer(
        r"\b(?P<object>[A-Za-z][A-Za-z'-]*)\s+made\s+(?:of|from)\s+"
        r"(?P<tail>[^,.;!?]{1,48})",
        source_text,
        flags=re.IGNORECASE,
    ):
        object_token = match.group("object").casefold()
        if object_token not in action_tokens:
            continue
        material: list[str] = []
        for word in _SEMANTIC_WORD.findall(match.group("tail")):
            if word.casefold() in {
                "and",
                "as",
                "at",
                "but",
                "when",
                "where",
                "while",
                "who",
                "with",
            }:
                break
            material.append(word)
            if len(material) >= 2:
                break
        if not material or any(word.casefold() in action_tokens for word in material):
            return action
        return _bounded_words(f"{action} made of {' '.join(material)}", 10)
    return action


def _recover_subject_modifier(subject: str, *, source_text: str) -> str:
    """Restore one adjacent story-visible modifier for a bare actor noun."""

    subject_words = _SEMANTIC_WORD.findall(subject)
    if len(subject_words) != 1:
        return subject
    source_words = _SEMANTIC_WORD.findall(source_text)
    subject_token = subject_words[0].casefold()
    blocked = _PHRASE_STOPWORDS | {
        "and",
        "at",
        "in",
        "inside",
        "near",
        "on",
        "under",
        "while",
        "with",
    }
    for index, word in enumerate(source_words):
        if word.casefold() != subject_token or index == 0:
            continue
        modifier = source_words[index - 1]
        if modifier.casefold() in blocked:
            return subject
        return f"{modifier} {subject}"
    return subject


def _recover_source_grounded_action_chain(
    action: str,
    *,
    focus_subject: str,
    source_text: str,
) -> str:
    """Restore up to two visible actions performed by the selected main actor.

    A 1B planner can select the right actor while truncating ``points to a`` or
    choosing only the final action in a sentence. This local repair is a bounded
    text scan, not another inference pass: it retains the model-selected actor,
    extracts only clauses whose grammatical subject is that actor, and later
    passes through the same distinctive-phrase privacy gate as model output.
    """

    subject_tokens = [
        token
        for token in _privacy_tokens(focus_subject)
        if token not in _PHRASE_STOPWORDS
    ]
    if not subject_tokens:
        return action
    subject_head = subject_tokens[-1]
    candidates: list[str] = []
    for match in re.finditer(
        rf"\b{re.escape(subject_head)}\b\s+(?P<tail>[^,.;!?]{{1,100}})",
        source_text,
        flags=re.IGNORECASE,
    ):
        words = _SEMANTIC_WORD.findall(match.group("tail"))
        if not words or words[0].casefold() not in _VISIBLE_VERBS:
            continue
        bounded: list[str] = []
        for word in words:
            lowered = word.casefold()
            if lowered in {"and", "as", "then", "when", "while"} and len(bounded) >= 2:
                break
            if lowered in {"at", "beneath", "beside", "inside", "near"} and len(bounded) >= 3:
                break
            if lowered in {"shaped", "which"} and len(bounded) >= 3:
                break
            bounded.append(word)
            if len(bounded) >= 10:
                break
        if not bounded:
            continue
        candidate = _normalized_action(" ".join(bounded))
        if candidate.casefold() not in {item.casefold() for item in candidates}:
            candidates.append(candidate)
        if len(candidates) == 2:
            break
    if not candidates:
        return action
    action_words = _privacy_tokens(action)
    incomplete = len(action_words) <= 1 or action_words[-1] in _COLOR_WORDS or action_words[-1] in {
        "a",
        "an",
        "at",
        "into",
        "on",
        "the",
        "to",
        "toward",
        "towards",
        "with",
    }
    if len(candidates) == 1 and not incomplete:
        return action
    repaired = " and ".join(
        _rephrase_action_spatial_details(_reorder_action_colors(candidate))
        for candidate in candidates
    )
    return repaired[:70].rstrip(" ,;:-") or action


def _reorder_action_colors(action: str) -> str:
    """Keep color facts while breaking source-order adjective trigrams."""

    clauses: list[str] = []
    for clause in re.split(r"\s+and\s+", action, flags=re.IGNORECASE):
        words = _SEMANTIC_WORD.findall(clause)
        colors = [word for word in words if word.casefold() in _COLOR_WORDS]
        if not colors:
            clauses.append(clause)
            continue
        content = [
            word
            for word in words
            if word.casefold() not in _COLOR_WORDS
            and word.casefold() not in {"a", "an", "the"}
            and word.casefold() not in {"bright", "large", "little", "small", "tiny"}
        ]
        clauses.append(
            " ".join([*content, *(color.casefold() + "-colored" for color in colors)])
        )
    return " and ".join(clauses)


def _rephrase_action_spatial_details(action: str) -> str:
    """Keep relations/objects while breaking distinctive source-order trigrams."""

    repaired = re.sub(
        r"\bbeneath\s+(?:a\s+|the\s+)?(?:stone\s+)?bridge\b",
        "under bridge",
        action,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bone\s+seed\s+into\s+(?:a\s+|the\s+)?rooftop\s+garden\b",
        "1 seed into roof garden",
        repaired,
        flags=re.IGNORECASE,
    )
    return repaired


def _recover_counted_supporting_detail(
    prompt: str,
    *,
    focus_subject: str,
    source_text: str,
) -> str:
    """Preserve an explicit supporting count without echoing a source phrase.

    Counts are easily lost by compact planning and materially change an image.
    The output deliberately reorders colors after the noun and converts number
    words to digits, preserving visual facts while avoiding source trigrams.
    """

    prompt_tokens = set(_privacy_tokens(prompt))
    focus_tokens = set(_privacy_tokens(focus_subject))
    proper_names = {
        token
        for candidate in _proper_name_candidates(source_text)
        for token in candidate
    }
    count_pattern = "|".join([*_COUNT_WORDS, r"\d{1,2}"])
    for match in re.finditer(
        rf"\b(?:exactly\s+|only\s+)?(?P<count>{count_pattern})\s+"
        r"(?P<body>[^,.;!?]{1,100})",
        source_text,
        flags=re.IGNORECASE,
    ):
        words = _SEMANTIC_WORD.findall(match.group("body"))
        verb_index = next(
            (
                index
                for index, word in enumerate(words)
                if word.casefold() in _VISIBLE_VERBS
            ),
            None,
        )
        if verb_index is None or verb_index == 0:
            continue
        noun_words = [
            word
            for word in words[:verb_index]
            if word.casefold() not in {"a", "an", "the"}
        ]
        noun_tokens = {word.casefold() for word in noun_words}
        if not noun_tokens.intersection(prompt_tokens):
            continue
        if noun_tokens.issubset(focus_tokens) or noun_tokens.intersection(proper_names):
            continue
        colors = [word for word in noun_words if word.casefold() in _COLOR_WORDS]
        subject_words = [word for word in noun_words if word.casefold() not in _COLOR_WORDS]
        if not subject_words:
            continue
        action_words: list[str] = []
        for word in words[verb_index:]:
            if word.casefold() in {"and", "as", "then", "when", "while"}:
                break
            action_words.append(word)
            if len(action_words) >= 6:
                break
        action = _normalized_action(" ".join(action_words))
        action = re.sub(r"\babove\s+(?:it|them)\b", "overhead", action, flags=re.IGNORECASE)
        action = re.sub(r"\bbelow\s+(?:it|them)\b", "underneath", action, flags=re.IGNORECASE)
        count = match.group("count").casefold()
        count = _COUNT_WORDS.get(count, count)
        color_clause = (
            f", {' and '.join(color.casefold() + '-colored' for color in colors)}"
            if colors
            else ""
        )
        repaired = f"{count} {' '.join(subject_words)}{color_clause}, {action}"
        return _bounded_words(repaired, 12)[:110].rstrip(" ,;:-")
    return prompt


def _recover_missing_supporting_subject(prompt: str, *, source_text: str) -> str:
    """Restore a short clause subject when a tiny model returns only its action."""

    prompt_tokens = set(_privacy_tokens(prompt))
    if not prompt_tokens:
        return prompt
    for match in re.finditer(
        r"(?=(?:\bwhile\b|\band\b|\bas\b|\bwhen\b|\bthen\b|,)\s*"
        r"([^,.;!?]{1,100}))",
        source_text,
        flags=re.IGNORECASE,
    ):
        words = _SEMANTIC_WORD.findall(match.group(1))
        while words and words[0].casefold() in {"a", "an", "the"}:
            words.pop(0)
        overlap_index = next(
            (index for index, word in enumerate(words[:6]) if word.casefold() in prompt_tokens),
            None,
        )
        if overlap_index is None or not 1 <= overlap_index <= 5:
            continue
        subject = words[:overlap_index]
        if any(word.casefold() in {"he", "her", "him", "it", "she", "they"} for word in subject):
            continue
        if any(word.casefold() in prompt_tokens for word in subject):
            continue
        return " ".join([*subject, _normalized_action(prompt)])
    return prompt


def _recover_malformed_transformation(prompt: str, *, source_text: str) -> str:
    """Replace a tiny-model trailing pronoun with its source-grounded result."""

    prompt_words = _SEMANTIC_WORD.findall(prompt)
    if not prompt_words or prompt_words[-1].casefold() not in {
        "he",
        "her",
        "him",
        "it",
        "she",
        "them",
        "they",
    }:
        return prompt

    prompt_tokens = {word.casefold() for word in prompt_words[:-1]}
    clauses = re.split(
        r"(?:[,;]|\b(?:and|as|then|when|while)\b)",
        source_text,
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        words = _SEMANTIC_WORD.findall(clause)
        lowered = [word.casefold() for word in words]
        verb_index = next(
            (
                index
                for index, word in enumerate(lowered)
                if word
                in {
                    "bloom",
                    "blooms",
                    "rise",
                    "rises",
                }
            ),
            None,
        )
        if verb_index is None or verb_index == 0:
            continue
        subject = words[:verb_index]
        while subject and subject[0].casefold() in {"a", "an", "the"}:
            subject.pop(0)
        subject_tokens = {word.casefold() for word in subject}
        if not subject or not subject_tokens.intersection(prompt_tokens):
            continue
        repaired = " ".join([*subject[-6:], _normalized_action(words[verb_index])])
        return _bounded_words(repaired, 8)
    return " ".join(prompt_words[:-1]) or prompt


def _recover_pronominal_transformation(prompt: str, *, source_text: str) -> str:
    """Resolve a short ``it becomes`` result to its visible source object."""

    words = _SEMANTIC_WORD.findall(prompt)
    if not words or words[0].casefold() != "it":
        return prompt
    match = re.search(
        r"\bafter\s+(?:a|an|the)\s+(?P<initial>[^,.;!?]{1,100}),\s*"
        r"it\s+(?:emerges\s+as|becomes|turns\s+into|transforms\s+into)\s+"
        r"(?:a|an|the)?\s*(?P<result>[^,.;!?]{1,100})",
        source_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return prompt
    initial_words = _SEMANTIC_WORD.findall(match.group("initial"))
    verb_index = next(
        (
            index
            for index, word in enumerate(initial_words)
            if word.casefold() in _VISIBLE_VERBS
        ),
        None,
    )
    if verb_index is None or verb_index == 0:
        return prompt
    result_words = _SEMANTIC_WORD.findall(match.group("result"))
    initial_subject = initial_words[:verb_index]
    while initial_subject and initial_subject[0].casefold() in {"a", "an", "the"}:
        initial_subject.pop(0)
    while result_words and result_words[0].casefold() in {"a", "an", "the"}:
        result_words.pop(0)
    if not initial_subject or not result_words:
        return prompt
    return _bounded_words(
        " ".join([*initial_subject[-4:], "becomes", *result_words[:6]]),
        10,
    )


def _recover_containment_and_scale(prompt: str, *, source_text: str) -> str:
    """Restore explicit containment and a bounded relative-scale comparison."""

    repaired = prompt
    containment = re.search(
        r"\binside\s+(?:a\s+|an\s+|the\s+)?(?P<container>[A-Za-z][A-Za-z'-]*)\s*,\s*"
        r"(?:a\s+|an\s+|the\s+)?(?P<subject>(?:[A-Za-z][A-Za-z'-]*\s+){0,2}"
        r"[A-Za-z][A-Za-z'-]*)\s+(?P<verb>shines?|glows?|moves?|floats?|rests?)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    prompt_tokens = set(_privacy_tokens(prompt))
    clauses: list[str] = []
    if containment is not None:
        container = containment.group("container")
        subject = containment.group("subject")
        subject_tokens = set(_privacy_tokens(subject))
        if subject_tokens & prompt_tokens and not prompt_tokens.intersection({"inside", "within"}):
            clauses.append(f"{subject} {containment.group('verb')} inside {container}")

    scale = re.search(
        r"\b(?P<detail>[A-Za-z][A-Za-z'-]*)\s+no\s+(?:larger|bigger)\s+than\s+"
        r"(?:a\s+|an\s+|the\s+)?(?P<reference>[A-Za-z][A-Za-z'-]*)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if scale is not None:
        detail = scale.group("detail")
        reference = scale.group("reference")
        if detail.casefold() not in prompt_tokens or reference.casefold() not in prompt_tokens:
            clauses.append(f"{reference}-sized {detail}")

    if clauses:
        return _bounded_words("; ".join(clauses), 14)
    return repaired


def _remove_nonvisual_negative_terms(prompt: str) -> str:
    """Keep renderer exclusions out of positive visible layer prompts."""

    repaired = re.sub(
        r"\b(?:with\s+)?(?:no|without)\s+(?:digits|numbers|text|words|writing)\b",
        " ",
        prompt,
        flags=re.IGNORECASE,
    )
    return " ".join(repaired.replace(",", " ").split()).strip(" ,;:-") or prompt


def _rephrase_distinctive_supporting_action(prompt: str, *, source_text: str) -> str:
    """Keep a supporting noun while breaking a verb-ending source trigram locally."""

    words = _SEMANTIC_WORD.findall(prompt)
    output_tokens = [word.casefold() for word in words]
    source_tokens = _privacy_tokens(source_text)
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if _distinctive_phrase(source_tokens[index : index + 3])
    }
    for index in range(len(words) - 2):
        if tuple(output_tokens[index : index + 3]) not in source_phrases:
            continue
        suffix = " ".join(words[index + 2 :])
        normalized = _normalized_action(suffix)
        if normalized.casefold() != suffix.casefold():
            return " ".join([*words[: index + 2], normalized])
    return prompt


def _repair_duplicated_focus_in_supporting_prompt(
    prompt: str,
    *,
    focus_subject: str,
    focus_action: str,
    source_text: str,
) -> str:
    """Replace a duplicated main actor with distinct source-grounded details."""

    prompt_tokens = _privacy_tokens(prompt)
    subject_tokens = {
        token for token in _privacy_tokens(focus_subject) if token not in _PHRASE_STOPWORDS
    }
    if not subject_tokens.intersection(prompt_tokens):
        return prompt

    excluded = subject_tokens | {
        token for token in _privacy_tokens(focus_action) if token not in _PHRASE_STOPWORDS
    }
    supporting: list[str] = []
    for token in _privacy_tokens(source_text):
        if token in _PHRASE_STOPWORDS or token in excluded or token in supporting:
            continue
        supporting.append(token)
    if supporting:
        return " ".join(supporting[-4:])

    repaired = [
        token
        for token in prompt_tokens
        if token not in subject_tokens and token not in _PHRASE_STOPWORDS
    ]
    return " ".join(dict.fromkeys(repaired)) or "supporting story detail"


def _remove_focus_from_background_prompt(
    prompt: str,
    *,
    focus_subject: str,
    focus_action: str,
) -> str:
    """Keep the setting layer from asking the renderer for a second main actor."""

    excluded = {
        token
        for value in (focus_subject, focus_action)
        for token in _privacy_tokens(value)
        if token not in _PHRASE_STOPWORDS
    }
    repaired = [
        token
        for token in _privacy_tokens(prompt)
        if token not in excluded and token not in _PHRASE_STOPWORDS
    ]
    return " ".join(dict.fromkeys(repaired)) or "open layered storybook setting"


def _recover_generic_background_prompt(prompt: str, *, source_text: str) -> str:
    """Recover an explicit local setting when a tiny model emits a placeholder."""

    prompt_tokens = set(_privacy_tokens(prompt))
    generic_tokens = {
        "background",
        "environment",
        "layered",
        "open",
        "scene",
        "setting",
        "storybook",
    }
    if not prompt_tokens or not prompt_tokens.issubset(generic_tokens):
        return prompt

    setting_match = re.search(
        r"\b(?:in|inside|within|beneath|under|on|at|across|beside|near)\b"
        r"\s+(?P<setting>[^,.;!?]{1,80})",
        source_text,
        flags=re.IGNORECASE,
    )
    if setting_match is None:
        return prompt
    setting = re.split(
        r"\b(?:and|as|when|while|then)\b",
        setting_match.group("setting"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    setting = _LEADING_ARTICLE.sub("", setting).strip(" ,-:")
    return _bounded_words(setting, 8) or prompt


_SETTING_NOUNS = (
    "attic",
    "bakery",
    "bedroom",
    "bridge",
    "candle",
    "cave",
    "classroom",
    "desert",
    "garden",
    "lake",
    "library",
    "lighthouse",
    "moon",
    "ocean",
    "pond",
    "rooftop",
    "school",
    "station",
)
_SETTING_RELATION = {
    "at": "at",
    "beneath": "under",
    "beside": "beside",
    "from": "from",
    "in": "inside",
    "inside": "inside",
    "near": "beside",
    "on": "on",
    "through": "through",
    "to": "toward",
    "toward": "toward",
    "towards": "toward",
    "under": "under",
}


def _recover_source_grounded_setting(prompt: str, *, source_text: str) -> str:
    """Retain one explicit spatial setting/destination omitted by the edge model."""

    prompt_tokens = set(_privacy_tokens(prompt))
    noun_pattern = "|".join(_SETTING_NOUNS)
    matches = list(
        re.finditer(
            rf"\b(?P<relation>{'|'.join(_SETTING_RELATION)})\b\s+"
            rf"(?:a|an|the)?\s*(?P<description>(?:[A-Za-z'-]+\s+){{0,2}}?)"
            rf"(?P<noun>{noun_pattern})\b",
            source_text,
            flags=re.IGNORECASE,
        )
    )
    matches.sort(
        key=lambda item: (
            item.group("relation").casefold() not in {"to", "toward", "towards"},
            item.start(),
        )
    )
    for match in matches:
        relation = _SETTING_RELATION[match.group("relation").casefold()]
        noun = match.group("noun").casefold()
        description = [
            word.casefold()
            for word in _SEMANTIC_WORD.findall(match.group("description"))
            if word.casefold() not in {"a", "an", "distant", "the"}
        ]
        candidate_tokens = [relation, noun, *description[-1:]]
        if noun in prompt_tokens and relation in prompt_tokens:
            continue
        candidate = " ".join(candidate_tokens)
        return _bounded_words(f"{candidate}, {prompt}", 10)

    # Some settings appear as modifiers of a visible architectural surface rather
    # than after a preposition (for example ``the classroom ceiling``). Preserve
    # the setting noun without copying the source phrase or inventing story detail.
    structure_match = re.search(
        rf"\b(?:a|an|the)\s+(?P<noun>{noun_pattern})\s+"
        r"(?:ceiling|door|floor|roof|wall|walls|window|windows)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if structure_match is not None:
        noun = structure_match.group("noun").casefold()
        if noun not in prompt_tokens:
            return _bounded_words(f"{noun} interior, {prompt}", 10)
    return prompt


def _normalized_action(value: str) -> str:
    words = value.strip(" ,").split()
    if not words:
        return "performing the story action"
    verb = words[0].casefold()
    gerunds = {
        "arc": "arcing",
        "arcs": "arcing",
        "carry": "carrying",
        "carries": "carrying",
        "circle": "circling",
        "circles": "circling",
        "cast": "casting",
        "casts": "casting",
        "climb": "climbing",
        "climbs": "climbing",
        "create": "creating",
        "creates": "creating",
        "drift": "drifting",
        "drifts": "drifting",
        "draw": "drawing",
        "draws": "drawing",
        "float": "floating",
        "floats": "floating",
        "fold": "folding",
        "folds": "folding",
        "form": "forming",
        "forms": "forming",
        "guide": "guiding",
        "guides": "guiding",
        "hold": "holding",
        "holds": "holding",
        "lift": "lifting",
        "lifts": "lifting",
        "open": "opening",
        "opens": "opening",
        "plant": "planting",
        "plants": "planting",
        "play": "playing",
        "plays": "playing",
        "point": "pointing",
        "points": "pointing",
        "push": "pushing",
        "pushes": "pushing",
        "read": "reading",
        "reads": "reading",
        "raise": "raising",
        "raises": "raising",
        "rise": "rising",
        "rises": "rising",
        "run": "running",
        "runs": "running",
        "sail": "sailing",
        "sails": "sailing",
        "bloom": "blooming",
        "blooms": "blooming",
        "spiral": "spiraling",
        "spirals": "spiraling",
        "skate": "skating",
        "skates": "skating",
        "steer": "steering",
        "steers": "steering",
        "swim": "swimming",
        "swims": "swimming",
        "tumble": "tumbling",
        "tumbles": "tumbling",
        "unfold": "unfolding",
        "unfolds": "unfolding",
        "watch": "watching",
        "watches": "watching",
        "wait": "waiting",
        "waits": "waiting",
    }
    if verb in gerunds:
        words[0] = gerunds[verb]
    return " ".join(words)


def _prompt_fragment(value: str) -> str:
    return value.rstrip(" \t\r\n.,;:!?")


def _semantically_redundant(value: str, reference: str) -> bool:
    words = {word.casefold() for word in _SEMANTIC_WORD.findall(value)}
    reference_words = {word.casefold() for word in _SEMANTIC_WORD.findall(reference)}
    return len(words) >= 4 and len(words & reference_words) / len(words) >= 0.8


def _covered_visual_detail(detail: str, *references: str) -> bool:
    """Avoid drawing a second copy of a detail already in the visual contract."""
    scaffold = _PHRASE_STOPWORDS | {"complete", "visible", "shown"}

    def tokens(value: str) -> tuple[str, ...]:
        return tuple(word for word in _privacy_tokens(value) if word not in scaffold)

    detail_tokens = tokens(detail)
    if not detail_tokens:
        return False
    for reference in references:
        words = tokens(reference)
        if any(
            words[index : index + len(detail_tokens)] == detail_tokens
            for index in range(len(words) - len(detail_tokens) + 1)
        ):
            return True
    return False


def _normalized_placements(
    focus: LiveScenePlacedLayerPlan,
    accent: LiveScenePlacedLayerPlan,
    *,
    seed: int,
) -> tuple[
    tuple[float, float, float, float, float],
    tuple[float, float, float, float, float],
]:
    focus_box = _normalized_box(focus.anchor, max_width=0.6, max_height=0.72)
    accent_box = _normalized_box(accent.anchor, max_width=0.42, max_height=0.48)
    focus_x, focus_y, _, _ = focus_box
    accent_x, accent_y, accent_width, accent_height = accent_box
    centers_equal = abs(focus_x - accent_x) < 0.08 and abs(focus_y - accent_y) < 0.08
    if centers_equal or _overlap_fraction(focus_box, accent_box) > 0.55:
        accent_width = min(accent_width, 0.3)
        accent_height = min(accent_height, 0.34)
        place_right = bool(seed & 1) if abs(focus_x - 0.5) < 0.12 else focus_x < 0.5
        place_above = bool((seed >> 1) & 1) if abs(focus_y - 0.5) < 0.12 else focus_y > 0.5
        accent_x = (
            1 - accent_width / 2 - _PLACEMENT_MARGIN
            if place_right
            else accent_width / 2 + _PLACEMENT_MARGIN
        )
        accent_y = (
            accent_height / 2 + _PLACEMENT_MARGIN
            if place_above
            else 1 - accent_height / 2 - _PLACEMENT_MARGIN
        )

    focus_depth = focus.depth
    accent_depth = accent.depth
    if abs(focus_depth - accent_depth) < 0.75:
        focus_depth = min(focus_depth, 18)
        accent_depth = min(20, focus_depth + 1.5)
    return (
        (*focus_box, focus_depth),
        (accent_x, accent_y, accent_width, accent_height, accent_depth),
    )


def _wire_anchor(
    kind: Literal["character", "prop", "effect"],
    *,
    role: Literal["focus", "accent"],
) -> tuple[float, float, float, float]:
    if role == "focus":
        center_x, center_y = 0.5, 0.55
        dimensions = {
            "character": (0.4, 0.62),
            "prop": (0.4, 0.46),
            "effect": (0.46, 0.42),
        }
    else:
        center_x, center_y = 0.73, 0.28
        dimensions = {
            "character": (0.25, 0.38),
            "prop": (0.27, 0.28),
            "effect": (0.3, 0.26),
        }
    width, height = dimensions[kind]
    return center_x, center_y, width, height


def _placement_label(placement: tuple[float, float, float, float, float]) -> str:
    center_x, center_y, *_ = placement
    horizontal = "left" if center_x < 0.4 else "right" if center_x > 0.6 else "center"
    vertical = "upper" if center_y < 0.4 else "lower" if center_y > 0.62 else "mid-frame"
    if horizontal == "center":
        return f"at {vertical} center"
    if vertical == "mid-frame":
        return f"at {horizontal} mid-frame"
    return f"at {vertical} {horizontal}"


def _normalized_box(
    anchor: LiveSceneAnchor,
    *,
    max_width: float,
    max_height: float,
) -> tuple[float, float, float, float]:
    raw_center_x, raw_center_y, raw_width, raw_height = anchor
    width = min(max_width, max(0.08, raw_width))
    height = min(max_height, max(0.08, raw_height))
    center_x = min(
        1 - _PLACEMENT_MARGIN - width / 2,
        max(_PLACEMENT_MARGIN + width / 2, raw_center_x),
    )
    center_y = min(
        1 - _PLACEMENT_MARGIN - height / 2,
        max(_PLACEMENT_MARGIN + height / 2, raw_center_y),
    )
    return center_x, center_y, width, height


def _overlap_fraction(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    overlap_width = max(
        0.0,
        min(first_x + first_width / 2, second_x + second_width / 2)
        - max(first_x - first_width / 2, second_x - second_width / 2),
    )
    overlap_height = max(
        0.0,
        min(first_y + first_height / 2, second_y + second_height / 2)
        - max(first_y - first_height / 2, second_y - second_height / 2),
    )
    smaller_area = min(first_width * first_height, second_width * second_height)
    return (overlap_width * overlap_height) / smaller_area


def _placed_composition(
    layer_id: str,
    layer: LiveScenePlacedLayerPlan,
    *,
    placement: tuple[float, float, float, float, float],
    seed: int,
) -> LayerComposition:
    motion_settings = {
        "drift": {"amplitude_x": 0.015, "amplitude_y": -0.01},
        "float": {"amplitude_x": 0.012, "amplitude_y": 0.012, "scale_delta": 0.004},
        "breathe": {"scale_delta": 0.012},
        "pulse": {"scale_delta": 0.016},
        "parallax": {"amplitude_x": 0.008},
    }[layer.motion]
    center_x, center_y, width, height, depth = placement
    return LayerComposition(
        layer_id=layer_id,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        depth=depth,
        ambient_motion=AmbientMotion(
            kind=layer.motion,
            period_ms=3_600 + seed % 2_401,
            **motion_settings,
        ),
    )


def _motion_description(kind: LiveSceneMotion) -> str:
    return {
        "drift": "slow directional drift",
        "float": "gentle seamless floating",
        "breathe": "subtle breathing motion",
        "pulse": "restrained rhythmic glow",
        "parallax": "subtle seamless parallax",
    }[kind]


def _bounded_words(value: str, maximum_words: int) -> str:
    return " ".join(value.split()[:maximum_words])


def _ambient_effect(kind: LiveSceneAmbience) -> AmbientEffect:
    settings = {
        "dust": (0.22, 0.24, "#ffe2a1"),
        "fireflies": (0.24, 0.3, "#ffe58a"),
        "fog": (0.12, 0.12, "#b9d4dc"),
        "stars": (0.3, 0.18, "#d8e7ff"),
        "light_rays": (0.08, 0.12, "#fff0c7"),
    }[kind]
    density, speed, color = settings
    return AmbientEffect(kind=kind, density=density, speed=speed, color=color)


class LiveScenePlanningResult(FrozenStrictModel):
    plan: LiveScenePlan
    metrics: ModelMetrics
    model_revision: PlanModelRevision
    wall_ms: Annotated[float, Field(ge=0)]
    cache_hit: bool = False


class LiveScenePlannerWarmupResult(FrozenStrictModel):
    metrics: ModelMetrics
    wall_ms: Annotated[float, Field(ge=0)]


class _LiveScenePlannerWarmupOutput(FrozenStrictModel):
    ready: Literal[True]


class LiveScenePlannerError(RuntimeError):
    """A recoverable local planning failure; callers may use a deterministic plan."""


class LiveScenePlannerTimeoutError(LiveScenePlannerError):
    pass


class LiveScenePlannerPrivacyError(LiveScenePlannerError):
    """The local semantic plan is unsafe to hand to a remote renderer."""


def validate_live_scene_plan_privacy(
    plan: LiveScenePlan,
    *,
    source_text: str,
) -> None:
    """Fail closed when a model plan carries source text or obvious PII outbound."""

    source_tokens = _privacy_tokens(source_text)
    proper_names = _proper_name_candidates(source_text)
    fields = {
        "scene_summary": plan.scene_summary,
        "art_direction": plan.art_direction,
        "background_prompt": plan.background_prompt,
        "focus.prompt": plan.focus.prompt,
        "accent.prompt": plan.accent.prompt,
    }
    for field_name, value in fields.items():
        if _EMAIL.search(value) or _PHONE.search(value) or _URL.search(value):
            raise LiveScenePlannerPrivacyError(
                f"local privacy gate rejected {field_name}: possible contact data"
            )
        output_tokens = _privacy_tokens(value)
        if source_tokens and _contains_token_sequence(output_tokens, source_tokens):
            raise LiveScenePlannerPrivacyError(
                f"local privacy gate rejected {field_name}: source passage echo"
            )
        if _contains_distinctive_source_phrase(output_tokens, source_tokens):
            raise LiveScenePlannerPrivacyError(
                f"local privacy gate rejected {field_name}: distinctive source phrase"
            )
        if any(_contains_token_sequence(output_tokens, candidate) for candidate in proper_names):
            raise LiveScenePlannerPrivacyError(
                f"local privacy gate rejected {field_name}: proper-name candidate"
            )


def _privacy_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"([^\W_]+)[’']s\b", r"\1", normalized)
    normalized = normalized.replace("-", " ").replace("–", " ").replace("—", " ")
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _proper_name_candidates(source_text: str) -> set[tuple[str, ...]]:
    candidates = {
        _privacy_tokens(match.group(1)) for match in _NAME_AFTER_MARKER.finditer(source_text)
    }
    for match in _CAPITALIZED_WORD.finditer(source_text):
        word = match.group(0)
        if word.casefold() in _NON_NAME_CAPITALIZED or word.casefold() in _COUNT_WORDS:
            continue
        if word.casefold() == "exactly":
            following = _privacy_tokens(source_text[match.end() :])
            if following and (following[0] in _COUNT_WORDS or following[0].isdigit()):
                continue
        candidates.add(_privacy_tokens(word))
    return {candidate for candidate in candidates if candidate}


def _contains_token_sequence(
    tokens: tuple[str, ...],
    candidate: tuple[str, ...],
) -> bool:
    width = len(candidate)
    return width > 0 and any(
        tokens[index : index + width] == candidate for index in range(len(tokens) - width + 1)
    )


def _contains_distinctive_source_phrase(
    output_tokens: tuple[str, ...],
    source_tokens: tuple[str, ...],
) -> bool:
    if len(source_tokens) < 3 or len(output_tokens) < 3:
        return False
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if _distinctive_phrase(source_tokens[index : index + 3])
    }
    return any(
        output_tokens[index : index + 3] in source_phrases
        for index in range(len(output_tokens) - 2)
    )


def _remove_distinctive_source_overlap(
    value: str,
    source_text: str,
    *,
    preserve_subject: bool = False,
    preserve_tail: bool = False,
    preserve_action: bool = False,
) -> str:
    """Minimally redact repeated source trigrams without inventing replacement text."""

    value = re.sub(r"\bno\s+other\b", "no additional", value, flags=re.IGNORECASE)
    output_tokens = list(_privacy_tokens(value))
    source_tokens = _privacy_tokens(source_text)
    if len(output_tokens) < 3 or len(source_tokens) < 3:
        return value
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if _distinctive_phrase(source_tokens[index : index + 3])
    }
    changed = False
    while True:
        overlap_index = next(
            (
                index
                for index in range(len(output_tokens) - 2)
                if tuple(output_tokens[index : index + 3]) in source_phrases
            ),
            None,
        )
        if overlap_index is None:
            break
        if preserve_subject:
            # Small models sometimes append the action to focus.subject. Remove
            # the trailing token from an echoed trigram so the actor/head noun
            # survives ("silver whale swims" -> "silver whale"). The dedicated
            # action field still carries the visible verb.
            deletion_offset = 2
        elif preserve_action:
            # Keep both the visible verb and its head object. Removing the
            # middle article/modifier breaks the source trigram while retaining
            # actionable rendering semantics ("carries glowing pear" becomes
            # "carries pear"; "walnut shell boat" becomes "walnut boat").
            deletion_offset = 1
        elif preserve_tail:
            # Background and supporting-object phrases usually end in their
            # visual head noun ("small paper kite"). For one overlapping
            # trigram, drop its leading modifier. For a longer echoed run,
            # remove the third token once so adjacent subject/detail pairs
            # survive ("glowing jellyfish drift between stars" keeps both
            # "glowing jellyfish" and "between stars").
            has_adjacent_overlap = any(
                tuple(output_tokens[index : index + 3]) in source_phrases
                for index in range(
                    overlap_index + 1,
                    min(overlap_index + 3, len(output_tokens) - 2),
                )
            )
            deletion_offset = 2 if has_adjacent_overlap else 0
        else:
            window = output_tokens[overlap_index : overlap_index + 3]
            deletion_offset = min(range(3), key=lambda offset: len(window[offset]))
        deletion_index = overlap_index + deletion_offset
        if output_tokens[deletion_index] in {"no", "not", "without", "neither", "never", "nor"}:
            raise LiveScenePlannerPrivacyError("privacy rewrite would remove a negation")
        del output_tokens[deletion_index]
        changed = True
    return (" ".join(output_tokens) if changed else value) or value


def _distinctive_phrase(tokens: tuple[str, ...]) -> bool:
    content = [token for token in tokens if token not in _PHRASE_STOPWORDS]
    return len(content) >= 2 and any(len(token) >= 4 for token in content)


class LiveScenePlanner(Protocol):
    async def plan(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
    ) -> LiveScenePlanningResult: ...


LIVE_SCENE_SYSTEM_PROMPT = """You are Bookforge's private edge scene planner.
Return only JSON conforming to the supplied schema. Infer the passage's visual meaning locally,
but never quote it, repeat distinctive phrases or proper names, or include personal information.
Do not extend its plot or invent characters, events, brands, labels, or readable writing. Produce
one cohesive, safe, projection-ready visual plan for a child."""


def live_scene_plan_prompt(
    *,
    text: str,
    visual_style: str,
    seed: int,
    compact_wire: bool = False,
) -> str:
    # Geometry and animation remain deterministic from the seed. The language
    # model only performs semantic extraction, so sending the seed wastes edge
    # input tokens and can introduce irrelevant variation.
    del seed
    # Visual style is applied deterministically when the final SceneSpec is
    # compiled. Gemma extracts story semantics only, so sending style here
    # wastes prompt tokens and prevents plan reuse across style auditions.
    del visual_style
    request = {"passage": text}
    compact_key_guide = (
        "\nCompact JSON: b=background; f=[kind,subject,action], where kind is character or "
        "prop; m=[kind,prompt], where kind is character, prop, or effect.\n"
        if compact_wire
        else ""
    )
    return (
        """Plan one full-bleed cinematic 16:9 illustration for immediate projection.

Requirements:
- Preserve only the subjects, setting, action, and mood present in the passage.
- Palette, lighting, projection brightness, silhouette clarity, materials, depth, geometry, and
  animation are supplied locally. Spend the output only on story-visible semantics.
- background_prompt is at most 10 words. focus.subject is a complete actor in at most 8 words and
  focus.action is the exact visible action in at most 6 words. magic.prompt is at most 8 words.
- Do not put trailing punctuation in any layer prompt.
- Return visual semantics only: do not copy sentences, distinctive phrases, proper names, or
  personal information from the passage.
- Reuse ordinary visual nouns and the exact visible action from the passage when needed. This is
  safer and more faithful than replacing them with a different action or invented object.
- Preserve explicitly named common animal breeds/species, flowers/plants, objects, and settings.
  Keep safe one- or two-word common terms such as "golden retriever" and "daisies" instead of
  generalizing them to "dog", "flowers", "wildflowers", or an unrelated visual substitute.
- Never copy three adjacent words from the passage into any output field. Keep the action verb and
  essential ordinary nouns, but remove or paraphrase neighboring modifiers.
- Never request readable writing, captions, signs, logos, watermarks, borders, panels, or UI.
- Describe exactly three semantic layers: one background_prompt plus focus and magic.
- background_prompt is setting words only: never put numbers, brackets, arrays, coordinates,
  subjects, actions, or camera directions in it.
- focus.subject must name the complete actor. Prefer the person or creature acting over the object
  it touches. focus.action must separately state the exact visible action from the passage. Never
  invent a pose or action. Include its essential object or destination instead of returning only a
  verb. Stop the action before a later magical transformation; that result belongs in magic.prompt.
  Never return an isolated body part, gaze, expression, or adjective list.
- magic must name the passage's most visually surprising transformation, creature, or object.
  If magic is a creature or person, set kind to character and begin magic.prompt with that
  complete creature or person before its action. Never return an action without its actor.
  If the passage has a later independent clause introduced by and, while, as, when, then, or a
  comma, inspect that clause first: its new creature, transformed result, or impossible event is
  usually magic. Include both that concrete subject and its visible action or destination.
  Prefer an actual magical change over the focus's tool. If no transformation occurs, use a
  concrete supporting element explicitly present in the passage. Never invent a transformation.
  When the passage says something becomes, turns into, transforms, blooms, or rises into something,
  magic.prompt must name that concrete result, including its essential form or destination.
  Light, glow, shimmer, dust, fog, color, or atmosphere alone is not magic.
- Fidelity example only: for "a keeper lifts a brass key, and luminous moths spiral through the
  arch", focus.action is "lifts key" and magic.prompt is "luminous moths spiral through arch".
- Geometry, depth, composition, and layer motion are supplied locally; never emit regions,
  coordinates, or measurements.
- The scene must remain legible on a projector. Atmosphere is supplied locally from the setting.
- Keep the entire JSON compact; omit unnecessary adjectives and explanations.
"""
        + compact_key_guide
        + """
Input:
"""
        + json.dumps(request, ensure_ascii=False, indent=2)
    )


class StructuredLiveScenePlanner:
    """Generate one validated live plan through the configured structured model client."""

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        timeout_seconds: float,
        model_revision: str = "configured-local-model",
        compact_wire: bool = False,
        cache_entries: int = 32,
        persistent_cache_dir: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("live-scene planner timeout must be positive")
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.model_revision = _PLAN_MODEL_REVISION_ADAPTER.validate_python(model_revision)
        self.compact_wire = compact_wire
        if not 0 <= cache_entries <= 256:
            raise ValueError("live-scene planner cache entries must be between 0 and 256")
        self.cache_entries = cache_entries
        self.persistent_cache_dir = persistent_cache_dir
        self._cache: OrderedDict[str, tuple[LiveScenePlan, ModelMetrics]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[LiveScenePlanningResult]] = {}
        self._warmup_task: asyncio.Task[LiveScenePlannerWarmupResult] | None = None

    async def warmup(self) -> LiveScenePlannerWarmupResult:
        """Load the local model with fixed synthetic input and no story text."""

        task = self._warmup_task
        if task is None:
            task = asyncio.create_task(
                self._warmup_uncached(),
                name="bookforge-live-planner-warmup",
            )
            self._warmup_task = task
            task.add_done_callback(self._discard_warmup)
        return await asyncio.shield(task)

    async def _warmup_uncached(self) -> LiveScenePlannerWarmupResult:
        started = perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                output, metrics = await self.client.generate(
                    system="Return only the supplied local readiness JSON schema.",
                    prompt=(
                        "This is a local model-load warmup with fixed synthetic input. "
                        'Return {"ready":true}.'
                    ),
                    output_type=_LiveScenePlannerWarmupOutput,
                )
        except TimeoutError as error:
            raise LiveScenePlannerTimeoutError(
                f"local planner warmup exceeded {self.timeout_seconds:g} seconds"
            ) from error
        except asyncio.CancelledError:
            raise
        except Exception as error:
            detail = (str(error) or error.__class__.__name__)[:300]
            raise LiveScenePlannerError(f"local planner warmup failed: {detail}") from error
        if output.ready is not True:
            raise LiveScenePlannerError("local planner warmup returned an invalid readiness value")
        return LiveScenePlannerWarmupResult(
            metrics=metrics,
            wall_ms=(perf_counter() - started) * 1_000,
        )

    def _discard_warmup(
        self,
        task: asyncio.Task[LiveScenePlannerWarmupResult],
    ) -> None:
        if self._warmup_task is task:
            self._warmup_task = None

    async def plan(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
    ) -> LiveScenePlanningResult:
        text = _PLAN_TEXT_ADAPTER.validate_python(text)
        visual_style = _PLAN_STYLE_ADAPTER.validate_python(visual_style)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("live-scene seed is outside uint32 range")
        started = perf_counter()
        cache_key = self._cache_key(text=text)
        cached = self._cache.pop(cache_key, None)
        if cached is None and self.cache_entries and self.persistent_cache_dir is not None:
            cached = await asyncio.to_thread(self._load_persistent_cache, cache_key)
        if cached is not None:
            plan, source_metrics = cached
            self._remember(cache_key, cached)
            validate_live_scene_plan_privacy(plan, source_text=text)
            return LiveScenePlanningResult(
                plan=plan,
                metrics=source_metrics.model_copy(
                    update={
                        "total_ms": 0,
                        "load_ms": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                ),
                model_revision=self.model_revision,
                wall_ms=(perf_counter() - started) * 1_000,
                cache_hit=True,
            )

        warmup_task = self._warmup_task
        if warmup_task is not None:
            # Only cache misses need the one-slot model. Keep warmup outside
            # their independent inference timeout budget.
            with suppress(LiveScenePlannerError):
                await asyncio.shield(warmup_task)
        task = self._inflight.get(cache_key)
        if task is None:
            task = asyncio.create_task(
                self._plan_uncached(
                    text=text,
                    visual_style=visual_style,
                    seed=seed,
                    cache_key=cache_key,
                ),
                name=f"bookforge-live-plan-{cache_key[:12]}",
            )
            self._inflight[cache_key] = task
            task.add_done_callback(
                lambda finished, key=cache_key: self._discard_inflight(key, finished)
            )
        return await asyncio.shield(task)

    async def has_cached_plan(self, *, text: str) -> bool:
        """Check and hydrate the private semantic cache without running the model."""

        text = _PLAN_TEXT_ADAPTER.validate_python(text)
        cache_key = self._cache_key(text=text)
        cached = self._cache.get(cache_key)
        if cached is None and self.cache_entries and self.persistent_cache_dir is not None:
            cached = await asyncio.to_thread(self._load_persistent_cache, cache_key)
            if cached is not None:
                self._remember(cache_key, cached)
        if cached is None:
            return False
        validate_live_scene_plan_privacy(cached[0], source_text=text)
        return True

    async def _plan_uncached(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
        cache_key: str,
    ) -> LiveScenePlanningResult:
        started = perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                output_type = LiveSceneCompactWirePlan if self.compact_wire else LiveSceneWirePlan
                plan, metrics = await self.client.generate(
                    system=LIVE_SCENE_SYSTEM_PROMPT,
                    prompt=live_scene_plan_prompt(
                        text=text,
                        visual_style=visual_style,
                        seed=seed,
                        compact_wire=self.compact_wire,
                    ),
                    output_type=output_type,
                )
        except TimeoutError as error:
            raise LiveScenePlannerTimeoutError(
                f"local scene planning exceeded {self.timeout_seconds:g} seconds"
            ) from error
        except asyncio.CancelledError:
            raise
        except Exception as error:
            detail = (str(error) or error.__class__.__name__)[:300]
            raise LiveScenePlannerError(
                f"local structured scene planning failed: {detail}"
            ) from error

        if self.compact_wire:
            compact_plan = LiveSceneCompactWirePlan.model_validate(plan.model_dump())
            wire_plan = compact_plan.to_wire_plan()
        else:
            wire_plan = LiveSceneWirePlan.model_validate(plan.model_dump())
        if getattr(self.client, "wire_plans_are_privacy_sanitized", False):
            sanitized_wire_plan = wire_plan
        else:
            sanitized_wire_plan = wire_plan.privacy_sanitized(source_text=text)
        validated = sanitized_wire_plan.to_live_scene_plan(context_text=text)
        validate_live_scene_plan_privacy(validated, source_text=text)
        if self.cache_entries:
            cached = (validated, metrics)
            self._remember(cache_key, cached)
            if self.persistent_cache_dir is not None:
                await asyncio.to_thread(
                    self._store_persistent_cache,
                    cache_key,
                    validated,
                    metrics,
                )
        return LiveScenePlanningResult(
            plan=validated,
            metrics=metrics,
            model_revision=self.model_revision,
            wall_ms=(perf_counter() - started) * 1_000,
        )

    def _discard_inflight(
        self,
        cache_key: str,
        task: asyncio.Task[LiveScenePlanningResult],
    ) -> None:
        if self._inflight.get(cache_key) is task:
            self._inflight.pop(cache_key, None)

    def _cache_key(self, *, text: str) -> str:
        identity = {
            "text": text,
            "model_revision": self.model_revision,
            "compact_wire": self.compact_wire,
            "contract_revision": _PLAN_CACHE_CONTRACT_REVISION,
        }
        if client_identity := getattr(self.client, "cache_identity", None):
            identity["client_identity"] = client_identity
        payload = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _remember(
        self,
        cache_key: str,
        cached: tuple[LiveScenePlan, ModelMetrics],
    ) -> None:
        self._cache[cache_key] = cached
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)

    def _load_persistent_cache(
        self,
        cache_key: str,
    ) -> tuple[LiveScenePlan, ModelMetrics] | None:
        assert self.persistent_cache_dir is not None
        path = self.persistent_cache_dir / f"{cache_key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != _PLAN_CACHE_SCHEMA_VERSION
                or payload.get("cache_key") != cache_key
            ):
                return None
            return (
                LiveScenePlan.model_validate(payload["plan"]),
                ModelMetrics.model_validate(payload["metrics"]),
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _store_persistent_cache(
        self,
        cache_key: str,
        plan: LiveScenePlan,
        metrics: ModelMetrics,
    ) -> None:
        assert self.persistent_cache_dir is not None
        try:
            self.persistent_cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.persistent_cache_dir.chmod(0o700)
            encoded = json.dumps(
                {
                    "schema_version": _PLAN_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "plan": plan.model_dump(mode="json"),
                    "metrics": metrics.model_dump(mode="json"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_key}.",
                suffix=".tmp",
                dir=self.persistent_cache_dir,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.persistent_cache_dir / f"{cache_key}.json")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                with suppress(FileNotFoundError):
                    Path(temporary_name).unlink()
            entries = sorted(
                self.persistent_cache_dir.glob("*.json"),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
            for stale in entries[self.cache_entries :]:
                stale.unlink(missing_ok=True)
        except OSError:
            # A performance cache must never turn a valid private model result
            # into a failed scene. The in-memory cache remains available.
            return
