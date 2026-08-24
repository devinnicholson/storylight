"""Strict, latency-bounded structured planning for live generated scenes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import OrderedDict
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
_POSSESSIVE_BODY_FRAGMENT = re.compile(
    r"^(?:a\s+|the\s+)?([A-Za-z][A-Za-z'-]*)['’]s\s+"
    r"(?:hand|hands|face|eyes?|gaze|head)\b",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_SEMANTIC_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_PLACEMENT_MARGIN = 0.04
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
        "but",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
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

    kind: Literal["prop", "effect"]
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
        return self.model_copy(
            update={
                "background_prompt": _remove_distinctive_source_overlap(
                    self.background_prompt, source_text
                ),
                "focus": self.focus.model_copy(
                    update={
                        "subject": _remove_distinctive_source_overlap(
                            self.focus.subject, source_text
                        ),
                        "action": _remove_distinctive_source_overlap(
                            self.focus.action, source_text
                        ),
                    }
                ),
                "magic": self.magic.model_copy(
                    update={
                        "prompt": _remove_distinctive_source_overlap(self.magic.prompt, source_text)
                    }
                ),
            }
        )

    def to_live_scene_plan(self, *, context_text: str = "") -> LiveScenePlan:
        focus_prompt = _normalized_wire_focus(self.focus)
        accent_prompt = _bounded_words(self.magic.prompt, 8)
        return LiveScenePlan(
            scene_summary=_derived_scene_summary(focus_prompt, accent_prompt),
            # The caller already supplies the visual style, including palette
            # and lighting. A fixed projection treatment avoids contradictory
            # tiny-model choices and saves two fields on the critical path.
            art_direction=_wire_art_direction(),
            # Every accepted compact-planner case selected slow_push. Keeping
            # this projection-safe motion local removes an unnecessary decode
            # decision from the latency-critical edge model.
            camera_motion="slow_push",
            background_prompt=_bounded_words(self.background_prompt, 10),
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


class LiveSceneCompactWireFocus(FrozenStrictModel):
    """Alias-key equivalent of :class:`LiveSceneWireFocus` for edge decoding."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    kind: Literal["character", "prop"] = Field(alias="k")
    subject: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
    ] = Field(alias="s")
    action: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=70),
    ] = Field(alias="a")

    def to_wire_focus(self) -> LiveSceneWireFocus:
        return LiveSceneWireFocus(
            kind=self.kind,
            subject=self.subject,
            action=self.action,
        )


class LiveSceneCompactWireMagic(FrozenStrictModel):
    """Alias-key equivalent of :class:`LiveSceneWireMagic` for edge decoding."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    kind: Literal["prop", "effect"] = Field(alias="k")
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=110),
    ] = Field(alias="p")

    def to_wire_magic(self) -> LiveSceneWireMagic:
        return LiveSceneWireMagic(kind=self.kind, prompt=self.prompt)


class LiveSceneCompactWirePlan(FrozenStrictModel):
    """Short-key wire contract for opt-in, lower-latency Jetson inference."""

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
    focus: LiveSceneCompactWireFocus = Field(alias="f")
    magic: LiveSceneCompactWireMagic = Field(alias="m")

    def to_wire_plan(self) -> LiveSceneWirePlan:
        return LiveSceneWirePlan(
            background_prompt=self.background_prompt,
            focus=self.focus.to_wire_focus(),
            magic=self.magic.to_wire_magic(),
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
        background_prompt = _normalized_background_prompt(
            self.background_prompt,
            art_direction=art_direction,
            scene_summary=scene_summary,
        )
        focus_prompt = _bounded_words(self.focus.prompt, 18)
        accent_prompt = _bounded_words(self.accent.prompt, 18)
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
        composition_clause = (
            "Composition: place the main subject "
            f"{_placement_label(focus_placement)}, clearly larger and nearer; place the "
            f"supporting detail {_placement_label(accent_placement)}, smaller and separated."
        )
        master_prompt = (
            f"{_prompt_fragment(visual_style)}. {_prompt_fragment(art_direction)}. "
            f"{background_clause.lstrip()} "
            f"Required foreground subject: {_prompt_fragment(focus_prompt)}. "
            f"Required supporting visual: {_prompt_fragment(accent_prompt)}. "
            "Show the background, subject, and supporting visual simultaneously. "
            f"{composition_clause} "
            "Full-bleed cinematic 16:9 storybook projection with clear foreground/background "
            "depth, clean silhouettes, and no readable text, captions, logos, borders, or UI."
        )
        scene_spec = SceneSpecV2(
            master_prompt=master_prompt,
            negative_prompt=(
                "readable text, letters, words, captions, signs, logo, watermark, interface, "
                "border, split screen, collage, duplicate subject, distorted anatomy"
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
    return _prompt_fragment(_bounded_words(candidate, 18)) or "cinematic storybook setting"


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


def _normalized_wire_focus(layer: LiveSceneWireFocus) -> str:
    subject = _bounded_words(layer.subject, 8)
    action = _normalized_action(_bounded_words(layer.action, 6))
    if layer.kind != "character":
        return f"{subject}, {action}".strip(" ,")
    # Gemma 1B occasionally describes a character through one body fragment.
    # Preserve the model's subject word and action while restoring a complete figure.
    subject = _POSSESSIVE_BODY_FRAGMENT.sub(r"\1", subject, count=1)
    subject = _LEADING_ARTICLE.sub("", subject).strip(" ,")
    return f"a complete visible {subject}, {action}".strip(" ,")


def _normalized_action(value: str) -> str:
    words = value.strip(" ,").split()
    if not words:
        return "performing the story action"
    verb = words[0].casefold()
    gerunds = {
        "carry": "carrying",
        "carries": "carrying",
        "climb": "climbing",
        "climbs": "climbing",
        "create": "creating",
        "creates": "creating",
        "hold": "holding",
        "holds": "holding",
        "lift": "lifting",
        "lifts": "lifting",
        "open": "opening",
        "opens": "opening",
        "plant": "planting",
        "plants": "planting",
        "read": "reading",
        "reads": "reading",
        "unfold": "unfolding",
        "unfolds": "unfolding",
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
        if word.casefold() in _NON_NAME_CAPITALIZED:
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


def _remove_distinctive_source_overlap(value: str, source_text: str) -> str:
    """Minimally redact repeated source trigrams without inventing replacement text."""

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
        window = output_tokens[overlap_index : overlap_index + 3]
        shortest_offset = min(range(3), key=lambda offset: len(window[offset]))
        del output_tokens[overlap_index + shortest_offset]
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
    request = {"passage": text, "visual_style": visual_style}
    compact_key_guide = (
        "\nCompact JSON keys: b=background_prompt; f={k=kind,s=subject,a=action}; "
        "m={k=kind,p=prompt}.\n"
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
  Prefer an actual magical change over the focus's tool. If no transformation occurs, use a
  concrete supporting element explicitly present in the passage. Never invent a transformation.
  When the passage says something becomes, turns into, transforms, blooms, or rises into something,
  magic.prompt must name that concrete result, including its essential form or destination.
  Light, glow, shimmer, dust, fog, color, or atmosphere alone is not magic.
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
        self._cache: OrderedDict[str, tuple[LiveScenePlan, ModelMetrics]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[LiveScenePlanningResult]] = {}

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
        cache_key = self._cache_key(text=text, visual_style=visual_style)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            started = perf_counter()
            plan, source_metrics = cached
            self._cache[cache_key] = cached
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
        sanitized_wire_plan = wire_plan.privacy_sanitized(source_text=text)
        validated = sanitized_wire_plan.to_live_scene_plan(context_text=f"{text} {visual_style}")
        validate_live_scene_plan_privacy(validated, source_text=text)
        if self.cache_entries:
            self._cache[cache_key] = (validated, metrics)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
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

    def _cache_key(self, *, text: str, visual_style: str) -> str:
        payload = json.dumps(
            {
                "text": text,
                "visual_style": visual_style,
                "model_revision": self.model_revision,
                "compact_wire": self.compact_wire,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()
