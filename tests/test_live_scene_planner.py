import asyncio

import pytest

from bookforge.domain import ModelMetrics
from bookforge.live_scene_planner import (
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlannerError,
    LiveScenePlannerPrivacyError,
    LiveScenePlannerTimeoutError,
    StructuredLiveScenePlanner,
    validate_live_scene_plan_privacy,
)


def _plan() -> LiveScenePlan:
    return LiveScenePlan(
        scene_summary="A child follows luminous paper birds toward a floating school.",
        art_direction=(
            "Low-angle moonlit composition, cyan paper birds sweeping diagonally through indigo "
            "clouds, warm book light, watercolor fibers"
        ),
        camera_motion="slow_push",
        background_prompt="Indigo cloudscape and a distant floating school in warm window light",
        focus=LiveScenePlacedLayerPlan(
            kind="character",
            prompt="A child in profile holding a luminous open storybook",
            anchor=(0.32, 0.58, 0.36, 0.68),
            depth=4,
            motion="breathe",
        ),
        accent=LiveScenePlacedLayerPlan(
            kind="effect",
            prompt="A sweeping arc of cyan origami birds becoming constellations",
            anchor=(0.68, 0.38, 0.5, 0.42),
            depth=2,
            motion="float",
        ),
        ambience=["stars"],
    )


def test_compact_plan_normalizes_to_canonical_scene_spec_and_layers() -> None:
    plan = _plan()
    page = plan.to_page(
        source_text="A child opened a book and the birds showed the way.",
        visual_style="luminous watercolor paper theater",
        seed=17,
    )

    assert len(plan.model_dump_json()) < 900
    assert page.scene_summary == plan.scene_summary
    assert [layer.layer_id for layer in page.layers] == [
        "scene-background",
        "scene-focus",
        "scene-accent",
    ]
    assert [layer.kind for layer in page.layers] == ["background", "character", "effect"]
    assert page.scene_spec is not None
    assert plan.art_direction in page.scene_spec.master_prompt
    assert "luminous watercolor paper theater" in page.scene_spec.master_prompt
    assert "A child opened a book and the birds showed the way." not in (
        page.scene_spec.master_prompt
    )
    assert plan.background_prompt in page.scene_spec.master_prompt
    assert plan.focus.prompt in page.scene_spec.master_prompt
    assert plan.accent.prompt in page.scene_spec.master_prompt
    assert page.source_text == "A child opened a book and the birds showed the way."
    assert "readable text" in page.scene_spec.negative_prompt
    assert page.scene_spec.camera.kind == "slow_push"
    assert page.scene_spec.camera.duration_ms == 8_017
    assert [item.layer_id for item in page.scene_spec.composition] == [
        "scene-background",
        "scene-focus",
        "scene-accent",
    ]
    focus = page.scene_spec.composition[1]
    assert (focus.center_x, focus.center_y, focus.depth) == (0.32, 0.58, 4)
    assert focus.ambient_motion.kind == "breathe"


def test_plan_normalizes_raw_anchor_inside_projection_canvas() -> None:
    payload = _plan().model_dump()
    payload["focus"]["anchor"] = (0.05, 0.95, 0.8, 0.95)

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="A book opens.",
        visual_style="paper theater",
        seed=1,
    )

    focus = page.scene_spec.composition[1]  # type: ignore[union-attr]
    assert (focus.width, focus.height) == (0.6, 0.72)
    assert (focus.center_x, focus.center_y) == pytest.approx((0.34, 0.6))


def test_plan_preserves_projector_overscan_margin_for_animated_layers() -> None:
    payload = _plan().model_dump()
    payload["accent"]["anchor"] = (0.99, 0.99, 0.42, 0.48)

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="A book opens.",
        visual_style="paper theater",
        seed=2,
    )

    accent = page.scene_spec.composition[2]  # type: ignore[union-attr]
    assert (accent.center_x, accent.center_y) == pytest.approx((0.75, 0.72))
    assert accent.center_x + accent.width / 2 == pytest.approx(0.96)
    assert accent.center_y + accent.height / 2 == pytest.approx(0.96)


def test_plan_repairs_duplicate_placements_and_coordinate_background_as_model_output() -> None:
    payload = _plan().model_dump()
    payload["background_prompt"] = (
        "[0.5, 0.5, 0.8, 0.8], cobalt sky over luminous clouds"
    )
    payload["focus"]["anchor"] = (0.5, 0.5, 0.8, 0.8)
    payload["accent"]["anchor"] = (0.5, 0.5, 0.8, 0.8)
    payload["focus"]["depth"] = 0.2
    payload["accent"]["depth"] = 0.2
    payload["focus"]["motion"] = "float"
    payload["accent"]["motion"] = "float"

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="Quenlora opens the cobalt gate.",
        visual_style="paper theater",
        seed=31,
    )

    assert page.scene_spec is not None
    background = page.layers[0].prompt
    assert "[" not in background
    assert not any(character.isdigit() for character in background)
    assert "Low-angle moonlit composition" in background
    assert page.scene_spec.master_prompt.count("Low-angle moonlit composition") == 1
    assert ".." not in page.scene_spec.master_prompt
    focus, accent = page.scene_spec.composition[1:]
    assert (focus.width, focus.height) == (0.6, 0.72)
    assert (accent.width, accent.height) == (0.3, 0.34)
    assert (focus.center_x, focus.center_y) != (accent.center_x, accent.center_y)
    assert focus.depth != accent.depth
    assert (accent.center_x, accent.center_y) == pytest.approx((0.81, 0.21))
    for placement in (focus, accent):
        assert 0.04 + placement.width / 2 <= placement.center_x <= (
            0.96 - placement.width / 2
        )
        assert 0.04 + placement.height / 2 <= placement.center_y <= (
            0.96 - placement.height / 2
        )


def test_plan_bounds_model_phrases_before_scene_spec_compilation() -> None:
    payload = _plan().model_dump()
    payload["scene_summary"] = " ".join(["star"] * 25)
    payload["art_direction"] = " ".join(["glow"] * 40)
    payload["background_prompt"] = " ".join(["sky"] * 25)

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="A book opens.",
        visual_style="paper theater",
        seed=1,
    )

    assert len(page.scene_summary.split()) == 18
    assert len(page.layers[0].prompt.split()) == 18
    assert page.scene_spec is not None
    assert page.scene_spec.master_prompt.count("glow") == 35


def test_plan_repairs_dangling_summary_participle_without_fallback() -> None:
    payload = _plan().model_dump()
    payload["scene_summary"] = (
        "A child lifts a book of birds. The birds rise toward the moon, illuminating."
    )

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="A private source sentence.",
        visual_style="paper theater",
        seed=2,
    )

    assert page.scene_summary == (
        "A child lifts a book of birds. The birds rise toward the moon"
    )
    assert "illuminating" not in page.scene_spec.master_prompt  # type: ignore[union-attr]


def test_privacy_gate_rejects_exact_source_echo() -> None:
    source = "At dawn the copper heron unlocks a silent observatory."
    payload = _plan().model_dump()
    payload["scene_summary"] = source

    with pytest.raises(LiveScenePlannerPrivacyError, match="source passage echo"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text=source,
        )


def test_privacy_gate_rejects_distinctive_three_token_source_phrase() -> None:
    source = "At dawn, a copper heron unlocks the cobalt orchard beneath quiet clouds."
    payload = _plan().model_dump()
    payload["focus"]["prompt"] = "A figure crosses the cobalt orchard beneath violet fog"

    with pytest.raises(LiveScenePlannerPrivacyError, match="distinctive source phrase"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text=source,
        )


@pytest.mark.parametrize(
    "unsafe_prompt",
    [
        "Warm paper lighting beside reader@example.com",
        "Warm paper lighting beside +1 (415) 555-0199",
    ],
)
def test_privacy_gate_rejects_obvious_contact_data(unsafe_prompt: str) -> None:
    payload = _plan().model_dump()
    payload["art_direction"] = unsafe_prompt

    with pytest.raises(LiveScenePlannerPrivacyError, match="possible contact data"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text="A reader opens a moonlit book.",
        )


def test_privacy_gate_rejects_source_proper_name_candidate() -> None:
    source = "A child named Quenlora opens a glowing book beneath the moon."
    payload = _plan().model_dump()
    payload["accent"]["prompt"] = "Quenlora beside a silver constellation"

    with pytest.raises(LiveScenePlannerPrivacyError, match="proper-name candidate"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text=source,
        )


def test_privacy_gate_accepts_visual_semantic_paraphrase() -> None:
    validate_live_scene_plan_privacy(
        _plan(),
        source_text=(
            "A child named Quenlora opens a silent volume; folded shapes glow above it."
        ),
    )


class _ModelStub:
    def __init__(self, *, delay_seconds: float = 0, failure: Exception | None = None) -> None:
        self.delay_seconds = delay_seconds
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    async def generate(self, *, system: str, prompt: str, output_type):
        self.calls.append({"system": system, "prompt": prompt, "output_type": output_type})
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failure is not None:
            raise self.failure
        return _plan(), ModelMetrics(
            backend="ollama",
            model="gemma3:1b",
            total_ms=9.5,
            input_tokens=211,
            output_tokens=168,
        )

    async def probe(self) -> tuple[bool, str]:
        return True, "ready"


def test_structured_planner_uses_live_schema_and_records_model_revision() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        model_revision="sha256:gemma-fixture",
    )

    result = asyncio.run(
        planner.plan(
            text="A child opens a silent book and origami birds light the sky.",
            visual_style="luminous watercolor paper theater",
            seed=23,
        )
    )

    assert result.plan == _plan()
    assert result.metrics.model == "gemma3:1b"
    assert result.model_revision == "sha256:gemma-fixture"
    assert result.wall_ms >= 0
    assert stub.calls[0]["output_type"] is LiveScenePlan
    assert "origami birds light the sky" in str(stub.calls[0]["prompt"])
    assert "luminous watercolor paper theater" in str(stub.calls[0]["prompt"])


def test_structured_planner_turns_timeout_and_model_failure_into_recoverable_errors() -> None:
    timeout_planner = StructuredLiveScenePlanner(
        _ModelStub(delay_seconds=0.05),  # type: ignore[arg-type]
        timeout_seconds=0.005,
    )
    broken_planner = StructuredLiveScenePlanner(
        _ModelStub(failure=ValueError("bad structured output")),  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    with pytest.raises(LiveScenePlannerTimeoutError, match="exceeded"):
        asyncio.run(
            timeout_planner.plan(text="A book opens.", visual_style="paper art", seed=1)
        )
    with pytest.raises(LiveScenePlannerError, match="bad structured output"):
        asyncio.run(
            broken_planner.plan(text="A book opens.", visual_style="paper art", seed=1)
        )
