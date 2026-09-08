import asyncio
from pathlib import Path

import pytest

from bookforge.domain import ModelMetrics
from bookforge.live_scene_grounding import prepare_grounded_wire
from bookforge.live_scene_planner import (
    LiveSceneCompactWirePlan,
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlannerError,
    LiveScenePlannerPrivacyError,
    LiveScenePlannerTimeoutError,
    LiveSceneWireFocus,
    LiveSceneWireMagic,
    LiveSceneWirePlan,
    StructuredLiveScenePlanner,
    _remove_distinctive_source_overlap,
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
        focus_label="child",
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
    )


def _wire_plan() -> LiveSceneWirePlan:
    return LiveSceneWirePlan(
        background_prompt="Indigo cloudscape and distant floating school with warm windows",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="A child in profile",
            action="holding a luminous open storybook",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="Cyan origami birds becoming constellations",
        ),
    )


def test_compact_wire_preserves_actor_setting_and_supporting_object() -> None:
    source = (
        "In a moonlit orchard, a striped badger carries a glowing pear while "
        "a small paper kite circles above the apple trees."
    )
    compact = LiveSceneCompactWirePlan.model_validate(
        {
            "b": "background",
            "f": ["character", "striped badger", "carries glowing pear"],
            "m": ["prop", "small paper kite"],
        }
    )

    sanitized = compact.to_wire_plan().privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert sanitized.background_prompt == "moonlit orchard"
    assert sanitized.focus.kind == "character"
    assert "badger" in plan.focus.prompt
    assert sanitized.magic.prompt == "paper kite"
    assert "kite" in plan.accent.prompt
    validate_live_scene_plan_privacy(plan, source_text=source)


@pytest.mark.parametrize(
    ("source", "action", "expected"),
    [
        (
            "A student lifts one folded butterfly from a desk, and flowers bloom.",
            "lifts",
            "lifts folded butterfly",
        ),
    ],
)
def test_wire_privacy_sanitizer_recovers_missing_action_object_locally(
    source: str,
    action: str,
    expected: str,
) -> None:
    payload = _wire_plan().model_dump()
    payload["focus"]["action"] = action

    sanitized = LiveSceneWirePlan.model_validate(payload).privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert sanitized.focus.action == expected
    assert (
        expected.replace("lifts", "lifting")
        .replace("swims", "swimming")
        .replace("climbs", "climbing")
        in plan.focus.prompt
    )
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_plan_recovers_spatial_relation_and_complete_colored_object() -> None:
    source = (
        "A white rabbit waits beneath a stone bridge holding a red umbrella, while "
        "three paper lanterns float high above the bridge."
    )
    wire = LiveSceneWirePlan(
        background_prompt="stone bridge overcast sky soft light red umbrella floating",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="white rabbit",
            action="waits a",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="3 paper lanterns all floating high above bridge",
        ),
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert "waiting under bridge" in sanitized.focus.action
    assert "umbrella" in sanitized.focus.action
    assert "red-colored" in sanitized.focus.action
    assert "lanterns floating" in sanitized.magic.prompt
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_plan_recovers_both_sides_of_pronominal_transformation() -> None:
    source = (
        "After a folded paper boat tumbles through a waterfall, it emerges as a white swan "
        "on a glowing lake."
    )
    wire = LiveSceneWirePlan(
        background_prompt="waterfall misty reflective surface soft light",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="white swan",
            action="emerges",
        ),
        magic=LiveSceneWireMagic(kind="effect", prompt="it as glowing lake"),
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert "paper boat" in plan.accent.prompt
    assert "swan" in plan.accent.prompt
    assert "lake" in plan.accent.prompt
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_plan_recovers_destination_and_small_model_action_objects() -> None:
    literacy_source = (
        "Each syllable a child reads aloud becomes a glowing firefly, and together the "
        "fireflies form a path from the bedroom to a distant library."
    )
    literacy = LiveSceneWirePlan(
        background_prompt="soft warm window light cozy garden",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="child",
            action="reading aloud",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="glowing firefly appears forming a path",
        ),
    ).privacy_sanitized(source_text=literacy_source)
    teacup_source = (
        "A field mouse sails a cracked blue teacup across a frozen pond while tiny "
        "snowflakes rise upward like lanterns."
    )
    teacup = LiveSceneWirePlan(
        background_prompt="frozen pond snow lanterns",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="field mouse",
            action="sails cracked blue",
        ),
        magic=LiveSceneWireMagic(kind="effect", prompt="snowflakes rising like lanterns"),
    ).privacy_sanitized(source_text=teacup_source)
    beetle_source = (
        "A green beetle pushes one seed into a rooftop garden, and a twisting tower of "
        "giant leaves grows around the chimneys."
    )
    beetle = LiveSceneWirePlan(
        background_prompt="rooftop garden twisting tower leaves",
        focus=LiveSceneWireFocus(kind="character", subject="green beetle", action="push"),
        magic=LiveSceneWireMagic(kind="effect", prompt="twisting leaves around chimneys"),
    ).privacy_sanitized(source_text=beetle_source)

    assert "library" in literacy.background_prompt
    assert "teacup" in teacup.focus.action
    assert "frozen pond" in teacup.focus.action
    assert "blue-colored" in teacup.focus.action
    assert "seed" in beetle.focus.action
    assert "roof garden" in beetle.focus.action


def test_wire_plan_repairs_possessive_body_fragment_to_complete_character() -> None:
    payload = _wire_plan().model_dump()
    payload["focus"]["subject"] = "child's hand"
    payload["focus"]["action"] = "opening a book beneath rising birds"

    plan = LiveSceneWirePlan.model_validate(payload).to_live_scene_plan()

    assert plan.focus.prompt == (
        "a complete visible child, shown opening a book beneath rising birds"
    )
    assert "hand" not in plan.focus.prompt


def test_wire_plan_removes_duplicated_actor_and_invented_action_from_support() -> None:
    source = "golden retriever running around a field of daisies"
    wire = LiveSceneWirePlan(
        background_prompt="field of daisies",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="golden retriever",
            action="running around",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="golden retriever leaps into a field daisies daisies",
        ),
    )

    sanitized = wire.privacy_sanitized(source_text=source)

    assert sanitized.magic.prompt == "field daisies"
    assert "retriever" not in sanitized.magic.prompt
    assert "leaps" not in sanitized.magic.prompt


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
    assert "Show all required visuals simultaneously" in (page.scene_spec.master_prompt)
    assert "Preserve the stated subject counts and actions" in page.scene_spec.master_prompt
    assert "Only for unspecified placement" in page.scene_spec.master_prompt
    assert "duplicate person" in page.scene_spec.negative_prompt
    assert "main subject at left" in page.scene_spec.master_prompt
    assert "supporting detail at upper right" in page.scene_spec.master_prompt
    assert "within the same continuous scene" in page.scene_spec.master_prompt
    assert "Never use an inset, panel, cutaway, collage, or split screen" in (
        page.scene_spec.master_prompt
    )
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


def test_bounded_action_retains_relation_after_ten_words_and_concise_assembly():
    action = "holds a small cup in the left paw while sitting below a tree"
    plan = LiveSceneWirePlan(
        background_prompt="quiet meadow",
        focus=LiveSceneWireFocus(kind="character", subject="mouse", action=action),
        magic=LiveSceneWireMagic(kind="effect", prompt="none"),
    ).to_live_scene_plan()
    assert "below a tree" in plan.focus.prompt
    page = plan.to_page(
        source_text="A peaceful evening.",
        visual_style="watercolor",
        seed=17,
        render_contract="concise",
    )
    assert "below a tree" in page.scene_spec.master_prompt
    assert "complete visible" not in page.scene_spec.master_prompt


def test_plan_repairs_duplicate_placements_and_coordinate_background_as_model_output() -> None:
    payload = _plan().model_dump()
    payload["background_prompt"] = "[0.5, 0.5, 0.8, 0.8], cobalt sky over luminous clouds"
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
        assert 0.04 + placement.width / 2 <= placement.center_x <= (0.96 - placement.width / 2)
        assert 0.04 + placement.height / 2 <= placement.center_y <= (0.96 - placement.height / 2)


def test_privacy_gate_rejects_distinctive_three_token_source_phrase() -> None:
    source = "At dawn, a copper heron unlocks the cobalt orchard beneath quiet clouds."
    payload = _plan().model_dump()
    payload["focus"]["prompt"] = "A figure crosses the cobalt orchard beneath violet fog"

    with pytest.raises(LiveScenePlannerPrivacyError, match="distinctive source phrase"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text=source,
        )


@pytest.mark.parametrize("unsafe_prompt", ["Warm paper lighting beside +1 (415) 555-0199"])
def test_privacy_gate_rejects_obvious_contact_data(unsafe_prompt: str) -> None:
    payload = _plan().model_dump()
    payload["art_direction"] = unsafe_prompt

    with pytest.raises(LiveScenePlannerPrivacyError, match="possible contact data"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text="A reader opens a moonlit book.",
        )


@pytest.mark.parametrize(
    ("source", "payload"),
    [
        ("In a room, a page reads orchid delta. A fox waits.", "orchid delta"),
        ("In a room, on a poster are the words orchid delta. A fox waits.", "orchid delta"),
    ],
)
def test_privacy_gate_rejects_printed_source_payload(source: str, payload: str) -> None:
    plan_payload = _plan().model_dump()
    plan_payload["accent"]["prompt"] = payload

    with pytest.raises(LiveScenePlannerPrivacyError, match="printed source payload"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(plan_payload),
            source_text=source,
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


@pytest.mark.parametrize(
    "source",
    [
        "A magician shows a fox a lantern.",
        "Each syllable a child reads aloud becomes a firefly.",
    ],
)
def test_privacy_gate_allows_narrative_read_show_say_and_display_verbs(source: str) -> None:
    validate_live_scene_plan_privacy(_plan(), source_text=source)


@pytest.mark.parametrize(
    ("source", "background"),
    [
        ("Inside a dusty attic, a moth opens a map.", "inside attic, dim paper rafters"),
        ("Exactly two red paper boats float on blue water.", "exactly two boats on water"),
    ],
)
def test_privacy_gate_does_not_treat_sentence_initial_relations_as_names(
    source: str,
    background: str,
) -> None:
    payload = _plan().model_dump()
    payload["background_prompt"] = background

    validate_live_scene_plan_privacy(
        LiveScenePlan.model_validate(payload),
        source_text=source,
    )


@pytest.mark.parametrize("name", ["Exactly"])
def test_privacy_gate_still_rejects_count_word_explicitly_used_as_name(name: str) -> None:
    payload = _plan().model_dump()
    payload["accent"]["prompt"] = f"{name} beside a silver constellation"
    with pytest.raises(LiveScenePlannerPrivacyError, match="proper-name candidate"):
        validate_live_scene_plan_privacy(
            LiveScenePlan.model_validate(payload),
            source_text=f"A child named {name} lifts a green lantern.",
        )


def test_generic_wire_privacy_preserves_absent_people() -> None:
    source = "One child holds a green book. No other people."
    wire = (
        _wire_plan()
        .model_copy(update={"magic": LiveSceneWireMagic(kind="effect", prompt="no other people")})
        .privacy_sanitized(source_text=source)
    )
    plan = wire.to_live_scene_plan(context_text=source)
    page = plan.to_page(source_text=source, visual_style="watercolor", seed=17)
    assert plan.accent.prompt == "no additional people"
    assert "Scene constraint: no additional people" in page.scene_spec.master_prompt
    assert "Required supporting visual: other people" not in page.scene_spec.master_prompt


@pytest.mark.parametrize("negation", ["no", "not"])
def test_privacy_rewrite_never_deletes_a_negation(negation: str) -> None:
    value = f"{negation} silver fox"
    with pytest.raises(LiveScenePlannerPrivacyError, match="remove a negation"):
        _remove_distinctive_source_overlap(
            value, f"The illustration has {value}.", preserve_tail=True
        )


def _grounded_wire(prompt: str) -> LiveSceneWirePlan:
    # Model fixtures preserve the requested facts; unrelated cloudscape fixtures
    # above remain useful for the direct privacy/render-format tests.
    if "whale" in prompt:
        subject, action, supporting = "whale", "carries a lantern through a library", "lantern"
    elif "silent book" in prompt:
        subject, action, supporting = (
            "child",
            "opens a silent book",
            "origami birds lighting the sky",
        )
    else:
        subject, action = "child", "opens a quiet book"
        supporting = "paper birds rising"
    return LiveSceneWirePlan(
        background_prompt="neutral background",
        focus=LiveSceneWireFocus(kind="character", subject=subject, action=action),
        magic=LiveSceneWireMagic(kind="effect", prompt=supporting),
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
        if output_type.__name__ == "_LiveScenePlannerWarmupOutput":
            return output_type.model_validate({"ready": True}), ModelMetrics(
                backend="ollama",
                model="gemma3:1b",
                total_ms=4.5,
                input_tokens=24,
                output_tokens=5,
            )
        plan = _grounded_wire(prompt)
        if output_type is LiveSceneCompactWirePlan:
            result = LiveSceneCompactWirePlan.model_validate(
                {
                    "b": plan.background_prompt,
                    "f": [
                        plan.focus.kind,
                        plan.focus.subject,
                        plan.focus.action,
                    ],
                    "m": [plan.magic.kind, plan.magic.prompt],
                }
            )
        else:
            result = plan
        return result, ModelMetrics(
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

    source = "A child opens a silent book and origami birds light the sky."
    assert result.plan == prepare_grounded_wire(
        _grounded_wire(source), source_text=source
    ).to_live_scene_plan(context_text=source)
    assert result.metrics.model == "gemma3:1b"
    assert result.model_revision == "sha256:gemma-fixture"
    assert result.wall_ms >= 0
    assert stub.calls[0]["output_type"] is LiveSceneWirePlan
    assert "origami birds light the sky" in str(stub.calls[0]["prompt"])
    assert "luminous watercolor paper theater" not in str(stub.calls[0]["prompt"])
    assert '"seed"' not in str(stub.calls[0]["prompt"])


def test_structured_planner_can_use_opt_in_short_key_contract() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        compact_wire=True,
    )

    result = asyncio.run(
        planner.plan(
            text="A child opens a quiet book while paper birds rise.",
            visual_style="luminous watercolor paper theater",
            seed=23,
        )
    )

    assert stub.calls[0]["output_type"] is LiveSceneCompactWirePlan
    assert "f=[kind,subject,action]" in str(stub.calls[0]["prompt"])
    assert result.plan.focus.prompt == (
        prepare_grounded_wire(
            _grounded_wire("A child opens a quiet book while paper birds rise."),
            source_text="A child opens a quiet book while paper birds rise.",
        )
        .to_live_scene_plan(context_text="A child opens a quiet book while paper birds rise.")
        .focus.prompt
    )


def test_structured_planner_coalesces_text_free_local_warmup() -> None:
    stub = _ModelStub(delay_seconds=0.01)
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    async def run_warmups():
        return await asyncio.gather(planner.warmup(), planner.warmup())

    first, second = asyncio.run(run_warmups())

    assert first == second
    assert first.metrics.model == "gemma3:1b"
    assert first.metrics.output_tokens == 5
    assert len(stub.calls) == 1
    assert stub.calls[0]["output_type"].__name__ == "_LiveScenePlannerWarmupOutput"
    assert "passage" not in str(stub.calls[0]["prompt"]).casefold()
    assert "story" not in str(stub.calls[0]["prompt"]).casefold()
    assert '{"ready":true}' in str(stub.calls[0]["prompt"])


@pytest.mark.parametrize("from_disk", [True])
def test_cached_plans_do_not_wait_for_model_warmup(tmp_path: Path, from_disk: bool) -> None:
    async def run() -> None:
        warmup_started = asyncio.Event()
        release_warmup = asyncio.Event()

        class PausedWarmupStub(_ModelStub):
            async def generate(self, *, system, prompt, output_type):
                if output_type.__name__ == "_LiveScenePlannerWarmupOutput":
                    warmup_started.set()
                    await release_warmup.wait()
                return await super().generate(system=system, prompt=prompt, output_type=output_type)

        stub = PausedWarmupStub()
        planner = StructuredLiveScenePlanner(
            stub,
            timeout_seconds=2,
            persistent_cache_dir=tmp_path / "plans",
        )
        request = {
            "text": "A child opens a quiet book while paper birds rise.",
            "visual_style": "paper theater",
            "seed": 23,
        }
        original = await planner.plan(**request)
        if from_disk:
            planner = StructuredLiveScenePlanner(
                stub,
                timeout_seconds=2,
                persistent_cache_dir=tmp_path / "plans",
            )
        warmup = asyncio.create_task(planner.warmup())
        await asyncio.wait_for(warmup_started.wait(), timeout=1)
        try:
            cached = await asyncio.wait_for(planner.plan(**request), timeout=0.5)
            assert not warmup.done()
            assert cached.cache_hit
            assert cached.plan == original.plan
            assert cached.metrics.input_tokens == cached.metrics.output_tokens == 0
            assert len(stub.calls) == 1
        finally:
            release_warmup.set()
            await warmup

    asyncio.run(run())


def test_structured_planner_cache_is_bounded_by_passage_and_reuses_new_styles() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        cache_entries=1,
    )
    text = "A child opens a quiet book while paper birds rise."

    first = asyncio.run(planner.plan(text=text, visual_style="paper theater", seed=1))
    restyled = asyncio.run(planner.plan(text=text, visual_style="oil pastel", seed=2))
    asyncio.run(
        planner.plan(
            text="A whale carries a lantern through a library.",
            visual_style="paper theater",
            seed=1,
        )
    )
    repeated = asyncio.run(planner.plan(text=text, visual_style="paper theater", seed=2))

    assert len(stub.calls) == 3
    assert first.cache_hit is False
    assert restyled.cache_hit is True
    assert restyled.plan == first.plan
    assert restyled.metrics.model == first.metrics.model
    assert restyled.metrics.total_ms == 0
    assert restyled.metrics.input_tokens == restyled.metrics.output_tokens == 0
    assert repeated.cache_hit is False


def test_private_persistent_plan_cache_survives_restart_without_storing_source(
    tmp_path: Path,
) -> None:
    text = "A child opens a quiet book while paper birds rise."
    cache_dir = tmp_path / "private-plans"
    first_stub = _ModelStub()
    first_planner = StructuredLiveScenePlanner(
        first_stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        model_revision="sha256:persistent-fixture",
        cache_entries=2,
        persistent_cache_dir=cache_dir,
    )

    first = asyncio.run(first_planner.plan(text=text, visual_style="paper theater", seed=1))
    cached_files = list(cache_dir.glob("*.json"))

    assert first.cache_hit is False
    assert len(first_stub.calls) == 1
    assert len(cached_files) == 1
    assert text.encode() not in cached_files[0].read_bytes()
    assert cache_dir.stat().st_mode & 0o777 == 0o700
    assert cached_files[0].stat().st_mode & 0o777 == 0o600

    restarted_stub = _ModelStub(failure=AssertionError("model must not run"))
    restarted_planner = StructuredLiveScenePlanner(
        restarted_stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        model_revision="sha256:persistent-fixture",
        cache_entries=2,
        persistent_cache_dir=cache_dir,
    )
    restored = asyncio.run(restarted_planner.plan(text=text, visual_style="bright clay", seed=2))

    assert restored.cache_hit is True
    assert restored.plan == first.plan
    assert restored.metrics.total_ms == 0
    assert restored.metrics.input_tokens == 0
    assert restored.metrics.output_tokens == 0
    assert restarted_stub.calls == []

    changed_revision_stub = _ModelStub()
    changed_revision_planner = StructuredLiveScenePlanner(
        changed_revision_stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        model_revision="sha256:new-model-revision",
        cache_entries=2,
        persistent_cache_dir=cache_dir,
    )
    changed = asyncio.run(
        changed_revision_planner.plan(text=text, visual_style="paper theater", seed=3)
    )

    assert changed.cache_hit is False
    assert len(changed_revision_stub.calls) == 1


def test_planner_coalesces_inflight_requests_and_survives_waiter_cancel() -> None:
    stub = _ModelStub(delay_seconds=0.03)
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
    )
    request = {
        "text": "A child opens a quiet book while paper birds rise.",
        "visual_style": "luminous watercolor paper theater",
        "seed": 23,
    }

    async def exercise():
        cancelled_waiter = asyncio.create_task(planner.plan(**request))
        await asyncio.sleep(0)
        surviving_waiter = asyncio.create_task(planner.plan(**request))
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        return await surviving_waiter

    result = asyncio.run(exercise())

    assert result.plan.focus.prompt
    assert len(stub.calls) == 1
    assert planner._inflight == {}  # noqa: SLF001


def test_persistent_planner_cache_tracks_client_instructions(tmp_path: Path) -> None:
    text = "A child opens a quiet book while paper birds rise."
    cached = []
    for identity in ("instruction-a", "instruction-a", "instruction-b"):
        stub = _ModelStub()
        stub.cache_identity = identity
        planner = StructuredLiveScenePlanner(
            stub,
            timeout_seconds=1,
            model_revision="same-weights",
            persistent_cache_dir=tmp_path / "plans",
        )
        cached.append(
            asyncio.run(planner.plan(text=text, visual_style="watercolor", seed=1)).cache_hit
        )
    assert cached == [False, True, False]


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
        asyncio.run(timeout_planner.plan(text="A book opens.", visual_style="paper art", seed=1))
    with pytest.raises(LiveScenePlannerError, match="bad structured output"):
        asyncio.run(broken_planner.plan(text="A book opens.", visual_style="paper art", seed=1))


def test_ungrounded_model_output_is_rejected_before_cache_or_scene_conversion(tmp_path):
    class InventingModel(_ModelStub):
        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _wire_plan(), ModelMetrics(backend="fixture", model="local", total_ms=1)

    model = InventingModel()
    planner = StructuredLiveScenePlanner(model, timeout_seconds=1, persistent_cache_dir=tmp_path)
    text = "A quick brown box. Don’t throw a little lazy dog."
    with pytest.raises(LiveScenePlannerError, match="unsupported visual facts") as error:
        asyncio.run(planner.plan(text=text, visual_style="watercolor", seed=1))
    assert text not in str(error.value)
    assert len(model.calls) == 1
    assert list(tmp_path.iterdir()) == []
    assert planner._inflight == {}


@pytest.mark.parametrize(
    "source,subject,action,supporting,required",
    [
        (
            "The pink fox jumped over the river stream.",
            "pink fox",
            "jumped over the river stream",
            "none",
            ("pink fox", "jumped over river stream"),
        ),
        (
            "A brown fox stands beside a stream. No dogs.",
            "brown fox",
            "stands beside a stream",
            "no dogs",
            ("brown fox", "stands beside stream", "Scene constraint: no dogs"),
        ),
    ],
)
def test_grounded_facts_survive_privacy_and_final_renderer_prompt(
    source, subject, action, supporting, required
):
    raw = LiveSceneWirePlan(
        background_prompt="neutral background",
        focus=LiveSceneWireFocus(kind="character", subject=subject, action=action),
        magic=LiveSceneWireMagic(kind="effect", prompt=supporting),
    )

    class Model(_ModelStub):
        async def generate(self, **kwargs):
            return raw, ModelMetrics(backend="fixture", model="local", total_ms=1)

    result = asyncio.run(
        StructuredLiveScenePlanner(Model(), timeout_seconds=1).plan(
            text=source, visual_style="watercolor", seed=1
        )
    )
    page = result.plan.to_page(source_text=source, visual_style="watercolor", seed=1)
    for phrase in required:
        assert phrase in page.scene_spec.master_prompt


def test_deferred_tensorrt_fallback_reaches_grounding_before_any_lossy_sanitizer(monkeypatch):
    import httpx

    import bookforge.live_scene_planner as planner_module
    from bookforge.tensorrt_slot_client import TensorRTSlotModelClient

    source = "A brown fox stands beside a stream. No dogs."
    raw = LiveSceneWirePlan(
        background_prompt="neutral background",
        focus=LiveSceneWireFocus(
            kind="character", subject="brown fox", action="stands beside a stream"
        ),
        magic=LiveSceneWireMagic(kind="effect", prompt="no dogs"),
    )
    seen = []

    def inspect_raw(wire, *, source_text):
        seen.append(wire.model_dump())
        return prepare_grounded_wire(wire, source_text=source_text)

    monkeypatch.setattr(planner_module, "prepare_grounded_wire", inspect_raw)

    async def exercise():
        class Fallback(_ModelStub):
            async def generate(self, **kwargs):
                self.calls.append(kwargs)
                return self.output, ModelMetrics(backend="fallback", model="fixture", total_ms=1)

        fallback = Fallback()

        def unavailable(request):
            raise httpx.ConnectError("offline fixture", request=request)

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="fixture",
            timeout_seconds=1,
            fallback=fallback,
            defer_fallback_sanitization=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(unavailable)
        )
        try:
            for bad_background in (
                None,
                "golden retriever in a flower field",
                "https://private.example/secret",
            ):
                fallback.output = (
                    raw
                    if bad_background is None
                    else raw.model_copy(update={"background_prompt": bad_background})
                )
                planner = StructuredLiveScenePlanner(client, timeout_seconds=1)
                if bad_background is not None:
                    with pytest.raises(LiveScenePlannerError, match="unsupported visual facts"):
                        await planner.plan(text=source, visual_style="watercolor", seed=1)
                else:
                    result = await planner.plan(text=source, visual_style="watercolor", seed=1)
                    prompt = result.plan.to_page(
                        source_text=source, visual_style="watercolor", seed=1
                    ).scene_spec.master_prompt
                    assert "stands beside stream" in prompt
                    assert "Scene constraint: no dogs" in prompt
            assert len(fallback.calls) == 3
        finally:
            await client.client.aclose()

    asyncio.run(exercise())
    assert len(seen) == 3
    assert all(item["focus"]["action"] == "stands beside a stream" for item in seen)
