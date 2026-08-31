import asyncio
from pathlib import Path

import pytest

from bookforge.domain import ModelMetrics
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


def test_wire_plan_is_compact_and_normalizes_safe_geometry_and_motion() -> None:
    wire = _wire_plan()
    plan = wire.to_live_scene_plan()

    assert len(wire.model_dump_json()) < 620
    assert plan.scene_summary == (
        "A complete visible child in profile, shown holding a luminous open storybook."
    )
    assert plan.art_direction == ("clear silhouettes, projection-bright midtones, tactile depth")
    assert plan.focus.anchor == (0.5, 0.55, 0.4, 0.62)
    assert plan.focus.depth == 2.5
    assert plan.focus.motion == "breathe"
    assert plan.accent.anchor == (0.73, 0.28, 0.3, 0.26)
    assert plan.accent.depth == 5
    assert plan.accent.motion == "pulse"
    assert plan.ambience == ["dust"]


def test_walnut_boat_action_compiles_to_explicit_visual_direction() -> None:
    plan = LiveSceneWirePlan(
        background_prompt="golden pond and reeds",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="young otter",
            action="steering walnut boat",
        ),
        magic=LiveSceneWireMagic(kind="effect", prompt="red dragonflies carrying ribbon light"),
    ).to_live_scene_plan()

    page = plan.to_page(
        source_text="An animal crosses water at sunrise.",
        visual_style="luminous layered paper theater",
        seed=27,
    )

    assert plan.focus_label == "young otter"
    assert "one hollow half of a brown walnut shell" in page.scene_spec.master_prompt
    assert "wrinkled brain-like walnut texture" in page.scene_spec.master_prompt
    assert "paws on a small tiller" in page.scene_spec.master_prompt
    assert page.layers[1].prompt.endswith("shown steering walnut boat")


def test_short_key_wire_contract_preserves_semantics_with_less_decode_text() -> None:
    compact = LiveSceneCompactWirePlan(
        background_prompt=_wire_plan().background_prompt,
        focus=("character", _wire_plan().focus.subject, _wire_plan().focus.action),
        magic=("effect", _wire_plan().magic.prompt),
    )

    assert compact.to_wire_plan() == _wire_plan()
    assert set(LiveSceneCompactWirePlan.model_json_schema()["properties"]) == {
        "b",
        "f",
        "m",
    }
    assert len(compact.model_dump_json(by_alias=True)) < len(_wire_plan().model_dump_json())


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


def test_wire_normalization_sanitizes_cross_field_summary_overlap() -> None:
    source = "A moonlit turtle climbs a staircase made of clouds."
    wire = LiveSceneWirePlan.model_validate(
        {
            "background_prompt": "moonlit clouds",
            "focus": {
                "kind": "character",
                "subject": "moonlit turtle",
                "action": "climbs staircase",
            },
            "magic": {"kind": "effect", "prompt": "cloud steps"},
        }
    )

    plan = wire.privacy_sanitized(source_text=source).to_live_scene_plan(context_text=source)

    validate_live_scene_plan_privacy(plan, source_text=source)
    assert "moonlit turtle climbs" not in plan.scene_summary.lower()
    assert "moonlit turtle climbs" not in plan.focus.prompt.lower()


def test_wire_privacy_sanitizer_preserves_primary_subject_head_noun() -> None:
    source = "A silver whale swims through a flooded library carrying a lantern."
    wire = LiveSceneWirePlan.model_validate(
        {
            "background_prompt": "flooded library",
            "focus": {
                "kind": "character",
                "subject": "silver whale swims through water",
                "action": "carries lantern",
            },
            "magic": {"kind": "effect", "prompt": "books become bright fish"},
        }
    )

    sanitized = wire.privacy_sanitized(source_text=source)

    assert "silver whale" in sanitized.focus.subject
    assert "silver whale swims" not in sanitized.focus.subject


@pytest.mark.parametrize(
    ("source", "action", "expected"),
    [
        (
            "A student lifts one folded butterfly from a desk, and flowers bloom.",
            "lifts",
            "lifts folded butterfly",
        ),
        (
            "A silver whale swims through a flooded library while books open.",
            "swims",
            "swims through flooded library",
        ),
        (
            "A turtle climbs a staircase made of clouds while jellyfish drift.",
            "climbs",
            "climbs staircase",
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


def test_wire_privacy_sanitizer_recovers_actor_and_compound_object_detail() -> None:
    source = (
        "At sunrise, a young otter steers a walnut-shell boat across a glassy pond "
        "while two red dragonflies carry a ribbon of morning light between the reeds."
    )
    wire = LiveSceneWirePlan.model_validate(
        {
            "background_prompt": "glassy pond and reeds",
            "focus": {
                "kind": "character",
                "subject": "otter",
                "action": "steers boat",
            },
            "magic": {
                "kind": "character",
                "prompt": "red dragonflies carrying ribbon of light",
            },
        }
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert sanitized.focus.subject == "young otter"
    assert sanitized.focus.action == "steers walnut boat"
    assert plan.focus.prompt == "a complete visible young otter, shown steering walnut boat"
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_plan_repairs_multiple_actor_actions_and_exact_supporting_count() -> None:
    source = (
        "At twilight, a small copper owl unfolds a bright blue paper map beside a "
        "glass lighthouse. Exactly two silver comet-fish arc above it while the owl "
        "points to a tiny red island shaped like a star."
    )
    wire = LiveSceneWirePlan(
        background_prompt="twilight glass lighthouse red island star map",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="copper owl",
            action="points to a",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="silver comet arc above",
        ),
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)
    page = plan.to_page(
        source_text=source,
        visual_style="luminous cut-paper theater",
        seed=17,
    )

    assert "unfolding" in sanitized.focus.action
    assert "paper map" in sanitized.focus.action
    assert "pointing" in sanitized.focus.action
    assert "island" in sanitized.focus.action
    assert "red-colored" in sanitized.focus.action
    assert sanitized.magic.prompt.startswith("2 comet-fish")
    assert "silver-colored" in sanitized.magic.prompt
    assert "arcing overhead" in sanitized.magic.prompt
    assert "2 comet-fish" in page.scene_spec.master_prompt
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


def test_wire_plan_recovers_omitted_supporting_subject_and_removes_negative_text() -> None:
    bakery_source = (
        "A round robot baker opens the oven, and a mountain of bread dough erupts "
        "with colorful confetti instead of smoke."
    )
    bakery = LiveSceneWirePlan(
        background_prompt="warm bakery interior oven",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="robot baker",
            action="opening oven",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="colorful confetti erupts from the oven",
        ),
    ).privacy_sanitized(source_text=bakery_source)
    underwater_source = (
        "An octopus conductor guides a tiny train through an underwater station, where "
        "bubbles swell into glowing clocks with no numbers."
    )
    underwater = LiveSceneWirePlan(
        background_prompt="vibrant coral reef softly illuminated",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="octopus",
            action="guides tiny train",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="tiny train, glowing clocks, no numbers",
        ),
    ).privacy_sanitized(source_text=underwater_source)

    assert "dough" in bakery.magic.prompt
    assert "confetti" in bakery.magic.prompt
    assert "station" in underwater.background_prompt
    assert "underwater" in underwater.background_prompt
    assert "numbers" not in underwater.magic.prompt
    assert "clocks" in underwater.magic.prompt


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


def test_wire_plan_restores_supporting_creature_omitted_by_small_model() -> None:
    source = (
        "On a frozen lake beneath the northern lights, a red fox skates in circles "
        "while a tiny owl watches from a snow-covered pine."
    )
    wire = LiveSceneWirePlan(
        background_prompt="covered pine frost shimmering winter scene",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="red fox",
            action="skates circles",
        ),
        magic=LiveSceneWireMagic(
            kind="effect",
            prompt="watches covered pine",
        ),
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert sanitized.magic.prompt == "tiny owl watching covered pine"
    assert "red fox" in plan.focus.prompt
    assert "skating circles" in plan.focus.prompt
    assert "tiny owl" in plan.accent.prompt
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_plan_preserves_supporting_creature_during_privacy_rephrase() -> None:
    source = "A red fox skates in circles while a tiny owl watches from a snow-covered pine."
    wire = LiveSceneWirePlan(
        background_prompt="frozen lake under aurora",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="red fox",
            action="skates circles",
        ),
        magic=LiveSceneWireMagic(
            kind="character",
            prompt="tiny owl watches from snow-covered pine",
        ),
    )

    sanitized = wire.privacy_sanitized(source_text=source)
    plan = sanitized.to_live_scene_plan(context_text=source)

    assert sanitized.magic.prompt == "tiny owl watching from covered pine"
    assert "tiny owl" in plan.accent.prompt
    assert "watching" in plan.accent.prompt
    validate_live_scene_plan_privacy(plan, source_text=source)


def test_wire_privacy_sanitizer_does_not_rewrite_complete_action() -> None:
    payload = _wire_plan().model_dump()
    payload["focus"]["action"] = "lifts folded butterfly"

    sanitized = LiveSceneWirePlan.model_validate(payload).privacy_sanitized(
        source_text="A student lifts one folded butterfly from a desk."
    )

    assert sanitized.focus.action == "lifts folded butterfly"


def test_wire_plan_repairs_possessive_body_fragment_to_complete_character() -> None:
    payload = _wire_plan().model_dump()
    payload["focus"]["subject"] = "child's hand"
    payload["focus"]["action"] = "opening a book beneath rising birds"

    plan = LiveSceneWirePlan.model_validate(payload).to_live_scene_plan()

    assert plan.focus.prompt == (
        "a complete visible child, shown opening a book beneath rising birds"
    )
    assert "hand" not in plan.focus.prompt


def test_wire_plan_preserves_safe_common_breed_action_and_flower() -> None:
    source = "golden retriever running around a field of daisies"
    wire = LiveSceneWirePlan(
        background_prompt="green field of daisies",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="golden retriever",
            action="running around field",
        ),
        magic=LiveSceneWireMagic(kind="prop", prompt="daisies"),
    )

    plan = wire.privacy_sanitized(source_text=source).to_live_scene_plan(context_text=source)
    validate_live_scene_plan_privacy(plan, source_text=source)

    assert "golden retriever" in plan.focus.prompt
    assert "shown running" in plan.focus.prompt
    assert plan.accent.prompt == "daisies"


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


def test_wire_plan_removes_main_actor_and_action_from_background() -> None:
    source = "golden retriever running around a field of daisies"
    wire = LiveSceneWirePlan(
        background_prompt="vast green field wildflowers retriever running",
        focus=LiveSceneWireFocus(
            kind="character",
            subject="golden retriever",
            action="running around",
        ),
        magic=LiveSceneWireMagic(kind="prop", prompt="daisies"),
    )

    sanitized = wire.privacy_sanitized(source_text=source)

    assert sanitized.background_prompt == "vast green field wildflowers"
    assert "retriever" not in sanitized.background_prompt
    assert "running" not in sanitized.background_prompt


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("open book", "opening book"),
        ("carries library", "carrying library"),
        ("plants brass seed", "planting brass seed"),
        ("climbs moonlit stairs", "climbing moonlit stairs"),
        ("swims through water", "swimming through water"),
    ],
)
def test_wire_plan_normalizes_visible_character_action(
    action: str,
    expected: str,
) -> None:
    payload = _wire_plan().model_dump()
    payload["focus"]["subject"] = "the child"
    payload["focus"]["action"] = action

    plan = LiveSceneWirePlan.model_validate(payload).to_live_scene_plan()

    assert plan.focus.prompt == f"a complete visible child, shown {expected}"


@pytest.mark.parametrize(
    ("background", "expected"),
    [
        ("midnight sky above silver clouds", ["stars"]),
        ("turquoise underwater canyon", ["light_rays"]),
        ("floating botanical garden", ["fireflies"]),
        ("saffron desert at sunrise", ["dust", "light_rays"]),
    ],
)
def test_wire_plan_derives_setting_specific_ambience(
    background: str,
    expected: list[str],
) -> None:
    payload = _wire_plan().model_dump()
    payload["background_prompt"] = background

    plan = LiveSceneWirePlan.model_validate(payload).to_live_scene_plan()

    assert plan.ambience == expected


def test_wire_plan_redacts_distinctive_source_phrase_without_inventing_text() -> None:
    source = "A turtle climbs moonlit stairs while glowing jellyfish drift between stars."
    payload = _wire_plan().model_dump()
    payload["magic"]["prompt"] = "glowing jellyfish drift between stars"
    wire = LiveSceneWirePlan.model_validate(payload).privacy_sanitized(source_text=source)
    plan = wire.to_live_scene_plan(context_text=source)

    assert "glowing" in plan.accent.prompt
    assert "jellyfish" in plan.accent.prompt
    assert "drifting" in plan.accent.prompt
    assert "stars" in plan.accent.prompt
    assert "drift" not in plan.accent.prompt.split()
    validate_live_scene_plan_privacy(plan, source_text=source)
    assert plan.ambience == ["stars"]


def test_wire_plan_recovers_transformation_subject_before_trailing_pronoun() -> None:
    source = (
        "A clockwork fox plants a brass seed in the snow, and a transparent forest "
        "of glass branches rises around it."
    )
    payload = _wire_plan().model_dump()
    payload["focus"] = {
        "kind": "character",
        "subject": "clockwork fox",
        "action": "plants seed",
    }
    payload["magic"] = {"kind": "character", "prompt": "clockwork fox branches rises it"}

    wire = LiveSceneWirePlan.model_validate(payload).privacy_sanitized(source_text=source)
    plan = wire.to_live_scene_plan(context_text=source)

    assert "forest" in plan.accent.prompt
    assert "branches" in plan.accent.prompt
    assert "rising" in plan.accent.prompt
    assert not plan.accent.prompt.endswith(" it")
    assert plan.accent.kind == "effect"
    validate_live_scene_plan_privacy(plan, source_text=source)


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
    assert "Show the background, subject, and supporting visual simultaneously" in (
        page.scene_spec.master_prompt
    )
    assert "exactly one main actor, shown once, performing every required action" in (
        page.scene_spec.master_prompt
    )
    assert "visually dominant single subject" in page.scene_spec.master_prompt
    assert "duplicate person" in page.scene_spec.negative_prompt
    assert "main subject at left" in page.scene_spec.master_prompt
    assert "supporting detail at upper right" in page.scene_spec.master_prompt
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


def test_open_landscape_prompt_rejects_giant_unrequested_structures() -> None:
    plan = _plan().model_copy(update={"background_prompt": "green field of daisies"})

    page = plan.to_page(
        source_text="A retriever runs through daisies.",
        visual_style="luminous layered paper theater",
        seed=23,
    )

    assert page.scene_spec is not None
    prompt = page.scene_spec.master_prompt
    assert "outdoor setting open and unobstructed" in prompt
    assert "walls, caves, portals, stage frames, monoliths" in prompt


def test_background_drops_repeated_foreground_actor_or_tool() -> None:
    plan = _plan().model_copy(
        update={
            "background_prompt": "moonlit observatory, brass telescope, starry open dome",
            "focus": _plan().focus.model_copy(
                update={"prompt": "a complete visible astronomer raises brass telescope"}
            ),
        }
    )

    page = plan.to_page(
        source_text="A figure studies the night as sea creatures cross overhead.",
        visual_style="luminous paper theater",
        seed=17,
    )

    assert page.layers[0].prompt == "moonlit observatory, starry open dome"
    assert "Background: moonlit observatory, starry open dome." in (page.scene_spec.master_prompt)


def test_background_drops_body_fragment_and_dangling_article() -> None:
    plan = _plan().model_copy(
        update={
            "background_prompt": (
                "A student's hand gently holding a butterfly, desk, classroom ceiling, a"
            ),
            "focus": _plan().focus.model_copy(
                update={"prompt": "a complete visible student lifting folded butterfly"}
            ),
        }
    )

    page = plan.to_page(
        source_text="A learner raises folded wings as flowers appear overhead.",
        visual_style="luminous paper theater",
        seed=17,
    )

    assert page.layers[0].prompt == "desk, classroom ceiling"
    assert "student's hand" not in page.scene_spec.master_prompt


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


def test_plan_strips_embedded_labeled_coordinates_from_background_prompt() -> None:
    payload = _plan().model_dump()
    payload["background_prompt"] = (
        "Dark, swirling clouds, soft blue glow, textured paper, subtle shadows, "
        "[center_x:0, center_y:0, width:1, height:1.5], depth:0.5"
    )

    page = LiveScenePlan.model_validate(payload).to_page(
        source_text="A book opens beneath a storm.",
        visual_style="paper theater",
        seed=32,
    )

    background = page.layers[0].prompt.lower()
    assert "dark, swirling clouds" in background
    assert "textured paper" in background
    for leaked_coordinate in (
        "[",
        "]",
        "center_x",
        "center_y",
        "width",
        "height",
        "depth",
    ):
        assert leaked_coordinate not in background
    assert not any(character.isdigit() for character in background)


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

    assert page.scene_summary == ("A child lifts a book of birds. The birds rise toward the moon")
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
        source_text=("A child named Quenlora opens a silent volume; folded shapes glow above it."),
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
        plan = _wire_plan()
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
    assert result.plan == _wire_plan().privacy_sanitized(source_text=source).to_live_scene_plan(
        context_text=source
    )
    assert result.metrics.model == "gemma3:1b"
    assert result.model_revision == "sha256:gemma-fixture"
    assert result.wall_ms >= 0
    assert stub.calls[0]["output_type"] is LiveSceneWirePlan
    assert "origami birds light the sky" in str(stub.calls[0]["prompt"])
    assert "luminous watercolor paper theater" not in str(stub.calls[0]["prompt"])
    assert '"seed"' not in str(stub.calls[0]["prompt"])
    assert "focus.action must separately state the exact visible action" in str(
        stub.calls[0]["prompt"]
    )
    assert "Never copy three adjacent words" in str(stub.calls[0]["prompt"])
    assert "most visually surprising transformation" in str(stub.calls[0]["prompt"])
    assert "exact visible action" in str(stub.calls[0]["prompt"])
    assert "Never invent a transformation" in str(stub.calls[0]["prompt"])
    assert "magic.prompt must name that concrete result" in str(stub.calls[0]["prompt"])
    assert "Stop the action before a later magical transformation" in str(stub.calls[0]["prompt"])
    assert "essential object or destination" in str(stub.calls[0]["prompt"])
    assert "inspect that clause first" in str(stub.calls[0]["prompt"])
    assert "luminous moths spiral through arch" in str(stub.calls[0]["prompt"])


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
        _wire_plan()
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


def test_structured_planner_does_not_spend_its_timeout_behind_warmup() -> None:
    stub = _ModelStub(delay_seconds=0.01)
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    async def warm_then_plan():
        warmup = asyncio.create_task(planner.warmup())
        await asyncio.sleep(0)
        plan = asyncio.create_task(
            planner.plan(
                text="A child opens a quiet book while paper birds rise.",
                visual_style="luminous watercolor paper theater",
                seed=23,
            )
        )
        return await warmup, await plan

    warmup, plan = asyncio.run(warm_then_plan())

    assert warmup.metrics.output_tokens == 5
    assert plan.plan.focus.prompt
    assert [call["output_type"].__name__ for call in stub.calls] == [
        "_LiveScenePlannerWarmupOutput",
        "LiveSceneWirePlan",
    ]


def test_structured_planner_reuses_privacy_gated_semantics_for_new_seed_and_style() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        cache_entries=2,
    )
    request = {
        "text": "A child opens a quiet book while paper birds rise.",
        "visual_style": "luminous watercolor paper theater",
    }

    first = asyncio.run(planner.plan(**request, seed=23))
    second = asyncio.run(
        planner.plan(
            text=request["text"],
            visual_style="bright clay animation",
            seed=24,
        )
    )

    assert len(stub.calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.plan == first.plan
    assert second.metrics.model == first.metrics.model
    assert second.metrics.total_ms == 0
    assert second.metrics.input_tokens == 0
    assert second.metrics.output_tokens == 0


def test_structured_planner_cache_is_bounded_by_passage_and_reuses_new_styles() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        cache_entries=1,
    )
    text = "A child opens a quiet book while paper birds rise."

    first = asyncio.run(planner.plan(text=text, visual_style="paper theater", seed=1))
    restyled = asyncio.run(planner.plan(text=text, visual_style="oil pastel", seed=1))
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
    assert repeated.cache_hit is False


def test_structured_planner_cache_can_be_disabled() -> None:
    stub = _ModelStub()
    planner = StructuredLiveScenePlanner(
        stub,  # type: ignore[arg-type]
        timeout_seconds=1,
        cache_entries=0,
    )

    for seed in (1, 2):
        result = asyncio.run(
            planner.plan(text="A book opens.", visual_style="paper art", seed=seed)
        )
        assert result.cache_hit is False
    assert len(stub.calls) == 2


def test_private_persistent_plan_cache_survives_restart_without_storing_source(
    tmp_path: Path,
) -> None:
    text = "A child opens a quiet book while paper birds rise above a floating school."
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
