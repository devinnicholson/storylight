from __future__ import annotations

from bookforge.live_scene_facts import adapt_live_scene_facts

SOURCE = "In a meadow, a rabbit carries a red basket. A silver kite appears."
SLOTS = {
    "SETTING": "meadow",
    "ACTOR": "rabbit",
    "ACTION": "carries red basket",
    "MAGIC": "silver kite",
}


def test_setting_articles_and_indefinite_one_preserve_the_source_graph():
    for scope in ("focal", "scene"):
        expected = adapt_live_scene_facts(SLOTS, source_text=SOURCE, scope=scope).facts
        assert expected is not None
        for changes in (
            {"SETTING": "a meadow"},
            {"SETTING": "the meadow"},
            {"ACTOR": "one rabbit"},
            {"ACTION": "carries one red basket"},
            {"MAGIC": "one silver kite"},
        ):
            result = adapt_live_scene_facts({**SLOTS, **changes}, source_text=SOURCE, scope=scope)
            assert result.facts == expected
        assert expected.setting.label == "meadow"
        assert all(node.count is None for node in (*expected.subjects, *expected.objects))
    source = "In an orchard, an owl carries a red basket. A silver kite appears."
    result = adapt_live_scene_facts(
        {**SLOTS, "SETTING": "an orchard", "ACTOR": "one owl"}, source_text=source, scope="scene"
    )
    assert result.facts is not None and result.facts.subjects[0].count is None
    for source in (
        "In a meadow, a rabbit carries a red basket. The rabbit stands beside the basket. "
        "A silver kite appears.",
        "In a meadow, a rabbit carries a red basket then the rabbit lifts a stone. "
        "A silver kite appears.",
    ):
        expected = adapt_live_scene_facts(SLOTS, source_text=source, scope="scene").facts
        actual = adapt_live_scene_facts(
            {**SLOTS, "ACTOR": "one rabbit"}, source_text=source, scope="scene"
        ).facts
        assert actual is not None and actual == expected


def test_indefinite_one_cannot_borrow_number_or_identity():
    for actor in ("the rabbit", "rabbits", "two rabbits", "a rabbits"):
        source = f"In a meadow, {actor} carries a red basket. A silver kite appears."
        assert (
            adapt_live_scene_facts(
                {**SLOTS, "ACTOR": "one rabbit"}, source_text=source, scope="scene"
            ).facts
            is None
        )
    for changes in (
        {"ACTOR": "two rabbits"},
        {"ACTION": "carries two red baskets"},
        {"MAGIC": "two silver kites"},
        {"ACTION": "carries one blue basket"},
    ):
        assert (
            adapt_live_scene_facts({**SLOTS, **changes}, source_text=SOURCE, scope="scene").facts
            is None
        )
    source = (
        "In a meadow, two rabbits carry a red basket. "
        "An owl holds a blue basket. A silver kite appears."
    )
    assert (
        adapt_live_scene_facts(
            {**SLOTS, "ACTOR": "one rabbit", "ACTION": "carry red basket"},
            source_text=source,
            scope="scene",
        ).facts
        is None
    )
    valid = adapt_live_scene_facts(
        {**SLOTS, "ACTOR": "two rabbits", "ACTION": "carry red basket"},
        source_text=source,
        scope="scene",
    ).facts
    assert valid is not None and valid.subjects[0].count == 2


def test_surface_normalization_retains_setting_and_privacy_boundaries():
    for setting in (
        "a forest",
        "a meadow beyond the wall",
        "a the meadow",
        "a reader@example.invalid",
    ):
        assert (
            adapt_live_scene_facts(
                {**SLOTS, "SETTING": setting}, source_text=SOURCE, scope="scene"
            ).facts
            is None
        )
    for suffix in (" The rabbit ponders a puzzle.", ' A sign reads "two blue birds".'):
        result = adapt_live_scene_facts(
            {**SLOTS, "SETTING": "the meadow", "ACTOR": "one rabbit"},
            source_text=SOURCE + suffix,
            scope="scene",
        )
        assert result.facts is None and SOURCE not in repr(result)
