from __future__ import annotations

from bookforge.live_scene_facts import adapt_live_scene_facts

SOURCE = "In a garden, one red basket is carried by two white badgers. A silver kite appears."
SLOTS = {
    "SETTING": "garden",
    "ACTOR": "two white badgers",
    "ACTION": "carried by 2 white badgers",
    "MAGIC": "silver kite",
}


def test_passive_fragment_recovers_the_unique_source_patient_and_number_spelling():
    for scope in ("focal", "scene"):
        facts = adapt_live_scene_facts(SLOTS, source_text=SOURCE, scope=scope).facts
        expected = adapt_live_scene_facts(
            {**SLOTS, "ACTION": "carry red basket"}, source_text=SOURCE, scope=scope
        ).facts
        assert facts is not None and facts == expected
        assert facts.subjects[0].count == 2 and facts.subjects[0].color == "white"
        basket = next(node for node in facts.objects if node.label == "basket")
        assert basket.count == 1 and basket.color == "red"
        relation = facts.relationships[0]
        assert relation.source == facts.subjects[0].ref and relation.target == basket.ref
    source = "In a garden, a red basket is carried by a badger. A silver kite appears."
    assert (
        adapt_live_scene_facts(
            {**SLOTS, "ACTOR": "one badger", "ACTION": "carried by 1 badger"},
            source_text=source,
            scope="scene",
        ).facts
        is not None
    )


def test_passive_fragment_requires_both_selected_actor_and_explicit_agent_to_match():
    for changes in (
        {"ACTOR": "one white badger"},
        {"ACTOR": "two red badgers"},
        {"ACTION": "carried by one white badger"},
        {"ACTION": "carried by two red badgers"},
        {"ACTION": "carried by two white owls"},
        {"ACTION": "held by two white badgers"},
        {"ACTION": "carried by"},
    ):
        result = adapt_live_scene_facts({**SLOTS, **changes}, source_text=SOURCE, scope="scene")
        assert result.facts is None and SOURCE not in repr(result)
    source = (
        "In a garden, one red basket is carried by two white badgers. "
        "One blue basket is carried by two red badgers. A silver kite appears."
    )
    assert (
        adapt_live_scene_facts(
            {**SLOTS, "ACTION": "carried by two badgers"}, source_text=source, scope="scene"
        ).facts
        is None
    )


def test_patient_recovery_rejects_active_ambiguous_or_unasserted_source_clauses():
    sources = (
        "In a garden, two white badgers carry one red basket. A silver kite appears.",
        "In a garden, one red basket is carried by two white badgers. "
        "One blue basket is carried by two white badgers. A silver kite appears.",
        "In a garden, one red basket is not carried by two white badgers. A silver kite appears.",
        "In a garden, an owl says, one red basket is carried by two white badgers. "
        "A silver kite appears.",
        'In a garden, a sign reads "one red basket is carried by two white badgers". '
        "A silver kite appears.",
        "In a garden, if one red basket is carried by two white badgers, a silver kite appears.",
        "In a garden, one red basket is carried by. A silver kite appears.",
    )
    for source in sources:
        for scope in ("focal", "scene"):
            assert adapt_live_scene_facts(SLOTS, source_text=source, scope=scope).facts is None
