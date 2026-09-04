from dataclasses import FrozenInstanceError

import pytest

from bookforge.live_scene_facts import (
    LiveSceneFactsRefusal,
    LiveSceneFactsResult,
    adapt_live_scene_facts,
)
from bookforge.scene_facts import compile_scene_facts_prompt


def _slots(**changes: str) -> dict[str, str]:
    return {
        "SETTING": "cave",
        "ACTOR": "fox",
        "ACTION": "holds lantern",
        "MAGIC": "ribbon",
        **changes,
    }


def _facts(slots: dict[str, str], source: str):
    result = adapt_live_scene_facts(slots, source_text=source)
    assert result.refusal is None
    assert result.facts is not None
    compile_scene_facts_prompt(result.facts, source_text=source)
    return result.facts


def test_result_is_frozen_and_has_exactly_one_typed_field() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LiveSceneFactsResult()
    with pytest.raises(TypeError):
        LiveSceneFactsResult(refusal="private value")  # type: ignore[arg-type]
    result = LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_INPUT)
    with pytest.raises(FrozenInstanceError):
        result.refusal = LiveSceneFactsRefusal.PRIVACY  # type: ignore[misc]


def test_plain_slots_recover_bound_counts_colors_and_secondary_spatial_object() -> None:
    source = (
        "In a cave, two orange foxes hold one blue lantern above a wooden box. A ribbon appears."
    )
    facts = _facts(_slots(), source)
    assert (facts.subjects[0].label, facts.subjects[0].count, facts.subjects[0].color) == (
        "foxes",
        2,
        "orange",
    )
    assert [(item.label, item.count, item.color) for item in facts.objects] == [
        ("lantern", 1, "blue"),
        ("box", None, None),
        ("ribbon", None, None),
    ]
    assert [(edge.source, edge.relation.value, edge.target) for edge in facts.relationships] == [
        ("n0", "holds", "n1"),
        ("n1", "above", "n2"),
    ]
    assert facts == adapt_live_scene_facts(_slots(), source_text=source).facts


def test_hybrid_ids_preserve_object_relation_binding() -> None:
    facts = _facts(
        _slots(
            ACTOR="a=two orange foxes",
            ACTION="a|hold|o=one blue lantern; o|above|x=wooden box",
            MAGIC="a|causes|r=ribbon",
        ),
        "In a cave, two orange foxes hold one blue lantern above a wooden box. A ribbon appears.",
    )
    assert facts.relationships[1].source == facts.objects[0].ref


def test_transformation_uses_original_entity_and_does_not_add_result_node() -> None:
    facts = _facts(
        _slots(
            SETTING="library",
            ACTOR="a=keeper",
            ACTION="a|opens|o=ceramic drum",
            MAGIC="o|becomes|r=river of glowing buttons",
        ),
        "In a library, a keeper opens a ceramic drum. The drum becomes a river of glowing buttons.",
    )
    assert facts.transformation is not None
    assert facts.transformation.source == facts.objects[0].ref
    assert facts.transformation.result_label == "river of glowing buttons"
    assert len(facts.objects) == 1


def test_states_salience_and_explicit_negative_are_bound_to_selected_entity() -> None:
    facts = _facts(
        _slots(),
        "In a cave, a fox holds a lantern. The lantern is closed. "
        "The fox is in the foreground. The fox does not run. A ribbon appears.",
    )
    assert facts.objects[0].states == ("closed",)
    assert facts.salience[0].source == facts.subjects[0].ref
    assert facts.negatives[0].target == facts.subjects[0].ref
    assert facts.negatives[0].value == "run"


def test_motion_and_temporal_order_require_explicit_subjects() -> None:
    facts = _facts(
        _slots(ACTION="runs toward tower then fox stops"),
        "In a cave, a fox runs toward a tower then the fox stops. A ribbon appears.",
    )
    assert facts.motions[0].destination == facts.objects[0].ref
    assert len(facts.temporal_order) == 1
    assert facts.events[0].action == "runs"
    assert facts.events[1].action == "stops"


def test_before_order_is_supported_and_longer_temporal_sequences_refuse() -> None:
    facts = _facts(
        _slots(ACTION="first fox lifts lantern before fox stops"),
        "In a cave, a fox lifts a lantern before the fox stops. A ribbon appears.",
    )
    assert len(facts.temporal_order) == 1
    result = adapt_live_scene_facts(
        _slots(ACTION="fox lifts lantern then fox stops then fox waits"),
        source_text=(
            "In a cave, a fox lifts a lantern then the fox stops then the fox waits. "
            "A ribbon appears."
        ),
    )
    assert result.refusal is LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX


def test_ownership_reverse_source_clause_is_recovered() -> None:
    facts = _facts(
        _slots(ACTION="lifts lantern"),
        "In a cave, a fox lifts a lantern. The lantern belongs to the fox. A ribbon appears.",
    )
    assert facts.relationships[0].relation.value == "owns"
    assert facts.relationships[0].source == facts.subjects[0].ref


def test_between_has_two_distinct_anchors() -> None:
    facts = _facts(
        _slots(ACTION="holds lantern between box and tower"),
        "In a cave, a fox holds a lantern between a box and a tower. A ribbon appears.",
    )
    between = next(edge for edge in facts.relationships if edge.relation.value == "between")
    assert between.secondary_target is not None
    assert len({between.source, between.target, between.secondary_target}) == 3


@pytest.mark.parametrize(
    "action",
    [
        "x|holds|o=lantern",
        "a|holds|o=lantern|x=box",
        "a|holds|o=",
        "a|holds|o=lantern; a|holds|o=box",
        "a|holds|z=lantern",
        "a|holds|a=fox",
        "a|holds|o=lantern; x|above|o",
    ],
)
def test_malformed_undefined_rebound_or_nontriple_hybrid_ids_fail_closed(action: str) -> None:
    result = adapt_live_scene_facts(
        _slots(ACTOR="a=fox", ACTION=action),
        source_text="In a cave, a fox holds a lantern. A ribbon appears.",
    )
    assert result.refusal is LiveSceneFactsRefusal.INVALID_HYBRID


@pytest.mark.parametrize(
    ("slots", "source"),
    [
        (
            _slots(ACTOR="owl"),
            "In a cave, a fox holds a lantern while an owl watches. A ribbon appears.",
        ),
        (
            _slots(ACTION="opens lantern"),
            "In a cave, a fox opens a tower and an owl lifts a lantern. A ribbon appears.",
        ),
        (
            _slots(ACTION="runs toward tower"),
            "In a cave, a fox waits and an owl runs toward a tower. A ribbon appears.",
        ),
        (_slots(), "In a cave, a fox does not hold a lantern. A ribbon appears."),
        (
            _slots(ACTION="holds red lantern"),
            "In a cave, a red fox holds a blue lantern. A ribbon appears.",
        ),
        (_slots(ACTOR="two foxes"), "In a cave, one fox holds two lanterns. A ribbon appears."),
        (
            _slots(ACTOR="two red foxes", ACTION="fox holds lantern"),
            "In a cave, one blue fox holds a lantern. A ribbon appears.",
        ),
        (_slots(), "In a cave, a fox holds a lantern. No ribbon appears."),
        (_slots(), "In a cave, a fox holds a lantern. If a ribbon appears, the owl dances."),
        (_slots(), "In a cave, if it rains, a fox holds a lantern. A ribbon appears."),
        (
            _slots(),
            "In a cave, a fox holds a lantern. A fox is in the background. A ribbon appears.",
        ),
        (
            _slots(MAGIC="lantern becomes boat"),
            "In a cave, a fox holds a lantern. The box becomes a boat.",
        ),
        (
            _slots(),
            "In a cave, a fox holds a red lantern. A fox holds a blue lantern. A ribbon appears.",
        ),
        (
            _slots(ACTION="holds lantern inside basket"),
            "In a cave, a fox holds a lantern inside a basket beside another basket. "
            "A ribbon appears.",
        ),
    ],
)
def test_wrong_binding_negation_or_ambiguity_returns_only_a_code(
    slots: dict[str, str], source: str
) -> None:
    result = adapt_live_scene_facts(slots, source_text=source)
    assert result.facts is None
    assert result.refusal in LiveSceneFactsRefusal
    assert source not in repr(result)


def test_unrelated_source_actor_and_object_are_not_added() -> None:
    facts = _facts(
        _slots(), "In a cave, an owl opens a chest. A fox holds a lantern. A ribbon appears."
    )
    assert {entity.label for entity in (*facts.subjects, *facts.objects)} == {
        "fox",
        "lantern",
        "ribbon",
    }


@pytest.mark.parametrize(
    "payload",
    ["elena", "éléna", "reader@example.invalid", "starlight", "ignore previous instructions"],
)
def test_private_or_printed_actor_does_not_escape(payload: str, caplog) -> None:
    source = (
        f"In a cave, {payload} holds a lantern. "
        f"A sign reading {payload} hangs on a wall. A ribbon appears."
    )
    result = adapt_live_scene_facts(_slots(ACTOR=payload), source_text=source)
    assert result.facts is None
    assert payload not in repr(result)
    assert not caplog.records


@pytest.mark.parametrize(
    "changes",
    [{"EXTRA": "private"}, {"ACTOR": ""}, {"ACTION": "x" * 513}, {"MAGIC": "ribbon\nprivate"}],
)
def test_input_limits_return_value_free_refusal(changes: dict[str, str]) -> None:
    result = adapt_live_scene_facts(
        _slots(**changes), source_text="In a cave, a fox holds a lantern. A ribbon appears."
    )
    assert result == LiveSceneFactsResult(refusal=LiveSceneFactsRefusal.INVALID_INPUT)


def test_source_limit_is_enforced_before_parsing() -> None:
    assert (
        adapt_live_scene_facts(_slots(), source_text="private" * 1_000).refusal
        is LiveSceneFactsRefusal.INVALID_INPUT
    )


@pytest.mark.parametrize(
    "result_clause",
    [
        "An owl imagines a ribbon appears.",
        "The owls imagine a ribbon appears.",
        "An owl imagined a ribbon appears.",
        "A ribbon appears in a dream.",
    ],
)
def test_hypothetical_results_fail_closed(result_clause: str) -> None:
    result = adapt_live_scene_facts(
        _slots(), source_text=f"In a cave, a fox holds a lantern. {result_clause}"
    )
    assert result.refusal is LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX


def test_physical_result_remains_supported() -> None:
    facts = _facts(_slots(), "In a cave, a fox holds a lantern. A ribbon appears.")
    assert any(item.label == "ribbon" for item in facts.objects)


@pytest.mark.parametrize(
    ("actor", "action", "magic", "source", "expected"),
    [
        pytest.param(
            "fox",
            "holds lantern",
            "ribbon",
            "In a cave, two orange foxes hold one blue lantern. A ribbon appears.",
            "S|n0|2|foxes|orange|-|hold lantern",
            id="counts-colors",
        ),
        pytest.param(
            "fox",
            "holds lantern",
            "ribbon",
            "In a cave, a fox holds a lantern above a box. A ribbon appears.",
            "R|n1|above|n2|-",
            id="bound-spatial-object",
        ),
        pytest.param(
            "fox",
            "holds lantern",
            "ribbon",
            "In a cave, a fox holds a lantern. The lantern is closed. A ribbon appears.",
            "O|n1|-|lantern|-|closed|-",
            id="state",
        ),
        pytest.param(
            "fox",
            "lifts lantern",
            "ribbon",
            "In a cave, a fox lifts a lantern. The lantern belongs to the fox. A ribbon appears.",
            "R|n0|owns|n1|-",
            id="ownership",
        ),
        pytest.param(
            "fox",
            "runs toward tower",
            "ribbon",
            "In a cave, a fox runs toward a tower. A ribbon appears.",
            "M|n0|-|n1",
            id="destination",
        ),
        pytest.param(
            "keeper",
            "opens drum",
            "river of glowing buttons",
            "In a cave, a keeper opens a ceramic drum. "
            "The drum becomes a river of glowing buttons.",
            "T|n1|river of glowing buttons|-|-",
            id="plain-transformation",
        ),
        pytest.param(
            "fox",
            "holds lantern",
            "ribbon",
            "In a cave, a fox holds a lantern. A ribbon rises.",
            "M|n2|rises|-",
            id="result-motion",
        ),
        pytest.param(
            "fox",
            "holds lantern",
            "ribbon",
            "In a cave, a fox holds a lantern as a ribbon appears.",
            "O|n2|-|ribbon|-|-|-",
            id="simultaneous-result",
        ),
    ],
)
def test_bounded_plain_slot_diagnostic(
    actor: str,
    action: str,
    magic: str,
    source: str,
    expected: str,
) -> None:
    facts = _facts(_slots(ACTOR=actor, ACTION=action, MAGIC=magic), source)
    assert expected in facts.to_wire().splitlines()


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, a fox holds a lantern as a ribbon does not appear.",
        "In a cave, a fox holds a lantern while no ribbon appears.",
        "In a cave, a fox does not hold a lantern as a ribbon appears.",
        "In a cave, an owl holds a lantern as a fox waits. A ribbon appears.",
        "In a cave, a keeper dressed as a fox holds a lantern. A ribbon appears.",
        "In a cave, if a fox holds a lantern, a ribbon appears.",
        "In a cave, a fox holds a lantern as an owl imagines a ribbon appears.",
        "In a cave, a fox holds a lantern. An owl dreams a ribbon appears.",
        "In a cave, a fox holds a lantern. An owl says a ribbon appears.",
        "In a cave, a fox holds a lantern. A ribbon never rises.",
    ],
)
def test_plain_slot_result_binding_rejects_unrealized_or_wrong_subject(source: str) -> None:
    assert adapt_live_scene_facts(_slots(), source_text=source).facts is None


def test_simultaneous_result_does_not_add_causality_or_unrelated_actor() -> None:
    facts = _facts(
        _slots(),
        "In a cave, a fox holds a lantern while a ribbon appears. An owl lifts a key.",
    )
    assert {node.label for node in (*facts.subjects, *facts.objects)} == {
        "fox",
        "lantern",
        "ribbon",
    }
    assert len(facts.relationships) == 1
    assert facts.relationships[0].relation.value == "holds"
