from dataclasses import FrozenInstanceError

import pytest

from bookforge.live_scene_facts import (
    LiveSceneFactsRefusal,
    LiveSceneFactsResult,
    adapt_live_scene_facts,
)
from bookforge.scene_facts import compile_scene_facts_prompt
from bookforge.tensorrt_slot_client import tensor_accepted_graph_wire_plan, tensor_slot_wire_plan


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


def test_explicit_passive_carry_preserves_binding_through_renderer() -> None:
    source = "In a cave, one blue lantern is carried by two orange foxes. A ribbon appears."
    slots = _slots(ACTION="carries lantern")
    facts = _facts(slots, source)
    assert (facts.subjects[0].label, facts.subjects[0].count, facts.subjects[0].color) == (
        "foxes",
        2,
        "orange",
    )
    assert (facts.objects[0].label, facts.objects[0].count, facts.objects[0].color) == (
        "lantern",
        1,
        "blue",
    )
    edge = facts.relationships[0]
    assert (edge.source, edge.relation.value, edge.target) == (
        facts.subjects[0].ref,
        "carries",
        facts.objects[0].ref,
    )
    wire = tensor_accepted_graph_wire_plan(
        "\n".join(f"{key}: {value}" for key, value in slots.items()), source_text=source
    )
    assert wire.scene_facts == facts
    page = wire.to_live_scene_plan(context_text=source).to_page(
        source_text=source, visual_style="watercolor", seed=0
    )
    assert page.scene_spec.master_prompt == compile_scene_facts_prompt(
        facts, source_text=source, visual_style="watercolor"
    )


@pytest.mark.parametrize(
    "clause",
    [
        "a lantern is not carried by a fox",
        "an owl claims, a lantern is carried by a fox",
        "if a lantern is carried by a fox, an owl watches",
        "a lantern is carried by an owl",
        "a fox is carried by a lantern",
        "a lantern is carried by an owl; a kettle is carried by a fox",
        "a lantern is carried by her",
        "a lantern is carried by a fox and an owl",
    ],
)
def test_passive_carry_refusal_preserves_accepted_fallback(clause: str) -> None:
    source = f"In a cave, {clause}. A ribbon appears."
    slots = _slots(ACTION="carries lantern")
    assert adapt_live_scene_facts(slots, source_text=source).facts is None
    raw = "\n".join(f"{key}: {value}" for key, value in slots.items())
    accepted = tensor_slot_wire_plan(raw, source_text=source)
    candidate = tensor_accepted_graph_wire_plan(raw, source_text=source)
    assert candidate.scene_facts is None
    assert candidate.model_dump(exclude={"scene_facts"}) == accepted.model_dump()


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


def test_ownership_reverse_source_clause_is_recovered() -> None:
    facts = _facts(
        _slots(ACTION="lifts lantern"),
        "In a cave, a fox lifts a lantern. The lantern belongs to the fox. A ribbon appears.",
    )
    assert facts.relationships[0].relation.value == "owns"
    assert facts.relationships[0].source == facts.subjects[0].ref


@pytest.mark.parametrize(
    "action",
    ["x|holds|o=lantern", "a|holds|o=lantern|x=box", "a|holds|o=lantern; a|holds|o=box"],
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
            _slots(ACTION="holds red lantern"),
            "In a cave, a red fox holds a blue lantern. A ribbon appears.",
        ),
        (_slots(ACTOR="two foxes"), "In a cave, one fox holds two lanterns. A ribbon appears."),
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


@pytest.mark.parametrize("payload", ["reader@example.invalid"])
def test_private_or_printed_actor_does_not_escape(payload: str, caplog) -> None:
    source = (
        f"In a cave, {payload} holds a lantern. "
        f"A sign reading {payload} hangs on a wall. A ribbon appears."
    )
    result = adapt_live_scene_facts(_slots(ACTOR=payload), source_text=source)
    assert result.facts is None
    assert payload not in repr(result)
    assert not caplog.records


@pytest.mark.parametrize("changes", [{"ACTOR": ""}, {"ACTION": "x" * 513}])
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


@pytest.mark.parametrize("result_clause", ["An owl imagines a ribbon appears."])
def test_hypothetical_results_fail_closed(result_clause: str) -> None:
    result = adapt_live_scene_facts(
        _slots(), source_text=f"In a cave, a fox holds a lantern. {result_clause}"
    )
    assert result.refusal is LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, a fox holds a lantern as a ribbon does not appear.",
        "In a cave, a keeper dressed as a fox holds a lantern. A ribbon appears.",
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


@pytest.mark.parametrize("predicate", ["rises"])
def test_magic_noun_and_explicit_physical_predicate_produce_same_graph(predicate: str) -> None:
    source = f"In a cave, a badger lifts a thimble. A comet {predicate}."
    base = _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet")
    assert _facts(base, source) == _facts({**base, "MAGIC": f"comet {predicate}"}, source)


@pytest.mark.parametrize("relation", ["above"])
def test_magic_result_keeps_its_direct_anchor_for_noun_and_clause_slots(relation: str) -> None:
    source = f"In a cave, a badger lifts a thimble. A comet appears {relation} a fountain."
    base = _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet")
    facts = _facts(base, source)
    assert facts == _facts({**base, "MAGIC": f"comet appears {relation} fountain"}, source)
    assert facts == _facts({**base, "MAGIC": f"comet {relation} fountain"}, source)
    entities = {node.ref: node.label for node in (*facts.subjects, *facts.objects)}
    edge = facts.relationships[0]
    assert (entities[edge.source], edge.relation.value, entities[edge.target]) == (
        "comet",
        relation,
        "fountain",
    )
    wrong = "below" if relation == "above" else "above"
    assert (
        adapt_live_scene_facts(
            {**base, "MAGIC": f"comet appears {wrong} fountain"}, source_text=source
        ).facts
        is None
    )


def test_unrelated_result_subject_cannot_supply_magic_anchor() -> None:
    source = "In a cave, a badger lifts a thimble. A comet appears. An owl rises above a fountain."
    base = _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet")
    facts = _facts(base, source)
    assert {node.label for node in (*facts.subjects, *facts.objects)} == {
        "badger",
        "thimble",
        "comet",
    }
    assert (
        adapt_live_scene_facts(
            {**base, "MAGIC": "comet appears above fountain"}, source_text=source
        ).facts
        is None
    )


@pytest.mark.parametrize(("link", "adverb"), [("calling forth", "deliberately")])
def test_explicit_action_linked_result_with_neutral_adverb(link: str, adverb: str) -> None:
    facts = _facts(
        _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet"),
        f"In a cave, a badger {adverb} lifts a thimble, {link} a comet.",
    )
    assert facts.subjects[0].actions == ("lifts thimble",)
    assert {node.label for node in facts.objects} == {"thimble", "comet"}


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, a badger allegedly lifts a thimble, causing a comet.",
        "In a cave, a badger does not lift a thimble, causing a comet.",
        "In a cave, a badger lifts a thimble. An owl waits, causing a comet.",
    ],
)
def test_action_linked_result_requires_actual_selected_antecedent(source: str) -> None:
    assert (
        adapt_live_scene_facts(
            _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet"), source_text=source
        ).facts
        is None
    )


def test_transformation_preserves_explicit_result_count() -> None:
    source = "In a cave, a badger lifts a thimble. The thimble becomes two kettles."
    facts = _facts(_slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="two kettles"), source)
    assert facts.transformation is not None
    assert facts.transformation.result_count == 2
    assert "exactly 2" in compile_scene_facts_prompt(facts, source_text=source)
    assert (
        adapt_live_scene_facts(
            _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="three kettles"),
            source_text=source,
        ).facts
        is None
    )


def test_repeated_action_temporal_order_is_bound_to_its_object() -> None:
    source = "In a cave, a badger lifts a thimble then the badger lifts a kettle. A comet appears."
    base = _slots(ACTOR="badger", MAGIC="comet")
    assert (
        adapt_live_scene_facts(
            {**base, "ACTION": "lifts thimble then badger lifts kettle"}, source_text=source
        ).facts
        is not None
    )
    assert (
        adapt_live_scene_facts(
            {**base, "ACTION": "lifts kettle then badger lifts thimble"}, source_text=source
        ).facts
        is None
    )


@pytest.mark.parametrize("verb", ["promises"])
def test_unlisted_speech_predicates_are_not_treated_as_visible_actions(verb: str) -> None:
    result = adapt_live_scene_facts(
        _slots(ACTOR="badger", ACTION=f"{verb} lantern", MAGIC="comet"),
        source_text=f"In a cave, a badger {verb} a lantern. A comet appears.",
    )
    assert result.refusal is LiveSceneFactsRefusal.UNSUPPORTED_SYNTAX


@pytest.mark.parametrize("verb", ["claims"])
def test_action_linked_result_does_not_discard_reported_clause_scope(verb: str) -> None:
    result = adapt_live_scene_facts(
        _slots(ACTOR="badger", ACTION="lifts thimble", MAGIC="comet"),
        source_text=(f"In a cave, an owl {verb}, a badger lifts a thimble, causing a comet."),
    )
    assert result.facts is None
