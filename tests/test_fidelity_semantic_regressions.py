from __future__ import annotations

import pytest

from bookforge.fidelity_evaluation import (
    FIDELITY_EVALUATOR_REVISION,
    _descriptor_phrases,
    _scene_graph_signature,
    evaluate_surface,
    extract_surface,
)
from bookforge.scene_facts import SceneFactsV2


def record(*expectations, passage=None, **updates):
    return {
        "record_id": "synthetic-semantic-control",
        "split": "synthetic",
        "passage": passage
        or (
            "In a quiet cave, a red fox holds a wooden cup above a stone box. "
            "Outside, rain falls softly while the animals stay sheltered."
        ),
        "expectations": list(expectations),
        **updates,
    }


def relation(subject="fox", predicate="holds", object_value="cup", **updates):
    return {
        "kind": "relation",
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "required": True,
        **updates,
    }


def facts():
    return SceneFactsV2.model_validate(
        {
            "setting": {"label": "quiet cave"},
            "subjects": [{"ref": "s1", "label": "red fox", "actions": ["holds the wooden cup"]}],
            "objects": [{"ref": "o1", "label": "wooden cup"}, {"ref": "o2", "label": "stone box"}],
            "relationships": [
                {"source": "s1", "relation": "holds", "target": "o1"},
                {"source": "o1", "relation": "above", "target": "o2"},
            ],
        }
    )


def equivalent(left, right):
    vocabulary = _descriptor_phrases(left) | _descriptor_phrases(right)
    return _scene_graph_signature(left, descriptor_vocabulary=vocabulary) == _scene_graph_signature(
        right, descriptor_vocabulary=vocabulary
    )


def test_semantic_articles_do_not_change_count_or_negation():
    expectation = {"kind": "slot", "slot": "ACTION", "alternatives": ["holds the two cups"]}
    expected = record(expectation, passage="A fox holds two cups inside a cave.")
    for action, passed in [
        ("holds two cups", True),
        ("holds the two cups", True),
        ("holds three cups", False),
        ("holds cups", False),
        ("does not hold two cups", False),
    ]:
        raw = f"SETTING: cave\nACTOR: fox\nACTION: {action}\nMAGIC: cups"
        evaluation = evaluate_surface(expected, raw, surface="raw")
        assert evaluation.expectation_results[0].passed is passed
        assert evaluation.evaluator_revision == FIDELITY_EVALUATOR_REVISION


@pytest.mark.parametrize("name", ["A", "An", "The"])
def test_privacy_names_and_forbidden_articles_keep_literal_matching(name):
    expected = record(
        {"kind": "slot", "slot": "ACTOR", "alternatives": ["fox"]},
        privacy_terms=[name],
        forbidden_terms=[name],
    )
    raw = f"SETTING: cave\nACTOR: {name} fox\nACTION: holds cup\nMAGIC: box"
    evaluation = evaluate_surface(expected, raw, surface="raw")
    assert evaluation.expectation_results[0].passed
    assert evaluation.privacy.privacy_term_leaks == (name,)
    assert evaluation.forbidden_hits == (name,)
    assert not evaluation.exact_example_pass


def test_privacy_source_echo_keeps_articles():
    passage = "A fox and an otter meet in the quiet cave after sunset."
    evaluation = evaluate_surface(
        record(passage=passage), {"master_prompt": passage}, surface="renderer"
    )
    assert evaluation.privacy.source_echo


def test_descriptor_partition_and_articles_preserve_entity_identity():
    original = facts()
    changed = original.model_dump()
    changed["setting"] = {"label": "cave", "attributes": ("quiet",)}
    changed["subjects"][0].update(label="fox", color="red", actions=("holds wooden cup",))
    changed["objects"][0].update(label="cup", attributes=("wooden",))
    changed["objects"][1].update(label="box", attributes=("stone",))
    assert equivalent(original, SceneFactsV2.model_validate(changed))


def test_descriptor_phrase_order_is_independent_of_attribute_order():
    original = facts().model_dump()
    original["objects"][0].update(label="cup", attributes=("wooden", "small"))
    changed = SceneFactsV2.model_validate(original).model_dump()
    changed["objects"][0]["attributes"] = ("small", "wooden")
    assert equivalent(SceneFactsV2.model_validate(original), SceneFactsV2.model_validate(changed))


@pytest.mark.parametrize("change", ["count", "color", "edge", "extra", "negative"])
def test_canonical_comparison_preserves_semantic_differences(change):
    original = facts()
    changed = original.model_dump()
    if change == "count":
        changed["objects"][0]["count"] = 2
    elif change == "color":
        changed["subjects"][0]["label"] = "blue fox"
    elif change == "edge":
        changed["relationships"][1]["relation"] = "under"
    elif change == "extra":
        changed["objects"] = (*changed["objects"], {"ref": "extra", "label": "key"})
    else:
        changed["negatives"] = ({"kind": "additional_subject", "value": "dragon"},)
    assert not equivalent(original, SceneFactsV2.model_validate(changed))


def test_reference_renaming_is_safe_but_ownership_swap_is_not():
    original = SceneFactsV2.model_validate(
        {
            "setting": {"label": "cave"},
            "subjects": [{"ref": "s1", "label": "fox"}, {"ref": "s2", "label": "otter"}],
            "objects": [{"ref": "o1", "label": "red cup"}, {"ref": "o2", "label": "blue cup"}],
            "relationships": [
                {"source": "s1", "relation": "owns", "target": "o1"},
                {"source": "s2", "relation": "owns", "target": "o2"},
            ],
        }
    )
    renamed = original.model_dump()
    renamed["subjects"][0]["ref"] = "newfox"
    renamed["relationships"][0]["source"] = "newfox"
    assert equivalent(original, SceneFactsV2.model_validate(renamed))
    swapped = original.model_dump()
    swapped["relationships"][0]["target"] = "o2"
    swapped["relationships"][1]["target"] = "o1"
    assert not equivalent(original, SceneFactsV2.model_validate(swapped))


def test_canonical_alias_collapse_refuses_instead_of_merging_nodes():
    graph = SceneFactsV2.model_validate(
        {
            "setting": {"label": "cave"},
            "subjects": [
                {"ref": "a", "label": "fox"},
                {"ref": "b", "label": "the fox"},
            ],
        }
    )
    with pytest.raises(ValueError, match="distinct"):
        _scene_graph_signature(graph)


def test_temporal_order_remains_bound_to_complete_event_identity():
    original = facts().model_dump()
    original["events"] = (
        {"ref": "e1", "source": "s1", "action": "lifts", "object": "o1"},
        {"ref": "e2", "source": "s1", "action": "opens", "object": "o2"},
    )
    original["temporal_order"] = ({"before": "e1", "after": "e2"},)
    graph = SceneFactsV2.model_validate(original)
    changed = graph.model_dump()
    changed["temporal_order"] = ({"before": "e2", "after": "e1"},)
    assert not equivalent(graph, SceneFactsV2.model_validate(changed))


@pytest.mark.parametrize("change", ["missing", "different", "style", "metadata_only", "extra"])
def test_renderer_metadata_cannot_spoof_emitted_prompt(change):
    graph = facts()
    source = record(relation())
    output = {
        "scene_facts": graph.model_dump(),
        "visual_style": "watercolor",
        "master_prompt": graph.to_renderer_prompt(
            source_text=source["passage"], visual_style="watercolor"
        ),
    }
    if change == "missing":
        del output["master_prompt"]
    elif change == "different":
        output["master_prompt"] = "An otter holds a key."
    elif change == "style":
        output["visual_style"] = "charcoal"
    elif change == "metadata_only":
        output = graph
    else:
        output["extra_prompt"] = "A dragon appears."
    evaluation = evaluate_surface(source, output, surface="renderer")
    assert not evaluation.schema_valid
    assert not evaluation.exact_example_pass


def test_verified_renderer_uses_bound_graph_and_requires_local_source():
    graph = facts()
    source = record(
        relation(), {"kind": "slot", "slot": "ACTION", "alternatives": ["holds wooden cup"]}
    )
    output = {
        "scene_facts": graph.model_dump(),
        "visual_style": "watercolor",
        "master_prompt": graph.to_renderer_prompt(
            source_text=source["passage"], visual_style="watercolor"
        ),
    }
    evaluation = evaluate_surface(source, output, surface="renderer")
    assert evaluation.schema_valid
    assert evaluation.exact_example_pass
    assert not extract_surface(output, surface="renderer").schema_valid


@pytest.mark.parametrize(
    "text,passed",
    [
        ("fox holds cup; otter watches", True),
        ("otter holds cup; fox watches", False),
        ("fox watches; otter holds cup", False),
        ("fox does not hold cup", False),
        ("fox holds box; cup is near otter", False),
    ],
)
def test_lexical_renderer_relation_requires_bound_unnegated_clause(text, passed):
    evaluation = evaluate_surface(record(relation()), {"master_prompt": text}, surface="renderer")
    assert evaluation.expectation_results[0].passed is passed


def test_raw_actor_action_binding_rejects_explicit_wrong_actor():
    expected = record(relation(alternatives=["holds the cup"]))
    for actor, action, passed in [
        ("fox", "holds cup", True),
        ("otter", "holds cup", False),
        ("fox", "otter holds cup", False),
        ("no fox", "holds cup", False),
        ("not a fox", "holds cup", False),
        ("statue of a fox", "holds cup", False),
    ]:
        raw = f"SETTING: cave\nACTOR: {actor}\nACTION: {action}\nMAGIC: box"
        evaluation = evaluate_surface(expected, raw, surface="raw")
        assert evaluation.expectation_results[0].passed is passed


@pytest.mark.parametrize("slot", ["SETTING", "MAGIC"])
def test_raw_relation_uses_its_declared_slot_without_borrowing_action(slot):
    expected = record(relation(slot=slot))
    slots = {"SETTING": "cave", "ACTOR": "fox", "ACTION": "otter holds cup", "MAGIC": "box"}
    slots[slot] = "fox holds cup"
    raw = "\n".join(f"{key}: {value}" for key, value in slots.items())
    assert evaluate_surface(expected, raw, surface="raw").passed_atoms == 1
    slots[slot] = "box"
    slots["ACTION"] = "fox holds cup"
    raw = "\n".join(f"{key}: {value}" for key, value in slots.items())
    assert evaluate_surface(expected, raw, surface="raw").passed_atoms == 0


def test_complete_action_alternative_binds_actor_without_relation_keyword_restriction():
    expected = record(
        relation(
            predicate="toward",
            object_value="cottage",
            slot="ACTION",
            alternatives=["travels toward the cottage"],
        )
    )
    raw = "SETTING: cave\nACTOR: fox\nACTION: travels toward cottage\nMAGIC: stars"
    assert evaluate_surface(expected, raw, surface="raw").passed_atoms == 1
    assert (
        evaluate_surface(
            expected, raw.replace("ACTOR: fox", "ACTOR: no fox"), surface="raw"
        ).passed_atoms
        == 0
    )


def test_unrelated_watch_and_spatial_edge_do_not_satisfy_action_slot():
    graph = facts().model_dump()
    graph["subjects"][0]["actions"] = ("watches stone box",)
    expected = record(
        {"kind": "slot", "slot": "ACTION", "alternatives": ["watches wooden cup above stone box"]}
    )
    evaluation = evaluate_surface(expected, graph, surface="postprocessed")
    assert not evaluation.expectation_results[0].passed


@pytest.mark.parametrize("binding", ["action", "event", "relation"])
def test_split_action_composes_only_through_its_bound_object(binding):
    graph = facts().model_dump()
    graph["subjects"][0]["actions"] = ()
    graph["relationships"] = (graph["relationships"][1],)
    if binding == "action":
        graph["subjects"][0]["actions"] = ("holds wooden cup",)
    elif binding == "event":
        graph["events"] = ({"ref": "event", "source": "s1", "action": "holds", "object": "o1"},)
    else:
        graph["relationships"] += ({"source": "s1", "relation": "holds", "target": "o1"},)
    expected = record(
        {
            "kind": "slot",
            "slot": "ACTION",
            "alternatives": ["red fox holds wooden cup above stone box"],
        }
    )
    assert evaluate_surface(expected, graph, surface="postprocessed").passed_atoms == 1
    if binding == "action":
        graph["subjects"][0]["actions"] = ("holds stone box",)
    elif binding == "event":
        graph["events"][0]["object"] = "o2"
    else:
        graph["relationships"][1]["target"] = "o2"
    assert evaluate_surface(expected, graph, surface="postprocessed").passed_atoms == 0


def test_split_action_cannot_borrow_another_actors_event():
    graph = facts().model_dump()
    graph["subjects"][0]["actions"] = ()
    graph["subjects"] += ({"ref": "s2", "label": "otter"},)
    graph["relationships"] = (graph["relationships"][1],)
    graph["events"] = ({"ref": "event", "source": "s2", "action": "holds", "object": "o1"},)
    expected = record(
        {
            "kind": "slot",
            "slot": "ACTION",
            "alternatives": ["red fox holds wooden cup above stone box"],
        }
    )
    assert evaluate_surface(expected, graph, surface="postprocessed").passed_atoms == 0


def test_transformation_count_is_visible_bound_and_not_optional_in_comparison():
    graph = SceneFactsV2.model_validate(
        {
            "setting": {"label": "cave"},
            "subjects": [{"ref": "fox", "label": "fox"}],
            "objects": [{"ref": "feather", "label": "feather"}],
            "transformation": {"source": "feather", "result_label": "boats", "result_count": 2},
        }
    )
    expected = record(
        {"kind": "count", "object": "boats", "count": 2},
        {"kind": "slot", "slot": "MAGIC", "alternatives": ["two boats"]},
        passage="In a cave, a fox lifts a feather. The feather becomes two boats.",
    )
    assert evaluate_surface(expected, graph, surface="postprocessed").semantic_atom_recall == 1
    for count in (None, 3):
        changed = graph.model_dump()
        changed["transformation"]["result_count"] = count
        other = SceneFactsV2.model_validate(changed)
        assert not equivalent(graph, other)
        assert evaluate_surface(expected, other, surface="postprocessed").semantic_atom_recall == 0


@pytest.mark.parametrize(
    "action,passed",
    [
        ("watches brass spool inside hamper", True),
        ("watches silver comb inside hamper", False),
        ("watches brass spool outside hamper; silver comb inside hamper", False),
        ("watches brass spool not inside hamper", False),
    ],
)
def test_relation_alias_keeps_subject_bound(action, passed):
    expected = record(
        relation(
            "brass spool", "inside", "woven hamper", slot="ACTION", alternatives=["inside a hamper"]
        )
    )
    output = f"SETTING: cave\nACTOR: sable vole\nACTION: {action}\nMAGIC: none"
    assert bool(evaluate_surface(expected, output, surface="raw").passed_atoms) is passed


@pytest.mark.parametrize("alternative", ["hamper", "inside no hamper", "inside hamper and another"])
def test_relation_alias_requires_positive_complete_relation_np(alternative):
    expected = record(
        relation("brass spool", "inside", "woven hamper", slot="ACTION", alternatives=[alternative])
    )
    output = "SETTING: cave\nACTOR: vole\nACTION: watches brass spool inside hamper\nMAGIC: none"
    assert evaluate_surface(expected, output, surface="raw").passed_atoms == 0


def test_relation_alias_cannot_drop_number():
    expected = record(
        relation(
            "brass spool",
            "inside",
            "two woven hampers",
            slot="ACTION",
            alternatives=["inside hamper"],
        )
    )
    output = "SETTING: cave\nACTOR: vole\nACTION: watches brass spool inside hamper\nMAGIC: none"
    assert evaluate_surface(expected, output, surface="raw").passed_atoms == 0


@pytest.mark.parametrize(
    "actor,action,passed",
    [
        ("sable vole", "runs to bronze gate", True),
        ("sable vole", "travels toward bronze gate", True),
        ("sable vole", "never runs to bronze gate", False),
        ("no sable vole", "runs to bronze gate", False),
        ("statue of sable vole", "runs to bronze gate", False),
        ("sable vole", "otter runs toward bronze gate", False),
    ],
)
def test_destination_movement_has_exact_actor_binding(actor, action, passed):
    expected = record(
        relation(
            "sable vole",
            "toward",
            "bronze gate",
            slot="ACTION",
            alternatives=["toward bronze gate"],
        )
    )
    output = f"SETTING: cave\nACTOR: {actor}\nACTION: {action}\nMAGIC: none"
    assert bool(evaluate_surface(expected, output, surface="raw").passed_atoms) is passed


@pytest.mark.parametrize(
    "magic,passed",
    [
        ("ribbon of moths rises", True),
        ("ribbon of moths ascends", True),
        ("ribbon of moths moves upward", True),
        ("ribbon of moths falls", False),
        ("choir rises; ribbon of moths falls", False),
        ("no ribbon of moths rises", False),
        ("ribbon of moths never rises", False),
    ],
)
def test_directional_magic_is_bound_to_its_subject(magic, passed):
    expected = record(
        relation("ribbon of moths", "moves", "upward", slot="MAGIC", alternatives=["rises"])
    )
    output = f"SETTING: cave\nACTOR: vole\nACTION: waits\nMAGIC: {magic}"
    assert bool(evaluate_surface(expected, output, surface="raw").passed_atoms) is passed


@pytest.mark.parametrize(
    "action,concepts,passed",
    [
        ("holds ceramic astrolabe above alcove", ["ceramic astrolabe"], True),
        ("carries ceramic astrolabe above alcove", ["ceramic astrolabe"], True),
        ("holds ceramic astrolabe above alcove", [], False),
        ("holds unknown box above alcove", ["ceramic astrolabe"], False),
        ("holds no ceramic astrolabe above alcove", ["ceramic astrolabe"], False),
        ("holds without ceramic astrolabe above alcove", ["ceramic astrolabe"], False),
        ("otter holds ceramic astrolabe above alcove", ["ceramic astrolabe"], False),
        ("holds ceramic astrolabe; otter waits above alcove", ["ceramic astrolabe"], False),
        ("holds ceramic astrolabe while otter waits above alcove", ["ceramic astrolabe"], False),
    ],
)
def test_holding_path_requires_known_single_positive_object(action, concepts, passed):
    expected = record(
        relation(
            "sable vole", "above", "bronze alcove", slot="ACTION", alternatives=["above alcove"]
        ),
        allowed_concepts=concepts,
    )
    output = f"SETTING: cave\nACTOR: sable vole\nACTION: {action}\nMAGIC: none"
    assert bool(evaluate_surface(expected, output, surface="raw").passed_atoms) is passed


def test_direct_relation_needs_no_intermediate_vocabulary():
    expected = record(relation("sable vole", "above", "bronze alcove", slot="ACTION"))
    output = "SETTING: cave\nACTOR: sable vole\nACTION: above bronze alcove\nMAGIC: none"
    assert evaluate_surface(expected, output, surface="raw").passed_atoms == 1


def test_literal_wire_renderer_preserves_slot_binding_and_refuses_extra_line():
    expected = record(
        relation("sable vole", "holds", "cup", slot="ACTION", alternatives=["holds cup"])
    )
    wire = "SETTING: cave\nACTOR: sable vole\nACTION: holds cup\nMAGIC: none"
    result = evaluate_surface(expected, {"master_prompt": wire}, surface="renderer")
    assert result.schema_valid and result.passed_atoms == 1
    malformed = evaluate_surface(
        expected, {"master_prompt": wire + "\nExtra: payload"}, surface="renderer"
    )
    assert not malformed.schema_valid


@pytest.mark.parametrize(
    "action",
    [
        "statue of fox moves toward box",
        "owl says fox moves toward box",
        "owl reports: fox moves toward box",
        "owl says, fox moves toward box",
    ],
)
def test_embedded_or_reported_actor_cannot_supply_movement(action):
    expected = record(relation("fox", "toward", "box", slot="ACTION"))
    output = f"SETTING: cave\nACTOR: fox\nACTION: {action}\nMAGIC: none"
    assert evaluate_surface(expected, output, surface="raw").passed_atoms == 0


@pytest.mark.parametrize("action", ["stands in the foreground", "fills the foreground"])
def test_salience_alone_does_not_supply_posture_or_extent(action):
    graph = SceneFactsV2.model_validate(
        {
            "setting": {"label": "cave"},
            "subjects": [{"ref": "s1", "label": "sable vole"}],
            "salience": [{"source": "s1", "layer": "foreground"}],
        }
    )
    expected = record({"kind": "slot", "slot": "ACTION", "alternatives": [action]})
    assert evaluate_surface(expected, graph, surface="postprocessed").passed_atoms == 0
    explicit = graph.model_dump()
    explicit["subjects"][0]["actions"] = [action]
    assert (
        evaluate_surface(
            expected, SceneFactsV2.model_validate(explicit), surface="postprocessed"
        ).passed_atoms
        == 1
    )
