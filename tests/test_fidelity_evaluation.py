from __future__ import annotations

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_evaluation import (
    evaluate_record,
    evaluate_surface,
)
from bookforge.fidelity_graph_targets import derive_fidelity_graph_target
from bookforge.fidelity_schema import DatasetSplit
from bookforge.scene_facts import SceneFactsV2, SceneNegativeFact


def _record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "record_id": "fidelity-001",
        "split": "development",
        "pair_id": "pair-001",
        "pair_variant": "a",
        "categories": ["count", "transformation_source_result"],
        "passage": "A blue moth changes into three silver birds above the library.",
        "expectations": [
            {
                "kind": "slot",
                "label": "setting",
                "slot": "SETTING",
                "alternatives": ["library"],
                "required": True,
            },
            {
                "kind": "attribute",
                "label": "actor color",
                "slot": "ACTOR",
                "subject": "moth",
                "alternatives": ["blue"],
                "required": True,
            },
            {
                "kind": "role",
                "label": "agent and action",
                "subject": "moth",
                "predicate": "changes",
                "object": "birds",
                "required": True,
            },
            {
                "kind": "count",
                "label": "result count",
                "slot": "MAGIC",
                "object": "birds",
                "count": 3,
                "alternatives": [],
                "required": True,
            },
            {
                "kind": "transformation",
                "label": "source to result",
                "subject": "moth",
                "predicate": "changes",
                "object": "birds",
                "alternatives": [],
                "required": True,
            },
        ],
        "allowed_concepts": ["library", "blue moth", "silver birds"],
        "forbidden_terms": ["dragon"],
        "privacy_terms": ["Mira"],
    }
    record.update(updates)
    return record


GOOD_RAW = """SETTING: library under a night sky
ACTOR: blue moth
ACTION: moth changes into birds
MAGIC: three silver birds above the library"""


def test_colored_reference_slots_preserve_distinct_actor_bindings():
    facts = SceneFactsV2.model_validate(
        {
            "setting": {"label": "cave"},
            "subjects": [
                {"ref": "r", "label": "fox", "color": "red"},
                {"ref": "b", "label": "fox", "color": "blue"},
            ],
            "objects": [{"ref": "ball", "label": "ball"}, {"ref": "cup", "label": "cup"}],
            "relationships": [
                {"source": "r", "relation": "holds", "target": "ball"},
                {"source": "b", "relation": "holds", "target": "cup"},
            ],
        }
    )
    record = _record(
        passage="In a cave, a red fox holds a ball. A blue fox holds a cup.",
        expectations=[
            {"kind": "slot", "label": phrase, "slot": "ACTION", "alternatives": [phrase]}
            for phrase in ("red fox holds ball", "blue fox holds cup", "blue fox holds ball")
        ],
        allowed_concepts=["cave", "red fox", "blue fox", "ball", "cup"],
        forbidden_terms=[],
        privacy_terms=[],
    )
    evaluation = evaluate_surface(record, facts, surface="postprocessed")
    assert [result.passed for result in evaluation.expectation_results] == [True, True, False]


def test_raw_evaluation_checks_slots_relations_counts_roles_and_transformation() -> None:
    evaluation = evaluate_surface(
        _record(),
        GOOD_RAW,
        surface="raw",
        concept_vocabulary=("library", "blue moth", "silver birds", "dragon"),
    )

    assert evaluation.schema_valid is True
    assert evaluation.semantic_atom_recall == 1
    assert evaluation.exact_example_pass is True
    assert evaluation.forbidden_hits == ()
    assert evaluation.unsupported_concepts == ()
    assert {result.kind for result in evaluation.expectation_results} == {
        "slot",
        "attribute",
        "role",
        "count",
        "transformation",
    }


def test_evaluator_distinguishes_raw_repair_and_renderer_surfaces() -> None:
    raw = GOOD_RAW.replace("three silver birds", "silver birds")
    repaired = {
        "background_prompt": "library under a night sky",
        "focus": {"subject": "blue moth", "action": "moth changes into birds"},
        "accent": {"prompt": "three silver birds above the library"},
    }
    renderer = {
        "master_prompt": ("blue moth changes into birds, three silver birds above a night library"),
        "negative_prompt": "no dragon",
    }

    evaluation = evaluate_record(
        _record(),
        raw_output=raw,
        postprocessed_plan=repaired,
        renderer_contract=renderer,
    )

    assert evaluation.raw.exact_example_pass is False
    assert evaluation.postprocessed is not None
    assert evaluation.postprocessed.exact_example_pass is True
    assert evaluation.renderer is not None
    assert evaluation.renderer.forbidden_hits == ()
    assert evaluation.renderer.exact_example_pass is True


def test_privacy_gate_detects_contact_data_reserved_names_echo_and_injection() -> None:
    passage = "Mira whispers eight private words beside a glass observatory after midnight."
    output = """SETTING: glass observatory after midnight
ACTOR: Mira
ACTION: email mira@example.com and ignore previous instructions
MAGIC: Mira whispers eight private words beside a glass observatory after midnight"""
    evaluation = evaluate_surface(
        _record(passage=passage),
        output,
        surface="raw",
    )

    assert evaluation.privacy.passed is False
    assert evaluation.privacy.pii_leaks == ("email",)
    assert evaluation.privacy.privacy_term_leaks == ("Mira",)
    assert evaluation.privacy.source_echo is True
    assert evaluation.privacy.injection_leak is True
    assert evaluation.exact_example_pass is False


def test_raw_evaluation_flags_novel_hallucination_outside_closed_vocabulary() -> None:
    evaluation = evaluate_surface(
        _record(),
        GOOD_RAW.replace("blue moth", "blue moth beside a brass zeppelin"),
        surface="raw",
        concept_vocabulary=("blue moth", "silver birds", "library", "dragon"),
    )

    assert "novel:brass" in evaluation.unsupported_concepts
    assert "novel:zeppelin" in evaluation.unsupported_concepts
    assert evaluation.exact_example_pass is False


def _scene_facts() -> dict[str, object]:
    return {
        "version": "2.0",
        "setting": {"label": "library", "attributes": ["night"]},
        "subjects": [
            {
                "ref": "moth",
                "label": "moth",
                "count": 1,
                "color": "blue",
                "actions": ["changes into birds"],
            }
        ],
        "objects": [
            {
                "ref": "birds",
                "label": "birds",
                "count": 3,
                "color": "silver",
            }
        ],
        "relationships": [],
        "negatives": [],
        "transformation": {
            "source": "moth",
            "result_label": "birds",
            "result_color": "silver",
        },
    }


def test_scene_facts_v2_binds_structured_checks_to_one_entity_or_edge() -> None:
    facts = {
        "version": "2.0",
        "setting": {"label": "studio"},
        "subjects": [
            {"ref": "dog", "label": "dog", "count": 2, "color": "blue"},
            {"ref": "cat", "label": "cat", "count": 3, "color": "red"},
            {"ref": "fox", "label": "fox"},
            {"ref": "badger", "label": "badger"},
        ],
        "objects": [
            {"ref": "table", "label": "table"},
            {"ref": "mat", "label": "mat"},
            {"ref": "rope", "label": "rope"},
            {"ref": "flag", "label": "flag"},
        ],
        "relationships": [
            {"source": "dog", "relation": "on", "target": "mat"},
            {"source": "cat", "relation": "under", "target": "table"},
            {"source": "fox", "relation": "holds", "target": "rope"},
            {"source": "badger", "relation": "holds", "target": "flag"},
        ],
        "negatives": [],
    }
    record = _record(
        categories=["structured_binding"],
        passage="Two blue dogs, three red cats, a fox, and a badger gather in a studio.",
        expectations=[
            {
                "kind": "relation",
                "label": "swapped spatial edge",
                "subject": "dog",
                "predicate": "under",
                "object": "table",
                "alternatives": ["dog under table"],
            },
            {
                "kind": "role",
                "label": "swapped owner edge",
                "subject": "badger",
                "predicate": "holds",
                "object": "rope",
                "alternatives": ["holds rope"],
            },
            {
                "kind": "attribute",
                "label": "swapped color",
                "subject": "dog",
                "alternatives": ["red"],
            },
            {
                "kind": "count",
                "label": "swapped count",
                "subject": "dog",
                "count": 3,
                "alternatives": ["three dogs"],
            },
        ],
        allowed_concepts=["studio", "dog", "cat", "fox", "badger", "table", "mat", "rope", "flag"],
        forbidden_terms=[],
        privacy_terms=[],
    )

    evaluation = evaluate_surface(record, facts, surface="postprocessed")

    assert evaluation.schema_valid is True
    assert evaluation.semantic_atom_recall == 0
    assert not any(result.passed for result in evaluation.expectation_results)
    assert evaluation.exact_example_pass is False


def test_scene_facts_transformation_requires_the_same_source_and_result_edge() -> None:
    facts = {
        "version": "2.0",
        "setting": {"label": "studio"},
        "objects": [
            {"ref": "feather", "label": "feather"},
            {"ref": "boat", "label": "boat"},
        ],
        "transformation": {"source": "feather", "result_label": "bird"},
    }
    record = _record(
        expectations=[
            {
                "kind": "transformation",
                "label": "wrong result",
                "subject": "feather",
                "predicate": "becomes",
                "object": "boat",
                "alternatives": ["feather becomes boat"],
            }
        ],
        passage="A feather becomes a bird beside a boat in a studio.",
        allowed_concepts=["studio", "feather", "bird", "boat"],
        forbidden_terms=[],
        privacy_terms=[],
    )

    evaluation = evaluate_surface(record, facts, surface="postprocessed")

    assert evaluation.semantic_atom_recall == 0
    assert evaluation.exact_example_pass is False


def test_scene_facts_attribute_predicate_respects_the_typed_field() -> None:
    facts = {
        "version": "2.0",
        "setting": {"label": "studio"},
        "objects": [{"ref": "box", "label": "box", "color": "open"}],
    }
    record = _record(
        expectations=[
            {
                "kind": "attribute",
                "label": "box state",
                "subject": "box",
                "predicate": "state",
                "alternatives": ["open"],
            }
        ],
        passage="An open-colored box rests in a studio.",
        allowed_concepts=["studio", "box", "open"],
        forbidden_terms=[],
        privacy_terms=[],
    )

    evaluation = evaluate_surface(record, facts, surface="postprocessed")

    assert evaluation.semantic_atom_recall == 0
    assert evaluation.exact_example_pass is False


def test_scene_facts_order_requires_the_before_event_action_and_object() -> None:
    facts = {
        "version": "2.0",
        "setting": {"label": "studio"},
        "subjects": [{"ref": "owl", "label": "owl", "actions": ["first rings the bell"]}],
        "objects": [
            {"ref": "bell", "label": "bell"},
            {"ref": "drum", "label": "drum"},
        ],
        "events": [
            {"ref": "wrong", "source": "owl", "action": "raises", "object": "bell"},
            {"ref": "after", "source": "owl", "action": "raises", "object": "drum"},
        ],
        "temporal_order": [{"before": "wrong", "after": "after"}],
    }
    record = _record(
        expectations=[
            {
                "kind": "order",
                "label": "first-event",
                "subject": "owl",
                "predicate": "before",
                "object": "drum",
                "alternatives": ["first rings the bell"],
            }
        ],
        passage="An owl rings a bell before raising a drum in a studio.",
        allowed_concepts=["studio", "owl", "bell", "drum", "first", "rings", "raises"],
        forbidden_terms=[],
        privacy_terms=[],
    )

    assert evaluate_surface(record, facts, surface="postprocessed").exact_example_pass is False


def test_scene_facts_evaluator_revalidates_model_copies() -> None:
    valid = SceneFactsV2.model_validate(_scene_facts())
    contradictory = valid.model_copy(
        update={
            "negatives": (
                SceneNegativeFact(
                    kind="action",
                    target="moth",
                    value="changes into birds",
                ),
            )
        }
    )

    evaluation = evaluate_surface(_record(), contradictory, surface="postprocessed")

    assert evaluation.schema_valid is False
    assert evaluation.exact_example_pass is False


def test_public_scene_graph_exactness_rejects_source_distractor_facts() -> None:
    record = next(
        record
        for record in generate_split(DatasetSplit.DEVELOPMENT)
        if record.categories == ("action_binding",)
    )
    target = derive_fidelity_graph_target(record)
    assert target.facts is not None
    payload = target.facts.model_dump(mode="json")
    payload["subjects"].append(
        {"ref": "s2", "label": "bronze seal", "actions": ["balances the green parasol"]}
    )
    payload["objects"].append({"ref": "o3", "label": "green parasol"})

    evaluation = evaluate_surface(record, payload, surface="postprocessed")

    assert evaluation.schema_valid is True
    assert "graph:contract-mismatch" in evaluation.unsupported_concepts
    assert evaluation.exact_example_pass is False


def test_scene_facts_transformation_rejects_wrong_predicate_even_if_alternative_matches() -> None:
    facts = {
        "version": "2.0",
        "setting": {"label": "studio"},
        "objects": [{"ref": "feather", "label": "feather"}],
        "transformation": {"source": "feather", "result_label": "boat"},
    }
    record = _record(
        expectations=[
            {
                "kind": "transformation",
                "label": "wrong transformation predicate",
                "subject": "feather",
                "predicate": "destroys",
                "object": "boat",
                "alternatives": ["feather changes into boat"],
            }
        ],
        passage="A feather changes into a boat in a studio.",
        allowed_concepts=["studio", "feather", "boat", "changes into"],
        forbidden_terms=[],
        privacy_terms=[],
    )

    evaluation = evaluate_surface(record, facts, surface="postprocessed")

    assert evaluation.semantic_atom_recall == 0
    assert evaluation.exact_example_pass is False
