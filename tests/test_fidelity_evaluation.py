from __future__ import annotations

from pydantic import BaseModel

from bookforge.fidelity_evaluation import (
    concept_vocabulary,
    evaluate_record,
    evaluate_surface,
    extract_surface,
)


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


def test_raw_schema_requires_exactly_four_ordered_unique_slots() -> None:
    malformed = """ACTOR: moth
SETTING: library
ACTION: changes
MAGIC: birds"""

    content = extract_surface(malformed, surface="raw")

    assert content.schema_valid is False
    assert set(content.slots) == {"SETTING", "ACTOR", "ACTION", "MAGIC"}


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


def test_negated_forbidden_concept_is_not_a_hallucination() -> None:
    evaluation = evaluate_surface(
        _record(forbidden_terms=["dragon"]),
        GOOD_RAW.replace("above the library", "above the library, without a dragon"),
        surface="raw",
    )

    assert evaluation.forbidden_hits == ()


def test_closed_world_vocabulary_flags_known_unsupported_concepts() -> None:
    evaluation = evaluate_surface(
        _record(),
        GOOD_RAW.replace("blue moth", "blue moth beside a dragon"),
        surface="raw",
        concept_vocabulary=("blue moth", "silver birds", "library", "dragon"),
    )

    assert evaluation.unsupported_concepts == ("dragon",)
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


def test_raw_novelty_check_never_rejects_exact_target_tokens() -> None:
    record = _record(
        target={
            "SETTING": "library under a night sky",
            "ACTOR": "blue moth",
            "ACTION": "moth changes into birds",
            "MAGIC": "three silver birds above the library",
        }
    )

    evaluation = evaluate_surface(
        record,
        GOOD_RAW,
        surface="raw",
        concept_vocabulary=concept_vocabulary([record]),
    )

    assert evaluation.unsupported_concepts == ()
    assert evaluation.exact_example_pass is True


class RecordModel(BaseModel):
    record_id: str
    allowed_concepts: list[str]


def test_vocabulary_adapter_accepts_mappings_and_pydantic_models() -> None:
    records = [
        RecordModel(record_id="one", allowed_concepts=["silver bird"]),
        {"record_id": "two", "allowed_concepts": ["blue moth", "silver bird"]},
    ]

    assert concept_vocabulary(records) == ("blue moth", "silver bird")


def test_order_expectation_rejects_reversed_semantics() -> None:
    record = _record(
        expectations=[
            {
                "kind": "order",
                "label": "temporal order",
                "subject": "opens the map",
                "predicate": "then",
                "object": "lights the lantern",
            }
        ]
    )
    correct = GOOD_RAW.replace("moth changes into birds", "opens the map then lights the lantern")
    reversed_output = GOOD_RAW.replace(
        "moth changes into birds", "lights the lantern then opens the map"
    )

    assert evaluate_surface(record, correct, surface="raw").semantic_atom_recall == 1
    assert evaluate_surface(record, reversed_output, surface="raw").semantic_atom_recall == 0
