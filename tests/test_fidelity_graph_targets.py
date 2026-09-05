from __future__ import annotations

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_evaluation import evaluate_surface
from bookforge.fidelity_graph_targets import (
    GraphTargetRefusal,
    derive_fidelity_graph_target,
)
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord, PairVariant


def _development_record(category: str, variant: PairVariant = PairVariant.A) -> FidelityRecord:
    return next(
        record
        for record in generate_split(DatasetSplit.DEVELOPMENT)
        if record.categories == (category,) and record.pair_variant is variant
    )


def test_public_derivation_is_deterministic_grounded_and_contract_exact() -> None:
    record = _development_record("spatial_relations")

    first = derive_fidelity_graph_target(record)
    second = derive_fidelity_graph_target(record)

    assert first == second
    assert first.eligible is True
    assert first.facts is not None
    first.facts.validate_source_grounding(source_text=record.passage)
    assert first.facts.to_wire(token_budget=128) == second.facts.to_wire(token_budget=128)
    relationship = first.facts.relationships[0]
    entities = {
        entity.ref: entity.label for entity in (*first.facts.subjects, *first.facts.objects)
    }
    specialized = record.expectations[-1]
    assert entities[relationship.source] == specialized.subject
    assert relationship.relation.value == "holds"
    held_ref = relationship.target
    spatial = first.facts.relationships[1]
    assert spatial.source == held_ref
    assert entities[spatial.target] == specialized.object

    evaluation = evaluate_surface(
        record.model_dump(mode="json", by_alias=True),
        first.facts,
        surface="postprocessed",
    )
    assert evaluation.exact_example_pass is True


def test_pronoun_proof_fails_closed_when_public_template_changes() -> None:
    record = _development_record("coreference")
    changed = record.model_copy(
        update={"passage": record.passage.replace("the latter", "the former")}
    )

    result = derive_fidelity_graph_target(changed)

    assert result.eligible is False
    assert result.refusal is GraphTargetRefusal.GROUNDING_REJECTED
    assert result.detail_codes == ("subjects[0].actions[0]",)


def test_typed_motion_and_order_are_exact_but_salience_does_not_prove_posture() -> None:
    for category in ("destination", "reversed_motion", "salience", "temporal_order"):
        record = _development_record(category)
        result = derive_fidelity_graph_target(record, token_budget=64)

        assert result.eligible is True
        assert result.facts is not None
        evaluation = evaluate_surface(
            record.model_dump(mode="json", by_alias=True),
            result.facts,
            surface="postprocessed",
        )
        if category == "salience":
            assert evaluation.exact_example_pass is False
            assert [atom.slot for atom in evaluation.expectation_results if not atom.passed] == [
                "ACTION"
            ]
        else:
            assert evaluation.exact_example_pass is True

    destination = derive_fidelity_graph_target(_development_record("destination"))
    reversed_motion = derive_fidelity_graph_target(_development_record("reversed_motion"))
    salience = derive_fidelity_graph_target(_development_record("salience"))
    temporal = derive_fidelity_graph_target(_development_record("temporal_order"))
    assert destination.facts is not None and destination.facts.motions[0].destination
    assert reversed_motion.facts is not None
    assert reversed_motion.facts.motions[0].direction.value == "rises"
    assert salience.facts is not None
    assert salience.facts.salience[0].layer.value == "foreground"
    assert temporal.facts is not None
    assert temporal.facts.temporal_order[0].before == "e1"


def test_private_names_are_withheld_from_graph_and_result_metadata() -> None:
    record = _development_record("proper_names")
    result = derive_fidelity_graph_target(record)

    assert result.facts is not None
    serialized = result.model_dump_json()
    for private_term in record.privacy_terms:
        assert private_term not in serialized
    assert record.passage not in serialized
    assert result.facts.subjects[0].label == "keeper"


def test_hidden_split_is_rejected_before_record_content_is_read() -> None:
    class HiddenRecordTrap:
        split = DatasetSplit.HIDDEN
        record_id = "hidden-placeholder"

        @property
        def categories(self) -> tuple[str, ...]:
            raise AssertionError("hidden record content was inspected")

    result = derive_fidelity_graph_target(HiddenRecordTrap())  # type: ignore[arg-type]

    assert result.eligible is False
    assert result.split is DatasetSplit.HIDDEN
    assert result.categories == ()
    assert result.refusal is GraphTargetRefusal.NON_PUBLIC_SPLIT


def test_ambiguous_typed_contract_fails_closed() -> None:
    record = _development_record("action_binding")
    ambiguous = record.model_copy(
        update={"expectations": (*record.expectations, record.expectations[-1])}
    )

    result = derive_fidelity_graph_target(ambiguous)

    assert result.eligible is False
    assert result.refusal is GraphTargetRefusal.AMBIGUOUS_CONTRACT
    assert result.detail_codes == ("specialized_expectation",)
