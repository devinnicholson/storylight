import pytest
from pydantic import ValidationError

from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsGroundingError,
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneObjectFact,
    SceneRelationshipFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
    SceneTransformationFact,
)


def _transformation(count=None):
    return SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        objects=(SceneObjectFact(ref="feather", label="feather"),),
        transformation=SceneTransformationFact(
            source="feather", result_label="boats", result_count=count
        ),
    )


def test_optional_transformation_count_preserves_legacy_wire():
    legacy = _transformation().to_wire()
    assert legacy.splitlines()[-1] == "T|feather|boats|-|-"
    assert SceneFactsV2.from_wire(legacy).transformation.result_count is None
    counted = _transformation(2)
    assert counted.to_wire().splitlines()[-1] == "T|feather|boats|-|-|2"
    assert SceneFactsV2.from_wire(counted.to_wire()) == counted
    assert "feather becomes exactly 2 boats" in counted.to_renderer_prompt(
        source_text="In a cave, a feather becomes two boats."
    )


@pytest.mark.parametrize("count", [0, True])
def test_transformation_count_is_strict_and_bounded(count):
    with pytest.raises(ValidationError):
        _transformation(count)


@pytest.mark.parametrize("count", ["２", "2|3"])
def test_transformation_count_wire_rejects_malformed_sixth_field(count):
    with pytest.raises(ValueError):
        SceneFactsV2.from_wire(_transformation().to_wire() + "|" + count)


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, two feathers become three boats.",
        "In a cave, a feather becomes three boats and two boats rise.",
        "In a cave, a feather does not become two boats.",
    ],
)
def test_transformation_count_cannot_borrow_source_or_distractor_count(source):
    with pytest.raises(SceneFactsGroundingError):
        _transformation(2).validate_source_grounding(source_text=source)


def test_transformation_count_does_not_bypass_private_result_validation():
    facts = _transformation(2).model_copy(
        update={
            "transformation": SceneTransformationFact(
                source="feather", result_label="password hunter2", result_count=2
            )
        }
    )
    with pytest.raises(SceneFactsPrivacyError):
        facts.to_renderer_prompt(source_text="In a cave, a feather becomes two password hunter2.")


@pytest.mark.parametrize("same_actor", [False, True])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_repeated_verb_temporal_order_binds_actor_and_object(same_actor, reversed_order):
    second_actor = "fox" if same_actor else "owl"
    subjects = (SceneSubjectFact(ref="fox", label="fox"),)
    if not same_actor:
        subjects += (SceneSubjectFact(ref="owl", label="owl"),)
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=subjects,
        objects=(
            SceneObjectFact(ref="feather", label="feather"),
            SceneObjectFact(ref="lantern", label="lantern"),
        ),
        events=(
            SceneEventFact(ref="e1", source="fox", action="lifts", object="feather"),
            SceneEventFact(ref="e2", source=second_actor, action="lifts", object="lantern"),
        ),
        temporal_order=(
            SceneTemporalOrderFact(
                before="e2" if reversed_order else "e1",
                after="e1" if reversed_order else "e2",
            ),
        ),
    )
    source = f"In a cave, a fox lifts a feather then the {second_actor} lifts a lantern."
    if reversed_order:
        with pytest.raises(SceneFactsGroundingError, match="temporal_order"):
            facts.validate_source_grounding(source_text=source)
    else:
        facts.validate_source_grounding(source_text=source)


@pytest.mark.parametrize(
    "clause,typed_event",
    [
        ("a fox holds a cup and otter holds a ball", False),
        ("a fox holds a cup next to a ball", False),
        ("a fox holds neither a cup nor a ball", False),
    ],
)
def test_action_object_cannot_be_borrowed_across_clause_or_relation(clause, typed_event):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(
            SceneSubjectFact(
                ref="fox", label="fox", actions=() if typed_event else ("holds ball",)
            ),
        ),
        objects=(SceneObjectFact(ref="ball", label="ball"),),
        events=(SceneEventFact(ref="e1", source="fox", action="holds", object="ball"),)
        if typed_event
        else (),
    )
    with pytest.raises(SceneFactsGroundingError):
        facts.validate_source_grounding(source_text=f"In a cave, {clause}.")


@pytest.mark.parametrize(
    "connector,second_head",
    [("then", ""), ("then", "otter "), ("only afterward", "does not ")],
)
def test_shared_subject_temporal_connector_requires_grounded_prior_event(connector, second_head):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(SceneSubjectFact(ref="fox", label="fox"),),
        objects=(
            SceneObjectFact(ref="feather", label="feather"),
            SceneObjectFact(ref="box", label="box"),
        ),
        events=(
            SceneEventFact(ref="e1", source="fox", action="lifts", object="feather"),
            SceneEventFact(ref="e2", source="fox", action="opens", object="box"),
        ),
        temporal_order=(SceneTemporalOrderFact(before="e1", after="e2"),),
    )
    source = f"In a cave, a fox first lifts a feather and {connector} {second_head}opens a box."
    if second_head:
        with pytest.raises(SceneFactsGroundingError):
            facts.validate_source_grounding(source_text=source)
    else:
        facts.validate_source_grounding(source_text=source)
        inverted = facts.model_copy(
            update={"temporal_order": (SceneTemporalOrderFact(before="e2", after="e1"),)}
        )
        with pytest.raises(SceneFactsGroundingError, match="temporal_order"):
            inverted.validate_source_grounding(source_text=source)
        with pytest.raises(SceneFactsGroundingError):
            facts.validate_source_grounding(
                source_text=f"In a cave, a fox beside a feather and {connector} opens a box."
            )


def _held_ball_facts():
    return SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(SceneSubjectFact(ref="fox", label="fox", actions=("holds ball",)),),
        objects=(SceneObjectFact(ref="ball", label="ball"),),
        events=(SceneEventFact(ref="e1", source="fox", action="holds", object="ball"),),
    )


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, an owl says, a fox holds a ball.",
        "In a cave, if a fox holds a ball, an owl smiles.",
        "In a cave, a fox holds a ball if an owl smiles.",
        "In a cave, a fox holds a ball, an owl says.",
        'An owl says, "a cat sleeps. A fox holds a ball."',
        "In a cave, a fox may hold a ball.",
    ],
)
def test_unasserted_action_cannot_ground_action_or_typed_event(source):
    with pytest.raises(SceneFactsGroundingError) as caught:
        _held_ball_facts().validate_source_grounding(
            source_text=f"A fox and a ball are in a cave. {source}"
        )
    assert {"subjects[0].actions[0]", "events[0]"}.issubset(caught.value.paths)


@pytest.mark.parametrize(
    "source",
    [
        "An owl says, a fox holds a ball; in a cave, a fox holds a ball.",
        "In a cave, a fox holds a ball, and an owl says a dragon flies.",
    ],
)
def test_independent_assertion_survives_reported_clause(source):
    _held_ball_facts().validate_source_grounding(source_text=source)


def test_colored_actor_cannot_borrow_action_event_or_relation_from_same_label():
    facts = _held_ball_facts().model_copy(
        update={
            "subjects": (
                SceneSubjectFact(ref="fox", label="fox", color="red", actions=("holds ball",)),
            ),
            "relationships": (
                SceneRelationshipFact(source="fox", relation="holds", target="ball"),
            ),
        }
    )
    with pytest.raises(SceneFactsGroundingError, match=r"subjects\[0\].identity"):
        facts.validate_source_grounding(
            source_text="In a cave, a red fox holds a cup. The blue fox holds a ball."
        )


@pytest.mark.parametrize("antecedent", ["the blue feather", "the red feather", "the feather"])
def test_transformation_preserves_colored_antecedent_identity(antecedent):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        objects=(SceneObjectFact(ref="feather", label="feather", color="red"),),
        transformation=SceneTransformationFact(source="feather", result_label="boat"),
    )
    source = f"In a cave, a fox holds a red feather. {antecedent} becomes a boat."
    if antecedent == "the blue feather":
        with pytest.raises(SceneFactsGroundingError, match=r"objects\[0\].identity"):
            facts.validate_source_grounding(source_text=source)
    else:
        facts.validate_source_grounding(source_text=source)
