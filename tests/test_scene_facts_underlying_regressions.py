import pytest
from pydantic import ValidationError

from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsGroundingError,
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneMotionFact,
    SceneNegativeFact,
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
    with pytest.raises(SceneFactsGroundingError) as caught:
        facts.validate_source_grounding(
            source_text="In a cave, a red fox holds a cup. The blue fox holds a ball."
        )
    assert {"subjects[0].actions[0]", "relationships[0]", "events[0]"}.issubset(caught.value.paths)


@pytest.mark.parametrize("antecedent", ["the blue feather", "the red feather", "the feather"])
def test_transformation_preserves_colored_antecedent_identity(antecedent):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        objects=(SceneObjectFact(ref="feather", label="feather", color="red"),),
        transformation=SceneTransformationFact(source="feather", result_label="boat"),
    )
    source = f"In a cave, a fox holds a red feather. {antecedent} becomes a boat."
    if antecedent == "the blue feather":
        with pytest.raises(SceneFactsGroundingError, match="transformation"):
            facts.validate_source_grounding(source_text=source)
    else:
        facts.validate_source_grounding(source_text=source)


def _colored_pairs():
    return SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(
            SceneSubjectFact(ref="red_fox", label="fox", color="red", actions=("holds blue ball",)),
            SceneSubjectFact(
                ref="blue_fox", label="fox", color="blue", actions=("holds red ball",)
            ),
        ),
        objects=(
            SceneObjectFact(ref="blue_ball", label="ball", color="blue"),
            SceneObjectFact(ref="red_ball", label="ball", color="red"),
        ),
        relationships=(
            SceneRelationshipFact(source="red_fox", relation="holds", target="blue_ball"),
            SceneRelationshipFact(source="blue_fox", relation="holds", target="red_ball"),
        ),
        events=(
            SceneEventFact(ref="first", source="red_fox", action="holds", object="blue_ball"),
            SceneEventFact(ref="second", source="blue_fox", action="holds", object="red_ball"),
        ),
        temporal_order=(SceneTemporalOrderFact(before="first", after="second"),),
    )


def test_colored_identities_roundtrip_and_render_bound_references():
    facts = _colored_pairs()
    source = "In a cave, a red fox holds a blue ball then a blue fox holds a red ball."
    prompt = facts.to_renderer_prompt(source_text=source)
    assert "Relations: red fox holds blue ball; blue fox holds red ball" in prompt
    assert "Order: red fox holds blue ball before blue fox holds red ball" in prompt
    assert "red red fox" not in prompt
    assert SceneFactsV2.from_wire(facts.to_wire()) == facts
    assert facts.subjects[0].label == "fox" and facts.subjects[0].color == "red"


@pytest.mark.parametrize(
    "source",
    [
        "In a cave, a red fox holds a red ball then a blue fox holds a blue ball.",
        "In a cave, a red fox stands beside a blue fox. "
        "A red ball stands beside a blue ball. The fox holds the ball.",
    ],
)
def test_colored_identities_reject_swapped_or_bare_references(source):
    with pytest.raises(SceneFactsGroundingError):
        _colored_pairs().to_renderer_prompt(source_text=source)


@pytest.mark.parametrize("second_color", [None, "red"])
def test_repeated_labels_require_distinct_explicit_colors(second_color):
    with pytest.raises(ValidationError, match="distinct colors"):
        SceneFactsV2(
            setting=SceneSettingFact(label="cave"),
            subjects=(
                SceneSubjectFact(ref="first", label="fox", color="red"),
                SceneSubjectFact(ref="second", label="fox", color=second_color),
            ),
        )


def test_motion_toward_destination_does_not_treat_carried_object_as_another_actor():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="valley"),
        subjects=(SceneSubjectFact(ref="kite", label="kite"),),
        objects=(
            SceneObjectFact(ref="ribbon", label="ribbon"),
            SceneObjectFact(ref="tower", label="tower"),
        ),
        motions=(SceneMotionFact(source="kite", direction="rises", destination="tower"),),
    )
    facts.validate_source_grounding(
        source_text="In a valley, a kite rises and pulls a ribbon toward a tower."
    )


@pytest.mark.parametrize(
    "count, clause", [(1, "one blue balloon rises"), (3, "three blue balloons rise")]
)
def test_rising_motion_agrees_in_number_without_borrowing_actor_or_direction(count, clause):
    source = f"In a valley, {clause} toward a tower. A fox stands."
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="valley"),
        subjects=(
            SceneSubjectFact(ref="balloon", label="balloon", count=count, color="blue"),
            SceneSubjectFact(ref="fox", label="fox"),
        ),
        objects=(SceneObjectFact(ref="tower", label="tower"),),
        motions=(SceneMotionFact(source="balloon", direction="rises", destination="tower"),),
    )
    facts.validate_source_grounding(source_text=source)
    with pytest.raises(SceneFactsGroundingError, match="motions"):
        facts.model_copy(update={"motions": (
            SceneMotionFact(source="fox", direction="rises", destination="tower"),
        )}).validate_source_grounding(source_text=source)
    with pytest.raises(SceneFactsGroundingError, match="motions"):
        facts.model_copy(update={"motions": (
            SceneMotionFact(source="balloon", direction="falls", destination="tower"),
        )}).validate_source_grounding(source_text=source)


def _coordinated_actors():
    return SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(
            SceneSubjectFact(ref="fox", label="fox", color="red", actions=("stand beside tree",)),
            SceneSubjectFact(ref="owl", label="owl", color="blue", actions=("stand beside tree",)),
        ),
        objects=(SceneObjectFact(ref="tree", label="tree"),),
        relationships=(
            SceneRelationshipFact(source="fox", relation="beside", target="tree"),
            SceneRelationshipFact(source="owl", relation="beside", target="tree"),
        ),
    )


def test_explicit_coordination_preserves_each_actor_and_shared_target():
    facts = _coordinated_actors()
    source = "A red fox and a blue owl stand beside a tree."
    facts.validate_source_grounding(source_text=source)
    prompt = facts.to_renderer_prompt(source_text=source)
    assert "red fox; action: stand beside tree" in prompt
    assert "blue owl; action: stand beside tree" in prompt
    assert "fox beside tree; owl beside tree" in prompt
    for changes in (
        {"subjects": tuple(
            subject.model_copy(update={"color": color})
            for subject, color in zip(facts.subjects, ("blue", "red"), strict=True)
        )},
        {"relationships": (SceneRelationshipFact(source="fox", relation="beside", target="owl"),)},
    ):
        with pytest.raises(SceneFactsGroundingError):
            facts.model_copy(update=changes).validate_source_grounding(source_text=source)


@pytest.mark.parametrize("source", [
    "A red fox sleeps and a blue owl stands beside a tree.",
    "A red fox does not stand beside a tree and a blue owl stands beside a tree.",
    "A red fox and a blue owl do not stand beside a tree.",
    "A red fox or a blue owl stands beside a tree.",
    "If a red fox and a blue owl stand beside a tree, a dog sleeps.",
])
def test_coordination_does_not_borrow_another_actors_action_or_negation(source):
    with pytest.raises(SceneFactsGroundingError):
        _coordinated_actors().validate_source_grounding(source_text=source)


def test_shared_negative_remains_negative_for_both_coordinated_actors():
    facts = _coordinated_actors().model_copy(update={
        "subjects": tuple(subject.model_copy(update={"actions": ()})
                          for subject in _coordinated_actors().subjects),
        "relationships": (),
        "negatives": tuple(SceneNegativeFact(kind="action", target=ref, value="stand beside a tree")
                           for ref in ("fox", "owl")),
    })
    facts.validate_source_grounding(
        source_text="A red fox and a blue owl do not stand beside a tree."
    )
    with pytest.raises(SceneFactsGroundingError, match="negatives"):
        facts.validate_source_grounding(
            source_text="A red fox and a blue owl stand beside a tree."
        )


def test_bare_in_does_not_borrow_a_later_object_as_its_container():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(SceneSubjectFact(ref="fox", label="fox"),),
        objects=(SceneObjectFact(ref="tree", label="tree"),),
        relationships=(SceneRelationshipFact(source="fox", relation="inside", target="tree"),),
    )
    facts.validate_source_grounding(source_text="A fox is in a tree.")
    for source in (
        "A fox is in front of a tree.",
        "A fox is in the shade of a tree.",
        "A fox is in a field beside a tree.",
    ):
        with pytest.raises(SceneFactsGroundingError, match="relationships"):
            facts.validate_source_grounding(source_text=source)
