import pytest

from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsGroundingError,
    SceneFactsV2,
    SceneMotionFact,
    SceneObjectFact,
    SceneSalienceFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
)


def _facts(**changes):
    return SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        subjects=(
            SceneSubjectFact(ref="a", label="fox"),
            SceneSubjectFact(ref="b", label="owl"),
        ),
        objects=(
            SceneObjectFact(ref="o", label="lantern"),
            SceneObjectFact(ref="x", label="tower"),
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "source,motion",
    [
        (
            "a fox waits and an owl flies toward a tower beside a lantern",
            SceneMotionFact(source="a", destination="x"),
        ),
        (
            "a fox rises beside a lantern and an owl waits near a tower",
            SceneMotionFact(source="a", direction="rises", destination="x"),
        ),
    ],
)
def test_motion_requires_bound_direction_and_destination(source, motion):
    with pytest.raises(SceneFactsGroundingError):
        _facts(motions=(motion,)).validate_source_grounding(source_text=f"In a forest, {source}.")


@pytest.mark.parametrize("other_action", ["lifts"])
def test_event_object_cannot_be_borrowed_from_another_actor(other_action):
    with pytest.raises(SceneFactsGroundingError):
        _facts(
            events=(SceneEventFact(ref="e", source="a", action="opens", object="o"),)
        ).validate_source_grounding(
            source_text=f"In a forest, a fox opens a tower and an owl {other_action} a lantern."
        )


@pytest.mark.parametrize("negation", ["not "])
def test_salience_checks_negation(negation):
    facts = _facts(salience=(SceneSalienceFact(source="a", layer="foreground"),))
    source = (
        f"In a forest, a fox is {negation}in the foreground beside a lantern "
        "while an owl waits near a tower."
    )
    with pytest.raises(SceneFactsGroundingError):
        facts.validate_source_grounding(source_text=source)


@pytest.mark.parametrize("preposed", [True, False])
def test_temporal_before_marker_must_bind_the_correct_event(preposed):
    facts = _facts(
        events=(
            SceneEventFact(ref="e1", source="a", action="opens", object="o"),
            SceneEventFact(ref="e2", source="b", action="lifts", object="x"),
        ),
        temporal_order=(SceneTemporalOrderFact(before="e1", after="e2"),),
    )
    source = (
        "In a forest, before a fox opens a lantern an owl lifts a tower."
        if preposed
        else "In a forest, a fox opens a lantern before an owl lifts a tower."
    )
    if preposed:
        with pytest.raises(SceneFactsGroundingError):
            facts.validate_source_grounding(source_text=source)
    else:
        facts.validate_source_grounding(source_text=source)


@pytest.mark.parametrize("negation", ["not "])
def test_object_state_checks_negation(negation):
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="forest"),
        objects=(SceneObjectFact(ref="o", label="lantern", states=("open",)),),
    )
    source = f"In a forest, a lantern is {negation}open."
    with pytest.raises(SceneFactsGroundingError):
        facts.validate_source_grounding(source_text=source)
