"""Relations must bind a predicate to its actual noun phrase, not a later clause."""

import pytest

from bookforge.scene_facts import (
    SceneFactsGroundingError,
    SceneFactsV2,
    SceneObjectFact,
    SceneRelationshipFact,
    SceneSettingFact,
    SceneSubjectFact,
)


def relation_facts(relation, *, between=False):
    return SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(SceneSubjectFact(ref="fox", label="fox"),),
        objects=tuple(SceneObjectFact(ref=label, label=label) for label in ("owl", "cup")),
        relationships=(
            SceneRelationshipFact(
                source="fox",
                relation=relation,
                target="owl",
                secondary_target="cup" if between else None,
            ),
        ),
    )


def assert_relation_refused(facts, source):
    with pytest.raises(SceneFactsGroundingError) as error:
        facts.validate_source_grounding(source_text=source)
    assert "relationships[0]" in error.value.paths


def test_actual_watch_while_does_not_invent_target():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(
            SceneSubjectFact(ref="m", label="meerkat", color="brown", actions=("watches",)),
            SceneSubjectFact(ref="p", label="porcupine", color="gray", actions=("digs",)),
        ),
        relationships=(SceneRelationshipFact(source="m", relation="looks_at", target="p"),),
    )
    source = "A brown meerkat watches while a gray porcupine digs."
    assert_relation_refused(facts, source)
    facts.model_copy(update={"relationships": ()}).validate_source_grounding(source_text=source)
    facts.validate_source_grounding(
        source_text="A brown meerkat watches a gray porcupine. The gray porcupine digs."
    )


@pytest.mark.parametrize(
    "relation,phrase",
    [
        ("looks_at", "watches"),
        ("looks_at", "looks at"),
        ("holds", "holds"),
        ("carries", "carries"),
        ("wears", "wears"),
        ("owns", "owns"),
        ("contains", "contains"),
        ("beside", "is beside"),
        ("behind", "is behind"),
        ("above", "is above"),
        ("under", "is under"),
        ("touches", "touches"),
        ("attached_to", "is attached to"),
        ("left_of", "is left of"),
        ("right_of", "is right of"),
        ("in_front_of", "is in front of"),
        ("inside", "is inside"),
        ("outside", "is outside"),
        ("on", "is on"),
    ],
)
def test_relation_marker_cannot_skip_clause_or_other_target(relation, phrase):
    facts = relation_facts(relation)
    facts.validate_source_grounding(source_text=f"The fox {phrase} an owl. A cup rests.")
    for tail in (
        "while the owl rests",
        "and the owl rests",
        "but the owl rests",
        "because the owl rests",
        "not the owl",
        "a cup near the owl",
        "a cup beside the owl",
        "a cup and an owl sleeps",
    ):
        assert_relation_refused(facts, f"The fox {phrase} {tail}. A cup rests.")


@pytest.mark.parametrize(
    "relation,phrase",
    [
        ("beside", "is beside"),
        ("next_to", "is next to"),
        ("touches", "touches"),
        ("attached_to", "is attached to"),
        ("contains", "is inside"),
        ("owns", "belongs to"),
    ],
)
def test_symmetric_and_inverse_markers_bind_both_sides(relation, phrase):
    facts = relation_facts(relation)
    facts.validate_source_grounding(source_text=f"The owl {phrase} the fox. A cup rests.")
    for text in (
        f"The owl {phrase} while the fox rests. A cup rests.",
        f"The owl {phrase} a cup near the fox.",
        f"The owl waits while a cup {phrase} the fox.",
        f"The owl does not {phrase} the fox. A cup rests.",
    ):
        assert_relation_refused(facts, text)


@pytest.mark.parametrize(
    "relation,phrase",
    [
        ("behind", "is behind"),
        ("above", "is above"),
        ("inside", "is inside"),
        ("contains", "contains"),
        ("owns", "owns"),
    ],
)
def test_directional_markers_cannot_reverse_endpoints(relation, phrase):
    assert_relation_refused(relation_facts(relation), f"The owl {phrase} the fox. A cup rests.")


def test_between_requires_second_target_in_same_coordinated_phrase():
    facts = relation_facts("between", between=True)
    facts.validate_source_grounding(source_text="The fox is between the owl and the cup.")
    for source in (
        "The fox is between the owl while the cup rests.",
        "The fox is between the owl and watches the cup.",
        "The fox is between the owl and not the cup.",
        "The fox is between the owl. A cup rests.",
    ):
        assert_relation_refused(facts, source)


def test_shared_subject_conjunction_keeps_each_explicit_target():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=tuple(SceneSubjectFact(ref=x, label=x) for x in ("fox", "owl")),
        objects=(SceneObjectFact(ref="cup", label="cup"),),
        relationships=tuple(
            SceneRelationshipFact(source=x, relation="looks_at", target="cup")
            for x in ("fox", "owl")
        ),
    )
    facts.validate_source_grounding(source_text="The fox and the owl watch the cup.")


def test_coordinated_targets_share_predicate_without_crossing_a_new_clause():
    facts = relation_facts("looks_at")
    facts.validate_source_grounding(source_text="The fox watches a cup and an owl.")
    with_both = facts.model_copy(
        update={
            "relationships": (
                *facts.relationships,
                SceneRelationshipFact(source="fox", relation="looks_at", target="cup"),
            )
        }
    )
    with_both.validate_source_grounding(source_text="The fox watches a cup and an owl.")
    assert_relation_refused(facts, "The fox watches a cup and the owl sleeps.")
    assert_relation_refused(facts, "The fox watches a cup near an owl.")


@pytest.mark.parametrize("distractor", ["a picture of", "a painting depicting", "a cup near"])
def test_omitted_distractor_is_not_an_explicit_target(distractor):
    facts = relation_facts("looks_at")
    facts = facts.model_copy(update={"objects": (facts.objects[0],)})
    assert_relation_refused(facts, f"The fox watches {distractor} an owl.")


def test_target_noun_phrase_retains_count_color_and_modifiers():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(SceneSubjectFact(ref="fox", label="fox"),),
        objects=(
            SceneObjectFact(
                ref="owls", label="owls", count=2, color="white", attributes=("small",)
            ),
            SceneObjectFact(ref="cup", label="cup", count=1, color="blue"),
        ),
        relationships=tuple(
            SceneRelationshipFact(source="fox", relation="looks_at", target=x)
            for x in ("owls", "cup")
        ),
    )
    facts.validate_source_grounding(
        source_text="The fox watches two small white owls and a blue cup."
    )


def test_coordinated_compound_target_does_not_promote_embedded_entity():
    facts = relation_facts("looks_at")
    facts = facts.model_copy(
        update={
            "objects": (
                *facts.objects,
                SceneObjectFact(ref="picture", label="picture of an owl"),
            )
        }
    )
    source = "The fox watches a picture of an owl and a cup. An owl rests."
    assert_relation_refused(facts, source)
    valid = facts.model_copy(
        update={
            "relationships": (
                SceneRelationshipFact(source="fox", relation="looks_at", target="picture"),
                SceneRelationshipFact(source="fox", relation="looks_at", target="cup"),
            )
        }
    )
    valid.validate_source_grounding(source_text=source)
