import pytest

from bookforge.bounded_description import REVISION, plan_bounded_description
from bookforge.live_scene_planner import LiveScenePlannerError
from bookforge.scene_facts import SceneFactsV2, SceneSettingFact, SceneSubjectFact


def test_classic_and_pink_descriptions_preserve_facts_through_final_graph_renderer():
    cases = (
        (
            "A quick brown fox jumps over a lazy dog.",
            "brown",
            ("quick",),
            "jumps over a lazy dog",
            "dog",
        ),
        (
            "The pink fox jumped over the river stream.",
            "pink",
            (),
            "jumped over the river stream",
            "river stream",
        ),
    )
    for text, color, attributes, action, target in cases:
        result = plan_bounded_description(text, "watercolor", 19)
        facts = result.plan.scene_facts
        assert facts.subjects[0].label == "fox"
        assert facts.subjects[0].color == color
        assert facts.subjects[0].attributes == attributes
        assert facts.subjects[0].actions == (action,)
        assert facts.objects[0].label == target
        assert facts.relationships[0].source == facts.subjects[0].ref
        assert facts.relationships[0].target == facts.objects[0].ref
        page = result.plan.to_page(source_text=text, visual_style="watercolor", seed=19)
        assert page.scene_spec.master_prompt == facts.to_renderer_prompt(
            source_text=text, visual_style="watercolor"
        )
        assert action in page.scene_spec.master_prompt
        assert "Setting:" not in page.scene_spec.master_prompt
        assert result.metrics.backend == "deterministic"
        assert result.metrics.model == result.model_revision == REVISION
        assert result.metrics.input_tokens == result.metrics.output_tokens == 0
        assert result.cache_hit is False


def test_two_subjects_counts_colors_and_both_actions_are_retained():
    text = "Two brown foxes jump over a white dog. Three pink birds sit beside a blue pond."
    result = plan_bounded_description(text, "oil pastel", 3)
    facts = result.plan.scene_facts
    assert [(s.label, s.count, s.color) for s in facts.subjects] == [
        ("foxes", 2, "brown"),
        ("birds", 3, "pink"),
    ]
    assert [(o.label, o.count, o.color) for o in facts.objects] == [
        ("dog", 1, "white"),
        ("pond", 1, "blue"),
    ]
    assert [s.actions for s in facts.subjects] == [
        ("jump over a white dog",),
        ("sit beside a blue pond",),
    ]
    prompt = result.plan.to_page(
        source_text=text, visual_style="oil pastel", seed=3
    ).scene_spec.master_prompt
    assert prompt.startswith("Style: oil pastel.")
    assert "exactly 2 brown foxes" in prompt and "exactly 3 pink birds" in prompt


def test_action_negation_and_simple_absence_are_explicit_and_grounded():
    text = "A brown fox stands beside a white dog. The fox does not jump over the dog."
    facts = plan_bounded_description(text, "watercolor", 1).plan.scene_facts
    assert facts.negatives[0].target == facts.subjects[0].ref
    assert facts.negatives[0].value == "jump over the dog"
    assert "Constraints: fox does not jump over the dog" in facts.to_renderer_prompt(
        source_text=text
    )
    text = "A brown fox stands beside a stream. No dogs."
    facts = plan_bounded_description(text, "watercolor", 1).plan.scene_facts
    assert facts.negatives[0].value == "dogs"
    assert all(o.label != "dogs" for o in facts.objects)
    assert "Constraints: no dogs" in facts.to_renderer_prompt(source_text=text)


@pytest.mark.parametrize(
    "text",
    [
        "A quick brown box. Do not throw a lazy dog.",
        "A fox jumps. It runs.",
        "If a fox jumps over a dog.",
        "A fox says jump over a dog.",
        "A fox jumps. A fox runs.",
        "A quick brown fox jumps. The lazy brown fox sleeps.",
        "A quick brown fox stands. The lazy brown fox does not jump.",
        "Two brown foxes stand. Three brown foxes do not jump.",
        "A fox jumps over a dog. The fox does not jump over the dog.",
        "A fox stands. No fox.",
        "A fox carries a dog beside a boat.",
        "A fox jumps. A dog runs. A bird sits.",
        "A child named Alice jumps over a dog.",
        "A fox carries password.",
        "A fox jumps over https://example.com.",
        "Thirteen foxes jump over a dog.",
        "",
    ],
)
def test_unsupported_ambiguous_private_or_contradictory_text_refuses_without_values(text):
    with pytest.raises(LiveScenePlannerError) as error:
        plan_bounded_description(text, "watercolor", 1)
    assert (
        str(error.value)
        == "Reviewed description is unsupported; use one or two explicit subject-action sentences"
    )


def test_unspecified_setting_is_the_only_exception_and_cannot_add_attributes():
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="unspecified"),
        subjects=(SceneSubjectFact(ref="f", label="fox", actions=("jumps",)),),
    )
    facts.validate_source_grounding(source_text="A fox jumps.")
    with pytest.raises(ValueError):
        SceneSettingFact(label="unspecified", attributes=("sunny",))
    with pytest.raises(ValueError):
        facts.model_copy(
            update={"setting": SceneSettingFact(label="meadow")}
        ).validate_source_grounding(source_text="A fox jumps.")
    facts.model_copy(
        update={"setting": SceneSettingFact(label="meadow")}
    ).validate_source_grounding(source_text="A fox jumps in a meadow.")
    with pytest.raises(LiveScenePlannerError):
        plan_bounded_description("A fox jumps.", "watercolor " * 20, 1)


@pytest.mark.parametrize(
    "text,expected_subjects,target_count",
    [
        (
            "The white golden retriever and the Merle Aussie are playing in the field.",
            [("golden retriever", "white", None), ("merle aussie", None, None)],
            None,
        ),
        ("A dog and a cat play in a field.", [("dog", None, 1), ("cat", None, 1)], 1),
        (
            "Two brown foxes and three ducks stand beside a red boat.",
            [("foxes", "brown", 2), ("ducks", None, 3)],
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "style", ["watercolor", "luminous watercolor paper theater, tactile fibers and cinematic depth"]
)
def test_coordinated_subjects_share_exact_predicate_and_one_target(
    text, expected_subjects, target_count, style
):
    result = plan_bounded_description(text, style, 1)
    facts = result.plan.scene_facts
    assert [(s.label, s.color, s.count) for s in facts.subjects] == expected_subjects
    assert len(facts.objects) == 1
    assert facts.objects[0].count == target_count
    assert facts.subjects[0].actions == facts.subjects[1].actions
    assert {edge.source for edge in facts.relationships} == {s.ref for s in facts.subjects}
    assert {edge.target for edge in facts.relationships} == {facts.objects[0].ref}
    prompt = result.plan.to_page(
        source_text=text, visual_style=style, seed=1
    ).scene_spec.master_prompt
    assert facts.subjects[0].actions[0] in prompt
    assert prompt == facts.to_renderer_prompt(source_text=text, visual_style=style)


def test_compound_breed_retains_explicit_color_in_single_subject_description():
    result = plan_bounded_description("A white golden retriever plays in a field.", "watercolor", 1)
    actor = result.plan.scene_facts.subjects[0]
    assert (actor.label, actor.color, actor.count) == ("golden retriever", "white", 1)
    assert actor.actions == ("plays in a field",)


@pytest.mark.parametrize(
    "text",
    [
        "The dog and the dog play in the field.",
        "A white and brown golden retriever plays in a field.",
        "A red fox and a blue owl play in a field near a pond.",
        "A red fox and a blue owl play in a field in a forest.",
        "A dog and a cat and a fox play in a field.",
        "A dog and a cat play in a field. A bird stands beside a boat.",
        "A dog and a cat do not play in the field.",
        "A dog and another dog play in a field.",
        "A dog plays in a field. A cat plays in a field.",
        "A white golden retriever and a black golden retriever play a dog beside a field.",
    ],
)
def test_unsupported_coordination_or_distinct_indefinite_targets_still_refuse(text):
    with pytest.raises(LiveScenePlannerError):
        plan_bounded_description(text, "watercolor", 1)
