import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsPrivacyError,
    SceneFactsV2,
    SceneObjectFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
    SceneTransformationFact,
)
from bookforge.scene_playback import build_display_plan, derive_display_steps, display_manifest

SOURCE = (
    "In a cave, a silver fox lifts a white feather. The feather becomes three golden birds."
)


def _transformation():
    return SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(
            SceneSubjectFact(ref="fox", label="fox", color="silver", actions=("lifts feather",)),
        ),
        objects=(SceneObjectFact(ref="feather", label="feather", color="white"),),
        transformation=SceneTransformationFact(
            source="feather", result_label="birds", result_count=3, result_color="golden"
        ),
    )


def _steps(selection=None):
    return derive_display_steps(
        _transformation(),
        source_text=SOURCE,
        source_page_id="page-05",
        phase_selection=selection,
    )


def test_transformation_steps_replace_source_without_inventing_after_action():
    before, after = _steps({"before": ["action:fox:0"], "after": []})
    assert before.facts.objects[0].label == "feather"
    assert before.facts.transformation is None
    assert before.facts.subjects[0].actions == ("lifts feather",)
    assert after.facts.subjects[0].actions == ()
    assert [(item.label, item.count, item.color) for item in after.facts.objects] == [
        ("birds", 3, "golden")
    ]
    assert after.facts.relationships == () and after.facts.transformation is None
    plan = build_display_plan(after, source_text=SOURCE, visual_style="watercolor")
    page = plan.to_page(
        source_text=SOURCE, visual_style="watercolor", seed=5, page_id=after.step_id
    )
    assert page.source_text == SOURCE
    assert page.page_id == "page-05-after"
    assert page.scene_spec.master_prompt == after.facts.to_renderer_prompt(
        source_text=SOURCE, visual_style="watercolor"
    )
    assert "feather" not in page.scene_spec.master_prompt
    assert "action:" not in page.scene_spec.master_prompt
    manifest = display_manifest((before, after))
    assert SOURCE not in json.dumps(manifest)
    assert manifest["semantic_accuracy_assessed"] is False
    with pytest.raises(ValueError, match="order"):
        display_manifest((after, before))
    with pytest.raises(ValueError, match="source differs"):
        build_display_plan(
            after, source_text=SOURCE + " A tree appears.", visual_style="watercolor"
        )


def test_two_events_split_in_proved_order_with_explicit_independent_motion():
    source = (
        "In a cave, a fox opens a basket then the fox lifts a lantern. Two golden birds fly."
        " The fox closes a box."
    )
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(
            SceneSubjectFact(
                ref="fox", label="fox", actions=("opens basket", "lifts lantern", "closes box")
            ),
            SceneSubjectFact(ref="birds", label="birds", count=2, color="golden", actions=("fly",)),
        ),
        objects=(
            SceneObjectFact(ref="basket", label="basket"),
            SceneObjectFact(ref="lantern", label="lantern"),
            SceneObjectFact(ref="box", label="box"),
        ),
        events=(
            SceneEventFact(ref="open", source="fox", action="opens", object="basket"),
            SceneEventFact(ref="lift", source="fox", action="lifts", object="lantern"),
        ),
        temporal_order=(SceneTemporalOrderFact(before="open", after="lift"),),
    )
    with pytest.raises(ValueError, match="unassigned"):
        derive_display_steps(
            facts,
            source_text=source,
            source_page_id="page-06",
            phase_selection={"first": [], "then": ["action:birds:0"]},
        )
    first, then = derive_display_steps(
        facts,
        source_text=source,
        source_page_id="page-06",
        phase_selection={"first": [], "then": ["action:birds:0", "action:fox:2"]},
    )
    assert first.facts.events == (facts.events[0],)
    assert then.facts.events == (facts.events[1],)
    assert first.facts.subjects[0].actions == ()
    assert then.facts.subjects[0].actions == ("closes box",)
    assert first.facts.subjects[1].actions == () and then.facts.subjects[1].actions == ("fly",)
    assert first.facts.temporal_order == then.facts.temporal_order == ()
    assert then.facts.objects[0].states == ()
    assert [entry["event_ref"] for entry in display_manifest((first, then))["steps"]] == [
        "open", "lift"
    ]
    with pytest.raises(ValueError, match="event reference"):
        replace(first, event_ref="lift")


def test_ordered_event_equivalence_keeps_differently_colored_object_actions():
    source = (
        "In a cave, a fox holds a blue ball then the fox lifts a lantern."
        " The fox holds a red ball."
    )
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(SceneSubjectFact(
            ref="fox", label="fox", actions=("holds blue ball", "holds red ball", "lifts lantern")
        ),),
        objects=(
            SceneObjectFact(ref="blue", label="ball", color="blue"),
            SceneObjectFact(ref="red", label="ball", color="red"),
            SceneObjectFact(ref="lantern", label="lantern"),
        ),
        events=(
            SceneEventFact(ref="hold", source="fox", action="holds", object="blue"),
            SceneEventFact(ref="lift", source="fox", action="lifts", object="lantern"),
        ),
        temporal_order=(SceneTemporalOrderFact(before="hold", after="lift"),),
    )
    with pytest.raises(ValueError, match="explicit phase selection"):
        derive_display_steps(facts, source_text=source, source_page_id="colored-order")
    first, then = derive_display_steps(
        facts, source_text=source, source_page_id="colored-order",
        phase_selection={"first": [], "then": ["action:fox:1"]},
    )
    assert first.facts.subjects[0].actions == ()
    assert then.facts.subjects[0].actions == ("holds red ball",)


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_page_id": "raw private source text"},
        {"source_sha256": "raw private source text"},
        {"parent_graph_sha256": "raw private source text"},
        {"event_ref": "raw private source text"},
        {"phase": "raw private source text"},
        {"ordinal": False},
        {"ordinal": 1},
    ],
)
def test_step_metadata_cannot_export_unbounded_text_or_invalid_phase(metadata):
    before, _ = _steps({"before": ["action:fox:0"], "after": []})
    with pytest.raises(ValueError):
        replace(before, **metadata)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {"before": [], "after": []},
        {"before": ["action:missing:0"], "after": []},
        {"before": [], "after": ["action:fox:0"]},
    ],
)
def test_phase_selection_cannot_omit_invent_or_transfer_dynamic_facts(selection):
    with pytest.raises(ValueError):
        _steps(selection)


def test_parent_graph_privacy_and_references_are_revalidated_before_splitting():
    with pytest.raises(SceneFactsPrivacyError):
        derive_display_steps(
            _transformation().model_copy(
                update={"transformation": SceneTransformationFact(
                    source="feather", result_label="password hunter2"
                )}
            ),
            source_text=SOURCE,
            source_page_id="page-05",
        )
    with pytest.raises(ValidationError):
        derive_display_steps(
            _transformation().model_copy(
                update={
                    "transformation": SceneTransformationFact(
                        source="missing", result_label="birds"
                    )
                }
            ),
            source_text=SOURCE,
            source_page_id="page-05",
        )
