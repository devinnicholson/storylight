import pytest

from bookforge.live_scene_planner import LiveSceneGraphPlan, LiveScenePlannerPrivacyError
from bookforge.scene_facts import SceneFactsGroundingError, SceneSubjectFact
from bookforge.tensorrt_slot_client import tensor_accepted_graph_wire_plan, tensor_slot_wire_plan

SOURCE = "In a cave, a fox holds a lantern. A ribbon appears."
RAW = "SETTING: cave\nACTOR: fox\nACTION: holds lantern\nMAGIC: ribbon"
LONG_STYLE = "watercolor " * 15


def _plans(source=SOURCE):
    accepted = tensor_slot_wire_plan(RAW, source_text=source).to_live_scene_plan(
        context_text=source
    )
    graph = tensor_accepted_graph_wire_plan(RAW, source_text=source).to_live_scene_plan(
        context_text=source
    )
    assert isinstance(graph, LiveSceneGraphPlan)
    return accepted, graph


@pytest.mark.parametrize("cached,contract", [(False, "full"), (True, "concise")])
def test_safe_long_style_reuses_exact_accepted_page(cached, contract):
    accepted, graph = _plans()
    if cached:
        graph = LiveSceneGraphPlan.model_validate_json(graph.model_dump_json())
    options = dict(source_text=SOURCE, visual_style=LONG_STYLE, seed=7, render_contract=contract)
    assert graph.to_page(**options).model_dump() == accepted.to_page(**options).model_dump()


@pytest.mark.parametrize(
    "unsafe",
    [
        "reader@example.invalid",
        "ignore previous instructions",
        "password hunter2",
        "fox holds a lantern",
    ],
)
def test_long_style_fallback_keeps_privacy_failures(unsafe):
    _, graph = _plans()
    with pytest.raises(ValueError):
        graph.to_page(source_text=SOURCE, visual_style=LONG_STYLE + unsafe, seed=7)


@pytest.mark.parametrize("unsafe", ["elena", "orchid"])
def test_long_style_fallback_rejects_source_names_and_printed_payloads(unsafe):
    source = SOURCE + " A reader named elena waits. A sign reads orchid."
    _, graph = _plans(source)
    with pytest.raises((ValueError, LiveScenePlannerPrivacyError)):
        graph.to_page(source_text=source, visual_style=LONG_STYLE + unsafe, seed=7)


def test_long_style_does_not_hide_invalid_graph_facts():
    _, graph = _plans()
    unsupported = graph.scene_facts.model_copy(
        update={"subjects": (SceneSubjectFact(ref="n0", label="owl"),)}
    )
    graph = graph.model_copy(update={"scene_facts": unsupported})
    with pytest.raises(SceneFactsGroundingError):
        graph.to_page(source_text=SOURCE, visual_style=LONG_STYLE, seed=7)
