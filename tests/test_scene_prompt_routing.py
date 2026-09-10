from __future__ import annotations

import asyncio
import json

import httpx

from storylight.scene_prompt_routing import scene_messages, select_scene_prompt
from storylight.tensorrt_slot_client import _slot_messages


def test_passive_route_requires_one_original_asserted_complete_clause():
    for source in (
        "In a meadow, one red basket is carried by two white badgers. A kite appears.",
        "Two red baskets were carried by a white badger. A kite appears.",
        "Inside an orchard, a basket was carried by an owl. A kite appears.",
    ):
        assert select_scene_prompt(source) == "accepted_passive"
        assert scene_messages(source) == _slot_messages(source)
    for source in (
        "A basket is not carried by a badger.",
        "A basket is 'not' carried by a badger.",
        'A sign reads "a basket is carried by a badger".',
        "An owl says a basket is carried by a badger.",
        "If a basket is carried by a badger, a kite appears.",
        "A basket might be carried by a badger.",
        "A basket is carried by a badger and an owl.",
        "A basket is carried by a badger then the badger lifts a stone.",
        "A basket is carried by a badger. A stone is carried by an owl.",
        "A badger carries a basket.",
        "A basket is held by a badger.",
        "A basket is carried by.",
        "A basket is carried by him.",
        "It is carried by a badger.",
    ):
        assert select_scene_prompt(source) == "scene_default"
    assert select_scene_prompt("a" * 4001) == "scene_default"


def test_route_selection_does_not_repair_a_contradictory_model_response():
    from storylight.live_scene_facts import adapt_live_scene_facts

    source = "In a meadow, one red basket is carried by two white badgers. A kite appears."
    assert select_scene_prompt(source) == "accepted_passive"
    result = adapt_live_scene_facts(
        {
            "SETTING": "meadow",
            "ACTOR": "two white badgers",
            "ACTION": "carried by a red basket",
            "MAGIC": "kite",
        },
        source_text=source,
        scope="scene",
    )
    assert result.facts is None


def test_scene_client_sends_one_source_selected_request_and_binds_both_prompts(monkeypatch):
    from storylight import tensorrt_slot_client as client_module
    from storylight.live_scene_planner import LiveSceneGraphWirePlan, live_scene_plan_prompt

    async def run():
        sources = (
            "In a meadow, one red basket is carried by two white badgers. A kite appears.",
            "In a meadow, two white badgers carry one red basket. A kite appears.",
        )
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": (
                "SETTING: meadow\nACTOR: two white badgers\n"
                "ACTION: carry red basket\nMAGIC: kite"
            )}, "finish_reason": "stop"}]})

        def client(scope):
            return client_module.TensorRTSlotModelClient(
                base_url="http://127.0.0.1:11435", model="llm", timeout_seconds=5,
                scene_facts_enabled=True, planning_scope=scope,
            )

        focal, scene = client("focal"), client("scene")
        await scene.client.aclose()
        scene.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11435", transport=httpx.MockTransport(handler)
        )
        try:
            for source in sources:
                plan, _ = await scene.generate(
                    system="ignored", output_type=LiveSceneGraphWirePlan,
                    prompt=live_scene_plan_prompt(text=source, visual_style="watercolor", seed=1),
                )
                assert plan.scene_facts is not None
            assert [row["messages"] for row in calls] == [scene_messages(s) for s in sources]
            assert len(calls) == 2
            original = client_module.scene_messages
            monkeypatch.setattr(client_module, "scene_messages", lambda source: [
                *original(source), {"role": "user", "content": "changed wrapper"},
            ])
            changed_focal, changed_scene = client("focal"), client("scene")
            try:
                assert changed_focal.cache_identity == focal.cache_identity
                assert changed_scene.cache_identity != scene.cache_identity
            finally:
                await changed_focal.client.aclose()
                await changed_scene.client.aclose()
        finally:
            await focal.client.aclose()
            await scene.client.aclose()

    asyncio.run(run())
