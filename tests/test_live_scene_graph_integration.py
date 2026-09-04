import asyncio
import json

import httpx
import pytest

from bookforge.api import _build_live_scene_planner_client
from bookforge.config import Settings
from bookforge.live_scene_planner import (
    LiveSceneGraphPlan,
    LiveSceneGraphWirePlan,
    LiveSceneWirePlan,
    StructuredLiveScenePlanner,
)
from bookforge.model_client import FakeModelClient, ModelUnavailableError
from bookforge.tensorrt_slot_client import (
    TENSORRT_SLOT_SYSTEM_PROMPT,
    TensorRTSlotModelClient,
    tensor_graph_wire_plan,
    tensor_slot_wire_plan,
)

SOURCE = "In a cave, two orange foxes hold one blue lantern above a wooden box. A ribbon appears."
HYBRID = (
    "SETTING: cave\nACTOR: a=two orange foxes\n"
    "ACTION: a|hold|o=one blue lantern; o|above|x=wooden box\n"
    "MAGIC: a|causes|r=ribbon"
)
ACCEPTED = "SETTING: cave\nACTOR: foxes\nACTION: hold lantern\nMAGIC: ribbon"


def test_graph_backend_is_explicit_and_keeps_accepted_wire_schema():
    assert set(LiveSceneWirePlan.model_fields) == {"background_prompt", "focus", "magic"}
    client = _build_live_scene_planner_client(
        Settings(_env_file=None, live_scene_planner_backend="tensorrt_graph"),
        fallback=FakeModelClient(),
    )
    assert client.scene_facts_enabled and client.protocol == "hybrid"
    asyncio.run(client.client.aclose())


@pytest.mark.parametrize("malformed", [HYBRID.replace("\n", " "), HYBRID + "\nMAGIC: ribbon"])
def test_graph_envelope_is_stricter_than_accepted_parser(malformed):
    with pytest.raises(ValueError, match="four ordered"):
        tensor_graph_wire_plan(malformed, source_text=SOURCE)


def test_graph_survives_live_planning_and_persistent_cache(tmp_path):
    async def run():
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": HYBRID}}],
                    "usage": {"completion_tokens": 50},
                },
            )

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            protocol="hybrid",
            scene_facts_enabled=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(respond)
        )
        try:
            for _ in range(2):
                planner = StructuredLiveScenePlanner(
                    client, timeout_seconds=5, persistent_cache_dir=tmp_path
                )
                result = await planner.plan(text=SOURCE, visual_style="watercolor", seed=0)
                assert isinstance(result.plan, LiveSceneGraphPlan)
                page = result.plan.to_page(source_text=SOURCE, visual_style="watercolor", seed=0)
                assert page.scene_spec.master_prompt == result.plan.scene_facts.to_renderer_prompt(
                    source_text=SOURCE, visual_style="watercolor"
                )
                assert SOURCE not in page.scene_spec.master_prompt
            assert len(calls) == 1
        finally:
            await client.client.aclose()

    asyncio.run(run())


def test_refused_graph_runs_one_accepted_fallback_and_counts_both_requests():
    async def run():
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            content = HYBRID.replace("two orange foxes", "three orange foxes")
            if len(calls) == 2:
                content = ACCEPTED
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"completion_tokens": 30},
                },
            )

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            protocol="hybrid",
            scene_facts_enabled=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(respond)
        )
        try:
            planner = StructuredLiveScenePlanner(client, timeout_seconds=5)
            result = await planner.plan(text=SOURCE, visual_style="watercolor", seed=0)
            expected = tensor_slot_wire_plan(ACCEPTED, source_text=SOURCE).to_live_scene_plan(
                context_text=SOURCE
            )
            assert result.plan.model_dump() == expected.model_dump()
            assert result.metrics.output_tokens == 60
            assert len(calls) == 2
            assert calls[1]["messages"][0]["content"] == TENSORRT_SLOT_SYSTEM_PROMPT
        finally:
            await client.client.aclose()

    asyncio.run(run())


def test_graph_response_validation_errors_are_value_free():
    async def run():
        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            protocol="hybrid",
            scene_facts_enabled=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": {"private": SOURCE}}}]}
                )
            ),
        )
        try:
            with pytest.raises(ModelUnavailableError) as error:
                await client.generate(
                    system="",
                    prompt="\nInput:\n" + json.dumps({"passage": SOURCE}),
                    output_type=LiveSceneGraphWirePlan,
                )
            assert str(error.value) == "TensorRT graph response failed validation"
            assert error.value.__cause__ is None
        finally:
            await client.client.aclose()

    asyncio.run(run())
