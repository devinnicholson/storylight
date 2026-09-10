import asyncio
import json

import httpx
import pytest

from storylight.api import _build_live_scene_planner_client
from storylight.config import Settings
from storylight.live_scene_planner import (
    LiveSceneGraphPlan,
    LiveSceneGraphWirePlan,
    LiveScenePlannerError,
    StructuredLiveScenePlanner,
)
from storylight.model_client import FakeModelClient, ModelUnavailableError
from storylight.tensorrt_slot_client import (
    TENSORRT_SLOT_SYSTEM_PROMPT,
    TensorRTSlotModelClient,
    tensor_accepted_graph_wire_plan,
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


@pytest.mark.parametrize("malformed", [HYBRID.replace("\n", " ")])
def test_graph_envelope_is_stricter_than_accepted_parser(malformed):
    with pytest.raises(ValueError, match="four ordered"):
        tensor_graph_wire_plan(malformed, source_text=SOURCE)


@pytest.mark.parametrize("protocol,content", [("slots", ACCEPTED)])
def test_graph_survives_live_planning_and_persistent_cache(tmp_path, protocol, content):
    async def run():
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"completion_tokens": 50},
                },
            )

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            protocol=protocol,
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


def test_accepted_graph_backend_uses_unchanged_accepted_prompt_and_distinct_cache():
    async def run():
        clients = [
            _build_live_scene_planner_client(
                Settings(_env_file=None, live_scene_planner_backend=backend),
                fallback=FakeModelClient(),
            )
            for backend in ("tensorrt_slots", "tensorrt_graph", "tensorrt_accepted_graph")
        ]
        try:
            assert clients[2].protocol == "slots"
            assert clients[2].scene_facts_enabled
            assert len({client.cache_identity for client in clients}) == 3
            assert not clients[0].scene_facts_enabled
        finally:
            for client in clients:
                await client.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("content", [ACCEPTED, ACCEPTED.replace("\n", " ")])
def test_accepted_graph_success_or_refusal_uses_exactly_one_request(content):
    async def run():
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                },
            )

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            scene_facts_enabled=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(respond)
        )
        try:
            wire, metrics = await client.generate(
                system="",
                prompt="\nInput:\n" + json.dumps({"passage": SOURCE}),
                output_type=LiveSceneGraphWirePlan,
            )
            expected = tensor_slot_wire_plan(content, source_text=SOURCE)
            assert wire.model_dump(exclude={"scene_facts"}) == expected.model_dump()
            assert (wire.scene_facts is not None) == (content == ACCEPTED)
            assert metrics.output_tokens == 30 and metrics.input_tokens == 100
            assert len(calls) == 1
            assert calls[0]["messages"][0]["content"] == TENSORRT_SLOT_SYSTEM_PROMPT
            assert calls[0]["max_tokens"] == 64
        finally:
            await client.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [ValueError])
def test_accepted_graph_late_compiler_refusal_preserves_accepted_wire(monkeypatch, error_type):
    def refuse(*args, **kwargs):
        raise error_type("compiler refusal")

    monkeypatch.setattr(LiveSceneGraphPlan, "to_page", refuse)
    wire = tensor_accepted_graph_wire_plan(ACCEPTED, source_text=SOURCE)
    assert wire.scene_facts is None
    assert (
        wire.model_dump(exclude={"scene_facts"})
        == tensor_slot_wire_plan(ACCEPTED, source_text=SOURCE).model_dump()
    )


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
            # The accepted fallback drops the source's counts, colors, and
            # spatial target. It must not reach the renderer after graph refusal.
            with pytest.raises(LiveScenePlannerError, match="unsupported visual facts"):
                await planner.plan(text=SOURCE, visual_style="watercolor", seed=0)
            assert len(calls) == 2
            assert calls[1]["messages"][0]["content"] == TENSORRT_SLOT_SYSTEM_PROMPT
        finally:
            await client.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["slots"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_graph_response_validation_errors_are_value_free(protocol, finish_reason):
    async def run():
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "message": {"content": {"private": SOURCE}},
                        }
                    ]
                },
            )

        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            protocol=protocol,
            scene_facts_enabled=True,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url,
            transport=httpx.MockTransport(respond),
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
            assert len(calls) == 1
        finally:
            await client.client.aclose()

    asyncio.run(run())
