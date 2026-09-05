from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from bookforge import tensorrt_slot_client
from bookforge.api import _build_live_scene_planner_client
from bookforge.config import Settings
from bookforge.domain import ModelMetrics
from bookforge.live_scene_planner import (
    LiveSceneWirePlan,
    live_scene_plan_prompt,
    validate_live_scene_plan_privacy,
)
from bookforge.model_client import FakeModelClient, ModelUnavailableError
from bookforge.planner_benchmark import CONTEST_CASES
from bookforge.tensorrt_slot_client import (
    TensorRTSlotModelClient,
    parse_tensor_graph_slots,
    tensor_accepted_graph_wire_plan,
    tensor_slot_wire_plan,
)

_GRAPH_SOURCE = "In a cave, a fox holds a lantern. A ribbon appears."
_GRAPH_SLOTS = "SETTING: cave\nACTOR: fox\nACTION: holds lantern\nMAGIC: ribbon"


@pytest.mark.parametrize("ending", ["<turn|>", "\n<end_of_turn>\n"])
def test_graph_parser_removes_one_terminal_control_marker_only(ending):
    raw = _GRAPH_SLOTS + ending
    assert parse_tensor_graph_slots(raw) == parse_tensor_graph_slots(_GRAPH_SLOTS)
    candidate = tensor_accepted_graph_wire_plan(raw, source_text=_GRAPH_SOURCE)
    assert candidate.scene_facts is not None
    assert candidate.model_dump(exclude={"scene_facts"}) == tensor_slot_wire_plan(
        _GRAPH_SLOTS, source_text=_GRAPH_SOURCE
    ).model_dump()


@pytest.mark.parametrize(
    "raw,strict_refusal",
    [
        (_GRAPH_SLOTS.replace("fox", "fox<turn|>"), True),
        (_GRAPH_SLOTS + "<turn|><end_of_turn>", True),
        (_GRAPH_SLOTS + "<end_of_turn> trailing", True),
        (_GRAPH_SLOTS + "<unknown|>", False),
        (_GRAPH_SLOTS + "|unsupported", False),
    ],
)
def test_control_marker_recovery_never_attaches_graph_to_malformed_content(raw, strict_refusal):
    if strict_refusal:
        with pytest.raises(ValueError):
            parse_tensor_graph_slots(raw)
    candidate = tensor_accepted_graph_wire_plan(raw, source_text=_GRAPH_SOURCE)
    assert candidate.scene_facts is None
    assert candidate.model_dump(exclude={"scene_facts"}) == tensor_slot_wire_plan(
        raw, source_text=_GRAPH_SOURCE
    ).model_dump()


class StubFallback:
    def __init__(self) -> None:
        self.probes = 0
        self.generations = 0

    async def probe(self) -> tuple[bool, str]:
        self.probes += 1
        return True, "ready"

    async def generate(self, *, system, prompt, output_type):
        self.generations += 1
        return output_type.model_validate(
            {
                "background_prompt": "inside a quiet cave",
                "focus": {
                    "kind": "character",
                    "subject": "child",
                    "action": "raising flashlight",
                },
                "magic": {"kind": "effect", "prompt": "dark birds scatter"},
            }
        ), ModelMetrics(backend="fallback", model="fallback", total_ms=3)


def test_tensorrt_slot_client_requires_loopback() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        TensorRTSlotModelClient(
            base_url="https://models.example.com",
            model="llm",
            timeout_seconds=5,
        )


def test_privacy_separator_keeps_relations_and_negative_coordination():
    separate = tensorrt_slot_client._semantic_privacy_separator
    assert "next to each other" in separate(
        "two boats side by side", source_text="Two boats side by side."
    )
    negative = separate("nothing glows or floats", source_text="Nothing glows or floats.")
    assert "nor" in negative
    assert "alternatively" not in negative


@pytest.mark.parametrize(
    "protocol,source,payload",
    [
        ("slots", "In a room, a page reads orchid delta. A fox waits.", "orchid delta"),
        ("hybrid", "In a room, a page reads orchid delta. A fox waits.", "orchid delta"),
    ],
)
def test_slot_protocols_reject_printed_source_payloads(
    protocol: str,
    source: str,
    payload: str,
) -> None:
    with pytest.raises(ValueError, match="printed source payload"):
        tensor_slot_wire_plan(
            f"SETTING: room\nACTOR: fox\nACTION: waits\nMAGIC: {payload}",
            source_text=source,
            protocol=protocol,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("protocol,name", [("hybrid", "élodie")])
def test_slot_protocols_reject_short_unicode_and_uncased_names(
    protocol: str,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="proper-name candidate"):
        tensor_slot_wire_plan(
            f"SETTING: room\nACTOR: {name}\nACTION: waits\nMAGIC: fireflies",
            source_text=f"{name} enters the room. A fox waits as fireflies appear.",
            protocol=protocol,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("protected_phrase", ["five opal compasses"])
def test_privacy_separator_refuses_to_delete_semantic_modifiers(
    protected_phrase: str,
) -> None:
    with pytest.raises(ValueError, match="protected source phrase"):
        tensorrt_slot_client._semantic_privacy_separator(
            protected_phrase,
            source_text=f"A fox watches {protected_phrase}.",
        )


def test_hybrid_wire_plan_preserves_bound_relation_without_id_artifacts() -> None:
    source = "In the coral reading room, a porcelain lynx holds a green parasol above a stone arch."
    plan = tensor_slot_wire_plan(
        """SETTING: coral reading room
ACTOR: a=porcelain lynx
ACTION: a|holds|o=green parasol; o|above|x=stone arch
MAGIC: green parasol above stone arch""",
        source_text=source,
        protocol="hybrid",
    )

    assert "porcelain lynx" in plan.focus.subject
    assert "green parasol" in plan.focus.action
    assert "green parasol over stone arch" in plan.focus.action
    assert "over stone arch" in plan.focus.action
    assert not re.search(r"\b[orx]\b", plan.focus.action)


def test_hybrid_wire_plan_does_not_rebalance_relation_into_magic() -> None:
    source = (
        "In a coral reading room, a porcelain lynx holds a green parasol above a stone arch "
        "while carrying a long silver telescope. A ring of silver stars appears."
    )
    plan = tensor_slot_wire_plan(
        """SETTING: coral reading room
ACTOR: a=porcelain lynx
ACTION: a|holds|o=green parasol while carrying a long silver telescope; o|above|x=stone arch
MAGIC: ring of silver stars""",
        source_text=source,
        protocol="hybrid",
    )

    assert "above" in plan.focus.action or "over" in plan.focus.action
    assert "ring" in plan.magic.prompt
    assert "silver stars" in plan.magic.prompt


@pytest.mark.parametrize("actor, action", [("a=keeper", "a|watches|o=basket; o=meadow")])
def test_hybrid_wire_plan_rejects_wrong_empty_or_rebound_ids(
    actor: str,
    action: str,
) -> None:
    with pytest.raises(ValueError, match="hybrid"):
        tensor_slot_wire_plan(
            f"SETTING: room\nACTOR: {actor}\nACTION: {action}\nMAGIC: lantern glows",
            source_text="In a room, a keeper holds a lantern as a meadow appears.",
            protocol="hybrid",
        )


def test_tensorrt_cache_identity_tracks_instruction_and_output_budget(monkeypatch) -> None:
    async def run() -> None:
        clients = []
        try:
            for tokens in (64, 64, 96):
                clients.append(
                    TensorRTSlotModelClient(
                        base_url="http://127.0.0.1:11435",
                        model="llm",
                        timeout_seconds=5,
                        max_output_tokens=tokens,
                    )
                )
            monkeypatch.setattr(
                tensorrt_slot_client,
                "TENSORRT_SLOT_SYSTEM_PROMPT",
                tensorrt_slot_client.TENSORRT_SLOT_SYSTEM_PROMPT + " Test revision.",
            )
            clients.append(
                TensorRTSlotModelClient(
                    base_url="http://127.0.0.1:11435",
                    model="llm",
                    timeout_seconds=5,
                )
            )
            assert clients[0].cache_identity == clients[1].cache_identity
            assert len({client.cache_identity for client in clients}) == 3
        finally:
            for client in clients:
                await client.client.aclose()

    asyncio.run(run())


def test_api_planner_client_keeps_default_or_builds_tensorrt_candidate() -> None:
    fallback = FakeModelClient()
    assert (
        _build_live_scene_planner_client(
            Settings(_env_file=None),
            fallback=fallback,
        )
        is fallback
    )

    candidate = _build_live_scene_planner_client(
        Settings(_env_file=None, live_scene_planner_backend="tensorrt_slots"),
        fallback=fallback,
    )
    assert isinstance(candidate, TensorRTSlotModelClient)
    asyncio.run(candidate.client.aclose())

    hybrid = _build_live_scene_planner_client(
        Settings(_env_file=None, live_scene_planner_backend="tensorrt_hybrid"),
        fallback=fallback,
    )
    assert isinstance(hybrid, TensorRTSlotModelClient)
    assert hybrid.protocol == "hybrid"
    asyncio.run(hybrid.client.aclose())

    with pytest.raises(ValueError, match="standard wire contract"):
        _build_live_scene_planner_client(
            Settings(
                _env_file=None,
                live_scene_planner_backend="tensorrt_slots",
                live_scene_planner_compact_wire=True,
            ),
            fallback=fallback,
        )


def test_tensorrt_slot_client_falls_back_only_when_endpoint_cannot_connect() -> None:
    fallback = StubFallback()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def run() -> tuple[LiveSceneWirePlan, object]:
        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            fallback=fallback,  # type: ignore[arg-type]
            fallback_ready_seconds=0,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11435",
            transport=httpx.MockTransport(handler),
        )
        try:
            case = CONTEST_CASES[11]
            return await client.generate(
                system="ignored",
                prompt=live_scene_plan_prompt(
                    text=case.text,
                    visual_style=case.visual_style,
                    seed=case.seed,
                ),
                output_type=LiveSceneWirePlan,
            )
        finally:
            await client.client.aclose()

    plan, metrics = asyncio.run(run())

    assert plan.background_prompt != "inside a quiet cave"
    validate_live_scene_plan_privacy(
        plan.to_live_scene_plan(context_text=CONTEST_CASES[11].text),
        source_text=CONTEST_CASES[11].text,
    )
    assert metrics.backend == "fallback"
    assert fallback.probes == 1
    assert fallback.generations == 1


def test_tensorrt_slot_client_fails_closed_on_ambiguous_server_error() -> None:
    fallback = StubFallback()

    async def run() -> None:
        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            fallback=fallback,  # type: ignore[arg-type]
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11435",
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )
        case = CONTEST_CASES[12]
        try:
            with pytest.raises(ModelUnavailableError, match="request failed"):
                await client.generate(
                    system="ignored",
                    prompt=live_scene_plan_prompt(
                        text=case.text,
                        visual_style=case.visual_style,
                        seed=case.seed,
                    ),
                    output_type=LiveSceneWirePlan,
                )
        finally:
            await client.client.aclose()

    asyncio.run(run())
    assert fallback.probes == 0
    assert fallback.generations == 0


def test_tensorrt_slot_client_sends_accepted_prompt_and_returns_wire_plan() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "llm",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "SETTING: darkest night\n"
                                "ACTOR: child\n"
                                "ACTION: opens silent book\n"
                                "MAGIC: every unwritten letter becomes a luminous origami "
                                "bird, forming a bridge of constellations toward a floating "
                                "school above the clouds<turn|>"
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 276, "completion_tokens": 44},
            },
        )

    async def run() -> tuple[LiveSceneWirePlan, object]:
        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11435",
            transport=httpx.MockTransport(handler),
        )
        case = CONTEST_CASES[0]
        try:
            return await client.generate(
                system="ignored generic schema instruction",
                prompt=live_scene_plan_prompt(
                    text=case.text,
                    visual_style=case.visual_style,
                    seed=case.seed,
                ),
                output_type=LiveSceneWirePlan,
            )
        finally:
            await client.client.aclose()

    wire_plan, metrics = asyncio.run(run())

    assert observed["max_tokens"] == 64
    assert observed["temperature"] == 0
    assert observed["messages"][0]["role"] == "system"  # type: ignore[index]
    assert observed["messages"][-1]["content"].startswith("STORY:\nOn the darkest")  # type: ignore[index]
    assert wire_plan.magic.prompt.endswith("school above clouds")
    assert metrics.backend == "tensorrt-edge-llm"
    assert metrics.input_tokens == 276
    assert metrics.output_tokens == 44


@pytest.mark.parametrize("finish_reason", ["length", "tool_calls"])
def test_tensorrt_slot_client_rejects_incomplete_generation_without_fallback(
    finish_reason: str,
) -> None:
    fallback = StubFallback()

    async def run() -> None:
        client = TensorRTSlotModelClient(
            base_url="http://127.0.0.1:11435",
            model="llm",
            timeout_seconds=5,
            fallback=fallback,
        )
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=client.base_url,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "finish_reason": finish_reason,
                                "message": {
                                    "content": "SETTING: cave\nACTOR: child\n"
                                    "ACTION: lifts lantern\nMAGIC: silver birds"
                                },
                            }
                        ],
                    },
                )
            ),
        )
        case = CONTEST_CASES[0]
        try:
            with pytest.raises(ModelUnavailableError, match="did not finish normally"):
                await client.generate(
                    system="ignored",
                    prompt=live_scene_plan_prompt(
                        text=case.text,
                        visual_style=case.visual_style,
                        seed=case.seed,
                    ),
                    output_type=LiveSceneWirePlan,
                )
        finally:
            await client.client.aclose()

    asyncio.run(run())
    assert fallback.generations == 0
    assert fallback.probes == 0


def test_tensorrt_slot_postprocessor_restores_passive_agent_and_destination() -> None:
    case = next(item for item in CONTEST_CASES if item.case_id == "owl_passive_key")
    wire_plan = tensor_slot_wire_plan(
        "SETTING: rain, snowy owl, round door, moon\n"
        "ACTOR: brass key\n"
        "ACTION: carried through rain by snowy owl and unlocks door\n"
        "MAGIC: unlocks round door",
        source_text=case.text,
    )

    assert wire_plan.focus.subject == "snowy owl"
    assert "carries brass key" in wire_plan.focus.action
    assert wire_plan.magic.prompt == "unlocks moon door"


def test_tensorrt_slot_postprocessor_preserves_explicit_negation() -> None:
    case = next(item for item in CONTEST_CASES if item.case_id == "bakery_volcano")
    wire_plan = tensor_slot_wire_plan(
        "SETTING: oven\n"
        "ACTOR: round robot baker\n"
        "ACTION: opens the oven\n"
        "MAGIC: mountain of bread dough erupts with colorful confetti instead of smoke",
        source_text=case.text,
    )

    assert "rather than smoke" in wire_plan.magic.prompt


def test_tensorrt_slot_postprocessor_keeps_static_book_state_and_absence() -> None:
    source = "One child stands on a wooden bridge holding an open green book. No other people."
    wire = tensor_slot_wire_plan(
        "SETTING: wooden bridge\nACTOR: child\nACTION: stands holding open green book\n"
        "MAGIC: No other people",
        source_text=source,
    )
    assert "opening" not in wire.focus.action
    assert "open" in wire.focus.action
    assert wire.magic.prompt == "no additional people"
    validate_live_scene_plan_privacy(
        wire.to_live_scene_plan(context_text=source), source_text=source
    )
