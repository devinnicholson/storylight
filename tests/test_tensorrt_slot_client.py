from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from bookforge.api import _build_live_scene_planner_client
from bookforge.config import Settings
from bookforge.domain import ModelMetrics
from bookforge.live_scene_planner import (
    LiveSceneWirePlan,
    live_scene_plan_prompt,
    validate_live_scene_plan_privacy,
)
from bookforge.model_client import FakeModelClient, ModelUnavailableError
from bookforge.planner_benchmark import CONTEST_CASES, _semantic_evidence
from bookforge.tensorrt_slot_client import (
    TensorRTSlotModelClient,
    tensor_slot_wire_plan,
)

EVIDENCE = Path("benchmarks/jetson-gemma4-tensorrt-repair-2026-09-01.json")


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


def test_production_slot_postprocessor_accepts_all_hardware_outputs_cleanly() -> None:
    evidence = json.loads(EVIDENCE.read_text())

    for case, result in zip(CONTEST_CASES, evidence["cases"], strict=True):
        wire_plan = tensor_slot_wire_plan(
            result["output_text"],
            source_text=case.text,
        )
        plan = wire_plan.to_live_scene_plan(context_text=case.text)
        validate_live_scene_plan_privacy(plan, source_text=case.text)
        generated_text = " ".join(
            (
                plan.scene_summary,
                plan.art_direction,
                plan.background_prompt,
                plan.focus.prompt,
                plan.accent.prompt,
            )
        )
        assert re.search(r"\bv(?=[A-Z])", generated_text) is None
        assert _semantic_evidence(case, generated_text=generated_text)[
            "automatic_semantic_pass"
        ], case.case_id


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
