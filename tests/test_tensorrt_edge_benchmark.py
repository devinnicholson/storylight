from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from storylight.planner_benchmark import CASES, CONTEST_CASES

SCRIPT = Path("deploy/jetson/benchmark-tensorrt-edge-llm.py")
SPEC = importlib.util.spec_from_file_location("tensorrt_edge_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_tensorrt_resident_url_must_be_loopback() -> None:
    assert BENCHMARK._require_loopback_base_url("http://127.0.0.1:11435/") == (
        "http://127.0.0.1:11435"
    )

    try:
        BENCHMARK._require_loopback_base_url("https://example.com")
    except ValueError as error:
        assert "loopback" in str(error)
    else:
        raise AssertionError("non-loopback resident URL was accepted")


def test_tensorrt_resident_payload_preserves_messages_and_token_bound() -> None:
    document = BENCHMARK._request_document(CASES[:1], prompt_profile="slots")
    request = document["requests"][0]

    payload = BENCHMARK._resident_request_payload(request, document, model="llm")

    assert payload["model"] == "llm"
    assert payload["messages"] == request["messages"]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.0
    assert payload["stream"] is False


def test_tensorrt_response_gate_includes_automatic_semantic_screen() -> None:
    case = next(item for item in CASES if item.case_id == "clockwork_fox")
    output = {
        "background_prompt": "snow",
        "focus": {
            "kind": "character",
            "subject": "clockwork fox",
            "action": "plants brass seed",
        },
        "magic": {"kind": "effect", "prompt": "glass branches rise"},
    }

    result = BENCHMARK._validate_responses(
        {"responses": [{"output_text": json.dumps(output), "finish_reason": "eos"}]},
        (case,),
    )[0]

    assert result["valid"] is True
    assert result["automatic_semantic_pass"] is True


def test_tensorrt_slot_protocol_fails_closed_on_missing_or_extra_lines() -> None:
    result = BENCHMARK._validate_responses(
        {
            "responses": [
                {
                    "output_text": "SETTING: snow\nACTOR: fox\nACTION: plants seed\nEXTRA: no",
                    "finish_reason": "eos",
                }
            ]
        },
        (CASES[0],),
        prompt_profile="slots",
    )[0]

    assert result["valid"] is False
    assert result["automatic_semantic_pass"] is False


def test_tensorrt_slot_protocol_separates_actor_from_action_object() -> None:
    case = next(item for item in CONTEST_CASES if item.case_id == "underwater_train")
    output = (
        "SETTING: underwater station\n"
        "ACTOR: octopus conductor, tiny train\n"
        "ACTION: guides train through station\n"
        "MAGIC: bubbles swell into glowing clocks with no numbers"
    )

    result = BENCHMARK._validate_responses(
        {"responses": [{"output_text": output, "finish_reason": "eos"}]},
        (case,),
        prompt_profile="repair",
    )[0]

    assert result["valid"] is True
    assert "octopus conductor" in result["focus"].casefold()
    assert "octopus conductor tiny train" not in result["focus"].casefold()
    assert result["automatic_semantic_pass"] is True


def test_tensorrt_slot_protocol_recovers_passive_actor_relationship() -> None:
    case = next(item for item in CONTEST_CASES if item.case_id == "owl_passive_key")
    output = (
        "SETTING: rain, snowy owl, round door, moon\n"
        "ACTOR: brass key\n"
        "ACTION: carried through rain by snowy owl and unlocks door\n"
        "MAGIC: unlocks round door"
    )

    result = BENCHMARK._validate_responses(
        {"responses": [{"output_text": output, "finish_reason": "eos"}]},
        (case,),
        prompt_profile="repair",
    )[0]

    assert result["valid"] is True
    assert "snowy owl" in result["focus"].casefold()
    assert "carrying brass key" in result["focus"].casefold()
    assert result["automatic_semantic_pass"] is True
