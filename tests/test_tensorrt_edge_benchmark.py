from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from bookforge.planner_benchmark import CASES, CONTEST_CASES

SCRIPT = Path("deploy/jetson/benchmark-tensorrt-edge-llm.py")
SPEC = importlib.util.spec_from_file_location("tensorrt_edge_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_tensorrt_contest_request_covers_all_twenty_semantic_cases() -> None:
    request = BENCHMARK._request_document(CONTEST_CASES, prompt_profile="production")

    assert len(request["requests"]) == 20
    assert request["max_generate_length"] == 180


def test_tensorrt_benchmark_parser_accepts_one_case_filter() -> None:
    parser = BENCHMARK._parser()

    args = parser.parse_args(
        [
            "--binary",
            "/tmp/llm_inference",
            "--engine-dir",
            "/tmp/engine",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--work-dir",
            "/tmp/work",
            "--output",
            "/tmp/report.json",
            "--case-id",
            "clockwork_fox",
        ]
    )

    assert args.case_id == "clockwork_fox"


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


def test_tensorrt_invalid_json_cannot_pass_semantics() -> None:
    result = BENCHMARK._validate_responses(
        {"responses": [{"output_text": "not-json", "finish_reason": "eos"}]},
        (CASES[0],),
    )[0]

    assert result["valid"] is False
    assert result["automatic_semantic_pass"] is False


def test_tensorrt_slot_protocol_builds_valid_semantic_plan() -> None:
    case = next(item for item in CASES if item.case_id == "clockwork_fox")
    output = (
        "SETTING: snow\n"
        "ACTOR: clockwork fox\n"
        "ACTION: plants brass seed\n"
        "MAGIC: glass branches rise"
    )

    result = BENCHMARK._validate_responses(
        {"responses": [{"output_text": output, "finish_reason": "eos"}]},
        (case,),
        prompt_profile="slots",
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


def test_tensorrt_slot_protocol_normalizes_inline_and_repeated_magic() -> None:
    slots = BENCHMARK._parse_slots(
        "SETTING: flooded library ACTOR: silver whale ACTION: swims through library "
        "MAGIC: carries lantern MAGIC: books open into bright fish"
    )

    assert slots == {
        "SETTING": "flooded library",
        "ACTOR": "silver whale",
        "ACTION": "swims through library",
        "MAGIC": "carries lantern; books open into bright fish",
    }


def test_tensorrt_slot_protocol_bounds_long_values_without_losing_tail() -> None:
    value = BENCHMARK._fit_wire_value(
        "every unwritten letter becomes a luminous origami bird forming a bridge of "
        "constellations toward a floating school above the clouds",
        maximum=80,
    )

    assert len(value) <= 80
    assert value.startswith("every unwritten")
    assert value.endswith("above the clouds")


def test_tensorrt_slot_privacy_separator_preserves_semantic_terms() -> None:
    value = BENCHMARK._privacy_separated_value(
        "a silver whale swims through a flooded library",
        source_text="A silver whale swims through a flooded library.",
    )

    assert "silver" in value.casefold()
    assert "whale" in value.casefold()
    assert "vSilver" in value
    tokens = BENCHMARK._privacy_tokens(value)
    assert ("silver", "whale", "swims") not in {
        tokens[index : index + 3] for index in range(len(tokens) - 2)
    }


def test_tensorrt_slot_privacy_separator_does_not_mask_names() -> None:
    value = BENCHMARK._privacy_separated_value(
        "Alice opens a silent book",
        source_text="Alice opens a silent book.",
    )

    assert "Alice" in value
    assert "vAlice" not in value
