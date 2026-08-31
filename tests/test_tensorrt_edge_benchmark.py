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
