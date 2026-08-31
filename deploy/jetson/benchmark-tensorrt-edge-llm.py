#!/usr/bin/env python3
"""Run Bookforge's fixed semantic cases through TensorRT Edge-LLM."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from bookforge.live_scene_planner import (
    LIVE_SCENE_SYSTEM_PROMPT,
    LiveSceneWirePlan,
    live_scene_plan_prompt,
    validate_live_scene_plan_privacy,
)
from bookforge.planner_benchmark import (
    CASES,
    CONTEST_CASES,
    BenchmarkCase,
    _semantic_evidence,
)

_WIRE_SCHEMA_INSTRUCTION = (
    "The supplied JSON schema is exactly: "
    '{"background_prompt":"setting, at most 10 words",'
    '"focus":{"kind":"character or prop","subject":"complete actor, at most 8 words",'
    '"action":"visible action, at most 6 words"},'
    '"magic":{"kind":"character, prop, or effect",'
    '"prompt":"surprising supporting element, at most 8 words"}} '
    "Return that object and nothing else."
)

_COMPACT_SYSTEM_PROMPT = (
    "You extract visual facts from a story for an illustration.\n"
    "Reply with exactly one JSON object and no markdown or explanation.\n"
    'Exact format: {"background_prompt":"setting",'
    '"focus":{"kind":"character or prop","subject":"actor",'
    '"action":"visible action"},"magic":{"kind":"character, prop, or effect",'
    '"prompt":"surprising second event"}}\n'
    "Use only story facts. Background is setting only. Focus is the main actor and its action. "
    "Magic is the later transformation, creature, or impossible event. Keep every value under "
    "10 words. Do not copy a name, personal information, or three adjacent source words. "
    "Do not invent anything."
)


def _messages(case: BenchmarkCase, *, prompt_profile: str) -> list[dict[str, str]]:
    if prompt_profile == "production":
        return [
            {
                "role": "system",
                "content": f"{LIVE_SCENE_SYSTEM_PROMPT}\n{_WIRE_SCHEMA_INSTRUCTION}",
            },
            {
                "role": "user",
                "content": live_scene_plan_prompt(
                    text=case.text,
                    visual_style=case.visual_style,
                    seed=case.seed,
                ),
            },
        ]
    return [
        {"role": "system", "content": _COMPACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                'Example story: "A keeper lifts a key, and bright moths spiral through an arch."\n'
                'Example JSON: {"background_prompt":"stone arch at dusk",'
                '"focus":{"kind":"character","subject":"a keeper","action":"lifts key"},'
                '"magic":{"kind":"character","prompt":"bright moths spiral through arch"}}\n'
                f'Story: "{case.text}"\nJSON:'
            ),
        },
    ]


def _request_document(
    cases: tuple[BenchmarkCase, ...],
    *,
    prompt_profile: str,
) -> dict[str, object]:
    requests: list[dict[str, object]] = []
    for case in cases:
        requests.append({"messages": _messages(case, prompt_profile=prompt_profile)})
    return {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": 128 if prompt_profile == "compact" else 180,
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "requests": requests,
    }


def _validate_responses(
    payload: dict[str, object],
    cases: tuple[BenchmarkCase, ...],
) -> list[dict[str, object]]:
    raw_responses = payload.get("responses")
    if not isinstance(raw_responses, list) or len(raw_responses) != len(cases):
        raise ValueError("TensorRT output did not contain one response per benchmark case")
    results: list[dict[str, object]] = []
    for case, response in zip(cases, raw_responses, strict=True):
        if not isinstance(response, dict):
            raise ValueError("TensorRT output response was not an object")
        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ValueError("TensorRT output response did not contain output_text")
        try:
            wire_plan = LiveSceneWirePlan.model_validate_json(output_text)
            plan = wire_plan.privacy_sanitized(source_text=case.text).to_live_scene_plan(
                context_text=case.text
            )
            validate_live_scene_plan_privacy(plan, source_text=case.text)
        except (ValidationError, ValueError) as error:
            results.append(
                {
                    "case_id": case.case_id,
                    "valid": False,
                    "finish_reason": response.get("finish_reason"),
                    "error": str(error)[:500],
                    "output_text": output_text,
                    "automatic_semantic_pass": False,
                    "semantic_checks": [],
                    "forbidden_checks": [],
                }
            )
            continue
        generated_text = " ".join(
            (
                plan.scene_summary,
                plan.art_direction,
                plan.background_prompt,
                plan.focus.prompt,
                plan.accent.prompt,
            )
        )
        results.append(
            {
                "case_id": case.case_id,
                "valid": True,
                "finish_reason": response.get("finish_reason"),
                "background": plan.background_prompt,
                "focus": plan.focus.prompt,
                "magic": plan.accent.prompt,
                "scene_summary": plan.scene_summary,
                "output_text": output_text,
                **_semantic_evidence(case, generated_text=generated_text),
            }
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-profile", choices=("production", "compact"), default="production")
    parser.add_argument("--suite", choices=("five", "contest"), default="five")
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    parser.add_argument("--candidate-revision", default="db09cd27ead7fee40cdee309693cf83601b9c899")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cases = CASES if args.suite == "five" else CONTEST_CASES
    args.work_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.work_dir / "requests.json"
    raw_output_path = args.work_dir / "responses.json"
    profile_path = args.work_dir / "profile.json"
    request_path.write_text(
        json.dumps(
            _request_document(cases, prompt_profile=args.prompt_profile),
            indent=2,
        )
        + "\n"
    )

    command = [
        str(args.binary),
        f"--engineDir={args.engine_dir}",
        f"--checkpointDir={args.checkpoint_dir}",
        f"--inputFile={request_path}",
        f"--outputFile={raw_output_path}",
        f"--profileOutputFile={profile_path}",
        f"--warmup={args.warmup}",
        "--dumpProfile",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    wall_ms = (time.perf_counter() - started) * 1_000
    if not raw_output_path.is_file():
        raise RuntimeError(
            "TensorRT Edge-LLM did not create an output file: "
            f"exit={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    raw_output = json.loads(raw_output_path.read_text())
    case_results = _validate_responses(raw_output, cases)
    schema_valid = all(case["valid"] for case in case_results)
    automatic_semantic_pass = all(
        bool(case["automatic_semantic_pass"]) for case in case_results
    )
    if completed.returncode != 0 or not schema_valid:
        result = "technical_fail"
    elif not automatic_semantic_pass:
        result = "automatic_semantic_fail_human_review_required"
    else:
        result = "technical_pass_human_semantic_review_required"
    report = {
        "schema_version": "1.0",
        "captured_at": datetime.now(UTC).isoformat(),
        "result": result,
        "runtime": {
            "backend": "tensorrt-edge-llm",
            "candidate_model": args.candidate_model,
            "candidate_revision": args.candidate_revision,
            "binary": str(args.binary),
            "engine_dir": str(args.engine_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "warmup_runs": args.warmup,
            "prompt_profile": args.prompt_profile,
            "suite": args.suite,
            "case_count": len(cases),
            "wall_ms": round(wall_ms, 3),
            "exit_code": completed.returncode,
        },
        "cases": case_results,
        "profile": json.loads(profile_path.read_text()) if profile_path.is_file() else None,
        "privacy": {
            "model_and_artifacts_local_only": True,
            "fixtures_are_synthetic": True,
            "structured_plan_privacy_gate_exercised": True,
            "modal_or_cloud_called": False,
        },
        "acceptance": {
            "all_outputs_schema_valid": schema_valid,
            "automatic_semantic_pass": automatic_semantic_pass,
            "automatic_checks_are_lexical_prescreen_only": True,
            "human_semantic_review_required": True,
            "candidate_not_promoted_by_this_benchmark": True,
            "production_backend_unchanged": "ollama/gemma3:1b-it-q4_K_M",
        },
        "diagnostics": {
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["result"] != "technical_pass_human_semantic_review_required":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
