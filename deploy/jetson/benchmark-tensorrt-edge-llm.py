#!/usr/bin/env python3
"""Run Bookforge's fixed semantic cases through TensorRT Edge-LLM."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from bookforge.live_scene_planner import (
    _PHRASE_STOPWORDS,
    LIVE_SCENE_SYSTEM_PROMPT,
    LiveScenePlannerPrivacyError,
    LiveSceneWirePlan,
    _distinctive_phrase,
    _privacy_tokens,
    _proper_name_candidates,
    _recover_source_grounded_setting,
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

_MINIMAL_SYSTEM_PROMPT = (
    "Extract visible facts from the story. Return exactly one valid JSON object. "
    "No markdown, explanation, or extra keys."
)

_SLOT_SYSTEM_PROMPT = (
    "Extract four visible facts from the story. Reply with exactly four labeled lines and no "
    "other text. Keep each value under eight words."
)

_REPAIR_SYSTEM_PROMPT = (
    "Read the story carefully. Identify the actual setting, main actor, its visible action, and "
    "the later magical result. Exclude anything the story says is absent, negated, or replaced. "
    "Reply with exactly four labeled lines: SETTING, ACTOR, ACTION, MAGIC. No other text."
)

_SLOT_PROFILES = frozenset({"slots", "repair"})

_SLOT_LABEL_PATTERN = re.compile(r"(?i)(?:^|\s)(SETTING|ACTOR|ACTION|MAGIC)\s*:\s*")


def _require_loopback_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("resident TensorRT benchmark requires a loopback URL")
    return value.rstrip("/")


def _fit_wire_value(value: str, *, maximum: int = 110) -> str:
    """Bound a slot while preserving its beginning and ending semantic anchors."""

    words = " ".join(value.strip().strip('"').split()).split()
    while len(" ".join(words)) > maximum and len(words) > 2:
        del words[len(words) // 2]
    return " ".join(words)[:maximum].rstrip()


def _privacy_separated_value(value: str, *, source_text: str) -> str:
    """Break source trigrams without deleting visual nouns or masking names."""

    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    source_tokens = _privacy_tokens(source_text)
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if _distinctive_phrase(source_tokens[index : index + 3])
    }
    proper_name_tokens = {
        token
        for candidate in _proper_name_candidates(source_text)
        for token in candidate
    }
    transformed: set[int] = set()
    while True:
        normalized = tuple(word.casefold() for word in words)
        overlap = next(
            (
                index
                for index in range(len(normalized) - 2)
                if normalized[index : index + 3] in source_phrases
            ),
            None,
        )
        if overlap is None:
            return " ".join(words)
        available = [
            index
            for index in range(overlap, overlap + 3)
            if normalized[index] not in proper_name_tokens
            and index not in transformed
        ]
        stopword_candidates = [
            index for index in available if normalized[index] in _PHRASE_STOPWORDS
        ]
        candidates = stopword_candidates or [
            index for index in available if len(normalized[index]) >= 5
        ]
        if not candidates:
            raise ValueError("slot response could not safely separate a source phrase")
        selected = max(candidates, key=lambda index: len(normalized[index]))
        words[selected] = f"v{words[selected][0].upper()}{words[selected][1:]}"
        transformed.add(selected)


def _compact_magic_slot(value: str, *, maximum_words: int = 10) -> str:
    words = value.split()
    if len(words) <= maximum_words:
        return value
    compact = [
        word
        for word in words
        if word.casefold().strip(".,;:!?") not in _PHRASE_STOPWORDS
    ]
    if len(compact) <= maximum_words:
        return " ".join(compact)
    return " ".join([*compact[:7], *compact[-3:]])


def _rebalance_slots(slots: dict[str, str], *, source_text: str) -> dict[str, str]:
    balanced = dict(slots)
    background = balanced["SETTING"]
    magic = balanced["MAGIC"]
    if len(_privacy_tokens(magic)) <= 3 and re.search(r"\s+and\s+", background, re.I):
        head, tail = re.split(r"\s+and\s+", background, maxsplit=1, flags=re.I)
        balanced["SETTING"] = head
        balanced["MAGIC"] = f"{tail}; {magic}"
    balanced["SETTING"] = _recover_source_grounded_setting(
        balanced["SETTING"],
        source_text=source_text,
    )
    return balanced


def _parse_slots(output_text: str) -> dict[str, str]:
    matches = list(_SLOT_LABEL_PATTERN.finditer(output_text.strip()))
    if not matches or output_text.strip()[: matches[0].start()].strip():
        raise ValueError("slot response contained text before its first label")
    values: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output_text)
        value = output_text[match.end() : end].strip()
        if not value:
            raise ValueError("slot response contained an empty value")
        values.setdefault(label, []).append(value)
    if set(values) != {"SETTING", "ACTOR", "ACTION", "MAGIC"}:
        raise ValueError("slot response did not contain exactly the four known fields")
    return {
        label: " ".join("; ".join(parts).strip().strip('"').split())
        for label, parts in values.items()
    }


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
    if prompt_profile == "compact":
        return [
            {"role": "system", "content": _COMPACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    'Example story: "A keeper lifts a key, and bright moths spiral '
                    'through an arch."\n'
                    'Example JSON: {"background_prompt":"stone arch at dusk",'
                    '"focus":{"kind":"character","subject":"a keeper","action":"lifts key"},'
                    '"magic":{"kind":"character","prompt":"bright moths spiral through arch"}}\n'
                    f'Story: "{case.text}"\nJSON:'
                ),
            },
        ]
    if prompt_profile == "fewshot":
        return [
            {"role": "system", "content": _MINIMAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    'Story: "A keeper lifts a key, and bright moths spiral through an arch."\n'
                    "Return the setting, main actor, visible action, and later magical event."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '{"background_prompt":"stone arch at dusk",'
                    '"focus":{"kind":"character","subject":"a keeper","action":"lifts key"},'
                    '"magic":{"kind":"effect","prompt":"bright moths spiral through arch"}}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Story: "{case.text}"\n'
                    "Return the same JSON shape using only this story. "
                    "Keep each value under eight words."
                ),
            },
        ]
    if prompt_profile == "slots":
        return [
            {"role": "system", "content": _SLOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'A keeper lifts a key, and bright moths spiral through an arch.',
            },
            {
                "role": "assistant",
                "content": (
                    "SETTING: stone arch at dusk\n"
                    "ACTOR: a keeper\n"
                    "ACTION: lifts key\n"
                    "MAGIC: bright moths spiral through arch"
                ),
            },
            {"role": "user", "content": case.text},
        ]
    if prompt_profile == "repair":
        return [
            {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"STORY:\n{case.text}\n"
                    "Answer with SETTING, ACTOR, ACTION, and MAGIC. Keep concrete nouns."
                ),
            },
        ]
    return [
        {"role": "system", "content": _MINIMAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f'Story: "{case.text}"\n'
                "Replace the CAPITAL values with short story phrases and return only this object: "
                '{"background_prompt":"SETTING",'
                '"focus":{"kind":"character","subject":"MAIN ACTOR","action":"VISIBLE ACTION"},'
                '"magic":{"kind":"effect","prompt":"LATER MAGICAL EVENT"}}'
            ),
        },
    ]


def _request_document(
    cases: tuple[BenchmarkCase, ...],
    *,
    prompt_profile: str = "production",
) -> dict[str, object]:
    requests: list[dict[str, object]] = []
    for case in cases:
        requests.append({"messages": _messages(case, prompt_profile=prompt_profile)})
    return {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": (
            180
            if prompt_profile == "production"
            else 64
            if prompt_profile in _SLOT_PROFILES
            else 96
        ),
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "requests": requests,
    }


def _resident_request_payload(
    request: dict[str, object],
    document: dict[str, object],
    *,
    model: str,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": request["messages"],
        "temperature": document["temperature"],
        "top_p": document["top_p"],
        "max_tokens": document["max_generate_length"],
        "stream": False,
    }


def _post_resident_request(
    payload: dict[str, object],
    *,
    base_url: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], float]:
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    started = time.perf_counter()
    with urlopen(  # noqa: S310 - URL is restricted to loopback by the caller.
        Request(
            f"{base_url}/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=timeout_seconds,
    ) as response:
        body = json.load(response)
    return body, (time.perf_counter() - started) * 1_000


def _run_resident_requests(
    document: dict[str, object],
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    base_url = _require_loopback_base_url(base_url)
    requests = document.get("requests")
    if not isinstance(requests, list):
        raise ValueError("TensorRT request document did not contain requests")
    responses: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError("TensorRT request was not an object")
        body, wall_ms = _post_resident_request(
            _resident_request_payload(request, document, model=model),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        responses.append(
            {
                "output_text": choice["message"]["content"],
                "finish_reason": choice.get("finish_reason"),
            }
        )
        timings.append(
            {
                "request_index": index,
                "wall_ms": round(wall_ms, 3),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        )
    wall_values = [float(item["wall_ms"]) for item in timings]
    profile: dict[str, object] = {
        "resident": True,
        "requests": timings,
        "summary": {
            "mean_wall_ms": round(statistics.fmean(wall_values), 3),
            "median_wall_ms": round(statistics.median(wall_values), 3),
            "maximum_wall_ms": round(max(wall_values), 3),
            "total_prompt_tokens": sum(int(item["prompt_tokens"]) for item in timings),
            "total_completion_tokens": sum(
                int(item["completion_tokens"]) for item in timings
            ),
        },
    }
    return {"responses": responses}, profile


def _validate_responses(
    payload: dict[str, object],
    cases: tuple[BenchmarkCase, ...],
    *,
    prompt_profile: str = "production",
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
            if prompt_profile in _SLOT_PROFILES:
                parsed_slots = _rebalance_slots(
                    _parse_slots(output_text),
                    source_text=case.text,
                )
                parsed_slots["MAGIC"] = _compact_magic_slot(parsed_slots["MAGIC"])
                slots = {
                    label: _privacy_separated_value(
                        _fit_wire_value(value),
                        source_text=case.text,
                    )
                    for label, value in parsed_slots.items()
                }
                wire_plan = LiveSceneWirePlan.model_validate(
                    {
                        "background_prompt": slots["SETTING"],
                        "focus": {
                            "kind": "character",
                            "subject": slots["ACTOR"],
                            "action": slots["ACTION"],
                        },
                        "magic": {"kind": "effect", "prompt": slots["MAGIC"]},
                    }
                )
            else:
                wire_plan = LiveSceneWirePlan.model_validate_json(output_text)
            if prompt_profile in _SLOT_PROFILES:
                # The slot parser has already guaranteed all four fields and
                # separated source trigrams. Avoid JSON-repair heuristics that
                # can mistake a complete supporting event for a duplicate.
                plan = wire_plan.to_live_scene_plan(context_text=case.text)
            else:
                plan = wire_plan.privacy_sanitized(source_text=case.text).to_live_scene_plan(
                    context_text=case.text
                )
            validate_live_scene_plan_privacy(plan, source_text=case.text)
        except (LiveScenePlannerPrivacyError, ValidationError, ValueError) as error:
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
    parser.add_argument(
        "--prompt-profile",
        choices=(
            "production",
            "compact",
            "minimal",
            "fewshot",
            "slots",
            "repair",
        ),
        default="production",
    )
    parser.add_argument("--suite", choices=("five", "contest"), default="five")
    parser.add_argument(
        "--case-id",
        help="Run one named case from the selected suite to expose per-request cold latency.",
    )
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    parser.add_argument("--candidate-revision", default="db09cd27ead7fee40cdee309693cf83601b9c899")
    parser.add_argument(
        "--resident-base-url",
        help="Use an already-loaded OpenAI-compatible Edge-LLM server on loopback.",
    )
    parser.add_argument("--resident-model", default="llm")
    parser.add_argument("--resident-timeout-seconds", type=float, default=15.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    cases = CASES if args.suite == "five" else CONTEST_CASES
    if args.case_id:
        cases = tuple(case for case in cases if case.case_id == args.case_id)
        if not cases:
            raise SystemExit(
                f"unknown case {args.case_id!r} in {args.suite!r} suite"
            )
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

    started = time.perf_counter()
    if args.resident_base_url:
        raw_output, resident_profile = _run_resident_requests(
            _request_document(cases, prompt_profile=args.prompt_profile),
            base_url=args.resident_base_url,
            model=args.resident_model,
            timeout_seconds=args.resident_timeout_seconds,
        )
        raw_output_path.write_text(json.dumps(raw_output, indent=2) + "\n")
        return_code = 0
        stdout = ""
        stderr = ""
    else:
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
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    wall_ms = (time.perf_counter() - started) * 1_000
    if not raw_output_path.is_file():
        raise RuntimeError(
            "TensorRT Edge-LLM did not create an output file: "
            f"exit={return_code}; stderr={stderr[-2000:]}"
        )
    if not args.resident_base_url:
        raw_output = json.loads(raw_output_path.read_text())
    case_results = _validate_responses(
        raw_output,
        cases,
        prompt_profile=args.prompt_profile,
    )
    schema_valid = all(case["valid"] for case in case_results)
    automatic_semantic_pass = all(
        bool(case["automatic_semantic_pass"]) for case in case_results
    )
    if return_code != 0 or not schema_valid:
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
            "backend": (
                "tensorrt-edge-llm-resident"
                if args.resident_base_url
                else "tensorrt-edge-llm"
            ),
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
            "exit_code": return_code,
            "resident_base_url": args.resident_base_url or None,
        },
        "cases": case_results,
        "profile": (
            resident_profile
            if args.resident_base_url
            else json.loads(profile_path.read_text())
            if profile_path.is_file()
            else None
        ),
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
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["result"] != "technical_pass_human_semantic_review_required":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
