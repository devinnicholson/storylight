#!/usr/bin/env python3
"""One-request resident-model smoke for a frozen synthetic story and passive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field, ValidationError

from bookforge.fidelity_evaluation import FIDELITY_EVALUATOR_REVISION
from bookforge.live_scene_facts import LiveSceneFactsRefusal, adapt_live_scene_facts
from bookforge.live_scene_planner import (
    LiveScenePlannerPrivacyError,
    validate_live_scene_plan_privacy,
)
from bookforge.privacy_policy import COUNT_WORDS, VISIBLE_VERBS
from bookforge.scene_facts import SceneFactsPrivacyError
from bookforge.tensorrt_slot_client import (
    parse_tensor_graph_slots,
    tensor_accepted_graph_wire_plan,
    tensor_slot_wire_plan,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_live_scene_facts as benchmark  # noqa: E402

MANIFEST_SHA256 = "a1fb181fa5e217a9b259b622c7340b64fc3788311ea50bdf3e973eb5265469ed"
PASSIVE_CONTROL = "In a cave, one blue lantern is carried by two orange foxes. A ribbon appears."

Stage = Literal[
    "raw_parse",
    "accepted_wire",
    "accepted_plan",
    "accepted_privacy",
    "accepted_render",
    "adapter",
    "candidate_integration",
    "candidate_plan",
    "candidate_privacy",
    "candidate_render",
    "candidate_grounding",
    "candidate_proof",
]
_ERROR_FIELDS = frozenset(
    [
        "background",
        "focus",
        "magic",
        "kind",
        "prompt",
        "label",
        "count",
        "color",
        "attributes",
        "actions",
        "states",
        "setting",
        "subjects",
        "objects",
        "relationships",
        "motions",
        "events",
        "temporal_order",
        "negatives",
        "transformation",
        "source",
        "target",
        "relation",
        "ref",
        "scene_facts",
        "layers",
        "scene_spec",
        "master_prompt",
        "subject",
        "action",
        "material",
        "result_label",
        "result_color",
        "result_count",
        "value",
        "placement",
        "position",
        "scale",
    ]
)
_ERROR_TYPES = frozenset(
    [
        "missing",
        "extra_forbidden",
        "literal_error",
        "string_type",
        "string_too_long",
        "string_too_short",
        "string_pattern_mismatch",
        "int_type",
        "int_parsing",
        "greater_than_equal",
        "less_than_equal",
        "too_long",
        "too_short",
        "value_error",
        "model_type",
        "enum",
        "finite_number",
        "bool_type",
    ]
)


class StageDiagnostic(benchmark.StrictModel):
    stage: Stage
    outcome: Literal["pass", "value_error", "validation_error", "privacy", "refused"]
    errors: list[dict[str, str | list[str | int]]] = Field(default_factory=list, max_length=8)


def failure(stage: Stage, error: Exception) -> StageDiagnostic:
    if isinstance(error, (LiveScenePlannerPrivacyError, SceneFactsPrivacyError)):
        return StageDiagnostic(stage=stage, outcome="privacy")
    if not isinstance(error, ValidationError):
        return StageDiagnostic(stage=stage, outcome="value_error")
    return StageDiagnostic(
        stage=stage,
        outcome="validation_error",
        errors=[
            {
                "loc": [
                    part
                    if (type(part) is int and 0 <= part <= 12)
                    or (isinstance(part, str) and part in _ERROR_FIELDS)
                    else "other"
                    for part in item["loc"][:8]
                ],
                "type": item["type"] if item["type"] in _ERROR_TYPES else "other",
            }
            for item in error.errors(include_url=False, include_context=False, include_input=False)[
                :8
            ]
        ],
    )


def slot_shapes(slots: Mapping[str, str]) -> dict:
    shapes = {}
    for label, value in slots.items():
        tokens = re.findall(r"[a-z]+", value.casefold())
        first = tokens[0] if tokens else ""
        kind = "other"
        if first in {"is", "are", "was", "were", "be"}:
            kind = "copula"
        elif first in {"a", "an", "the"}:
            kind = "determiner"
        elif first in COUNT_WORDS:
            kind = "count"
        elif first.endswith("ing"):
            kind = "gerund"
        elif first in VISIBLE_VERBS:
            kind = "visible_verb"
        shapes[label] = {
            "token_count": len(tokens),
            "first_token_kind": kind,
            "contains_not": "not" in tokens,
            "contains_comma": "," in value,
            "contains_coordinate": bool({"and", "or"}.intersection(tokens)),
        }
    return shapes


class Result(benchmark.StrictModel):
    kind: Literal["result"] = "result"
    index: Annotated[int, Field(ge=0, le=6)]
    status: Literal["ok", "request_failed"]
    generation_complete: bool = False
    raw_schema_valid: bool = False
    accepted_valid: bool = False
    accepted_privacy_pass: bool | None = None
    candidate_valid: bool = False
    candidate_privacy_pass: bool | None = None
    graph_valid: bool = False
    compiler_proof: bool = False
    fallback: bool = True
    raw_sha256: benchmark.Digest | None = None
    accepted_sha256: benchmark.Digest | None = None
    candidate_sha256: benchmark.Digest | None = None
    graph_sha256: benchmark.Digest | None = None
    learned_inference_ms: benchmark.Nonnegative | None = None
    accepted_construction_ms: benchmark.Nonnegative | None = None
    candidate_construction_ms: benchmark.Nonnegative | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    process_rss_peak_bytes: Annotated[int, Field(ge=0)] | None = None
    system_used_peak_bytes: Annotated[int, Field(ge=0)] | None = None
    memory_samples: Annotated[int, Field(ge=0)] = 0
    thermal_min_millicelsius: benchmark.Temperature | None = None
    thermal_max_millicelsius: benchmark.Temperature | None = None
    thermal_samples: Annotated[int, Field(ge=0)] = 0
    diagnostics: list[StageDiagnostic] = Field(default_factory=list)
    adapter_refusal: LiveSceneFactsRefusal | None = None
    slot_shapes: dict = Field(default_factory=dict)


def load_manifest(path: Path) -> tuple[dict, tuple[tuple[str, int], ...]]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("story manifest differs from frozen fixture")
    manifest = json.loads(data)
    cases = tuple((page["text"], page["seed"]) for page in manifest["pages"])
    return manifest, (*cases, (PASSIVE_CONTROL, 90407))


def context(args: argparse.Namespace, provenance: benchmark.Provenance) -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        Path(benchmark.__file__),
        *sorted((root / "src/bookforge").glob("*.py")),
    ]
    return {
        "kind": "header",
        "schema_version": 2,
        "benchmark": "lantern-bridge-resident-smoke-v2",
        "execution_mode": "live",
        "manifest_sha256": MANIFEST_SHA256,
        "control_sha256": benchmark.digest(PASSIVE_CONTROL),
        "implementation_sha256": benchmark.digest(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
        ),
        "evaluator_revision": FIDELITY_EVALUATOR_REVISION,
        "semantic_evaluation_performed": False,
        "provenance": provenance.model_dump(),
        "model_sha256": benchmark.digest(args.model),
        "endpoint_sha256": benchmark.digest(args.endpoint),
        "request_sha256": benchmark.digest(
            benchmark.request_payload("", "slots", args.model, args.max_output_tokens)
        ),
        "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout,
        "memory_pid": args.memory_pid,
        "expected_cases": 7,
        "retries": 0,
    }


def load_evidence(path: Path, expected: Mapping) -> tuple[set[int], list[Result]]:
    if not path.exists():
        benchmark.append_event(path, expected)
        return set(), []
    lines = path.read_text().splitlines()
    if not lines or json.loads(lines[0]) != expected:
        raise ValueError("incompatible smoke journal")
    started: set[int] = set()
    results: dict[int, Result] = {}
    for line in lines[1:]:
        event = json.loads(line)
        index = event.get("index")
        if type(index) is not int or not 0 <= index < 7:
            raise ValueError("invalid smoke index")
        if event.get("kind") == "start":
            if set(event) != {"kind", "index"} or index in started:
                raise ValueError("invalid smoke start")
            started.add(index)
        else:
            result = Result.model_validate_json(line)
            if index not in started or index in results:
                raise ValueError("orphan or duplicate smoke result")
            results[index] = result
    return started, list(results.values())


def run_case(
    client: httpx.Client,
    source: str,
    index: int,
    *,
    model: str,
    style: str,
    seed: int,
    capture: Callable[[dict], None] | None = None,
) -> tuple[Result, dict]:
    start = time.perf_counter()
    try:
        response = client.post(
            "/v1/chat/completions", json=benchmark.request_payload(source, "slots", model, 64)
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        raw = choice["message"]["content"]
        tokens = payload.get("usage", {}).get("completion_tokens")
        if not isinstance(raw, str) or (
            tokens is not None and (type(tokens) is not int or tokens < 0)
        ):
            raise ValueError("invalid resident response")
        if capture is not None and len(raw) > 16384:
            raise ValueError("private response exceeds bounded archive")
        complete = choice.get("finish_reason") in {None, "stop"}
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
        return Result(index=index, status="request_failed"), {}
    result = Result(
        index=index,
        status="ok",
        raw_sha256=benchmark.digest(raw),
        generation_complete=complete,
        output_tokens=tokens,
        learned_inference_ms=(time.perf_counter() - start) * 1000,
    )
    if capture is not None:
        capture(
            {
                "kind": "response",
                "index": index,
                "source_sha256": benchmark.digest(source),
                "raw": raw,
                "raw_sha256": result.raw_sha256,
                "generation_complete": complete,
            }
        )
    return construct_case(result, raw, source, style=style, seed=seed)


def construct_case(result: Result, raw: str, source: str, *, style: str, seed: int):
    if not result.generation_complete:
        return result, {}
    slots = None
    try:
        slots = parse_tensor_graph_slots(raw)
        result.raw_schema_valid = True
        result.diagnostics.append(StageDiagnostic(stage="raw_parse", outcome="pass"))
        result.slot_shapes = slot_shapes(slots)
    except ValueError as error:
        result.diagnostics.append(failure("raw_parse", error))
    contracts = {"index": result.index, "seed": seed, "visual_style": style}
    start = time.perf_counter()
    stage: Stage = "accepted_wire"
    try:
        accepted_wire = tensor_slot_wire_plan(raw, source_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "accepted_plan"
        accepted = accepted_wire.to_live_scene_plan(context_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "accepted_privacy"
        validate_live_scene_plan_privacy(accepted, source_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "accepted_render"
        prompt = accepted.to_page(
            source_text=source, visual_style=style, seed=seed
        ).scene_spec.master_prompt
        result.accepted_valid = True
        result.accepted_privacy_pass = True
        result.accepted_sha256 = benchmark.digest(prompt)
        contracts["accepted_master_prompt"] = prompt
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
    except (LiveScenePlannerPrivacyError, SceneFactsPrivacyError, ValueError) as error:
        result.diagnostics.append(failure(stage, error))
        if isinstance(error, (LiveScenePlannerPrivacyError, SceneFactsPrivacyError)):
            result.accepted_privacy_pass = False
    result.accepted_construction_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    stage = "candidate_integration"
    try:
        wire = tensor_accepted_graph_wire_plan(raw, source_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "candidate_plan"
        candidate = wire.to_live_scene_plan(context_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "candidate_privacy"
        validate_live_scene_plan_privacy(candidate, source_text=source)
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        stage = "candidate_render"
        prompt = candidate.to_page(
            source_text=source, visual_style=style, seed=seed
        ).scene_spec.master_prompt
        result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
        if wire.scene_facts is not None:
            stage = "candidate_grounding"
            wire.scene_facts.validate_source_grounding(source_text=source)
            result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
            result.graph_valid = True
            result.graph_sha256 = benchmark.digest(wire.scene_facts.model_dump(mode="json"))
            stage = "candidate_proof"
            result.compiler_proof = prompt == wire.scene_facts.to_renderer_prompt(
                source_text=source, visual_style=style
            )
            if not result.compiler_proof:
                raise ValueError("compiled graph prompt mismatch")
            result.diagnostics.append(StageDiagnostic(stage=stage, outcome="pass"))
            result.fallback = False
        result.candidate_valid = True
        result.candidate_privacy_pass = True
        result.candidate_sha256 = benchmark.digest(prompt)
        contracts["candidate_master_prompt"] = prompt
    except (LiveScenePlannerPrivacyError, SceneFactsPrivacyError, ValueError) as error:
        result.diagnostics.append(failure(stage, error))
        if isinstance(error, (LiveScenePlannerPrivacyError, SceneFactsPrivacyError)):
            result.candidate_privacy_pass = False
    result.candidate_construction_ms = (time.perf_counter() - start) * 1000
    if slots is not None:
        try:
            adapted = adapt_live_scene_facts(slots, source_text=source)
            result.adapter_refusal = adapted.refusal
            result.diagnostics.append(
                StageDiagnostic(
                    stage="adapter", outcome="pass" if adapted.facts is not None else "refused"
                )
            )
        except (LiveScenePlannerPrivacyError, SceneFactsPrivacyError, ValueError) as error:
            result.diagnostics.append(failure("adapter", error))
    return result, contracts


def aggregate(header: Mapping, started: set[int], results: Sequence[Result]) -> dict:
    complete = len(results) == 7 and all(row.status == "ok" for row in results)
    graphs = sum(
        row.candidate_valid and row.graph_valid and row.compiler_proof
        for row in results
    )
    privacy_failures = sum(
        row.accepted_privacy_pass is False or row.candidate_privacy_pass is False for row in results
    )
    return {
        "context": dict(header),
        "context_sha256": benchmark.digest(header),
        "complete": complete,
        "expected_cases": 7,
        "started_cases": len(started),
        "finished_cases": len(results),
        "interrupted_cases": len(started) - len(results),
        "request_failures": sum(row.status == "request_failed" for row in results),
        "accepted_valid_cases": sum(row.accepted_valid for row in results),
        "candidate_valid_cases": sum(row.candidate_valid for row in results),
        "compiler_proved_graph_cases": graphs,
        "privacy_failure_cases": privacy_failures,
        "decision": "offline_replay_only"
        if header.get("execution_mode") == "replay"
        else "eligible_for_development_gate"
        if complete and graphs > 0 and privacy_failures == 0
        else "stop_before_development_gate",
        "learned_inference_ms": benchmark.distribution(
            [row.learned_inference_ms for row in results if row.learned_inference_ms is not None]
        ),
        "output_tokens": benchmark.distribution(
            [row.output_tokens for row in results if row.output_tokens is not None]
        ),
        "results": [row.model_dump() for row in sorted(results, key=lambda row: row.index)],
        "semantic_accuracy_assessed": False,
        "visual_fidelity_assessed": False,
        "temporal_playback_assessed": False,
        "appliance_promotion_permitted": False,
        "hidden_evaluated": False,
        "paid_services_used": False,
        "gpu_allocator_peak_bytes": None,
        "measurement_scope": "offline reconstruction; no inference or performance measurements"
        if header.get("execution_mode") == "replay"
        else "HTTP inference and separate local construction; no evaluator timing",
    }


def private_path(path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = path.parent.stat()
    if (
        path.resolve().is_relative_to(root)
        or path.parent.resolve() != path.parent.absolute()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
        or path.is_symlink()
    ):
        raise ValueError("private responses require an external owner-only directory")


def create_private_archive(path: Path, header: Mapping) -> None:
    private_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps({"kind": "private_header", "context": dict(header)}) + "\n")


def load_private_replay(path: Path, evidence: Path, header: dict, cases) -> tuple[dict, list]:
    private_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 131072
        ):
            raise ValueError("invalid private response archive")
        data = stream.read()
    entries = [json.loads(line) for line in data.splitlines()]
    if not entries or set(entries[0]) != {"kind", "context"}:
        raise ValueError("invalid private response header")
    original = entries[0]["context"]
    if entries[0]["kind"] != "private_header" or original.get("execution_mode") != "live":
        raise ValueError("replay requires original live evidence")
    for key in (
        "schema_version",
        "benchmark",
        "manifest_sha256",
        "control_sha256",
        "provenance",
        "model_sha256",
        "endpoint_sha256",
        "request_sha256",
        "max_output_tokens",
        "expected_cases",
        "retries",
    ):
        if original.get(key) != header[key]:
            raise ValueError("private response context differs")
    if not evidence.is_file():
        raise ValueError("replay requires original evidence")
    started, results = load_evidence(evidence, original)
    by_index = {result.index: result for result in results if result.status == "ok"}
    seen = set()
    for entry in entries[1:]:
        if set(entry) != {
            "kind",
            "index",
            "source_sha256",
            "raw",
            "raw_sha256",
            "generation_complete",
        }:
            raise ValueError("invalid private response fields")
        index = entry["index"]
        if type(index) is not int or index not in by_index or index in seen:
            raise ValueError("invalid private response index")
        raw = entry["raw"]
        result = by_index[index]
        if (
            entry["kind"] != "response"
            or not isinstance(raw, str)
            or len(raw) > 16384
            or entry["source_sha256"] != benchmark.digest(cases[index][0])
            or entry["raw_sha256"] != benchmark.digest(raw)
            or entry["raw_sha256"] != result.raw_sha256
            or type(entry["generation_complete"]) is not bool
            or entry["generation_complete"] != result.generation_complete
        ):
            raise ValueError("private response does not match original evidence")
        seen.add(index)
    if seen != set(by_index) or not started:
        raise ValueError("private responses do not cover original successful requests")
    return {
        **header,
        "execution_mode": "replay",
        "private_archive_sha256": hashlib.sha256(data).hexdigest(),
        "original_context_sha256": benchmark.digest(original),
        "original_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "original_started_cases": len(started),
        "original_finished_cases": len(results),
        "original_request_failures": sum(row.status == "request_failed" for row in results),
        "original_interrupted_cases": len(started) - len(results),
    }, entries[1:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-output-tokens", type=int, choices=(64,), default=64)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--memory-pid", type=int)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--private-responses", type=Path)
    parser.add_argument("--replay-private", type=Path)
    parser.add_argument("--replay-evidence", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.endpoint = benchmark.validate_endpoint(args.endpoint)
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
            raise ValueError("invalid timeout")
        if args.memory_pid is not None and args.memory_pid <= 0:
            raise ValueError("invalid process ID")
        manifest, cases = load_manifest(args.manifest)
        provenance = benchmark.Provenance.model_validate_json(args.provenance.read_text())
        header = context(args, provenance)
        if bool(args.replay_private) != bool(args.replay_evidence):
            raise ValueError("replay requires archive and original evidence")
        if args.replay_private and (args.private_responses or args.contracts):
            raise ValueError("replay cannot capture private outputs")
        replay = None
        if args.replay_private:
            header, replay = load_private_replay(
                args.replay_private, args.replay_evidence, header, cases
            )
        if args.aggregate_only and not args.evidence.is_file():
            raise ValueError("aggregation requires an existing smoke journal")
        started, results = load_evidence(args.evidence, header)
        if not args.aggregate_only and replay is not None:
            for entry in replay:
                index = entry["index"]
                if index in started:
                    continue
                benchmark.append_event(args.evidence, {"kind": "start", "index": index})
                started.add(index)
                source, seed = cases[index]
                result, _ = construct_case(
                    Result(
                        index=index,
                        status="ok",
                        raw_sha256=entry["raw_sha256"],
                        generation_complete=entry["generation_complete"],
                    ),
                    entry["raw"],
                    source,
                    style=manifest["visual_style"],
                    seed=seed,
                )
                result.accepted_construction_ms = None
                result.candidate_construction_ms = None
                benchmark.append_event(args.evidence, result.model_dump())
                results.append(result)
        elif not args.aggregate_only:
            if args.private_responses is not None:
                if started:
                    raise ValueError("private capture requires a fresh live journal")
                create_private_archive(args.private_responses, header)
            if args.contracts is not None:
                fd = os.open(args.contracts, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            with httpx.Client(
                base_url=args.endpoint,
                timeout=args.timeout,
                trust_env=False,
                follow_redirects=False,
                transport=httpx.HTTPTransport(retries=0),
            ) as client:
                for index, (source, seed) in enumerate(cases):
                    if index in started:
                        continue
                    benchmark.append_event(args.evidence, {"kind": "start", "index": index})
                    started.add(index)
                    with benchmark.MemorySampler(args.memory_pid) as memory:
                        result, contracts = run_case(
                            client,
                            source,
                            index,
                            model=args.model,
                            style=manifest["visual_style"],
                            seed=seed,
                            capture=(
                                lambda entry: benchmark.append_event(args.private_responses, entry)
                            )
                            if args.private_responses is not None
                            else None,
                        )
                    result = result.model_copy(
                        update={
                            "process_rss_peak_bytes": memory.process_peak,
                            "system_used_peak_bytes": memory.system_peak,
                            "memory_samples": memory.samples,
                            "thermal_min_millicelsius": memory.thermal_min,
                            "thermal_max_millicelsius": memory.thermal_max,
                            "thermal_samples": memory.thermal_samples,
                        }
                    )
                    if args.contracts is not None and contracts:
                        benchmark.append_event(args.contracts, contracts)
                    benchmark.append_event(args.evidence, result.model_dump())
                    results.append(result)
                    args.output.write_text(
                        json.dumps(aggregate(header, started, results), indent=2, sort_keys=True)
                        + "\n"
                    )
        args.output.write_text(
            json.dumps(aggregate(header, started, results), indent=2, sort_keys=True) + "\n"
        )
        return 0
    except Exception:
        print("story smoke failed; check local configuration and sanitized evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
