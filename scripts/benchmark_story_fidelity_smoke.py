#!/usr/bin/env python3
"""One-request resident-model smoke for a frozen synthetic story and passive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field

from bookforge.fidelity_evaluation import FIDELITY_EVALUATOR_REVISION
from bookforge.live_scene_planner import (
    LiveScenePlannerPrivacyError,
    validate_live_scene_plan_privacy,
)
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
        "schema_version": 1,
        "benchmark": "lantern-bridge-resident-smoke-v1",
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
            result = Result.model_validate(event)
            if index not in started or index in results:
                raise ValueError("orphan or duplicate smoke result")
            results[index] = result
    return started, list(results.values())


def run_case(
    client: httpx.Client, source: str, index: int, *, model: str, style: str, seed: int
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
    if not complete:
        return result, {}
    try:
        parse_tensor_graph_slots(raw)
        result.raw_schema_valid = True
    except ValueError:
        pass
    contracts = {"index": index, "seed": seed, "visual_style": style}
    start = time.perf_counter()
    try:
        accepted = tensor_slot_wire_plan(raw, source_text=source).to_live_scene_plan(
            context_text=source
        )
        validate_live_scene_plan_privacy(accepted, source_text=source)
        prompt = accepted.to_page(
            source_text=source, visual_style=style, seed=seed
        ).scene_spec.master_prompt
        result.accepted_valid = True
        result.accepted_privacy_pass = True
        result.accepted_sha256 = benchmark.digest(prompt)
        contracts["accepted_master_prompt"] = prompt
    except (LiveScenePlannerPrivacyError, SceneFactsPrivacyError):
        result.accepted_privacy_pass = False
    except ValueError:
        pass
    result.accepted_construction_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    try:
        wire = tensor_accepted_graph_wire_plan(raw, source_text=source)
        candidate = wire.to_live_scene_plan(context_text=source)
        validate_live_scene_plan_privacy(candidate, source_text=source)
        prompt = candidate.to_page(
            source_text=source, visual_style=style, seed=seed
        ).scene_spec.master_prompt
        if wire.scene_facts is not None:
            wire.scene_facts.validate_source_grounding(source_text=source)
            result.graph_valid = True
            result.graph_sha256 = benchmark.digest(wire.scene_facts.model_dump(mode="json"))
            result.compiler_proof = prompt == wire.scene_facts.to_renderer_prompt(
                source_text=source, visual_style=style
            )
            if not result.compiler_proof:
                raise ValueError("compiled graph prompt mismatch")
            result.fallback = False
        result.candidate_valid = True
        result.candidate_privacy_pass = True
        result.candidate_sha256 = benchmark.digest(prompt)
        contracts["candidate_master_prompt"] = prompt
    except (LiveScenePlannerPrivacyError, SceneFactsPrivacyError):
        result.candidate_privacy_pass = False
    except ValueError:
        pass
    result.candidate_construction_ms = (time.perf_counter() - start) * 1000
    return result, contracts


def aggregate(header: Mapping, started: set[int], results: Sequence[Result]) -> dict:
    complete = len(results) == 7 and all(row.status == "ok" for row in results)
    graphs = sum(
        row.accepted_valid and row.candidate_valid and row.graph_valid and row.compiler_proof
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
        "decision": "eligible_for_development_gate"
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
        "measurement_scope": "HTTP inference and separate local construction; no evaluator timing",
    }


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
        if args.aggregate_only and not args.evidence.is_file():
            raise ValueError("aggregation requires an existing smoke journal")
        started, results = load_evidence(args.evidence, header)
        if not args.aggregate_only:
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
