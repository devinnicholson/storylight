#!/usr/bin/env python3
"""Bounded, restartable public-development TensorRT and live graph comparison.

Evidence contains hashes and scores, never passages, model text, or renderer contracts.
Interrupted requests are never retried. Use a new evidence file for a new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from bookforge.fidelity_dataset import DATASET_ID, generate_split
from bookforge.fidelity_evaluation import FIDELITY_EVALUATOR_REVISION, evaluate_surface
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from bookforge.live_scene_planner import (
    LiveScenePlannerPrivacyError,
    validate_live_scene_plan_privacy,
)
from bookforge.tensorrt_slot_client import _slot_messages, tensor_slot_wire_plan

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Temperature = Annotated[int, Field(ge=-100_000, le=200_000)]
SURFACES = ("accepted_raw", "hybrid_raw", "accepted_renderer", "graph_candidate", "final_renderer")
Mode = Literal["hybrid", "accepted_first"]


def surfaces_for(mode: str) -> tuple[str, ...]:
    return tuple(name for name in SURFACES if mode != "accepted_first" or name != "hybrid_raw")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Provenance(StrictModel):
    """Hash local hardware/thermal snapshots separately; do not embed shell output."""

    engine_sha256: Digest
    environment_sha256: Digest
    snapshot_sha256: Digest
    code_revision: Revision
    deployment_sha256: Digest
    peak_memory_bytes: Annotated[int, Field(ge=0)] | None = None


class Score(StrictModel):
    schema_valid: bool
    exact_pass: bool
    privacy_pass: bool
    required_atoms: Annotated[int, Field(ge=0)]
    passed_atoms: Annotated[int, Field(ge=0)]
    forbidden_count: Annotated[int, Field(ge=0)]
    unsupported_count: Annotated[int, Field(ge=0)]
    output_sha256: Digest
    latency_ms: Nonnegative
    output_tokens: Annotated[int, Field(ge=0)] | None
    generation_complete: bool = True
    required_atom_passes: list[bool] = Field(default_factory=list)


class CaseEvidence(StrictModel):
    kind: Literal["result"] = "result"
    index: Annotated[int, Field(ge=0, le=511)]
    status: Literal["ok", "request_failed", "processing_failed"]
    refusal: Literal["none", "adapter_refused", "parse_refused", "renderer_refused"]
    refusal_sha256: Digest | None = None
    adapter_refusal_code: (
        Literal[
            "invalid_input",
            "invalid_hybrid",
            "unsupported_syntax",
            "ambiguous_binding",
            "ungrounded",
            "privacy",
            "invalid_graph",
        ]
        | None
    ) = None
    fallback: bool
    surfaces: dict[str, Score]
    process_rss_peak_bytes: Annotated[int, Field(ge=0)] | None = None
    system_used_peak_bytes: Annotated[int, Field(ge=0)] | None = None
    memory_samples: Annotated[int, Field(ge=0)] = 0
    thermal_min_millicelsius: Temperature | None = None
    thermal_max_millicelsius: Temperature | None = None
    thermal_samples: Annotated[int, Field(ge=0)] = 0
    learned_inference_ms: Nonnegative | None = None
    graph_construction_ms: Nonnegative | None = None


class MemorySampler:
    """Sample Linux memory and thermal zones; no GPU allocator peak is measured."""

    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.process_peak: int | None = None
        self.system_peak: int | None = None
        self.samples = 0
        self.thermal_min: int | None = None
        self.thermal_max: int | None = None
        self.thermal_samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            values = {
                line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
            }
            used = values["MemTotal"] - values["MemAvailable"]
            self.system_peak = max(self.system_peak or 0, used)
        except (OSError, ValueError, KeyError, IndexError):
            pass
        if self.pid is not None:
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        self.process_peak = max(self.process_peak or 0, int(line.split()[1]) * 1024)
            except (OSError, ValueError, IndexError):
                pass
        self.samples += 1
        self._sample_thermal()

    def _sample_thermal(self) -> None:
        temperatures = []
        try:
            for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                try:
                    value = int(path.read_text().strip())
                    if -100_000 <= value <= 200_000:
                        temperatures.append(value)
                except (OSError, ValueError):
                    continue
        except OSError:
            return
        if temperatures:
            low, high = min(temperatures), max(temperatures)
            self.thermal_min = low if self.thermal_min is None else min(self.thermal_min, low)
            self.thermal_max = high if self.thermal_max is None else max(self.thermal_max, high)
            self.thermal_samples += 1

    def _run(self) -> None:
        while not self.stop.wait(0.1):
            self._sample()

    def __enter__(self) -> MemorySampler:
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self._sample()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_endpoint(endpoint: str) -> str:
    parsed = httpx.URL(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.host not in {"127.0.0.1", "localhost", "::1"}
        or parsed.userinfo
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("benchmark endpoint must be a plain loopback origin")
    return endpoint.rstrip("/")


def request_payload(
    source: str, protocol: Literal["slots", "hybrid"], model: str, max_output_tokens: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": _slot_messages(source, protocol=protocol),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_output_tokens,
        "stream": False,
    }


def context(
    records: Sequence[FidelityRecord],
    provenance: Provenance,
    *,
    model: str,
    max_output_tokens: int,
    limit: int,
    endpoint: str,
    timeout: float,
    memory_pid: int | None = None,
    mode: Mode = "hybrid",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/bookforge").glob("*.py"))]
    return {
        "kind": "header",
        "schema_version": 1,
        "evaluator_revision": FIDELITY_EVALUATOR_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_sha256": digest([record.model_dump(mode="json") for record in records]),
        "implementation_sha256": digest(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
        ),
        "provenance": provenance.model_dump(),
        "model_sha256": digest(model),
        "endpoint_sha256": digest(endpoint),
        "limit": limit,
        "timeout_seconds": timeout,
        "mode": mode,
        "requests": {
            protocol: digest(request_payload("", protocol, model, max_output_tokens))
            for protocol in (("slots",) if mode == "accepted_first" else ("slots", "hybrid"))
        },
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "retries": 0,
        "memory_pid": memory_pid,
    }


def append_event(path: Path, event: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_evidence(
    path: Path, expected: Mapping[str, object]
) -> tuple[set[int], list[CaseEvidence]]:
    if not path.exists():
        append_event(path, expected)
        return set(), []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or json.loads(lines[0]) != expected:
            raise ValueError
        started: set[int] = set()
        completed: set[int] = set()
        results: list[CaseEvidence] = []
        for line in lines[1:]:
            event = json.loads(line)
            index = event.get("index")
            if type(index) is not int or not 0 <= index < expected["limit"]:
                raise ValueError
            if event.get("kind") == "start":
                if set(event) != {"kind", "index"} or index in started:
                    raise ValueError
                started.add(index)
                continue
            result = CaseEvidence.model_validate(event)
            if (
                index not in started
                or index in completed
                or not set(result.surfaces) <= set(SURFACES)
            ):
                raise ValueError
            if result.status == "ok" and set(result.surfaces) != set(
                surfaces_for(expected.get("mode", "hybrid"))
            ):
                raise ValueError
            completed.add(index)
            results.append(result)
        return started, results
    except (ValueError, TypeError, KeyError):
        raise ValueError("evidence is incompatible, malformed, or incomplete on disk") from None


def score(
    record: FidelityRecord,
    output: object,
    *,
    surface: str,
    elapsed_ms: float,
    output_tokens: int | None,
    generation_complete: bool = True,
) -> Score:
    evaluated_output = (
        {"master_prompt": output} if surface == "renderer" and isinstance(output, str) else output
    )
    result = evaluate_surface(record, evaluated_output, surface=surface)
    schema_valid = result.schema_valid
    if surface == "raw":
        from bookforge.tensorrt_slot_client import parse_tensor_graph_slots

        try:
            parse_tensor_graph_slots(output)
        except (TypeError, ValueError):
            schema_valid = False
    serialized = output.model_dump(mode="json") if isinstance(output, BaseModel) else output
    return Score(
        schema_valid=schema_valid and generation_complete,
        exact_pass=result.exact_example_pass and schema_valid and generation_complete,
        privacy_pass=result.privacy.passed,
        required_atoms=result.required_atoms,
        passed_atoms=result.passed_atoms,
        forbidden_count=len(result.forbidden_hits),
        unsupported_count=len(result.unsupported_concepts),
        output_sha256=digest(serialized),
        latency_ms=elapsed_ms,
        output_tokens=output_tokens,
        generation_complete=generation_complete,
        required_atom_passes=[atom.passed for atom in result.expectation_results if atom.required],
    )


def infer(
    client: httpx.Client,
    record: FidelityRecord,
    protocol: Literal["slots", "hybrid"],
    model: str,
    max_output_tokens: int,
) -> tuple[str, float, int | None, bool]:
    started = time.perf_counter()
    response = client.post(
        "/v1/chat/completions",
        json=request_payload(record.passage, protocol, model, max_output_tokens),
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("invalid model envelope")
    tokens = payload.get("usage", {}).get("completion_tokens")
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError("invalid token count")
    return (
        content,
        (time.perf_counter() - started) * 1000,
        tokens,
        choice.get("finish_reason") in {None, "stop"},
    )


def safe_slots(text: str, source: str, protocol: Literal["slots", "hybrid"]) -> object:
    wire = tensor_slot_wire_plan(text, source_text=source, protocol=protocol)
    plan = wire.to_live_scene_plan(context_text=source)
    validate_live_scene_plan_privacy(plan, source_text=source)
    return renderer_contract(plan, source)


def renderer_contract(plan: Any, source: str) -> str:
    return plan.to_page(
        source_text=source, visual_style="luminous storybook illustration", seed=0
    ).scene_spec.master_prompt


def run_case(
    client: httpx.Client,
    record: FidelityRecord,
    index: int,
    *,
    model: str,
    max_output_tokens: int,
    mode: Mode = "hybrid",
) -> CaseEvidence:
    if mode == "accepted_first":
        return run_accepted_first_case(
            client, record, index, model=model, max_output_tokens=max_output_tokens
        )
    from bookforge.live_scene_facts import adapt_live_scene_facts
    from bookforge.tensorrt_slot_client import parse_tensor_graph_slots

    surfaces: dict[str, Score] = {}
    try:
        accepted, accepted_ms, accepted_tokens, accepted_complete = infer(
            client, record, "slots", model, max_output_tokens
        )
        surfaces["accepted_raw"] = score(
            record,
            accepted,
            surface="raw",
            elapsed_ms=accepted_ms,
            output_tokens=accepted_tokens,
            generation_complete=accepted_complete,
        )
        hybrid, hybrid_ms, hybrid_tokens, hybrid_complete = infer(
            client, record, "hybrid", model, max_output_tokens
        )
        surfaces["hybrid_raw"] = score(
            record,
            hybrid,
            surface="raw",
            elapsed_ms=hybrid_ms,
            output_tokens=hybrid_tokens,
            generation_complete=hybrid_complete,
        )
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
        return CaseEvidence(
            index=index, status="request_failed", refusal="none", fallback=False, surfaces=surfaces
        )
    start = time.perf_counter()
    try:
        if not accepted_complete:
            raise ValueError
        accepted_safe = safe_slots(accepted, record.passage, "slots")
    except ValueError:
        accepted_safe = {}
    surfaces["accepted_renderer"] = score(
        record,
        accepted_safe,
        surface="renderer",
        elapsed_ms=accepted_ms + (time.perf_counter() - start) * 1000,
        output_tokens=accepted_tokens,
    )
    start = time.perf_counter()
    refusal = "none"
    refusal_hash = None
    adapter_refusal_code = None
    facts = None
    try:
        if not hybrid_complete:
            raise ValueError
        result = adapt_live_scene_facts(
            parse_tensor_graph_slots(hybrid), source_text=record.passage
        )
        facts = result.facts
        if facts is None:
            refusal = "adapter_refused"
            refusal_hash = digest(result.refusal.value)
            adapter_refusal_code = result.refusal.value
    except ValueError:
        refusal = "parse_refused"
    graph_processing_ms = (time.perf_counter() - start) * 1000
    surfaces["graph_candidate"] = score(
        record,
        facts if facts is not None else {},
        surface="postprocessed",
        elapsed_ms=hybrid_ms + graph_processing_ms,
        output_tokens=hybrid_tokens,
    )
    fallback = facts is None
    start = time.perf_counter()
    try:
        if facts is not None:
            from bookforge.tensorrt_slot_client import tensor_graph_wire_plan

            plan = tensor_graph_wire_plan(hybrid, source_text=record.passage).to_live_scene_plan(
                context_text=record.passage
            )
            validate_live_scene_plan_privacy(plan, source_text=record.passage)
            final = renderer_contract(plan, record.passage)
        else:
            final = accepted_safe
    except ValueError:
        refusal = "renderer_refused"
        final = accepted_safe
        fallback = True
    surfaces["final_renderer"] = score(
        record,
        final,
        surface="renderer",
        elapsed_ms=hybrid_ms
        + ((time.perf_counter() - start) * 1000 if facts is not None else graph_processing_ms)
        + (surfaces["accepted_renderer"].latency_ms if fallback else 0),
        output_tokens=(
            hybrid_tokens + accepted_tokens
            if fallback and hybrid_tokens is not None and accepted_tokens is not None
            else None
            if fallback
            else hybrid_tokens
        ),
    )
    return CaseEvidence(
        index=index,
        status="ok",
        refusal=refusal,
        refusal_sha256=refusal_hash,
        adapter_refusal_code=adapter_refusal_code,
        fallback=fallback,
        surfaces=surfaces,
    )


def run_accepted_first_case(
    client: httpx.Client, record: FidelityRecord, index: int, *, model: str, max_output_tokens: int
) -> CaseEvidence:
    from bookforge.tensorrt_slot_client import tensor_accepted_graph_wire_plan

    try:
        raw, learned_ms, tokens, complete = infer(client, record, "slots", model, max_output_tokens)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
        return CaseEvidence(
            index=index, status="request_failed", refusal="none", fallback=False, surfaces={}
        )
    start = time.perf_counter()
    try:
        if not complete:
            raise ValueError
        accepted = safe_slots(raw, record.passage, "slots")
    except (ValueError, LiveScenePlannerPrivacyError):
        accepted = {}
    accepted_processing_ms = (time.perf_counter() - start) * 1000
    facts = None
    final = accepted
    refusal = "none"
    fallback = True
    start = time.perf_counter()
    try:
        if not complete:
            raise ValueError
        wire = tensor_accepted_graph_wire_plan(raw, source_text=record.passage)
        facts = wire.scene_facts
        refusal = "adapter_refused" if facts is None else "none"
    except (ValueError, LiveScenePlannerPrivacyError):
        refusal = "parse_refused"
    graph_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    if refusal != "parse_refused":
        try:
            plan = wire.to_live_scene_plan(context_text=record.passage)
            validate_live_scene_plan_privacy(plan, source_text=record.passage)
            final = renderer_contract(plan, record.passage)
            fallback = facts is None
        except (ValueError, LiveScenePlannerPrivacyError):
            refusal = "renderer_refused"
    final_processing_ms = graph_ms + (time.perf_counter() - start) * 1000
    if refusal in {"parse_refused", "renderer_refused"}:
        final_processing_ms += accepted_processing_ms
    # All evaluator calls follow measured production construction and compilation.
    outputs = {
        "accepted_raw": (raw, "raw", learned_ms),
        "accepted_renderer": (accepted, "renderer", learned_ms + accepted_processing_ms),
        "graph_candidate": (
            facts if facts is not None else {},
            "postprocessed",
            learned_ms + graph_ms,
        ),
        "final_renderer": (final, "renderer", learned_ms + final_processing_ms),
    }
    return CaseEvidence(
        index=index,
        status="ok",
        refusal=refusal,
        fallback=fallback,
        learned_inference_ms=learned_ms,
        graph_construction_ms=graph_ms,
        surfaces={
            name: score(
                record,
                output,
                surface=surface,
                elapsed_ms=elapsed,
                output_tokens=tokens,
                generation_complete=complete,
            )
            for name, (output, surface, elapsed) in outputs.items()
        },
    )


def distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": statistics.median(ordered) if ordered else None,
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
        "max": max(ordered, default=None),
    }


def compare_renderer_cases(
    results: Sequence[CaseEvidence], records: Sequence[FidelityRecord]
) -> dict[str, Any]:
    total: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    for row in results:
        if not {"accepted_renderer", "final_renderer"} <= row.surfaces.keys():
            continue
        accepted = row.surfaces["accepted_renderer"]
        final = row.surfaces["final_renderer"]
        comparable = (
            len(accepted.required_atom_passes) == accepted.required_atoms
            and len(final.required_atom_passes) == final.required_atoms
            and accepted.required_atoms == final.required_atoms
        )
        lost = gained = 0
        if comparable:
            lost = sum(
                a and not b
                for a, b in zip(
                    accepted.required_atom_passes, final.required_atom_passes, strict=True
                )
            )
            gained = sum(
                b and not a
                for a, b in zip(
                    accepted.required_atom_passes, final.required_atom_passes, strict=True
                )
            )
        counts = Counter(
            cases=1,
            exact_regressions=int(accepted.exact_pass and not final.exact_pass),
            exact_improvements=int(final.exact_pass and not accepted.exact_pass),
            schema_regressions=int(accepted.schema_valid and not final.schema_valid),
            privacy_regressions=int(accepted.privacy_pass and not final.privacy_pass),
            atom_comparable_cases=int(comparable),
            atoms_lost=lost,
            atoms_gained=gained,
            cases_with_atom_loss=int(lost > 0),
            cases_with_atom_gain=int(gained > 0),
        )
        total.update(counts)
        for category in records[row.index].categories:
            categories.setdefault(category, Counter()).update(counts)
    return {"overall": dict(total), "categories": dict(sorted(categories.items()))}


def aggregate(
    header: Mapping[str, Any],
    started: set[int],
    results: Sequence[CaseEvidence],
    records: Sequence[FidelityRecord],
) -> dict[str, Any]:
    surfaces = {}
    for name in surfaces_for(header.get("mode", "hybrid")):
        rows = [
            (records[result.index], result.surfaces[name])
            for result in results
            if name in result.surfaces
        ]
        categories: dict[str, Counter] = {}
        for record, value in rows:
            for category in record.categories:
                counts = categories.setdefault(
                    category,
                    Counter(
                        total=0,
                        exact=0,
                        privacy_failures=0,
                        schema_valid=0,
                        required_atoms=0,
                        passed_atoms=0,
                    ),
                )
                counts.update(
                    total=1,
                    exact=int(value.exact_pass),
                    privacy_failures=int(not value.privacy_pass),
                    schema_valid=int(value.schema_valid),
                    required_atoms=value.required_atoms,
                    passed_atoms=value.passed_atoms,
                )
        required_atoms = sum(row.required_atoms for _, row in rows)
        passed_atoms = sum(row.passed_atoms for _, row in rows)
        surfaces[name] = {
            "count": len(rows),
            "schema_valid": sum(row.schema_valid for _, row in rows),
            "exact_pass": sum(row.exact_pass for _, row in rows),
            "privacy_failures": sum(not row.privacy_pass for _, row in rows),
            "required_atoms": required_atoms,
            "passed_atoms": passed_atoms,
            "semantic_atom_recall": passed_atoms / required_atoms if required_atoms else None,
            "categories": {
                category: {
                    **counts,
                    "semantic_atom_recall": counts["passed_atoms"] / counts["required_atoms"]
                    if counts["required_atoms"]
                    else None,
                }
                for category, counts in sorted(categories.items())
            },
            "latency_ms": distribution([row.latency_ms for _, row in rows]),
            "output_tokens": distribution(
                [row.output_tokens for _, row in rows if row.output_tokens is not None]
            ),
            "output_tokens_unavailable": sum(row.output_tokens is None for _, row in rows),
        }
    complete = len(results) == 512 and all(result.status == "ok" for result in results)
    return {
        "schema_version": 1,
        "benchmark": "live-scene-facts-public-development",
        "context": dict(header),
        "context_sha256": digest(header),
        "complete": complete,
        "split": "development",
        "expected_cases": 512,
        "started_cases": len(started),
        "finished_cases": len(results),
        "interrupted_cases": len(started) - len(results),
        "failed_cases": sum(result.status != "ok" for result in results),
        "refusals": dict(sorted(Counter(result.refusal for result in results).items())),
        "adapter_refusals": dict(
            sorted(
                Counter(
                    result.adapter_refusal_code
                    for result in results
                    if result.adapter_refusal_code is not None
                ).items()
            )
        ),
        "refusal_code_hashes": dict(
            sorted(
                Counter(
                    result.refusal_sha256 for result in results if result.refusal_sha256 is not None
                ).items()
            )
        ),
        "fallback_cases": sum(result.fallback for result in results),
        "surfaces": surfaces,
        "final_vs_accepted": compare_renderer_cases(results, records),
        "learned_inference_ms": distribution(
            [row.learned_inference_ms for row in results if row.learned_inference_ms is not None]
        ),
        "graph_construction_ms": distribution(
            [row.graph_construction_ms for row in results if row.graph_construction_ms is not None]
        ),
        "graph_construction_scope": "accepted normalization and graph helper; excludes grading",
        "peak_memory_bytes": header["provenance"]["peak_memory_bytes"],
        "process_rss_peak_bytes": max(
            (
                row.process_rss_peak_bytes
                for row in results
                if row.process_rss_peak_bytes is not None
            ),
            default=None,
        ),
        "system_used_peak_bytes": max(
            (
                row.system_used_peak_bytes
                for row in results
                if row.system_used_peak_bytes is not None
            ),
            default=None,
        ),
        "memory_sample_attempts": sum(row.memory_samples for row in results),
        "memory_scope": "100 ms Linux VmRSS and MemTotal-MemAvailable samples during paired cases",
        "thermal_min_millicelsius": min(
            (
                row.thermal_min_millicelsius
                for row in results
                if row.thermal_min_millicelsius is not None
            ),
            default=None,
        ),
        "thermal_max_millicelsius": max(
            (
                row.thermal_max_millicelsius
                for row in results
                if row.thermal_max_millicelsius is not None
            ),
            default=None,
        ),
        "thermal_samples": sum(row.thermal_samples for row in results),
        "thermal_scope": "Linux thermal zones; 100 ms target cadence plus case boundaries",
        "gpu_allocator_peak_bytes": None,
        "peak_memory_source": "provided_device_measurement"
        if header["provenance"]["peak_memory_bytes"] is not None
        else "unavailable",
        "hidden_evaluated": False,
        "paid_services_used": False,
        "latency_scope": "HTTP inference plus local surface construction; nearest-rank p95",
        "semantic_scope": "raw slots, typed graph, and compiled renderer text scored separately",
        "fallback_scope": "same accepted response; one inference and measured local construction"
        if header.get("mode") == "accepted_first"
        else "captured accepted response; summed accepted and hybrid planning latency",
        "deterministic_recovery_is_model_improvement": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--memory-pid", type=int)
    parser.add_argument("--mode", choices=("hybrid", "accepted_first"), default="hybrid")
    args = parser.parse_args(argv)
    try:
        endpoint = validate_endpoint(args.endpoint)
        if not 1 <= args.limit <= 512 or not 1 <= args.max_output_tokens <= 128:
            raise ValueError
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
            raise ValueError
        if args.memory_pid is not None and args.memory_pid <= 0:
            raise ValueError
        provenance = Provenance.model_validate_json(args.provenance.read_text())
        records = tuple(generate_split(DatasetSplit.DEVELOPMENT))
        header = context(
            records,
            provenance,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            limit=args.limit,
            endpoint=endpoint,
            timeout=args.timeout,
            memory_pid=args.memory_pid,
            mode=args.mode,
        )
        started, results = load_evidence(args.evidence, header)
        if not args.aggregate_only:
            with httpx.Client(
                base_url=endpoint,
                timeout=args.timeout,
                trust_env=False,
                follow_redirects=False,
                transport=httpx.HTTPTransport(retries=0),
            ) as client:
                for index, record in enumerate(records[: args.limit]):
                    if index in started:
                        continue
                    append_event(args.evidence, {"kind": "start", "index": index})
                    started.add(index)
                    with MemorySampler(args.memory_pid) as memory:
                        result = run_case(
                            client,
                            record,
                            index,
                            model=args.model,
                            max_output_tokens=args.max_output_tokens,
                            mode=args.mode,
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
                    append_event(args.evidence, result.model_dump())
                    results.append(result)
                    args.output.write_text(
                        json.dumps(
                            aggregate(header, started, results, records), indent=2, sort_keys=True
                        )
                        + "\n"
                    )
        args.output.write_text(
            json.dumps(aggregate(header, started, results, records), indent=2, sort_keys=True)
            + "\n"
        )
        return 0
    except Exception:
        # Exceptions can contain the private request or model response; expose a fixed message.
        print("benchmark failed; check local configuration and sanitized evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
