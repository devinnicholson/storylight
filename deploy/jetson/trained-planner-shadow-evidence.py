#!/usr/bin/env python3
"""Validate and aggregate privacy-safe Jetson trained-planner shadow evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"[a-f0-9]{64}\Z")
MODEL_REVISION = re.compile(r"sha256:[a-f0-9]{64}\Z")
HIDDEN_SUMMARY_FIELDS = {
    "surface",
    "split",
    "records",
    "record_ids_sha256",
    "category_record_counts",
    "schema_valid_rate",
    "privacy_pass_rate",
    "semantic_atom_recall",
    "exact_example_pass_rate",
    "category_pass_rates",
    "counterfactual_pairs",
    "counterfactual_sensitivity",
    "unsupported_concept_rate",
    "pii_leaks",
    "privacy_term_leaks",
    "source_echoes",
    "injection_leaks",
    "forbidden_hits",
}
LEG_FIELDS = {
    "schema_version",
    "label",
    "order_index",
    "model_revision",
    "engine_sha256",
    "benchmark_sha256",
    "latency_seconds",
    "maximum_output_tokens",
    "automatic_semantic_pass",
    "planner_ready_seconds",
    "service_memory_peak_bytes",
    "minimum_available_memory_kib",
    "inference_swap_events",
    "oom_events",
    "external_planner_socket_events",
    "restart_failures",
    "kiosk_active",
    "projector_http_passed",
    "live_scene_passed",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"JSON document repeats key: {key}")
        document[key] = value
    return document


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_candidate_manifest(
    path: Path,
    *,
    expected_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
) -> dict[str, str]:
    """Validate the installed manifest and return its immutable serving lineage."""

    _sha(expected_sha256, label="candidate manifest checksum")
    _sha(config_sha256, label="configuration checksum")
    _sha(dataset_manifest_sha256, label="dataset manifest checksum")
    document = _load_json(path, label="candidate manifest")
    if sha256_path(path) != expected_sha256:
        raise ValueError("candidate manifest checksum mismatch")
    candidate_id = document.get("candidate_id")
    engine_sha256 = document.get("engine_sha256")
    model_revision = document.get("model_revision")
    training_run_id = document.get("training_run_id")
    if (
        not isinstance(candidate_id, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", candidate_id)
        or not isinstance(engine_sha256, str)
        or not SHA256.fullmatch(engine_sha256)
        or model_revision != f"sha256:{engine_sha256}"
    ):
        raise ValueError("candidate manifest has an invalid serving identity")
    if not isinstance(training_run_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{2,63}", training_run_id
    ):
        raise ValueError("candidate manifest has an invalid training run ID")
    if document.get("source_config_sha256") != config_sha256:
        raise ValueError("candidate manifest was trained with another configuration")
    if document.get("source_dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("candidate manifest was trained from another dataset manifest")
    return {
        "candidate_id": candidate_id,
        "engine_sha256": engine_sha256,
        "model_revision": str(model_revision),
        "training_run_id": training_run_id,
    }


def validate_hidden_report(
    path: Path,
    *,
    expected_sha256: str,
    candidate_revision: str,
) -> dict[str, object]:
    """Validate a checksum-bound aggregate report without accessing hidden records."""

    _sha(expected_sha256, label="hidden evaluation report checksum")
    if not MODEL_REVISION.fullmatch(candidate_revision):
        raise ValueError("candidate revision must be an engine SHA-256 revision")
    if path.is_symlink():
        raise ValueError("hidden evaluation report may not be a symbolic link")
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("hidden evaluation report is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("hidden evaluation report must be a regular file")
    if stat.S_IMODE(metadata.st_mode) not in {0o400, 0o440, 0o444, 0o600, 0o640, 0o644}:
        raise ValueError("hidden evaluation report has an unsafe file mode")
    actual_sha256 = sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("hidden evaluation report checksum mismatch")
    document = _load_json(path, label="hidden evaluation report")
    if set(document) != {"schema_version", "candidate_revision", "privacy", "summary"}:
        raise ValueError("hidden evaluation report has unexpected fields")
    if document["schema_version"] != "1.0" or document["candidate_revision"] != candidate_revision:
        raise ValueError("hidden evaluation report identifies another candidate")
    if document["privacy"] != {"passages_recorded": False, "outputs_recorded": False}:
        raise ValueError("hidden evaluation report does not prove aggregate-only retention")
    summary = document["summary"]
    if not isinstance(summary, dict) or set(summary) != HIDDEN_SUMMARY_FIELDS:
        raise ValueError("hidden evaluation report has an invalid summary contract")
    if (
        summary.get("surface") != "raw"
        or summary.get("split") != "hidden"
        or summary.get("records") != 512
        or summary.get("counterfactual_pairs") != 256
    ):
        raise ValueError("hidden evaluation report has the wrong held-out population")
    record_ids_sha256 = _sha(
        summary.get("record_ids_sha256"), label="hidden population record ID checksum"
    )
    categories = summary.get("category_record_counts")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("hidden evaluation report has no category population")
    if any(
        not isinstance(key, str) or type(value) is not int or value <= 0
        for key, value in categories.items()
    ):
        raise ValueError("hidden evaluation report has invalid category counts")
    canonical_summary = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    return {
        "report_sha256": actual_sha256,
        "summary_sha256": hashlib.sha256(canonical_summary).hexdigest(),
        "candidate_revision": candidate_revision,
        "records": 512,
        "pairs": 256,
        "record_ids_sha256": record_ids_sha256,
        "retains_passages": False,
        "retains_model_outputs": False,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_leg(
    path: Path,
    *,
    expected_label: str,
    expected_index: int,
    expected_revision: str,
    expected_engine_sha256: str,
) -> dict[str, Any]:
    document = _load_json(path, label="shadow leg evidence")
    if set(document) != LEG_FIELDS or document.get("schema_version") != "1.0":
        raise ValueError("shadow leg evidence has an invalid contract")
    if document.get("label") != expected_label or document.get("order_index") != expected_index:
        raise ValueError("shadow legs are not in the required ABBA order")
    if document.get("model_revision") != expected_revision:
        raise ValueError("shadow leg advertises the wrong model revision")
    if document.get("engine_sha256") != expected_engine_sha256:
        raise ValueError("shadow leg served the wrong engine checksum")
    _sha(document.get("benchmark_sha256"), label="benchmark checksum")
    latencies = document.get("latency_seconds")
    if (
        not isinstance(latencies, list)
        or not latencies
        or any(type(value) not in {int, float} or value < 0 for value in latencies)
    ):
        raise ValueError("shadow leg has invalid latency samples")
    integer_fields = (
        "maximum_output_tokens",
        "service_memory_peak_bytes",
        "minimum_available_memory_kib",
        "inference_swap_events",
        "oom_events",
        "external_planner_socket_events",
        "restart_failures",
    )
    if any(type(document[field]) is not int or document[field] < 0 for field in integer_fields):
        raise ValueError("shadow leg has invalid integer telemetry")
    if document["service_memory_peak_bytes"] == 0 or document["minimum_available_memory_kib"] == 0:
        raise ValueError("shadow leg is missing required memory telemetry")
    ready = document.get("planner_ready_seconds")
    if type(ready) not in {int, float} or ready < 0:
        raise ValueError("shadow leg has invalid planner readiness telemetry")
    boolean_fields = (
        "automatic_semantic_pass",
        "kiosk_active",
        "projector_http_passed",
        "live_scene_passed",
    )
    if any(type(document[field]) is not bool for field in boolean_fields):
        raise ValueError("shadow leg has invalid boolean telemetry")
    return document


def build_leg_evidence(
    benchmark_path: Path,
    monitor_path: Path,
    *,
    label: str,
    order_index: int,
    model_revision: str,
    engine_sha256: str,
    planner_ready_seconds: float,
    kiosk_active: bool,
    projector_http_passed: bool,
    live_scene_passed: bool,
) -> dict[str, object]:
    benchmark = _load_json(benchmark_path, label="planner benchmark")
    runtime = benchmark.get("runtime")
    contracts = benchmark.get("contracts")
    acceptance = benchmark.get("acceptance")
    if (
        benchmark.get("schema_version") != "1.2"
        or not isinstance(runtime, dict)
        or runtime.get("model_revision") != model_revision
        or runtime.get("base_url") != "http://127.0.0.1:11435"
        or not isinstance(contracts, list)
        or len(contracts) != 1
        or not isinstance(acceptance, dict)
    ):
        raise ValueError("planner benchmark does not match the shadow leg")
    contract = contracts[0]
    if not isinstance(contract, dict) or contract.get("contract") != "standard":
        raise ValueError("shadow benchmark must use the standard TensorRT contract")
    cases = contract.get("cases")
    summary = contract.get("summary")
    if not isinstance(cases, list) or not cases or not isinstance(summary, dict):
        raise ValueError("planner benchmark has no measured cases")
    latency_seconds: list[float] = []
    output_tokens: list[int] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("planner benchmark has an invalid case")
        latency = case.get("planning_ms")
        tokens = case.get("output_tokens")
        if (
            type(latency) not in {int, float}
            or latency < 0
            or type(tokens) is not int
            or tokens < 0
        ):
            raise ValueError("planner benchmark has invalid latency or token telemetry")
        latency_seconds.append(float(latency) / 1000)
        output_tokens.append(tokens)
    monitor_rows: list[tuple[int, int, int, int, int, int]] = []
    for line_number, line in enumerate(monitor_path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6 or any(not field.isdigit() for field in fields):
            raise ValueError(f"runtime monitor line {line_number} is invalid")
        parsed = [int(field) for field in fields]
        monitor_rows.append((parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5]))
    if not monitor_rows:
        raise ValueError("runtime monitor has no samples")
    available, memory_peak, swap_used, external, oom, restarts = zip(*monitor_rows, strict=True)
    return {
        "schema_version": "1.0",
        "label": label,
        "order_index": order_index,
        "model_revision": model_revision,
        "engine_sha256": engine_sha256,
        "benchmark_sha256": sha256_path(benchmark_path),
        "latency_seconds": latency_seconds,
        "maximum_output_tokens": max(output_tokens),
        "automatic_semantic_pass": acceptance.get("automatic_semantic_pass") is True,
        "planner_ready_seconds": planner_ready_seconds,
        "service_memory_peak_bytes": max(memory_peak),
        "minimum_available_memory_kib": min(available),
        "inference_swap_events": int(any(swap_used)),
        "oom_events": int(any(oom)),
        "external_planner_socket_events": int(any(external)),
        "restart_failures": max(restarts),
        "kiosk_active": kiosk_active,
        "projector_http_passed": projector_http_passed,
        "live_scene_passed": live_scene_passed,
    }


def build_shadow_evidence(
    legs: Sequence[Mapping[str, Any]],
    *,
    candidate_manifest: Mapping[str, object],
    candidate_manifest_sha256: str,
    accepted_engine_sha256: str,
    hidden_binding: Mapping[str, object],
    restoration: Mapping[str, object],
    run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    int4_export_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the runtime contract and its hash-chained orchestration artifact."""

    _sha(candidate_manifest_sha256, label="candidate manifest checksum")
    _sha(accepted_engine_sha256, label="accepted engine checksum")
    _sha(config_sha256, label="configuration checksum")
    _sha(dataset_manifest_sha256, label="dataset manifest checksum")
    _sha(int4_export_sha256, label="INT4 export artifact checksum")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", run_id):
        raise ValueError("invalid orchestration run ID")
    candidate_id = candidate_manifest.get("candidate_id")
    candidate_engine_sha256 = candidate_manifest.get("engine_sha256")
    candidate_revision = candidate_manifest.get("model_revision")
    if (
        not isinstance(candidate_id, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", candidate_id)
        or not isinstance(candidate_engine_sha256, str)
        or not SHA256.fullmatch(candidate_engine_sha256)
        or candidate_revision != f"sha256:{candidate_engine_sha256}"
    ):
        raise ValueError("candidate manifest has an invalid serving identity")
    training_run_id = candidate_manifest.get("training_run_id")
    if not isinstance(training_run_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{2,63}", training_run_id
    ):
        raise ValueError("candidate manifest has an invalid training run ID")
    if candidate_manifest.get("source_config_sha256") != config_sha256:
        raise ValueError("candidate manifest was trained with another configuration")
    if candidate_manifest.get("source_dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("candidate manifest was trained from another dataset manifest")
    if len(legs) != 4 or [leg["label"] for leg in legs] != [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]:
        raise ValueError("runtime evidence requires exactly four ABBA legs")
    if hidden_binding.get("candidate_revision") != candidate_revision:
        raise ValueError("hidden evaluation binding identifies another candidate")
    expected_restoration = {
        "accepted_engine_sha256": accepted_engine_sha256,
        "engine_sha256_after": accepted_engine_sha256,
        "config_sha256_before": restoration.get("config_sha256_after"),
        "endpoint_ready": True,
        "unit_active": True,
    }
    restoration_demonstrated = all(
        restoration.get(key) == value for key, value in expected_restoration.items()
    )
    candidate_legs = [leg for leg in legs if leg["label"] == "candidate"]
    baseline_legs = [leg for leg in legs if leg["label"] == "baseline"]
    candidate_latency = [float(value) for leg in candidate_legs for value in leg["latency_seconds"]]
    baseline_latency = [float(value) for leg in baseline_legs for value in leg["latency_seconds"]]
    candidate_p95 = _percentile(candidate_latency, 0.95)
    baseline_p95 = _percentile(baseline_latency, 0.95)
    if baseline_p95 <= 0:
        raise ValueError("baseline latency must be positive")
    p95_regression = (candidate_p95 - baseline_p95) / baseline_p95
    all_runtime_legs = list(legs)
    runtime = {
        "maximum_output_tokens": max(int(leg["maximum_output_tokens"]) for leg in candidate_legs),
        "p50_seconds": round(_percentile(candidate_latency, 0.50), 6),
        "p95_seconds": round(candidate_p95, 6),
        "maximum_seconds": round(max(candidate_latency), 6),
        "p95_regression_fraction": round(p95_regression, 6),
        "unified_memory_peak_gb": round(
            max(int(leg["service_memory_peak_bytes"]) for leg in candidate_legs) / 1024**3,
            6,
        ),
        "available_memory_mib": min(
            int(leg["minimum_available_memory_kib"]) for leg in candidate_legs
        )
        // 1024,
        "inference_swap_events": sum(int(leg["inference_swap_events"]) for leg in all_runtime_legs),
        "oom_events": sum(int(leg["oom_events"]) for leg in all_runtime_legs),
        "external_planner_socket_events": sum(
            int(leg["external_planner_socket_events"]) for leg in all_runtime_legs
        ),
        "restart_failures": sum(int(leg["restart_failures"]) for leg in all_runtime_legs),
        "planner_ready_seconds": round(
            max(float(leg["planner_ready_seconds"]) for leg in candidate_legs), 6
        ),
        "projector_flow_passed": all(
            bool(leg["kiosk_active"])
            and bool(leg["projector_http_passed"])
            and bool(leg["live_scene_passed"])
            for leg in all_runtime_legs
        ),
        "restoration_demonstrated": restoration_demonstrated,
    }
    checks = {
        "maximum_output_tokens": runtime["maximum_output_tokens"] <= 64,
        "p50_seconds": runtime["p50_seconds"] <= 1.70,
        "p95_seconds": runtime["p95_seconds"] <= 2.00,
        "maximum_seconds": runtime["maximum_seconds"] <= 2.25,
        "p95_regression_fraction": runtime["p95_regression_fraction"] <= 0.05,
        "unified_memory_peak_gb": runtime["unified_memory_peak_gb"] <= 4.0,
        "available_memory_mib": runtime["available_memory_mib"] >= 768,
        "zero_runtime_failures": sum(
            int(runtime[field])
            for field in (
                "inference_swap_events",
                "oom_events",
                "external_planner_socket_events",
                "restart_failures",
            )
        )
        == 0,
        "planner_ready_seconds": runtime["planner_ready_seconds"] <= 90,
        "projector_flow": runtime["projector_flow_passed"] is True,
        "restoration": runtime["restoration_demonstrated"] is True,
        "contest_semantics": all(bool(leg["automatic_semantic_pass"]) for leg in candidate_legs),
    }
    candidate_identity = {
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "engine_sha256": candidate_engine_sha256,
        "model_revision": candidate_revision,
    }
    runtime_evidence = {
        "schema_version": "story-fidelity-runtime-v1",
        "candidate_identity": candidate_identity,
        "runtime": runtime,
    }
    runtime_sha256 = hashlib.sha256(_payload(runtime_evidence)).hexdigest()
    stage_evidence = {
        "schema_version": "1.0",
        "stage": "jetson-shadow",
        "producer": "bookforge-jetson-shadow-recorder",
        "run_id": run_id,
        "training_run_id": training_run_id,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "status": "succeeded",
        "inputs": {"int4-export": int4_export_sha256},
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_identity": candidate_identity,
        "shadow_status": "passed" if all(checks.values()) else "rejected",
        "evidence_sha256": {
            "runtime": runtime_sha256,
            "candidate_manifest": candidate_manifest_sha256,
            "hidden_summary": str(hidden_binding["report_sha256"]),
        },
    }
    return runtime_evidence, stage_evidence


def _payload(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _write_new(path: Path, document: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hidden = subparsers.add_parser("validate-hidden")
    hidden.add_argument("--report", type=Path, required=True)
    hidden.add_argument("--expected-sha256", required=True)
    hidden.add_argument("--candidate-revision", required=True)
    leg = subparsers.add_parser("make-leg")
    leg.add_argument("--benchmark", type=Path, required=True)
    leg.add_argument("--monitor", type=Path, required=True)
    leg.add_argument("--label", choices=("baseline", "candidate"), required=True)
    leg.add_argument("--order-index", type=int, required=True)
    leg.add_argument("--model-revision", required=True)
    leg.add_argument("--engine-sha256", required=True)
    leg.add_argument("--planner-ready-seconds", type=float, required=True)
    leg.add_argument("--kiosk-active", action="store_true")
    leg.add_argument("--projector-http-passed", action="store_true")
    leg.add_argument("--live-scene-passed", action="store_true")
    leg.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--leg", action="append", type=Path, required=True)
    aggregate.add_argument("--candidate-manifest", type=Path, required=True)
    aggregate.add_argument("--candidate-manifest-sha256", required=True)
    aggregate.add_argument("--accepted-engine-sha256", required=True)
    aggregate.add_argument("--hidden-report", type=Path, required=True)
    aggregate.add_argument("--hidden-report-sha256", required=True)
    aggregate.add_argument("--restoration", type=Path, required=True)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--config-sha256", required=True)
    aggregate.add_argument("--dataset-manifest-sha256", required=True)
    aggregate.add_argument("--int4-export-sha256", required=True)
    aggregate.add_argument("--runtime-output", type=Path, required=True)
    aggregate.add_argument("--stage-output", type=Path, required=True)
    candidate = subparsers.add_parser("validate-candidate")
    candidate.add_argument("--manifest", type=Path, required=True)
    candidate.add_argument("--expected-sha256", required=True)
    candidate.add_argument("--config-sha256", required=True)
    candidate.add_argument("--dataset-manifest-sha256", required=True)
    args = parser.parse_args()

    if args.command == "make-leg":
        document = build_leg_evidence(
            args.benchmark,
            args.monitor,
            label=args.label,
            order_index=args.order_index,
            model_revision=args.model_revision,
            engine_sha256=args.engine_sha256,
            planner_ready_seconds=args.planner_ready_seconds,
            kiosk_active=args.kiosk_active,
            projector_http_passed=args.projector_http_passed,
            live_scene_passed=args.live_scene_passed,
        )
        _write_new(args.output, document)
        return
    if args.command == "validate-hidden":
        hidden_binding = validate_hidden_report(
            args.report,
            expected_sha256=args.expected_sha256,
            candidate_revision=args.candidate_revision,
        )
        print(json.dumps(hidden_binding, indent=2, sort_keys=True))
        return
    if args.command == "validate-candidate":
        lineage = validate_candidate_manifest(
            args.manifest,
            expected_sha256=args.expected_sha256,
            config_sha256=args.config_sha256,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
        )
        print(json.dumps(lineage, indent=2, sort_keys=True))
        return
    if len(args.leg) != 4:
        parser.error("aggregate requires exactly four --leg paths in ABBA order")
    candidate_manifest = _load_json(args.candidate_manifest, label="candidate manifest")
    if sha256_path(args.candidate_manifest) != args.candidate_manifest_sha256:
        raise ValueError("candidate manifest checksum mismatch")
    candidate_revision = candidate_manifest.get("model_revision")
    if not isinstance(candidate_revision, str) or not MODEL_REVISION.fullmatch(candidate_revision):
        raise ValueError("candidate manifest has an invalid model revision")
    hidden_binding = validate_hidden_report(
        args.hidden_report,
        expected_sha256=args.hidden_report_sha256,
        candidate_revision=candidate_revision,
    )
    accepted_revision = f"sha256:{args.accepted_engine_sha256}"
    expected = (
        ("baseline", accepted_revision, args.accepted_engine_sha256),
        ("candidate", candidate_revision, candidate_revision.removeprefix("sha256:")),
        ("candidate", candidate_revision, candidate_revision.removeprefix("sha256:")),
        ("baseline", accepted_revision, args.accepted_engine_sha256),
    )
    legs = [
        validate_leg(
            path,
            expected_label=label,
            expected_index=index,
            expected_revision=revision,
            expected_engine_sha256=engine,
        )
        for index, (path, (label, revision, engine)) in enumerate(
            zip(args.leg, expected, strict=True)
        )
    ]
    restoration = _load_json(args.restoration, label="restoration evidence")
    runtime_evidence, stage_evidence = build_shadow_evidence(
        legs,
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        accepted_engine_sha256=args.accepted_engine_sha256,
        hidden_binding=hidden_binding,
        restoration=restoration,
        run_id=args.run_id,
        config_sha256=args.config_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        int4_export_sha256=args.int4_export_sha256,
    )
    runtime_sha256 = _write_new(args.runtime_output, runtime_evidence)
    evidence_sha256 = stage_evidence["evidence_sha256"]
    if not isinstance(evidence_sha256, dict) or runtime_sha256 != evidence_sha256.get("runtime"):
        raise RuntimeError("runtime evidence changed before publication")
    stage_sha256 = _write_new(args.stage_output, stage_evidence)
    print(
        json.dumps(
            {
                "runtime_output": str(args.runtime_output),
                "runtime_sha256": runtime_sha256,
                "stage_output": str(args.stage_output),
                "stage_sha256": stage_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
