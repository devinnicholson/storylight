#!/usr/bin/env python3
"""Preflight, execute once, or recompute the fixed Klein transport latency comparison."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookforge.klein_latency_client import (  # noqa: E402
    METRIC_TIMES,
    TIMINGS,
    LatencyClient,
    validate_payload,
)
from deploy import klein_latency_protocol as protocol  # noqa: E402
from scripts import render_fidelity_display as render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BATCH = ROOT / "benchmarks/scene-routing-2026-09-04/visual-batch.json"
ORIGINAL_JOURNAL = ROOT / "benchmarks/scene-routing-2026-09-04/render-journal.jsonl"
ORIGINAL_JOURNAL_SHA256 = "dc13a7ad79b29da69f05a3438c568e0b2c7aa55d000a857ce5a6807924bf1738"
PAIRS = (
    "candidate-page-01-still",
    "candidate-page-02-still",
    "candidate-page-04-still",
    "accepted-page-03",
    "accepted-page-05",
    "accepted-page-06",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, maximum=131072) -> dict:
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("bounded input differs")
    return protocol.decode_json(path.read_bytes())


def load_inputs(manifest_path: Path, authorization_path: Path):
    manifest = read_json(manifest_path)
    authorization = read_json(authorization_path)
    required = {
        "schema_version",
        "experiment_id",
        "original_batch_sha256",
        "expected_identity",
        "instrumentation_sha256",
        "deployment_sha256",
        "runtime_sha256",
        "operations",
        "expires_at",
        "protocol_sha256",
        "http_server_sha256",
    }
    if (
        set(manifest) != required
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("manifest fields differ")
    experiment = manifest["experiment_id"]
    if (
        not isinstance(experiment, str)
        or not 1 <= len(experiment) <= 80
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in experiment
        )
    ):
        raise ValueError("experiment identity differs")
    if type(manifest["expires_at"]) is not int or manifest["expires_at"] <= 0:
        raise ValueError("manifest expiry differs")
    for key, path in {
        "instrumentation_sha256": ROOT / "deploy/klein_latency_runtime.py",
        "deployment_sha256": ROOT / "deploy/modal_klein_latency.py",
        "runtime_sha256": ROOT / "deploy/klein_scene_runtime.py",
        "protocol_sha256": ROOT / "deploy/klein_latency_protocol.py",
        "http_server_sha256": ROOT / "deploy/klein_latency_http.py",
    }.items():
        if manifest[key] != file_hash(path):
            raise ValueError("implementation proof differs")
    batch = render.read_batch(ORIGINAL_BATCH, manifest["original_batch_sha256"])
    if (
        manifest["expected_identity"] != render.expected_identity()
        or file_hash(ORIGINAL_JOURNAL) != ORIGINAL_JOURNAL_SHA256
    ):
        raise ValueError("original renderer proof differs")
    journal = [protocol.decode_json(line) for line in ORIGINAL_JOURNAL.read_bytes().splitlines()]
    token_counts = journal[0]["tokens"]["token_counts"]
    rows = {row.id: row for row in batch.requests}
    schedule = [(transport, None) for transport in ("sdk", "http")]
    for index, pair in enumerate(PAIRS):
        schedule.extend(
            (transport, pair)
            for transport in (("sdk", "http") if index % 2 == 0 else ("http", "sdk"))
        )
    expected = []
    for transport, pair in schedule:
        request = {
            "request_id": hashlib.sha256(
                f"{experiment}:{transport}:{pair or 'warmup'}".encode()
            ).hexdigest()[:32],
            "operation": "render" if pair else "prewarm",
            "prompt": rows[pair].prompt if pair else "",
            "seed": rows[pair].seed if pair else 0,
        }
        protocol.validate_request(request)
        expected.append(
            {
                "transport": transport,
                "request": request,
                "pair_id": pair,
                "expected_bucket": (128 if token_counts[pair] <= 128 else 256) if pair else None,
            }
        )
    if manifest["operations"] != expected:
        raise ValueError("manifest is not the fixed fourteen-operation schedule")
    if (
        set(authorization)
        != {
            "schema_version",
            "manifest_sha256",
            "reservation_id",
            "reserved_usd",
            "maximum_operations",
            "ledger_sha256",
        }
        or type(authorization["schema_version"]) is not int
        or authorization["schema_version"] != 1
        or authorization["manifest_sha256"] != file_hash(manifest_path)
        or type(authorization["maximum_operations"]) is not int
        or authorization["maximum_operations"] != 14
        or type(authorization["reserved_usd"]) not in (int, float)
        or authorization["reserved_usd"] != 4.58
        or not isinstance(authorization["reservation_id"], str)
        or not 1 <= len(authorization["reservation_id"]) <= 200
        or not isinstance(authorization["ledger_sha256"], str)
        or len(authorization["ledger_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in authorization["ledger_sha256"])
    ):
        raise ValueError("canonical reservation proof differs")
    return manifest, authorization


def write_exclusive(path: Path, content: bytes) -> None:
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append(path: Path, event: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def header(args, manifest, authorization):
    return {
        "kind": "header",
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": file_hash(args.manifest),
        "authorization_sha256": file_hash(args.authorization),
        "reservation_sha256": protocol.digest(authorization["reservation_id"]),
        "ledger_sha256": authorization["ledger_sha256"],
        "reserved_usd": authorization["reserved_usd"],
        "harness_sha256": file_hash(Path(__file__)),
        "client_sha256": file_hash(ROOT / "src/bookforge/klein_latency_client.py"),
        "maximum_operations": 14,
        "automatic_retries": 0,
        "physical_display_measured": False,
        "billing_preparation": "centrally reserved before Jetson; excluded from per-image timings",
        "submission_timing": "SDK call-handle creation; HTTP request through response headers",
    }


def claim(args, evidence_header):
    path = args.authorization
    info, parent = path.stat(), path.parent.stat()
    if (
        path.resolve() != path.absolute()
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise ValueError("authorization must be in an owner-only directory")
    directory = Path.home() / ".local/state/bookforge/renderer-latency-attempts"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.resolve() != directory.absolute() or directory.stat().st_uid != os.getuid():
        raise ValueError("attempt directory differs from private operator home")
    directory.chmod(0o700)
    marker = directory / f"{evidence_header['experiment_id']}.json"
    write_exclusive(
        marker,
        json.dumps(
            {
                **evidence_header,
                "kind": "attempt",
                "output_path_sha256": protocol.digest(str(args.output.resolve())),
            },
            sort_keys=True,
        ).encode(),
    )


async def execute(args, manifest, evidence_header):
    journal = args.output / "journal.jsonl"
    client = LatencyClient(manifest)
    client.headers()
    write_exclusive(journal, (json.dumps(evidence_header, sort_keys=True) + "\n").encode())
    try:
        for ordinal, operation in enumerate(manifest["operations"]):
            if time.time() >= manifest["expires_at"]:
                raise ValueError("experiment authorization expired")
            append(
                journal,
                {"kind": "start", "ordinal": ordinal, "request_sha256": protocol.digest(operation)},
            )
            begun = time.perf_counter()
            try:
                payload, timings = await client.invoke(
                    operation["transport"], operation["request"], operation["expected_bucket"]
                )
                storage = time.perf_counter()
                directory = args.output / f"operation-{ordinal:02}"
                directory.mkdir(mode=0o700)
                for role in ("master", "depth"):
                    if payload[role]:
                        write_exclusive(directory / f"{role}.jpg", payload[role])
                timings["storage_seconds"] = time.perf_counter() - storage
                timings["total_artifact_ready_seconds"] = time.perf_counter() - begun
                metadata = {
                    key: payload[key]
                    for key in (
                        "request_id",
                        "identity",
                        "deployment_sha256",
                        "location",
                        "model_load_seconds",
                        "instrumentation_sha256",
                        "server_seconds",
                        "startup_seconds",
                        "cache_setup_seconds",
                    )
                }
                if operation["pair_id"] is None:
                    metadata.update(
                        {
                            key: payload[key]
                            for key in ("warmup_seconds", "renders", "instrumentation_sha256")
                        }
                    )
                else:
                    metadata.update({key: payload[key] for key in ("metrics", "warm_state")})
                append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "ok",
                        "timings": timings,
                        "payload": metadata,
                    },
                )
            except BaseException:
                append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "failed",
                        "code": client.last_failure or "local_operation_failed",
                        "timings": client.last_timings or {},
                    },
                )
                raise
        append(journal, {"kind": "complete", "operations": 14})
    finally:
        cleaned = await client.cleanup()
        append(
            journal,
            {
                "kind": "cleanup",
                "known_calls_cancelled": cleaned,
                "external_app_stop_required": not cleaned,
            },
        )


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "p50": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
        "p95": values[math.ceil(len(values) * 0.95) - 1],
        "max": values[-1],
    }


def aggregate(args, manifest, evidence_header):
    journal = args.output / "journal.jsonl"
    if journal.stat().st_size > 524288:
        raise ValueError("journal exceeds bound")
    events = [protocol.decode_json(line) for line in journal.read_bytes().splitlines()]
    if not events or events[0] != evidence_header:
        raise ValueError("journal provenance differs")
    results, starts = [], []
    complete, cleanup = False, None
    for event in events[1:]:
        if cleanup is not None:
            raise ValueError("journal continues after cleanup")
        if event["kind"] == "start":
            index = len(starts)
            if (
                index >= 14
                or complete
                or (results and results[-1]["status"] != "ok")
                or len(results) != index
            ):
                raise ValueError("journal request chronology differs")
            if event != {
                "kind": "start",
                "ordinal": index,
                "request_sha256": protocol.digest(manifest["operations"][index]),
            }:
                raise ValueError("journal request differs")
            starts.append(event)
        elif event["kind"] == "result":
            index = len(results)
            if (
                len(starts) != index + 1
                or event.get("ordinal") != index
                or event.get("status") not in {"ok", "failed"}
            ):
                raise ValueError("journal result chronology differs")
            if event["status"] == "ok":
                operation = manifest["operations"][index]
                payload = dict(event["payload"])
                for role in ("master", "depth"):
                    path = args.output / f"operation-{index:02}/{role}.jpg"
                    payload[role] = path.read_bytes() if operation["pair_id"] else b""
                validate_payload(
                    payload, operation["request"], manifest, operation["expected_bucket"]
                )
                if set(event["timings"]) != set(TIMINGS) or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in event["timings"].values()
                ):
                    raise ValueError("journal timings differ")
            results.append(event)
        elif event == {"kind": "complete", "operations": 14}:
            if complete or len(results) != 14 or any(row["status"] != "ok" for row in results):
                raise ValueError("journal completion differs")
            complete = True
        elif event.get("kind") == "cleanup" and cleanup is None:
            if (
                set(event) != {"kind", "known_calls_cancelled", "external_app_stop_required"}
                or any(
                    type(event[name]) is not bool
                    for name in ("known_calls_cancelled", "external_app_stop_required")
                )
                or event["known_calls_cancelled"] == event["external_app_stop_required"]
            ):
                raise ValueError("journal cleanup status differs")
            cleanup = event
        else:
            raise ValueError("unknown journal event")
    measured = [
        (operation, result)
        for operation, result in zip(manifest["operations"], results, strict=False)
        if operation["pair_id"] and result["status"] == "ok"
    ]
    by_transport = {
        transport: distribution(
            [
                result["timings"]["total_artifact_ready_seconds"]
                for operation, result in measured
                if operation["transport"] == transport
            ]
        )
        for transport in ("sdk", "http")
    }
    by_bucket = {
        str(bucket): {
            transport: distribution(
                [
                    result["timings"]["total_artifact_ready_seconds"]
                    for operation, result in measured
                    if operation["transport"] == transport
                    and operation["expected_bucket"] == bucket
                ]
            )
            for transport in ("sdk", "http")
        }
        for bucket in (128, 256)
    }
    pairs = []
    for pair in PAIRS:
        selected = {
            operation["transport"]: result
            for operation, result in measured
            if operation["pair_id"] == pair
        }
        if set(selected) == {"sdk", "http"}:
            pairs.append(
                {
                    "pair_id": pair,
                    "master_equal": selected["sdk"]["payload"]["metrics"]["master_sha256"]
                    == selected["http"]["payload"]["metrics"]["master_sha256"],
                    "depth_equal": selected["sdk"]["payload"]["metrics"]["depth_sha256"]
                    == selected["http"]["payload"]["metrics"]["depth_sha256"],
                }
            )
    locations = [result["payload"]["location"] for _, result in measured]
    comparable = bool(locations) and all(
        location["cloud"] not in {None, "unknown"}
        and location["compute_region"] not in {None, "unknown"}
        and (location["cloud"], location["compute_region"])
        == (locations[0]["cloud"], locations[0]["compute_region"])
        for location in locations
    )
    warm = bool(measured) and all(
        result["payload"]["metrics"]["bucket_was_warm"] for _, result in measured
    )
    sdk, http = by_transport["sdk"], by_transport["http"]
    container_stable = all(
        len(
            {
                result["payload"]["location"]["container_sha256"]
                for operation, result in zip(manifest["operations"], results, strict=False)
                if operation["transport"] == transport and result["status"] == "ok"
            }
        )
        == 1
        for transport in ("sdk", "http")
    ) and all(
        location["container_sha256"] != hashlib.sha256(b"unavailable").hexdigest()
        for location in locations
    )
    improvement = 1 - http["p50"] / sdk["p50"] if sdk["p50"] and http["p50"] is not None else None
    passed = (
        complete
        and len(measured) == 12
        and len(pairs) == 6
        and comparable
        and container_stable
        and cleanup
        == {"kind": "cleanup", "known_calls_cancelled": True, "external_app_stop_required": False}
        and warm
        and all(pair["master_equal"] and pair["depth_equal"] for pair in pairs)
        and improvement is not None
        and improvement >= 0.25
        and http["p95"] <= sdk["p95"]
        and http["max"] <= sdk["max"]
    )
    return {
        "schema_version": 1,
        "kind": "klein-latency-summary",
        "header": evidence_header,
        "journal_sha256": file_hash(journal),
        "started": len(starts),
        "results": len(results),
        "complete": complete,
        "failures": sum(result["status"] != "ok" for result in results),
        "unresolved_requests": len(starts) - len(results),
        "artifact_ready_seconds": by_transport,
        "by_bucket": by_bucket,
        "pairs": pairs,
        "compute_location_comparable": comparable,
        "container_stable_through_warmup_and_measurement": container_stable,
        "client_stage_seconds": {
            transport: {
                name: distribution(
                    [
                        result["timings"][name]
                        for operation, result in measured
                        if operation["transport"] == transport
                    ]
                )
                for name in TIMINGS
            }
            for transport in ("sdk", "http")
        },
        "server_stages": {
            transport: {
                name: distribution(
                    [
                        result["payload"]["metrics"][name]
                        for operation, result in measured
                        if operation["transport"] == transport
                        and result["payload"]["metrics"][name] is not None
                    ]
                )
                for name in (*METRIC_TIMES, "cuda_image_seconds")
            }
            for transport in ("sdk", "http")
        },
        "all_measured_buckets_warm": warm,
        "http_p50_improvement_fraction": improvement,
        "decision": "pass" if passed else "reject",
        "cleanup": cleanup,
        "external_app_shutdown_verified": False,
        "physical_display_measured": False,
        "visual_quality_measured": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "authorization", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, authorization = load_inputs(args.manifest, args.authorization)
        if (args.execute and args.authorization_sha256 is None) or (
            args.authorization_sha256 is not None
            and args.authorization_sha256 != file_hash(args.authorization)
        ):
            raise ValueError("operator-issued authorization proof differs")
        evidence_header = header(args, manifest, authorization)
        if args.aggregate_only:
            summary = aggregate(args, manifest, evidence_header)
            write_exclusive(
                args.output / "recomputed-summary.json",
                (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
            )
            return 0
        if (
            args.output.exists()
            or args.output.is_symlink()
            or not args.output.parent.is_dir()
            or not time.time() < manifest["expires_at"] <= time.time() + 7200
        ):
            raise ValueError("fresh output and unexpired manifest required")
        if args.execute:
            if (
                sys.platform != "linux"
                or platform.machine() != "aarch64"
                or not Path("/etc/nv_tegra_release").is_file()
            ):
                raise ValueError("execute requires the Jetson")
            LatencyClient(manifest).headers()
            claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        if not args.execute:
            write_exclusive(
                args.output / "preflight.json",
                (
                    json.dumps(
                        {**evidence_header, "mode": "preflight", "generation_calls": 0},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
            return 0
        try:
            asyncio.run(execute(args, manifest, evidence_header))
        finally:
            summary = aggregate(args, manifest, evidence_header)
            write_exclusive(
                args.output / "summary.json",
                (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
            )
        return 0
    except BaseException:
        print("latency experiment stopped; inspect sanitized local journal and cleanup status")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
