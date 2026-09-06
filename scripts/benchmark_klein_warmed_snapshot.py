#!/usr/bin/env python3
"""Verify three warmed-snapshot calls; correctness, latency and platform audit stay separate."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts import benchmark_klein_import_snapshot as probe  # noqa: E402

cold, legacy, preparation = probe.cold, probe.legacy, probe.preparation
require = cold.require
EXPERIMENT = "klein-warmed-snapshot-20260905-a"
MAX_OPERATIONS, MAX_CAPTURES, COST_CEILING_USD = 3, 1, 1.15
BASE_SNAPSHOT = {
    "capture_id",
    "capture_container_sha256",
    "activation_id",
    "imports_seconds",
    "versions",
    "cuda_available",
}


def read_manifest(path):
    value = legacy.protocol.decode_json(preparation.read_code(path))
    require(
        set(value)
        == {
            "schema_version",
            "experiment_id",
            "status",
            "expires_at",
            "runtime_sha256",
            "deployment_sha256",
            "client_sha256",
            "image_id",
            "cache_id",
            "expected_identity",
            "operations",
            "cases",
            "max_operations",
            "max_captures",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(
        value["experiment_id"] == EXPERIMENT
        and value["image_id"] == cold.IMAGE_ID
        and value["cache_id"] == cold.CACHE_ID
    )
    require(type(value["max_operations"]) is int and value["max_operations"] == MAX_OPERATIONS)
    require(type(value["max_captures"]) is int and value["max_captures"] == MAX_CAPTURES)
    require(value["status"] in ("draft", "authorized"))
    require(
        value["expires_at"] is None
        if value["status"] == "draft"
        else type(value["expires_at"]) is int and value["expires_at"] > 0
    )
    identity, cases = cold.frozen_cases()
    require(preparation.encoded(value["expected_identity"]) == preparation.encoded(identity))
    require(preparation.encoded(value["cases"]) == preparation.encoded(cases))
    for field, source in {
        "runtime_sha256": ROOT / "deploy/klein_scene_runtime.py",
        "deployment_sha256": ROOT / "deploy/modal_klein_warmed_snapshot.py",
        "client_sha256": Path(__file__),
    }.items():
        require(value[field] == legacy.file_hash(source))
    require(isinstance(value["operations"], list) and len(value["operations"]) == MAX_OPERATIONS)
    ids = set()
    for ordinal, operation in enumerate(value["operations"]):
        require(
            isinstance(operation, dict) and set(operation) == {"ordinal", "request_id", "variant"}
        )
        require(
            type(operation["ordinal"]) is int
            and operation["ordinal"] == ordinal
            and operation["variant"] == "warmed_snapshot"
        )
        require(
            isinstance(operation["request_id"], str)
            and re.fullmatch(r"[a-f0-9]{32}", operation["request_id"]) is not None
            and operation["request_id"] not in ids
        )
        ids.add(operation["request_id"])
    return value


def read_authorization(path, proof, manifest_path):
    raw = preparation.read_code(path)
    require(hashlib.sha256(raw).hexdigest() == proof and path.resolve() == path.absolute())
    require(path.stat().st_uid == os.getuid() and stat.S_IMODE(path.stat().st_mode) == 0o600)
    require(
        path.parent.stat().st_uid == os.getuid()
        and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    )
    value = legacy.protocol.decode_json(raw)
    require(
        set(value)
        == {
            "schema_version",
            "manifest_sha256",
            "reservation_id",
            "reserved_usd",
            "maximum_operations",
            "ledger_sha256",
        }
    )
    require(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["manifest_sha256"] == legacy.file_hash(manifest_path)
    )
    require(
        type(value["reserved_usd"]) in (int, float) and value["reserved_usd"] == COST_CEILING_USD
    )
    require(
        type(value["maximum_operations"]) is int and value["maximum_operations"] == MAX_OPERATIONS
    )
    require(
        isinstance(value["reservation_id"], str)
        and 1 <= len(value["reservation_id"]) <= 200
        and cold.is_hash(value["ledger_sha256"])
    )
    return value


def header(args, manifest, authorization):
    result = cold.header(args, manifest, authorization)
    result["support_sha256"].update(
        {
            str(Path(module.__file__).relative_to(ROOT)): legacy.file_hash(Path(module.__file__))
            for module in (cold, probe)
        }
    )
    return {
        **result,
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "maximum_operations": MAX_OPERATIONS,
        "maximum_captures": MAX_CAPTURES,
        "qualification": (
            "two later warmed restores; correctness and latency separate; platform audit required"
        ),
    }


def validate_payload(payload, operation, manifest):
    require(
        isinstance(payload, dict)
        and set(payload)
        == {"request_id", "variant", "identity", "location", "stages", "samples", "snapshot"}
    )
    location, snapshot = payload["location"], payload["snapshot"]
    require(
        isinstance(location, dict)
        and set(location) == {"cloud", "compute_region", "container_sha256"}
    )
    require(location["cloud"] in {"CLOUD_PROVIDER_AWS", "CLOUD_PROVIDER_GCP", "CLOUD_PROVIDER_OCI"})
    require(
        isinstance(location["compute_region"], str)
        and re.fullmatch(r"us-[a-z0-9-]{1,48}", location["compute_region"]) is not None
    )
    require(
        isinstance(snapshot, dict)
        and set(snapshot) == BASE_SNAPSHOT | {"initialization", "warmups"}
    )
    # The actual US placement is checked above; reuse the existing byte/metric validator's
    # fixed-region view without changing the response recorded in evidence.
    view = {
        **payload,
        "location": {**location, "cloud": "CLOUD_PROVIDER_AWS", "compute_region": "us-east-1"},
        "snapshot": {key: snapshot[key] for key in BASE_SNAPSHOT},
    }
    probe.validate_payload(view, operation, manifest)
    initialization = snapshot["initialization"]
    require(
        isinstance(initialization, dict)
        and set(initialization) == {"model_load_seconds", "cache_setup_seconds", "warmup_seconds"}
    )
    require(all(cold.finite(value) for value in initialization.values()))
    require(payload["stages"]["imports_seconds"] == snapshot["imports_seconds"])
    require(
        all(
            payload["stages"][key] == initialization[key]
            for key in ("model_load_seconds", "cache_setup_seconds")
        )
    )
    require(isinstance(snapshot["warmups"], list) and len(snapshot["warmups"]) == 4)
    for index, row in enumerate(snapshot["warmups"]):
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "repeat",
                "case_index",
                "sequence_bucket",
                "total_seconds",
                "master_sha256",
                "depth_sha256",
                "hashes_match",
            }
        )
        require(
            type(row["repeat"]) is int
            and row["repeat"] == index // 2
            and type(row["case_index"]) is int
            and row["case_index"] == index % 2
        )
        case = manifest["cases"][index % 2]
        require(
            type(row["sequence_bucket"]) is int
            and row["sequence_bucket"] == case["expected_bucket"]
        )
        require(cold.finite(row["total_seconds"]) and row["hashes_match"] is True)
        require(
            all(row[f"{role}_sha256"] == case[f"{role}_sha256"] for role in ("master", "depth"))
        )


def qualification(payloads, manifest):
    state = probe.qualification(payloads, manifest)
    state["capture_limit_exceeded"] = state["captures_observed"] > MAX_CAPTURES
    return state


def rejected(state):
    return (
        state["capture_limit_exceeded"]
        or state["activation_reused"]
        or state["container_reused"]
        or not state["all_reference_hashes_match"]
    )


class WarmedClient(probe.SnapshotClient):
    async def cycle(self, operation):
        started = time.perf_counter()
        lookup = 0.0
        self.last_failure = None
        self.last_timings = dict.fromkeys(cold.TIMINGS, 0.0)
        try:
            remaining = self.deadline - time.time()
            require(remaining > 0)
            async with asyncio.timeout(min(210, remaining)):
                if self.method is None:
                    if self.modal is None:
                        self.modal = importlib.import_module("modal")
                    cls = self.modal.Cls.from_name(
                        "bookforge-klein-warmed-snapshot", "WarmedSnapshot"
                    )
                    await asyncio.wait_for(cls.hydrate.aio(), min(10, remaining))
                    self.method = cls().cycle
                    lookup = time.perf_counter() - started
                return await super().cycle(operation)
        except BaseException:
            self.last_failure = self.last_failure or "local_operation_failed"
            raise
        finally:
            self.last_timings["lookup_seconds"] += lookup
            self.last_timings["total_artifact_ready_seconds"] = time.perf_counter() - started


async def execute(args, manifest, evidence_header, client):
    journal = args.output / "journal.jsonl"
    legacy.write_exclusive(journal, (json.dumps(evidence_header, sort_keys=True) + "\n").encode())
    payloads = []
    try:
        for ordinal, operation in enumerate(manifest["operations"]):
            if time.time() >= min(args.deadline_unix, manifest["expires_at"]):
                legacy.append(journal, {"kind": "stop", "reason": "deadline"})
                return
            legacy.append(
                journal,
                {
                    "kind": "start",
                    "ordinal": ordinal,
                    "request_sha256": legacy.protocol.digest(operation),
                },
            )
            started = time.perf_counter()
            try:
                payload, timings = await client.cycle(operation)
                begun = time.perf_counter()
                validate_payload(payload, operation, manifest)
                timings["validation_seconds"] = time.perf_counter() - begun
                begun = time.perf_counter()
                (args.output / f"operation-{ordinal:02}").mkdir(mode=0o700)
                for index, sample in enumerate(payload["samples"]):
                    for role in ("master", "depth"):
                        legacy.write_exclusive(
                            cold.artifact_path(args.output, ordinal, index, role), sample[role]
                        )
                timings["storage_seconds"] = time.perf_counter() - begun
                timings["total_artifact_ready_seconds"] = time.perf_counter() - started
                safe = {
                    **payload,
                    "samples": [
                        {
                            key: value
                            for key, value in sample.items()
                            if key not in {"master", "depth"}
                        }
                        for sample in payload["samples"]
                    ],
                }
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "ok",
                        "timings": timings,
                        "payload": safe,
                    },
                )
                payloads.append(safe)
            except BaseException:
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "failed",
                        "code": client.last_failure
                        if client.last_failure in probe.FAILURES
                        else "local_operation_failed",
                        "timings": client.last_timings or {},
                    },
                )
                raise
            state = qualification(payloads, manifest)
            if rejected(state) or state["observed_restores"] == 2:
                legacy.append(
                    journal,
                    {
                        "kind": "stop",
                        "reason": "evidence_rejected" if rejected(state) else "two_restores",
                    },
                )
                return
        legacy.append(journal, {"kind": "stop", "reason": "operation_limit"})
    finally:
        cleaned = await client.cleanup()
        legacy.append(
            journal,
            {
                "kind": "cleanup",
                "known_calls_cancelled": cleaned,
                "external_app_stop_required": not cleaned,
            },
        )


def aggregate(args, manifest, evidence_header):
    journal = args.output / "journal.jsonl"
    require(journal.is_file() and not journal.is_symlink() and journal.stat().st_size <= 1_000_000)
    events = [legacy.protocol.decode_json(line) for line in journal.read_bytes().splitlines()]
    require(bool(events) and preparation.encoded(events[0]) == preparation.encoded(evidence_header))
    started, results, pending, stop, cleanup = 0, [], None, None, None
    failed = False
    for event in events[1:]:
        require(cleanup is None)
        kind = event.get("kind")
        if kind == "cleanup":
            require(set(event) == {"kind", "known_calls_cancelled", "external_app_stop_required"})
            require(
                type(event["known_calls_cancelled"]) is bool
                and type(event["external_app_stop_required"]) is bool
                and event["known_calls_cancelled"] is not event["external_app_stop_required"]
            )
            cleanup = event
            continue
        require(stop is None)
        if kind == "start":
            require(not failed and pending is None and started < MAX_OPERATIONS)
            require(
                set(event) == {"kind", "ordinal", "request_sha256"}
                and type(event["ordinal"]) is int
                and event["ordinal"] == started
            )
            require(
                event["request_sha256"] == legacy.protocol.digest(manifest["operations"][started])
            )
            before = qualification([row["payload"] for row in results], manifest)
            require(not results or (not rejected(before) and before["observed_restores"] < 2))
            pending = started
            started += 1
        elif kind == "result":
            require(
                pending is not None
                and type(event.get("ordinal")) is int
                and event["ordinal"] == pending
            )
            require(event.get("status") in ("ok", "failed"))
            require(
                set(event)
                == (
                    {"kind", "ordinal", "status", "timings", "payload"}
                    if event["status"] == "ok"
                    else {"kind", "ordinal", "status", "timings", "code"}
                )
            )
            require(
                isinstance(event["timings"], dict)
                and set(event["timings"]) in (set(cold.TIMINGS), set())
                and all(cold.finite(value) for value in event["timings"].values())
            )
            if event["status"] == "ok":
                require(set(event["timings"]) == set(cold.TIMINGS))
                payload = copy.deepcopy(event["payload"])
                require(isinstance(payload["samples"], list) and len(payload["samples"]) == 4)
                for index, sample in enumerate(payload["samples"]):
                    require(set(sample) == {"repeat", "case_index", "metrics"})
                    for role in ("master", "depth"):
                        path = cold.artifact_path(args.output, pending, index, role)
                        require(
                            path.resolve() == path.absolute()
                            and path.is_file()
                            and path.stat().st_size <= cold.MAX_ARTIFACT_BYTES
                        )
                        sample[role] = path.read_bytes()
                validate_payload(payload, manifest["operations"][pending], manifest)
            else:
                require(event["code"] in probe.FAILURES)
                failed = True
            results.append(event)
            pending = None
        elif kind == "stop":
            require(
                set(event) == {"kind", "reason"}
                and event["reason"]
                in ("deadline", "two_restores", "operation_limit", "evidence_rejected")
                and pending is None
                and not failed
            )
            stop = event["reason"]
        else:
            raise ValueError("warmed_snapshot_journal_invalid")
    successful = [row for row in results if row["status"] == "ok"]
    payloads = [row["payload"] for row in successful]
    state = qualification(payloads, manifest)
    if stop == "two_restores":
        require(state["observed_restores"] == 2)
    if stop == "operation_limit":
        require(started == MAX_OPERATIONS)
    if stop == "evidence_rejected":
        require(rejected(state))
    bad = (
        failed
        or pending is not None
        or (bool(payloads) and rejected(state))
        or cleanup is None
        or not cleanup["known_calls_cancelled"]
    )
    correct = not bad and stop == "two_restores" and state["observed_restores"] == 2
    restore_rows = []
    for index, row in enumerate(successful):
        if (
            qualification(payloads[: index + 1], manifest)["observed_restores"]
            > qualification(payloads[:index], manifest)["observed_restores"]
        ):
            restore_rows.append(row)
    first_buckets = [
        {
            "ordinal": row["ordinal"],
            "first_128_seconds": row["payload"]["samples"][0]["metrics"]["total_seconds"],
            "first_256_seconds": row["payload"]["samples"][1]["metrics"]["total_seconds"],
        }
        for row in restore_rows
    ]
    cycles = legacy.distribution(
        [row["timings"]["total_artifact_ready_seconds"] for row in restore_rows]
    )
    bucket_gate = correct and all(
        row[key] <= 2.5
        for row in first_buckets
        for key in ("first_128_seconds", "first_256_seconds")
    )
    client_gate = correct and cycles["p50"] is not None and cycles["p50"] <= 15
    return {
        "schema_version": 1,
        "kind": "klein-warmed-snapshot-summary",
        "header": evidence_header,
        "journal_sha256": legacy.file_hash(journal),
        "started": started,
        "results": len(results),
        "failures": sum(row["status"] == "failed" for row in results),
        "unresolved_requests": int(pending is not None),
        **state,
        "stop_reason": stop,
        "cleanup": cleanup,
        "correctness_qualified": correct,
        "first_bucket_latency_pass": bucket_gate,
        "restore_cycle_latency_pass": client_gate,
        "latency_pass": bucket_gate and client_gate,
        "restore_first_buckets": first_buckets,
        "restore_client_cycle_seconds": cycles,
        "decision": "pending_platform_audit" if correct else "reject" if bad else "inconclusive",
        "platform_fallback_audit_pending": True,
        "comparison_is_causal": False,
        "timing_scope": (
            "four-render client cycles; captured initialization is not additive to restored work"
        ),
        "physical_display_measured": False,
        "external_app_shutdown_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--deadline-unix", type=float)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        authorization = None
        if manifest["status"] == "authorized":
            require(args.authorization is not None and args.authorization_sha256 is not None)
            authorization = read_authorization(
                args.authorization, args.authorization_sha256, args.manifest
            )
        else:
            require(
                not args.run
                and not args.aggregate_only
                and args.authorization is None
                and args.authorization_sha256 is None
            )
        if args.aggregate_only:
            result = aggregate(args, manifest, header(args, manifest, authorization))
            legacy.write_exclusive(
                args.output / "recomputed-summary.json", preparation.encoded(result)
            )
            return 0
        require(
            not args.output.exists()
            and not args.output.is_symlink()
            and args.output.parent.is_dir()
        )
        if not args.run:
            args.output.mkdir(mode=0o700)
            legacy.write_exclusive(
                args.output / "preflight.json",
                preparation.encoded(
                    {
                        "schema_version": 1,
                        "kind": "klein-warmed-snapshot-preflight",
                        "manifest_sha256": legacy.file_hash(args.manifest),
                        "status": manifest["status"],
                        "maximum_operations": MAX_OPERATIONS,
                        "maximum_captures": MAX_CAPTURES,
                        "cost_ceiling_usd": COST_CEILING_USD,
                        "generation_calls": 0,
                    }
                ),
            )
            return 0
        require(cold.finite(args.deadline_unix) and 0 < args.deadline_unix - time.time() <= 240)
        require(time.time() < manifest["expires_at"] <= time.time() + 7200)
        evidence_header = header(args, manifest, authorization)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        client = WarmedClient(manifest, deadline=min(args.deadline_unix, manifest["expires_at"]))
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("warmed-snapshot probe stopped; inspect sanitized evidence and supervised cleanup")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
