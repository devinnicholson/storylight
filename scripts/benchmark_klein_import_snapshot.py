#!/usr/bin/env python3
"""Qualify two observed import-snapshot restores in at most five non-retrying calls."""

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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts import benchmark_klein_cold_start as cold  # noqa: E402

legacy, preparation = cold.legacy, cold.preparation
require = cold.require
EXPERIMENT = "klein-import-snapshot-20260905-a"
COST_CEILING_USD = 1.67
MAX_OPERATIONS, MAX_CAPTURES = 5, 3
COLD_SUMMARY = ROOT / "benchmarks/renderer-cold-start-2026-09-05/summary.json"
COLD_SUMMARY_SHA256 = "68e38672e7c75cefbe28012575af073d645ade688ad3bba8e656ea548ddff111"
FAILURES = cold.FAILURES | {"snapshot_drain_failed", "snapshot_deadline_exceeded"}


def read_manifest(path):
    require(legacy.file_hash(COLD_SUMMARY) == COLD_SUMMARY_SHA256)
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
    for field, path in {
        "runtime_sha256": ROOT / "deploy/klein_scene_runtime.py",
        "deployment_sha256": ROOT / "deploy/modal_klein_import_snapshot.py",
        "client_sha256": Path(__file__),
    }.items():
        require(value[field] == legacy.file_hash(path))
    require(isinstance(value["operations"], list) and len(value["operations"]) == MAX_OPERATIONS)
    ids = set()
    for ordinal, operation in enumerate(value["operations"]):
        require(set(operation) == {"ordinal", "request_id", "variant"})
        require(
            type(operation["ordinal"]) is int
            and operation["ordinal"] == ordinal
            and operation["variant"] == "snapshot"
        )
        require(
            isinstance(operation["request_id"], str)
            and re.fullmatch(r"[a-f0-9]{32}", operation["request_id"]) is not None
        )
        require(operation["request_id"] not in ids)
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
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["manifest_sha256"] == legacy.file_hash(manifest_path))
    require(
        type(value["reserved_usd"]) in (int, float) and value["reserved_usd"] == COST_CEILING_USD
    )
    require(
        type(value["maximum_operations"]) is int and value["maximum_operations"] == MAX_OPERATIONS
    )
    require(isinstance(value["reservation_id"], str) and 1 <= len(value["reservation_id"]) <= 200)
    require(cold.is_hash(value["ledger_sha256"]))
    return value


def header(args, manifest, authorization):
    result = cold.header(args, manifest, authorization)
    result["support_sha256"].update(
        {"scripts/benchmark_klein_cold_start.py": legacy.file_hash(Path(cold.__file__))}
    )
    return {
        **result,
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "maximum_operations": MAX_OPERATIONS,
        "maximum_captures": MAX_CAPTURES,
        "historical_cold_summary_sha256": COLD_SUMMARY_SHA256,
        "qualification": (
            "two restores of previously observed capture IDs into distinct later containers"
        ),
    }


def validate_payload(payload, operation, manifest):
    require(
        isinstance(payload, dict)
        and set(payload)
        == {"request_id", "variant", "identity", "location", "stages", "samples", "snapshot"}
    )
    cold.validate_payload(
        {key: value for key, value in payload.items() if key != "snapshot"}, operation, manifest
    )
    snapshot = payload["snapshot"]
    require(
        isinstance(snapshot, dict)
        and set(snapshot)
        == {
            "capture_id",
            "capture_container_sha256",
            "activation_id",
            "imports_seconds",
            "versions",
            "cuda_available",
        }
    )
    for field in ("capture_id", "activation_id"):
        require(
            isinstance(snapshot[field], str)
            and re.fullmatch(r"[a-f0-9]{32}", snapshot[field]) is not None
        )
    require(
        cold.is_hash(snapshot["capture_container_sha256"])
        and snapshot["capture_container_sha256"] != hashlib.sha256(b"unavailable").hexdigest()
    )
    require(cold.finite(snapshot["imports_seconds"]) and snapshot["cuda_available"] is True)
    require(
        preparation.encoded(snapshot["versions"])
        == preparation.encoded(
            {
                key: manifest["expected_identity"][key]
                for key in ("torch", "diffusers", "transformers", "triton")
            }
        )
    )


def qualification(payloads, manifest):
    captures, activations, containers = {}, set(), set()
    restored = 0
    repeated_activation = False
    repeated_container = False
    reference_match = True
    for payload in payloads:
        snapshot = payload["snapshot"]
        capture, activation = snapshot["capture_id"], snapshot["activation_id"]
        container = payload["location"]["container_sha256"]
        metadata = {key: value for key, value in snapshot.items() if key != "activation_id"}
        prior = captures.get(capture)
        require(prior is None or preparation.encoded(prior) == preparation.encoded(metadata))
        if activation in activations:
            repeated_activation = True
        if container in containers:
            repeated_container = True
        if (
            prior is not None
            and activation not in activations
            and container not in containers
            and container != snapshot["capture_container_sha256"]
        ):
            restored += 1
        captures[capture] = metadata
        activations.add(activation)
        containers.add(container)
        reference_match &= all(
            sample["metrics"][f"{role}_sha256"]
            == manifest["cases"][sample["case_index"]][f"{role}_sha256"]
            for sample in payload["samples"]
            for role in ("master", "depth")
        )
    return {
        "observed_restores": restored,
        "captures_observed": len(captures),
        "recapture_observed": len(captures) > 1,
        "recapture_count": max(0, len(captures) - 1),
        "capture_limit_exceeded": len(captures) > MAX_CAPTURES,
        "activation_reused": repeated_activation,
        "container_reused": repeated_container,
        "all_reference_hashes_match": bool(payloads) and reference_match,
    }


class SnapshotClient(cold.ColdClient):
    def __init__(self, manifest, *, deadline, modal_module=None):
        super().__init__(manifest, modal_module=modal_module)
        self.deadline = deadline
        self.method = None
        self.calls = 0

    async def cycle(self, operation):
        self.last_failure = None
        timings = dict.fromkeys(cold.TIMINGS, 0.0)
        self.last_timings = timings
        started = time.perf_counter()
        try:
            remaining = self.deadline - time.time()
            require(remaining > 0)
            async with asyncio.timeout(min(310, remaining)):
                begun = time.perf_counter()
                if self.method is None:
                    if self.modal is None:
                        self.modal = importlib.import_module("modal")
                    cls = self.modal.Cls.from_name(
                        "bookforge-klein-import-snapshot", "ImportSnapshot"
                    )
                    await asyncio.wait_for(cls.hydrate.aio(), min(10, remaining))
                    self.method = cls().cycle
                timings["lookup_seconds"] = time.perf_counter() - begun
                if self.calls:
                    begun = time.perf_counter()
                    drain_until = min(self.deadline, time.time() + 30)
                    while True:
                        left = drain_until - time.time()
                        if left <= 0:
                            self.last_failure = "snapshot_drain_failed"
                            raise TimeoutError
                        stats = await asyncio.wait_for(
                            self.method.get_current_stats.aio(), min(5, left)
                        )
                        if stats.num_total_runners == 0 and stats.backlog == 0:
                            break
                        await asyncio.sleep(min(1, left))
                    timings["readiness_seconds"] = time.perf_counter() - begun
                require(time.time() < self.deadline)

                async def spawn(*, request):
                    return await self.method.spawn.aio(request["request_id"])

                self.instance = SimpleNamespace(
                    invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn))
                )
                self.calls += 1
                return await self.sdk(operation, timings), timings
        except BaseException:
            self.last_failure = self.last_failure or "local_operation_failed"
            raise
        finally:
            timings["total_artifact_ready_seconds"] = time.perf_counter() - started


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
                        if client.last_failure in FAILURES
                        else "local_operation_failed",
                        "timings": client.last_timings or {},
                    },
                )
                raise
            state = qualification(payloads, manifest)
            if (
                state["capture_limit_exceeded"]
                or state["activation_reused"]
                or state["container_reused"]
                or not state["all_reference_hashes_match"]
            ):
                legacy.append(journal, {"kind": "stop", "reason": "evidence_rejected"})
                return
            if state["observed_restores"] >= 2:
                legacy.append(journal, {"kind": "stop", "reason": "two_restores"})
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
            before = qualification(
                [row["payload"] for row in results if row["status"] == "ok"], manifest
            )
            require(
                before["observed_restores"] < 2
                and not before["capture_limit_exceeded"]
                and not before["activation_reused"]
                and not before["container_reused"]
            )
            require(not results or before["all_reference_hashes_match"])
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
            )
            require(all(cold.finite(value) for value in event["timings"].values()))
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
                require(event["code"] in FAILURES)
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
            raise ValueError("snapshot_journal_invalid")
    payloads = [row["payload"] for row in results if row["status"] == "ok"]
    state = qualification(payloads, manifest)
    if stop == "two_restores":
        require(state["observed_restores"] == 2)
    if stop == "operation_limit":
        require(started == MAX_OPERATIONS)
    if stop == "evidence_rejected":
        require(
            state["capture_limit_exceeded"]
            or state["activation_reused"]
            or state["container_reused"]
            or not state["all_reference_hashes_match"]
        )
    clean = cleanup is not None and cleanup["known_calls_cancelled"]
    rejected = (
        failed
        or pending is not None
        or state["capture_limit_exceeded"]
        or state["activation_reused"]
        or state["container_reused"]
        or (bool(payloads) and not state["all_reference_hashes_match"])
        or not clean
    )
    qualified = not rejected and stop == "two_restores" and state["observed_restores"] == 2
    metrics = {
        key: legacy.distribution([payload["stages"][key] for payload in payloads])
        for key in cold.STAGES
    }
    metrics["client_cycle_seconds"] = legacy.distribution(
        [row["timings"]["total_artifact_ready_seconds"] for row in results if row["status"] == "ok"]
    )
    return {
        "schema_version": 1,
        "kind": "klein-import-snapshot-summary",
        "header": evidence_header,
        "journal_sha256": legacy.file_hash(journal),
        "started": started,
        "results": len(results),
        "failures": sum(row["status"] == "failed" for row in results),
        "unresolved_requests": int(pending is not None),
        **state,
        "stop_reason": stop,
        "cleanup": cleanup,
        "metrics": metrics,
        "decision": "qualified_for_comparison"
        if qualified
        else "reject"
        if rejected
        else "inconclusive",
        "comparison_is_causal": False,
        "provider_restore_logs_reviewed": False,
        "platform_fallback_audit_pending": True,
        "worker_timing_scope": "postrestore four-render work; captured imports are not additive",
        "historical_cold_cycle_median_seconds": 43.53517280006781,
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
                        "kind": "klein-import-snapshot-preflight",
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
        require(cold.finite(args.deadline_unix) and 0 < args.deadline_unix - time.time() <= 480)
        require(time.time() < manifest["expires_at"] <= time.time() + 7200)
        evidence_header = header(args, manifest, authorization)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        client = SnapshotClient(manifest, deadline=min(args.deadline_unix, manifest["expires_at"]))
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("import-snapshot probe stopped; inspect sanitized evidence and supervised cleanup")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
