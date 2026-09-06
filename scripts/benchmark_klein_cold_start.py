#!/usr/bin/env python3
"""Bounded six-container cold-start comparison; preflight and aggregation make no cloud calls."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from bookforge.finite_modal_provider import _jpeg_dimensions  # noqa: E402
from bookforge.klein_latency_client import TIMINGS, LatencyClient, finite  # noqa: E402
from scripts import benchmark_klein_latency as legacy  # noqa: E402
from scripts import prepare_klein_region_comparison as preparation  # noqa: E402

SOURCE_MANIFEST = ROOT / "benchmarks/renderer-region-2026-09-05-c/manifest.json"
SOURCE_JOURNAL = ROOT / "benchmarks/renderer-region-2026-09-05-c/journal.jsonl"
SOURCE_MANIFEST_SHA256 = "d5b7e8da3140f1bcce5249c19115cd2e243d3c8538faf59be88da837337d1202"
SOURCE_JOURNAL_SHA256 = "b2b9269b450b344dc06d6b435f7490cd10ec2935dac0487c46387f4c13226278"
EXPERIMENT = "klein-cold-start-20260905-a"
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
COST_CEILING_USD = 2.84
STAGES = (
    "imports_seconds",
    "model_load_seconds",
    "cache_setup_seconds",
    "worker_seconds",
    "process_peak_rss_gib",
)
METRICS = (
    "image_seconds",
    "depth_seconds",
    "encoding_seconds",
    "total_seconds",
    "peak_allocated_gib",
    "peak_reserved_gib",
)
SCHEDULE = (
    ("baseline", 0),
    ("candidate", 0),
    ("candidate", 1),
    ("baseline", 1),
    ("baseline", 2),
    ("candidate", 2),
)
FAILURES = {
    "sdk_submission_unknown",
    "sdk_submission_cleanup_required",
    "sdk_result_failed",
    "sdk_result_cleanup_required",
    "local_operation_failed",
}
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def require(condition, code="cold_evidence_invalid"):
    if not condition:
        raise ValueError(code)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def frozen_cases():
    require(legacy.file_hash(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256)
    require(legacy.file_hash(SOURCE_JOURNAL) == SOURCE_JOURNAL_SHA256)
    source = legacy.protocol.decode_json(SOURCE_MANIFEST.read_bytes())
    results = {
        row["ordinal"]: row
        for row in (
            legacy.protocol.decode_json(line) for line in SOURCE_JOURNAL.read_bytes().splitlines()
        )
        if row["kind"] == "result"
    }
    cases = []
    for ordinal in (2, 9):
        operation = source["operations"][ordinal]
        require(results[ordinal]["status"] == "ok")
        metrics = results[ordinal]["payload"]["metrics"]
        cases.append(
            {
                "prompt": operation["request"]["prompt"],
                "seed": operation["request"]["seed"],
                "expected_bucket": operation["expected_bucket"],
                "master_sha256": metrics["master_sha256"],
                "depth_sha256": metrics["depth_sha256"],
            }
        )
    return source["expected_identity"], cases


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
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(
        value["experiment_id"] == EXPERIMENT
        and value["image_id"] == IMAGE_ID
        and value["cache_id"] == CACHE_ID
    )
    require(value["status"] in ("draft", "authorized"))
    require(
        value["expires_at"] is None
        if value["status"] == "draft"
        else type(value["expires_at"]) is int and value["expires_at"] > 0
    )
    identity, cases = frozen_cases()
    require(preparation.encoded(value["expected_identity"]) == preparation.encoded(identity))
    require(preparation.encoded(value["cases"]) == preparation.encoded(cases))
    for field, filename in {
        "runtime_sha256": ROOT / "deploy/klein_scene_runtime.py",
        "deployment_sha256": ROOT / "deploy/modal_klein_cold_start.py",
        "client_sha256": Path(__file__),
    }.items():
        require(value[field] == legacy.file_hash(filename))
    operations = value["operations"]
    require(isinstance(operations, list) and len(operations) == 6)
    ids = set()
    for row, (variant, pair) in zip(operations, SCHEDULE, strict=True):
        require(isinstance(row, dict) and set(row) == {"request_id", "variant", "pair_index"})
        require(
            row["variant"] == variant
            and type(row["pair_index"]) is int
            and row["pair_index"] == pair
        )
        require(
            isinstance(row["request_id"], str)
            and re.fullmatch(r"[a-f0-9]{32}", row["request_id"]) is not None
        )
        require(row["request_id"] not in ids)
        ids.add(row["request_id"])
    return value


def read_authorization(path, proof, manifest_path):
    raw = preparation.read_code(path)
    require(hashlib.sha256(raw).hexdigest() == proof)
    # Reuse the private-file guard without changing the regional reservation profile.
    import os
    import stat

    require(path.resolve() == path.absolute() and path.stat().st_uid == os.getuid())
    require(stat.S_IMODE(path.stat().st_mode) == 0o600)
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
    require(type(value["maximum_operations"]) is int and value["maximum_operations"] == 6)
    require(isinstance(value["reservation_id"], str) and 1 <= len(value["reservation_id"]) <= 200)
    require(is_hash(value["ledger_sha256"]))
    return value


def header(args, manifest, authorization):
    return {
        "kind": "header",
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": legacy.file_hash(args.manifest),
        "authorization_sha256": legacy.file_hash(args.authorization),
        "reservation_sha256": legacy.protocol.digest(authorization["reservation_id"]),
        "ledger_sha256": authorization["ledger_sha256"],
        "reserved_usd": authorization["reserved_usd"],
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "transport_sha256": legacy.file_hash(ROOT / "src/bookforge/klein_latency_client.py"),
        "support_sha256": {
            name: legacy.file_hash(ROOT / name)
            for name in (
                "scripts/benchmark_klein_latency.py",
                "scripts/prepare_klein_region_comparison.py",
                "deploy/klein_latency_protocol.py",
            )
        },
        "maximum_operations": 6,
        "automatic_retries": 0,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_journal_sha256": SOURCE_JOURNAL_SHA256,
        "physical_display_measured": False,
    }


def validate_payload(payload, operation, manifest):
    require(
        isinstance(payload, dict)
        and set(payload) == {"request_id", "variant", "identity", "location", "stages", "samples"}
    )
    require(
        payload["request_id"] == operation["request_id"]
        and payload["variant"] == operation["variant"]
    )
    require(
        preparation.encoded(payload["identity"])
        == preparation.encoded(manifest["expected_identity"])
    )
    location = payload["location"]
    require(
        isinstance(location, dict)
        and set(location) == {"cloud", "compute_region", "container_sha256"}
    )
    require(location["cloud"] == "CLOUD_PROVIDER_AWS" and location["compute_region"] == "us-east-1")
    require(
        is_hash(location["container_sha256"])
        and location["container_sha256"] != hashlib.sha256(b"unavailable").hexdigest()
    )
    require(isinstance(payload["stages"], dict) and set(payload["stages"]) == set(STAGES))
    require(all(finite(value) for value in payload["stages"].values()))
    require(isinstance(payload["samples"], list) and len(payload["samples"]) == 4)
    for index, sample in enumerate(payload["samples"]):
        require(
            isinstance(sample, dict)
            and set(sample) == {"repeat", "case_index", "metrics", "master", "depth"}
        )
        require(type(sample["repeat"]) is int and sample["repeat"] == index // 2)
        require(type(sample["case_index"]) is int and sample["case_index"] == index % 2)
        case, metrics = manifest["cases"][index % 2], sample["metrics"]
        require(
            isinstance(metrics, dict)
            and set(metrics)
            == {*METRICS, "seed", "token_count", "sequence_bucket", "master_sha256", "depth_sha256"}
        )
        require(type(metrics["seed"]) is int and metrics["seed"] == case["seed"])
        require(
            type(metrics["sequence_bucket"]) is int
            and metrics["sequence_bucket"] == case["expected_bucket"]
        )
        require(
            type(metrics["token_count"]) is int
            and 0 < metrics["token_count"] <= case["expected_bucket"]
        )
        require((128 if metrics["token_count"] <= 128 else 256) == case["expected_bucket"])
        require(all(finite(metrics[name]) for name in METRICS))
        for role in ("master", "depth"):
            content = sample[role]
            require(type(content) is bytes and 0 < len(content) <= MAX_ARTIFACT_BYTES)
            require(hashlib.sha256(content).hexdigest() == metrics[f"{role}_sha256"])
            require(_jpeg_dimensions(content) == (1024, 576))


class ColdClient(LatencyClient):
    async def cycle(self, operation):
        self.last_failure = None
        timings = dict.fromkeys(TIMINGS, 0.0)
        self.last_timings = timings
        started = time.perf_counter()
        try:
            async with asyncio.timeout(310):
                begun = time.perf_counter()
                if self.modal is None:
                    self.modal = importlib.import_module("modal")
                function = self.modal.Function.from_name(
                    "bookforge-klein-cold-start", f"{operation['variant']}_cycle"
                )
                await asyncio.wait_for(function.hydrate.aio(), 10)
                timings["lookup_seconds"] = time.perf_counter() - begun

                async def spawn(*, request):
                    return await function.spawn.aio(request["request_id"])

                self.instance = SimpleNamespace(
                    invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn))
                )
                return await self.sdk(operation, timings), timings
        except BaseException:
            self.last_failure = self.last_failure or "local_operation_failed"
            raise
        finally:
            timings["total_artifact_ready_seconds"] = time.perf_counter() - started


def artifact_path(output, ordinal, sample, role):
    return output / f"operation-{ordinal:02}" / f"sample-{sample}-{role}.jpg"


async def execute(args, manifest, evidence_header, client):
    journal = args.output / "journal.jsonl"
    legacy.write_exclusive(journal, (json.dumps(evidence_header, sort_keys=True) + "\n").encode())
    try:
        for ordinal, operation in enumerate(manifest["operations"]):
            require(time.time() < manifest["expires_at"], "authorization_expired")
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
                            artifact_path(args.output, ordinal, index, role), sample[role]
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
            except BaseException:
                code = (
                    client.last_failure
                    if client.last_failure in FAILURES
                    else "local_operation_failed"
                )
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "failed",
                        "code": code,
                        "timings": client.last_timings or {},
                    },
                )
                raise
        legacy.append(journal, {"kind": "complete", "operations": 6})
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
    rows = [legacy.protocol.decode_json(line) for line in journal.read_bytes().splitlines()]
    require(bool(rows) and preparation.encoded(rows[0]) == preparation.encoded(evidence_header))
    started, results, complete, cleanup = 0, [], False, None
    pending = None
    failed = False
    for event in rows[1:]:
        require(cleanup is None)
        kind = event.get("kind")
        if kind == "cleanup":
            require(set(event) == {"kind", "known_calls_cancelled", "external_app_stop_required"})
            require(
                type(event["known_calls_cancelled"]) is bool
                and type(event["external_app_stop_required"]) is bool
            )
            require(event["known_calls_cancelled"] is not event["external_app_stop_required"])
            cleanup = event
        elif kind == "start":
            require(not failed and not complete and pending is None and started < 6)
            require(set(event) == {"kind", "ordinal", "request_sha256"})
            require(type(event["ordinal"]) is int and event["ordinal"] == started)
            require(
                event["request_sha256"] == legacy.protocol.digest(manifest["operations"][started])
            )
            pending = started
            started += 1
        elif kind == "result":
            require(
                not complete
                and pending is not None
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
            timings = event["timings"]
            require(isinstance(timings, dict) and set(timings) in (set(TIMINGS), set()))
            require(all(finite(value) for value in timings.values()))
            if event["status"] == "ok":
                require(set(timings) == set(TIMINGS))
                payload = copy.deepcopy(event["payload"])
                for index, sample in enumerate(payload["samples"]):
                    require(set(sample) == {"repeat", "case_index", "metrics"})
                    for role in ("master", "depth"):
                        path = artifact_path(args.output, pending, index, role)
                        require(
                            path.resolve() == path.absolute()
                            and path.is_file()
                            and path.stat().st_size <= MAX_ARTIFACT_BYTES
                        )
                        sample[role] = path.read_bytes()
                validate_payload(payload, manifest["operations"][pending], manifest)
            else:
                require(event["code"] in FAILURES)
                failed = True
            results.append(event)
            pending = None
        elif kind == "complete":
            require(
                preparation.encoded(event)
                == preparation.encoded({"kind": "complete", "operations": 6})
                and not complete
                and not failed
                and pending is None
                and len(results) == 6
            )
            complete = True
        else:
            raise ValueError("journal_sequence_invalid")
    successful = [row for row in results if row["status"] == "ok"]
    containers = {row["payload"]["location"]["container_sha256"] for row in successful}
    references_match = bool(successful) and all(
        sample["metrics"][f"{role}_sha256"]
        == manifest["cases"][sample["case_index"]][f"{role}_sha256"]
        for row in successful
        for sample in row["payload"]["samples"]
        for role in ("master", "depth")
    )
    pairs = []
    for pair_index in range(3):
        selected = [
            row
            for row in successful
            if manifest["operations"][row["ordinal"]]["pair_index"] == pair_index
        ]
        equal = len(selected) == 2 and all(
            selected[0]["payload"]["samples"][index]["metrics"][f"{role}_sha256"]
            == selected[1]["payload"]["samples"][index]["metrics"][f"{role}_sha256"]
            for index in range(4)
            for role in ("master", "depth")
        )
        pairs.append(
            {"pair_index": pair_index, "complete": len(selected) == 2, "hashes_equal": equal}
        )
    variants = {}
    for variant in ("baseline", "candidate"):
        selected = [row for row in successful if row["payload"]["variant"] == variant]
        values = {name: [row["payload"]["stages"][name] for row in selected] for name in STAGES}
        values["client_total_seconds"] = [
            row["timings"]["total_artifact_ready_seconds"] for row in selected
        ]
        values["client_residual_seconds"] = [
            row["timings"]["total_artifact_ready_seconds"]
            - row["payload"]["stages"]["worker_seconds"]
            for row in selected
        ]
        for bucket, index in ((128, 0), (256, 1)):
            for repeat in (0, 1):
                values[f"{'first' if repeat == 0 else 'warm'}_{bucket}_seconds"] = [
                    row["payload"]["samples"][repeat * 2 + index]["metrics"]["total_seconds"]
                    for row in selected
                ]
        variants[variant] = {name: legacy.distribution(items) for name, items in values.items()}
    baseline, candidate = (
        variants[key]["client_total_seconds"] for key in ("baseline", "candidate")
    )
    improvement = (
        1 - candidate["p50"] / baseline["p50"]
        if baseline["p50"] and candidate["p50"] is not None
        else None
    )
    fresh = len(containers) == 6
    passed = (
        complete
        and not failed
        and fresh
        and references_match
        and cleanup is not None
        and cleanup["known_calls_cancelled"]
        and improvement is not None
        and improvement >= 0.25
        and candidate["max"] <= baseline["max"]
    )
    return {
        "schema_version": 1,
        "kind": "klein-cold-start-summary",
        "header": evidence_header,
        "journal_sha256": legacy.file_hash(journal),
        "started": started,
        "results": len(results),
        "failures": sum(row["status"] == "failed" for row in results),
        "unresolved_requests": int(pending is not None),
        "complete": complete,
        "six_distinct_containers": fresh,
        "all_reference_hashes_match": references_match,
        "pairs": pairs,
        "variants": variants,
        "candidate_p50_improvement_fraction": improvement,
        "cleanup": cleanup,
        "decision": "pass" if passed else "reject",
        "external_app_shutdown_verified": False,
        "physical_display_measured": False,
        "sample_scope": (
            "three paired cold four-render cycles; first-bucket timings are server-only; "
            "descriptive, not an SLA"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
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
                not args.execute
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
        if not args.execute:
            args.output.mkdir(mode=0o700)
            legacy.write_exclusive(
                args.output / "preflight.json",
                preparation.encoded(
                    {
                        "schema_version": 1,
                        "kind": "klein-cold-start-preflight",
                        "manifest_sha256": legacy.file_hash(args.manifest),
                        "planned_operations": 6,
                        "planned_images": 24,
                        "cost_ceiling_usd": COST_CEILING_USD,
                        "status": manifest["status"],
                        "generation_calls": 0,
                        "authorization_verified": authorization is not None,
                    }
                ),
            )
            return 0
        require(
            sys.platform == "linux"
            and platform.machine() == "aarch64"
            and Path("/etc/nv_tegra_release").is_file()
        )
        require(time.time() < manifest["expires_at"] <= time.time() + 7200)
        evidence_header = header(args, manifest, authorization)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        client = ColdClient(manifest)
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("cold-start comparison stopped; inspect sanitized evidence and cleanup status")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
