#!/usr/bin/env python3
"""Run the separately authorized regional comparison using the existing latency evidence gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookforge.klein_region_client import RegionClient  # noqa: E402
from scripts import benchmark_klein_latency as legacy  # noqa: E402
from scripts import prepare_klein_region_comparison as preparation  # noqa: E402


def header(args, manifest, authorization):
    result = legacy.header(args, manifest, authorization)
    return {
        **result,
        "legacy_harness_sha256": result["harness_sha256"],
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "region_client_sha256": manifest["region_client_sha256"],
        "region_deployment_sha256": manifest["region_deployment_sha256"],
        "comparison_profile": manifest["comparison_profile"],
        "placement": manifest["placement"],
    }


async def execute(args, manifest, evidence_header, client: RegionClient):
    client.headers()
    journal = args.output / "journal.jsonl"
    legacy.write_exclusive(journal, (json.dumps(evidence_header, sort_keys=True) + "\n").encode())
    try:
        for ordinal, operation in enumerate(manifest["operations"]):
            if time.time() >= manifest["expires_at"]:
                raise ValueError("regional authorization expired")
            legacy.append(
                journal,
                {
                    "kind": "start",
                    "ordinal": ordinal,
                    "request_sha256": legacy.protocol.digest(operation),
                },
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
                        legacy.write_exclusive(directory / f"{role}.jpg", payload[role])
                timings["storage_seconds"] = time.perf_counter() - storage
                timings["total_artifact_ready_seconds"] = time.perf_counter() - begun
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "ok",
                        "timings": timings,
                        "payload": {
                            key: value
                            for key, value in payload.items()
                            if key not in {"master", "depth"}
                        },
                    },
                )
            except BaseException:
                legacy.append(
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
        legacy.append(journal, {"kind": "complete", "operations": 14})
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
    expected = {**manifest, "deployment_sha256": manifest["region_deployment_sha256"]}
    result = legacy.aggregate(args, expected, evidence_header)
    events = [
        legacy.protocol.decode_json(line)
        for line in (args.output / "journal.jsonl").read_bytes().splitlines()
    ]
    locations = [
        event["payload"]["location"]
        for event in events
        if event.get("kind") == "result" and event.get("status") == "ok"
    ]
    placement_verified = bool(locations) and all(
        location["cloud"] == manifest["placement"]["expected_cloud"]
        and location["compute_region"] == manifest["placement"]["expected_compute_region"]
        and location["routing_region"] == manifest["placement"]["routing_region"]
        for location in locations
    )
    return {
        **result,
        "kind": "klein-region-latency-summary",
        "placement_verified": placement_verified,
        "decision": result["decision"] if placement_verified else "reject",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = preparation.validate_manifest(args.manifest)
        authorization = None
        if manifest["status"] == "authorized":
            if args.authorization is None or args.authorization_sha256 is None:
                raise ValueError("active comparison requires centrally issued authorization")
            authorization = preparation.read_authorization(
                args.authorization, args.authorization_sha256, preparation.read_code(args.manifest)
            )
        elif (
            args.execute
            or args.aggregate_only
            or args.authorization is not None
            or args.authorization_sha256 is not None
        ):
            raise ValueError("draft comparisons are preflight-only")
        if args.aggregate_only:
            result = aggregate(args, manifest, header(args, manifest, authorization))
            legacy.write_exclusive(
                args.output / "recomputed-summary.json", preparation.encoded(result)
            )
            return 0
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("regional comparison requires a fresh output directory")
        if not args.execute:
            args.output.mkdir(mode=0o700)
            legacy.write_exclusive(
                args.output / "preflight.json",
                preparation.encoded(
                    {
                        "schema_version": 1,
                        "kind": "klein-region-preflight",
                        "status": manifest["status"],
                        "manifest_sha256": legacy.file_hash(args.manifest),
                        "planned_operations": 14,
                        "cost_ceiling_usd": preparation.COST_CEILING_USD,
                        "authorization_verified": authorization is not None,
                        "generation_calls": 0,
                        "placement": manifest["placement"],
                        "physical_display_measured": False,
                    }
                ),
            )
            return 0
        if (
            sys.platform != "linux"
            or platform.machine() != "aarch64"
            or not Path("/etc/nv_tegra_release").is_file()
        ):
            raise ValueError("regional execution requires the Jetson")
        if not time.time() < manifest["expires_at"] <= time.time() + 7200:
            raise ValueError("regional execution requires a fresh bounded activation")
        client = RegionClient(manifest)
        client.headers()
        evidence_header = header(args, manifest, authorization)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("regional comparison stopped; inspect sanitized local inputs and cleanup status")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
