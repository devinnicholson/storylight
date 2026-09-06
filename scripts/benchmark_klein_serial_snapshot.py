#!/usr/bin/env python3
"""Qualify one serial-compiler snapshot capture without changing the frozen warmed probe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import os
import re
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts import benchmark_klein_warmed_snapshot as warm  # noqa: E402

cold, legacy, preparation = warm.cold, warm.legacy, warm.preparation
require = cold.require
EXPERIMENT = "klein-serial-snapshot-20260905-a"
MAX_OPERATIONS, MAX_CAPTURES, COST_CEILING_USD = 3, 1, 1.05
validate_payload, qualification, execute = warm.validate_payload, warm.qualification, warm.execute


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
        "deployment_sha256": ROOT / "deploy/modal_klein_serial_snapshot.py",
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
    result = warm.header(args, manifest, authorization)
    result["support_sha256"][str(Path(warm.__file__).relative_to(ROOT))] = legacy.file_hash(
        Path(warm.__file__)
    )
    return {
        **result,
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "qualification": (
            "serial compiler; two warmed restores; correctness and latency separate; "
            "compiler configuration and platform audit required"
        ),
    }


def aggregate(args, manifest, evidence_header):
    result = warm.aggregate(args, manifest, evidence_header)
    return {
        **result,
        "kind": "klein-serial-snapshot-summary",
        "compiler_configuration_audit_pending": True,
    }


class SerialClient(warm.WarmedClient):
    async def cycle(self, operation):
        started = time.perf_counter()
        lookup = 0.0
        self.last_failure = None
        self.last_timings = dict.fromkeys(cold.TIMINGS, 0.0)
        try:
            remaining = self.deadline - time.time()
            require(remaining > 0)
            async with asyncio.timeout(min(180, remaining)):
                if self.method is None:
                    if self.modal is None:
                        self.modal = importlib.import_module("modal")
                    cls = self.modal.Cls.from_name(
                        "bookforge-klein-serial-snapshot", "SerialSnapshot"
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
                        "kind": "klein-serial-snapshot-preflight",
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
        require(cold.finite(args.deadline_unix) and 0 < args.deadline_unix - time.time() <= 180)
        require(time.time() < manifest["expires_at"] <= time.time() + 7200)
        evidence_header = header(args, manifest, authorization)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        client = SerialClient(manifest, deadline=min(args.deadline_unix, manifest["expires_at"]))
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("serial-snapshot probe stopped; inspect sanitized evidence and supervised cleanup")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
