#!/usr/bin/env python3
"""Prepare an inactive region-controlled comparison from the frozen fourteen-operation run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.klein_latency_protocol import decode_json  # noqa: E402

SOURCE = ROOT / "benchmarks/renderer-latency-2026-09-05/manifest.json"
SOURCE_SHA256 = "b9a817866d93b57b732b6571a5a06d117d43779d5593d8139c57157f7bcb81ce"
PLACEMENT = {
    "cloud": "aws",
    "compute_region": "us-east",
    "routing_region": "us-east",
    "expected_cloud": "CLOUD_PROVIDER_AWS",
    "expected_compute_region": "us-east-1",
}
REGION_DEPLOYMENT = ROOT / "deploy/modal_klein_region.py"
REGION_CLIENT = ROOT / "src/bookforge/klein_region_client.py"
DEPLOYMENTS = {
    "sdk": {"app": "bookforge-klein-region-sdk", "class": "RegionStudio"},
    "http": {"app": "bookforge-klein-region-http", "class": "RegionServer"},
}
SUPPORT = {
    "deployment_sha256": "modal_klein_latency.py",
    "runtime_sha256": "klein_scene_runtime.py",
    "instrumentation_sha256": "klein_latency_runtime.py",
    "protocol_sha256": "klein_latency_protocol.py",
    "http_server_sha256": "klein_latency_http.py",
}
COST_CEILING_USD = 5.96
BAKED_IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def read_authorization(path: Path, proof: str, manifest_bytes: bytes) -> dict:
    raw = read_code(path)
    if (
        hashlib.sha256(raw).hexdigest() != proof
        or path.resolve() != path.absolute()
        or path.stat().st_uid != os.getuid()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or path.parent.stat().st_uid != os.getuid()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("operator-issued private authorization proof differs")
    value = decode_json(raw)
    if (
        set(value)
        != {
            "schema_version",
            "manifest_sha256",
            "reservation_id",
            "reserved_usd",
            "maximum_operations",
            "ledger_sha256",
        }
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
        or type(value["reserved_usd"]) not in (int, float)
        or value["reserved_usd"] != COST_CEILING_USD
        or type(value["maximum_operations"]) is not int
        or value["maximum_operations"] != 14
        or not isinstance(value["reservation_id"], str)
        or not 1 <= len(value["reservation_id"]) <= 200
        or not isinstance(value["ledger_sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["ledger_sha256"])
    ):
        raise ValueError("regional reservation differs from fixed comparison")
    return value


def read_code(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("code proof requires a bounded regular file")
    return path.read_bytes()


def prepare(experiment_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", experiment_id):
        raise ValueError("invalid experiment identifier")
    source_bytes = read_code(SOURCE)
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("frozen source manifest proof differs")
    source = json.loads(source_bytes)
    if experiment_id == source["experiment_id"]:
        raise ValueError("the comparison requires a new experiment identity")
    deployment_sha = hashlib.sha256(read_code(REGION_DEPLOYMENT)).hexdigest()
    if deployment_sha == source["deployment_sha256"]:
        raise ValueError("comparison requires the separate region-controlled deployment")
    pins = {}
    for field, name in SUPPORT.items():
        pins[field] = hashlib.sha256(read_code(ROOT / "deploy" / name)).hexdigest()
        if pins[field] != source[field]:
            raise ValueError("shared renderer and protocol must retain their frozen implementation")
    result = copy.deepcopy(source)
    result.update(
        schema_version=2,
        status="draft",
        comparison_profile="region-controlled-v1",
        experiment_id=experiment_id,
        expires_at=None,
        source_manifest_sha256=SOURCE_SHA256,
        region_deployment_sha256=deployment_sha,
        region_client_sha256=hashlib.sha256(read_code(REGION_CLIENT)).hexdigest(),
        baked_image_id=BAKED_IMAGE_ID,
        placement=copy.deepcopy(PLACEMENT),
        deployments=copy.deepcopy(DEPLOYMENTS),
        **pins,
    )
    for operation in result["operations"]:
        identity = f"{experiment_id}:{operation['transport']}:{operation['pair_id'] or 'warmup'}"
        operation["request"]["request_id"] = hashlib.sha256(identity.encode()).hexdigest()[:32]
    return result


def validate_manifest(path: Path) -> dict:
    value = decode_json(read_code(path))
    expected = prepare(value["experiment_id"])
    if value.get("status") == "authorized":
        if type(value.get("expires_at")) is not int or value["expires_at"] <= 0:
            raise ValueError("active comparison expiry differs")
        expected.update(status="authorized", expires_at=value["expires_at"])
    if encoded(value) != encoded(expected):
        raise ValueError("regional comparison differs from its frozen requests and implementation")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activate-until", type=int)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("draft output requires a fresh file and an existing parent")
        result = prepare(args.experiment_id)
        if args.activate_until is not None:
            if (
                not time.time() < args.activate_until <= time.time() + 7200
                or args.authorization is None
                or args.authorization_sha256 is None
            ):
                raise ValueError(
                    "activation requires bounded expiry and centrally issued authorization"
                )
            result.update(status="authorized", expires_at=args.activate_until)
            read_authorization(args.authorization, args.authorization_sha256, encoded(result))
        elif args.authorization is not None or args.authorization_sha256 is not None:
            raise ValueError("draft preparation does not consume authorization")
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded(result).decode())
            stream.flush()
            os.fsync(stream.fileno())
        return 0
    except Exception:
        print("region comparison draft refused; verify frozen local inputs and fresh output")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
