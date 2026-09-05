#!/usr/bin/env python3
"""Freeze a reduced, source-free comparison from the already approved image batch."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_SHA256 = "4601956b9a3a39a4f5b9b802b694db70aea737e6182d03d1411499326a1b1331"
PAIRS = (
    "candidate-page-01-still",
    "candidate-page-02-still",
    "candidate-page-04-still",
    "accepted-page-03",
    "accepted-page-05",
    "accepted-page-06",
)


def freeze(experiment_id: str, expires_at: int) -> dict:
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", experiment_id):
        raise ValueError("invalid experiment identifier")
    if type(expires_at) is not int or not 0 < expires_at <= time.time() + 7200:
        raise ValueError("expiry must be within the two-hour maximum horizon")
    batch_bytes = (ROOT / "benchmarks/scene-routing-2026-09-04/visual-batch.json").read_bytes()
    if hashlib.sha256(batch_bytes).hexdigest() != BATCH_SHA256:
        raise ValueError("original batch proof differs")
    rows = {row["id"]: row for row in json.loads(batch_bytes)["requests"]}
    operations = []

    def append(transport, pair_id, bucket):
        row = rows[pair_id] if pair_id else None
        key = f"{experiment_id}:{transport}:{pair_id or 'warmup'}"
        operations.append(
            {
                "transport": transport,
                "pair_id": pair_id,
                "expected_bucket": bucket,
                "request": {
                    "request_id": hashlib.sha256(key.encode()).hexdigest()[:32],
                    "operation": "render" if row else "prewarm",
                    "prompt": row["prompt"] if row else "",
                    "seed": row["seed"] if row else 0,
                },
            }
        )

    for transport in ("sdk", "http"):
        append(transport, None, None)
    for index, pair in enumerate(PAIRS):
        for transport in ("sdk", "http") if index % 2 == 0 else ("http", "sdk"):
            append(transport, pair, 128 if index < 3 else 256)
    identity = json.loads(
        (ROOT / "benchmarks/overnight-20260904/renderer-summary.json").read_text()
    )["batches"]["development"]["identity"]
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "expires_at": expires_at,
        "original_batch_sha256": BATCH_SHA256,
        "expected_identity": identity,
        "operations": operations,
    }
    for field, name in (
        ("runtime_sha256", "klein_scene_runtime.py"),
        ("instrumentation_sha256", "klein_latency_runtime.py"),
        ("deployment_sha256", "modal_klein_latency.py"),
        ("protocol_sha256", "klein_latency_protocol.py"),
        ("http_server_sha256", "klein_latency_http.py"),
    ):
        manifest[field] = hashlib.sha256((ROOT / "deploy" / name).read_bytes()).hexdigest()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--expires-at", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.experiment_id, args.expires_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
