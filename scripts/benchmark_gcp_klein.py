#!/usr/bin/env python3
"""Finite native Cloud Run qualification: one request per public case, no retries."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bookforge.gcp_scene_provider import GoogleImpersonatedIdentityTokenSource  # noqa: E402
from deploy import klein_latency_protocol as protocol  # noqa: E402
from scripts import benchmark_klein_cold_start as cold  # noqa: E402

FIXTURE = ROOT / "experiments/renderer-fidelity/modal_compare.py"
FIXTURE_SHA256 = "766badfbbd886486622c28f069a01a2d38ed16ef550def8bd06e47f82a4f190d"
MAX_BYTES = 4 * 1024 * 1024
SOURCE_PATHS = {
    "runtime": ROOT / "deploy/klein_scene_runtime.py",
    "worker": ROOT / "deploy/gcp_klein_worker/app.py",
    "weights": ROOT / "deploy/gcp_klein_worker/klein_weights.py",
}
INVOKER = "bookforge-renderer@your-gcp-project.iam.gserviceaccount.com"
TIMINGS = {
    "image_seconds",
    "depth_seconds",
    "encoding_seconds",
    "total_seconds",
    "peak_allocated_gib",
    "peak_reserved_gib",
}


def require(ok):
    if not ok:
        raise ValueError("qualification_invalid")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_cases():
    require(sha(FIXTURE.read_bytes()) == FIXTURE_SHA256)
    _, retained = cold.frozen_cases()
    rows = [
        {"case_id": f"retained-{i}", "prompt": row["prompt"], "seed": row["seed"]}
        for i, row in enumerate(retained)
    ]
    constants = {}
    for node in ast.parse(FIXTURE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in {"PROMPTS", "STYLE", "SUFFIX"}:
                constants[name] = ast.literal_eval(node.value)
    for index, prompt in enumerate(constants["PROMPTS"]):
        rows.append(
            {
                "case_id": f"watercolor-{index}",
                "prompt": constants["STYLE"] + prompt + constants["SUFFIX"],
                "seed": 20260903 + index,
            }
        )
    return rows


def schedule(manifest):
    return manifest["cases"] + manifest["cases"][:2]


def prepare_manifest(experiment_id, identity, sources, *, runtime_source: Path | None = None):
    value = {
        "schema_version": 1,
        "status": "draft",
        "experiment_id": experiment_id,
        "expires_at": None,
        "max_requests": 10,
        "expected_identity": identity,
        "cases": public_cases(),
        "sources": sources,
    }
    validate_manifest(value, active=False, runtime_source=runtime_source)
    return value


def validate_manifest(value, *, active, runtime_source: Path | None = None):
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "experiment_id",
            "expires_at",
            "max_requests",
            "expected_identity",
            "cases",
            "sources",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(type(value["max_requests"]) is int and value["max_requests"] == 10)
    require(
        isinstance(value["experiment_id"], str)
        and re.fullmatch(r"[a-z0-9-]{1,80}", value["experiment_id"])
    )
    require(encoded(value["cases"]) == encoded(public_cases()))
    require(set(value["sources"]) == {"runtime", "worker", "weights"})
    require(all(is_hash(v) for v in value["sources"].values()))
    paths = SOURCE_PATHS if runtime_source is None else SOURCE_PATHS | {"runtime": runtime_source}
    require(value["sources"] == {k: sha(p.read_bytes()) for k, p in paths.items()})
    identity, _ = cold.frozen_cases()
    if runtime_source is not None:
        identity = identity | {"runtime_sha256": sha(runtime_source.read_bytes())}
    require(set(value["expected_identity"]) == set(identity))
    for key, expected in identity.items():
        if key not in {"gpu", "capability"}:
            require(encoded(value["expected_identity"][key]) == encoded(expected))
    require(value["sources"]["runtime"] == identity["runtime_sha256"])
    gpus = {
        "NVIDIA L4": [8, 9],
        "NVIDIA RTX PRO 6000 Blackwell": [12, 0],
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": [12, 0],
    }
    require(value["expected_identity"]["gpu"] in gpus)
    require(value["expected_identity"]["capability"] == gpus[value["expected_identity"]["gpu"]])
    capability = value["expected_identity"]["capability"]
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(n) is int and 0 <= n <= 20 for n in capability)
    )
    if active:
        require(value["status"] == "authorized")
        require(type(value["expires_at"]) is int and value["expires_at"] > time.time())
    else:
        require(value["status"] in {"draft", "authorized"})
        require(value["expires_at"] is None or type(value["expires_at"]) is int)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def finite(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def validate_payload(payload, case, manifest, manifest_sha256, service, revision):
    require(
        set(payload)
        == {
            "schema_version",
            "case_id",
            "instance_id",
            "manifest_sha256",
            "identity",
            "metrics",
            "master_b64",
            "depth_b64",
            "cold",
            "load_seconds",
            "compile_seconds",
            "worker_seconds",
            "bucket_was_warm",
            "request_ordinal",
            "revision",
            "service",
        }
    )
    require(type(payload["schema_version"]) is int and payload["schema_version"] == 1)
    require(payload["case_id"] == case["case_id"])
    require(payload["manifest_sha256"] == manifest_sha256)
    require(payload["service"] == service and payload["revision"] == revision)
    require(encoded(payload["identity"]) == encoded(manifest["expected_identity"]))
    require(
        isinstance(payload["instance_id"], str)
        and re.fullmatch(r"[a-f0-9]{32}", payload["instance_id"])
    )
    require(type(payload["cold"]) is bool and type(payload["bucket_was_warm"]) is bool)
    require(type(payload["request_ordinal"]) is int and 1 <= payload["request_ordinal"] <= 10)
    require(payload["cold"] == (payload["request_ordinal"] == 1))
    require(all(finite(payload[k]) for k in ("load_seconds", "compile_seconds", "worker_seconds")))
    metrics = payload["metrics"]
    require(
        set(metrics)
        == TIMINGS | {"seed", "sequence_bucket", "token_count", "master_sha256", "depth_sha256"}
    )
    require(all(finite(metrics[k]) for k in TIMINGS))
    require(type(metrics["seed"]) is int and metrics["seed"] == case["seed"])
    require(type(metrics["token_count"]) is int and 0 < metrics["token_count"] <= 512)
    bucket = next(n for n in (128, 256, 512) if metrics["token_count"] <= n)
    require(type(metrics["sequence_bucket"]) is int and metrics["sequence_bucket"] == bucket)
    require(payload["worker_seconds"] >= metrics["total_seconds"])
    assets = {}
    for role in ("master", "depth"):
        raw = base64.b64decode(payload[role + "_b64"], validate=True)
        require(0 < len(raw) < MAX_BYTES and is_hash(metrics[role + "_sha256"]))
        require(sha(raw) == metrics[role + "_sha256"])
        with Image.open(io.BytesIO(raw)) as image:
            require(image.format == "JPEG" and image.size == (1024, 576))
            image.load()
        assets[role] = raw
    return assets


def write_exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append(path, row):
    with path.open("ab") as stream:
        stream.write(encoded(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def context(manifest, manifest_sha256, service, revision):
    return {
        "kind": "header",
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "experiment_id": manifest["experiment_id"],
        "service": service,
        "revision": revision,
        "harness_sha256": sha(Path(__file__).read_bytes()),
        "support_sha256": {
            str(p.relative_to(ROOT)): sha(p.read_bytes())
            for p in (
                FIXTURE,
                Path(cold.__file__),
                Path(cold.legacy.__file__),
                Path(cold.preparation.__file__),
                cold.SOURCE_MANIFEST,
                cold.SOURCE_JOURNAL,
                ROOT / "deploy/klein_latency_protocol.py",
                ROOT / "src/bookforge/gcp_scene_provider.py",
            )
        },
        "claims": "client artifact-ready only; no browser, projector or full product latency",
    }


def validate_endpoint(endpoint, service, revision):
    url = urlsplit(endpoint)
    require(
        url.scheme == "https"
        and url.hostname is not None
        and url.hostname.endswith(".run.app")
        and not url.username
        and not url.password
        and url.port is None
        and url.path in {"", "/"}
        and not url.query
        and not url.fragment
    )
    require(all(re.fullmatch(r"[a-z][a-z0-9-]{0,100}", v) for v in (service, revision)))


async def run(
    manifest,
    raw_manifest,
    output,
    endpoint,
    service,
    revision,
    *,
    client,
    token_source,
    runtime_source: Path | None = None,
):
    validate_manifest(manifest, active=True, runtime_source=runtime_source)
    validate_endpoint(endpoint, service, revision)
    require(not output.exists())
    token = await token_source(endpoint)
    require(
        isinstance(token, str)
        and token
        and token.isascii()
        and all(32 < ord(c) < 127 for c in token)
    )
    claim_root = Path.home() / ".local/state/bookforge/gcp-klein-attempts"
    claim_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not claim_root.is_symlink())
    os.chmod(claim_root, 0o700)
    manifest_hash = sha(raw_manifest)
    header = context(manifest, manifest_hash, service, revision)
    write_exclusive(claim_root / f"{manifest['experiment_id']}-{revision}.json", encoded(header))
    output.mkdir(mode=0o700, parents=True)
    journal = output / "journal.jsonl"
    write_exclusive(journal, encoded(header) + b"\n")
    instances, primed = {}, set()
    for ordinal, case in enumerate(schedule(manifest)):
        started = time.perf_counter()
        append(journal, {"kind": "start", "ordinal": ordinal, "case_sha256": protocol.digest(case)})
        try:
            require(time.time() < manifest["expires_at"])
            async with client.stream(
                "POST",
                endpoint.rstrip("/") + "/generate",
                headers={"Authorization": "Bearer " + token},
                json={"case_id": case["case_id"], "seed": case["seed"]},
            ) as response:
                require(response.status_code == 200)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    require(len(data) <= MAX_BYTES)
            payload = protocol.decode_json(bytes(data))
            assets = validate_payload(payload, case, manifest, manifest_hash, service, revision)
            instance = payload["instance_id"]
            require(payload["request_ordinal"] == instances.get(instance, 1))
            bucket_key = (instance, payload["metrics"]["sequence_bucket"])
            require(payload["bucket_was_warm"] == (bucket_key in primed))
            primed.add(bucket_key)
            instances[instance] = payload["request_ordinal"] + 1
            for role, asset in assets.items():
                write_exclusive(output / f"{ordinal}-{role}.jpg", asset)
            row = {
                "kind": "result",
                "ordinal": ordinal,
                "status": "ok",
                "payload": {k: v for k, v in payload.items() if not k.endswith("_b64")},
                "client_artifact_ready_seconds": time.perf_counter() - started,
            }
            append(journal, row)
        except (Exception, asyncio.CancelledError):
            append(
                journal,
                {
                    "kind": "result",
                    "ordinal": ordinal,
                    "status": "failed",
                    "error": "request_or_evidence_failed",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            break
    append(journal, {"kind": "complete"})


def aggregate(manifest, raw_manifest, output, service, revision):
    rows = [
        protocol.decode_json(line) for line in (output / "journal.jsonl").read_bytes().splitlines()
    ]
    require(rows and rows[0] == context(manifest, sha(raw_manifest), service, revision))
    require(rows[-1] == {"kind": "complete"})
    middle = rows[1:-1]
    require(len(middle) % 2 == 0 and 0 < len(middle) <= 20)
    instances, buckets, results = {}, set(), []
    _, retained = cold.frozen_cases()
    for ordinal in range(len(middle) // 2):
        start, row = middle[2 * ordinal : 2 * ordinal + 2]
        case = schedule(manifest)[ordinal]
        require(
            start == {"kind": "start", "ordinal": ordinal, "case_sha256": protocol.digest(case)}
        )
        require(row["kind"] == "result" and row["ordinal"] == ordinal)
        if row["status"] == "failed":
            require(
                ordinal == len(middle) // 2 - 1 and row["error"] == "request_or_evidence_failed"
            )
            results.append({"ordinal": ordinal, "case_id": case["case_id"], "status": "failed"})
            continue
        require(row["status"] == "ok" and finite(row["client_artifact_ready_seconds"]))
        payload = row["payload"]
        assets = {
            role: (output / f"{ordinal}-{role}.jpg").read_bytes() for role in ("master", "depth")
        }
        validate_payload(
            {**payload, **{k + "_b64": base64.b64encode(v).decode() for k, v in assets.items()}},
            case,
            manifest,
            sha(raw_manifest),
            service,
            revision,
        )
        instance, bucket = payload["instance_id"], payload["metrics"]["sequence_bucket"]
        require(payload["request_ordinal"] == instances.get(instance, 1))
        instances[instance] = payload["request_ordinal"] + 1
        first_bucket = (instance, bucket) not in buckets
        require(payload["bucket_was_warm"] == (not first_bucket))
        buckets.add((instance, bucket))
        results.append(
            {
                "ordinal": ordinal,
                "case_id": case["case_id"],
                "status": "ok",
                "cold_worker": payload["cold"],
                "instance_id": instance,
                "bucket": bucket,
                "first_bucket_on_worker": first_bucket,
                "client_artifact_ready_seconds": row["client_artifact_ready_seconds"],
                "metrics": payload["metrics"],
                "load_seconds": payload["load_seconds"],
                "compile_seconds": payload["compile_seconds"],
                "worker_seconds": payload["worker_seconds"],
                "historical_exact": {
                    role: sha(raw)
                    == retained[int(case["case_id"].split("-")[-1])][role + "_sha256"]
                    for role, raw in assets.items()
                }
                if case["case_id"].startswith("retained-")
                else None,
                "human_quality": "ungraded",
            }
        )
    eligible = [
        r["client_artifact_ready_seconds"]
        for r in results
        if r["status"] == "ok" and not r["first_bucket_on_worker"] and not r["cold_worker"]
    ]
    completed = sum(r["status"] == "ok" for r in results)
    return {
        "schema_version": 1,
        "kind": "gcp-klein-qualification",
        "manifest_sha256": sha(raw_manifest),
        "journal_sha256": sha((output / "journal.jsonl").read_bytes()),
        "declared_cases": 10,
        "completed_cases": completed,
        "decision": "ungraded" if completed == 10 else "reject",
        "worker_count": len(instances),
        "cases": results,
        "later_bucket_client_seconds": {
            "count": len(eligible),
            "median": statistics.median(eligible) if eligible else None,
            "max": max(eligible) if eligible else None,
        },
        "limitations": [
            "No p95 estimate from this small screen.",
            "First occurrence of every bucket on every worker is priming, not established warm.",
            "No blind quality, browser display, cold reliability or full product latency claim.",
            "Token acquisition precedes timings; deployment, billing and cleanup are external.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", "--base-url", dest="endpoint")
    parser.add_argument("--service")
    parser.add_argument("--revision")
    parser.add_argument("--proof-manifest-sha256")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help="Explicit reviewed runtime source matching both manifest hash pins",
    )
    parser.add_argument("--experiment-id", default="gcp-klein-20260906-a")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            value = prepare_manifest(
                args.experiment_id,
                protocol.decode_json(args.identity.read_bytes()),
                protocol.decode_json(args.sources.read_bytes()),
                runtime_source=args.runtime_source,
            )
            write_exclusive(args.manifest, encoded(value))
            print(
                json.dumps(
                    {
                        "status": "draft",
                        "manifest_sha256": sha(encoded(value)),
                        "unique_cases": 8,
                        "requests": 10,
                    }
                )
            )
            return
        raw = args.manifest.read_bytes()
        value = protocol.decode_json(raw)
        validate_manifest(value, active=args.run, runtime_source=args.runtime_source)
        if args.proof_manifest_sha256:
            require(sha(raw) == args.proof_manifest_sha256)
        if args.run:
            require(args.proof_manifest_sha256 is not None)

            async def execute():
                async with httpx.AsyncClient(
                    trust_env=False, follow_redirects=False, timeout=httpx.Timeout(240, connect=15)
                ) as client:
                    await asyncio.wait_for(
                        run(
                            value,
                            raw,
                            args.output,
                            args.endpoint,
                            args.service,
                            args.revision,
                            client=client,
                            token_source=GoogleImpersonatedIdentityTokenSource(INVOKER),
                            runtime_source=args.runtime_source,
                        ),
                        timeout=600,
                    )

            asyncio.run(execute())
        if args.run or args.aggregate_only:
            summary = aggregate(value, raw, args.output, args.service, args.revision)
            write_exclusive(
                args.output
                / ("recomputed-summary.json" if args.aggregate_only else "summary.json"),
                encoded(summary),
            )
            print(
                json.dumps(
                    {"decision": summary["decision"], "completed_cases": summary["completed_cases"]}
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "status": "preflight",
                        "manifest_sha256": sha(raw),
                        "unique_cases": 8,
                        "requests": 10,
                    }
                )
            )
    except Exception:
        print('{"status":"failed","error":"qualification_failed"}')
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
