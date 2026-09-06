#!/usr/bin/env python3
"""Run a frozen twelve-image text/reference comparison once; never infer human acceptance."""

from __future__ import annotations

import argparse
import asyncio
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
from scripts import benchmark_klein_cold_start as cold  # noqa: E402

legacy, preparation = cold.legacy, cold.preparation
require = cold.require
EXPERIMENT = "simple-scenes-20260905-a"
FIXTURE = ROOT / "experiments/renderer-fidelity/simple-scene-controls-v1.json"
FIXTURE_SHA256 = "fc0a6ab8585aa7c71111fac50e02edd6787b6d436feb6393696429e4c8d29bba"
MAX_OPERATIONS, COST_CEILING_USD = 12, 1.75
PROFILE = {
    "width": 1024,
    "height": 576,
    "steps": 4,
    "guidance": 1.0,
    "max_reference_bytes": 2000000,
}
PINS = {
    "runtime_sha256": ROOT / "deploy/klein_scene_runtime.py",
    "reference_runtime_sha256": ROOT / "deploy/klein_reference_runtime.py",
    "deployment_sha256": ROOT / "deploy/modal_klein_simple_scenes.py",
    "client_sha256": Path(__file__),
}


def fixture():
    require(legacy.file_hash(FIXTURE) == FIXTURE_SHA256)
    value = legacy.protocol.decode_json(preparation.read_code(FIXTURE))
    require(set(value) == {"schema_version", "experiment_id", "profile", "controls"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["experiment_id"] == EXPERIMENT)
    require(preparation.encoded(value["profile"]) == preparation.encoded(PROFILE))
    require(isinstance(value["controls"], list) and len(value["controls"]) == 4)
    return value


def operations(value):
    rows = []
    for index, control in enumerate(value["controls"]):
        order = (
            ("text_next", "reference_next") if index % 2 == 0 else ("reference_next", "text_next")
        )
        base_id = hashlib.sha256(f"{EXPERIMENT}:{control['id']}:base".encode()).hexdigest()[:32]
        for variant in ("base", *order):
            prompt = control["before_prompt" if variant == "base" else "after_prompt"]
            rows.append(
                {
                    "ordinal": len(rows),
                    "request_id": hashlib.sha256(
                        f"{EXPERIMENT}:{control['id']}:{variant}".encode()
                    ).hexdigest()[:32],
                    "control_id": control["id"],
                    "variant": variant,
                    "seed": control["seed"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reference_request_id": base_id if variant == "reference_next" else None,
                }
            )
    return rows


def load_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    identity, _ = cold.frozen_cases()
    snapshot = snapshot_download(
        identity["model"],
        revision=identity["model_revision"],
        local_files_only=True,
    )
    return AutoTokenizer.from_pretrained(
        str(Path(snapshot) / "tokenizer"),
        local_files_only=True,
        trust_remote_code=False,
    )


def token_preflight():
    tokenizer = load_tokenizer()
    counts = {}
    for control in fixture()["controls"]:
        for state in ("before", "after"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": control[f"{state}_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            count = len(tokenizer(text)["input_ids"])
            require(0 < count <= 256)
            counts[f"{control['id']}:{state}"] = count
    return {
        "tokenizer_sha256": legacy.protocol.digest(
            {
                "backend": tokenizer.backend_tokenizer.to_str(),
                "chat_template": tokenizer.chat_template,
            }
        ),
        "token_counts": counts,
        "text_tokens_only": True,
        "reference_shape_qualified": False,
    }


def prepare_manifest():
    identity, _ = cold.frozen_cases()
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "status": "draft",
        "expires_at": None,
        "fixture_sha256": legacy.file_hash(FIXTURE),
        **{key: legacy.file_hash(path) for key, path in PINS.items()},
        "expected_identity": identity,
        "operations": operations(fixture()),
        "maximum_operations": MAX_OPERATIONS,
    }


def read_manifest(path):
    value = legacy.protocol.decode_json(preparation.read_code(path))
    expected = prepare_manifest()
    require(value.get("status") in {"draft", "authorized"})
    require(
        value.get("expires_at") is None
        if value["status"] == "draft"
        else type(value.get("expires_at")) is int and value["expires_at"] > 0
    )
    expected.update(status=value["status"], expires_at=value["expires_at"])
    require(preparation.encoded(value) == preparation.encoded(expected))
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


def header(args, manifest, authorization, tokens=None):
    result = cold.header(args, manifest, authorization)
    result["support_sha256"]["scripts/benchmark_klein_cold_start.py"] = legacy.file_hash(
        Path(cold.__file__)
    )
    return {
        **result,
        "harness_sha256": legacy.file_hash(Path(__file__)),
        "maximum_operations": MAX_OPERATIONS,
        "token_preflight": token_preflight() if tokens is None else tokens,
        "warm_excluded_ordinals": [0, 2],
        "timing_population": "later requests; additional shapes may incur first execution work",
        "timing_gate_seconds": 2.5,
        "timing_scope": (
            "client start through reference read/upload, inference, download, "
            "validation and storage"
        ),
    }


def validate_tokens(payload, operation, evidence_header):
    state = "before" if operation["variant"] == "base" else "after"
    key = f"{operation['control_id']}:{state}"
    require(
        payload["metrics"]["token_count"] == evidence_header["token_preflight"]["token_counts"][key]
    )


def artifact(output, ordinal, role):
    return output / f"operation-{ordinal:02}" / f"{role}.jpg"


def read_artifact(path, checksum, limit=cold.MAX_ARTIFACT_BYTES):
    require(path.resolve() == path.absolute() and path.is_file() and path.stat().st_size <= limit)
    data = path.read_bytes()
    require(
        hashlib.sha256(data).hexdigest() == checksum and cold._jpeg_dimensions(data) == (1024, 576)
    )
    return data


def validate_payload(payload, operation, manifest, reference_sha256):
    require(
        isinstance(payload, dict)
        and set(payload) == {"request_id", "identity", "location", "metrics", "master", "depth"}
    )
    require(payload["request_id"] == operation["request_id"])
    require(
        preparation.encoded(payload["identity"])
        == preparation.encoded(manifest["expected_identity"])
    )
    location = payload["location"]
    require(
        isinstance(location, dict)
        and set(location) == {"cloud", "compute_region", "container_sha256"}
    )
    require(location["cloud"] in {"CLOUD_PROVIDER_AWS", "CLOUD_PROVIDER_GCP", "CLOUD_PROVIDER_OCI"})
    require(
        isinstance(location["compute_region"], str)
        and re.fullmatch(r"us-[a-z0-9-]{1,48}", location["compute_region"])
    )
    require(cold.is_hash(location["container_sha256"]))
    metrics = payload["metrics"]
    require(
        isinstance(metrics, dict)
        and set(metrics)
        == set(cold.METRICS)
        | {
            "seed",
            "token_count",
            "sequence_bucket",
            "master_sha256",
            "depth_sha256",
            "reference_sha256",
            "reference_runtime_sha256",
            "reference_conditioned",
            "reference_width",
            "reference_height",
        }
    )
    require(all(cold.finite(metrics[key]) for key in cold.METRICS))
    require(type(metrics["seed"]) is int and metrics["seed"] == operation["seed"])
    require(type(metrics["token_count"]) is int and 0 < metrics["token_count"] <= 256)
    require(
        type(metrics["sequence_bucket"]) is int
        and metrics["sequence_bucket"] == (128 if metrics["token_count"] <= 128 else 256)
    )
    require(metrics["reference_runtime_sha256"] == manifest["reference_runtime_sha256"])
    require(metrics["reference_conditioned"] is (operation["variant"] == "reference_next"))
    require(metrics["reference_sha256"] == reference_sha256)
    for dimension, expected in (("width", 1024), ("height", 576)):
        actual = metrics[f"reference_{dimension}"]
        require(
            (type(actual) is int and actual == expected) if reference_sha256 else actual is None
        )
    require((reference_sha256 is not None) == (operation["variant"] == "reference_next"))
    for role in ("master", "depth"):
        data = payload[role]
        require(isinstance(data, bytes) and 0 < len(data) <= cold.MAX_ARTIFACT_BYTES)
        require(
            hashlib.sha256(data).hexdigest() == metrics[f"{role}_sha256"]
            and cold._jpeg_dimensions(data) == (1024, 576)
        )


class SimpleClient(cold.LatencyClient):
    def __init__(self, manifest, *, deadline, modal_module=None):
        super().__init__(manifest, modal_module=modal_module)
        self.deadline = deadline

    async def render(self, operation, reference, reference_sha256):
        self.last_failure = None
        self.last_timings = dict.fromkeys(cold.TIMINGS, 0.0)
        timings = self.last_timings
        try:
            remaining = self.deadline - time.time()
            require(remaining > 0)
            async with asyncio.timeout(min(310, remaining)):
                begun = time.perf_counter()
                if self.instance is None:
                    self.modal = self.modal or importlib.import_module("modal")
                    cls = self.modal.Cls.from_name(
                        "bookforge-klein-simple-scenes", "SimpleSceneRenderer"
                    )
                    await asyncio.wait_for(cls.hydrate.aio(), 10)
                    self.instance = cls()
                timings["lookup_seconds"] = time.perf_counter() - begun
                begun = time.perf_counter()
                task = asyncio.create_task(
                    self.instance.render.spawn.aio(
                        operation["request_id"],
                        reference_jpeg=reference,
                        reference_sha256=reference_sha256,
                    )
                )
                try:
                    call = await asyncio.wait_for(asyncio.shield(task), 30)
                except BaseException:
                    self.last_failure = "sdk_submission_unknown"
                    task.add_done_callback(self.late_submission)
                    self.pending_cleanup.add(task)
                    task.add_done_callback(self.pending_cleanup.discard)
                    raise
                finally:
                    timings["submission_seconds"] = time.perf_counter() - begun
                begun = time.perf_counter()
                try:
                    payload = await call.get.aio()
                except BaseException:
                    self.last_failure = "sdk_result_failed"
                    try:
                        await asyncio.shield(self.cancel_call(call))
                    except BaseException:
                        self.last_failure = "sdk_result_cleanup_required"
                    raise
                finally:
                    timings["result_download_seconds"] = time.perf_counter() - begun
                return payload, timings
        except BaseException:
            self.last_failure = self.last_failure or "local_operation_failed"
            raise


async def execute(args, manifest, evidence_header, client):
    journal = args.output / "journal.jsonl"
    legacy.write_exclusive(journal, (json.dumps(evidence_header, sort_keys=True) + "\n").encode())
    completed = {}
    try:
        for operation in manifest["operations"]:
            require(time.time() < min(args.deadline_unix, manifest["expires_at"]))
            ordinal = operation["ordinal"]
            reference_sha = (
                completed[operation["reference_request_id"]]["metrics"]["master_sha256"]
                if operation["reference_request_id"]
                else None
            )
            legacy.append(
                journal,
                {
                    "kind": "start",
                    "ordinal": ordinal,
                    "request_sha256": legacy.protocol.digest(operation),
                    "reference_sha256": reference_sha,
                },
            )
            started = time.perf_counter()
            try:
                reference = None
                if reference_sha:
                    reference = read_artifact(
                        artifact(args.output, ordinal - ordinal % 3, "master"),
                        reference_sha,
                        PROFILE["max_reference_bytes"],
                    )
                payload, timings = await client.render(operation, reference, reference_sha)
                begun = time.perf_counter()
                validate_payload(payload, operation, manifest, reference_sha)
                validate_tokens(payload, operation, evidence_header)
                timings["validation_seconds"] = time.perf_counter() - begun
                begun = time.perf_counter()
                artifact(args.output, ordinal, "master").parent.mkdir(mode=0o700)
                for role in ("master", "depth"):
                    legacy.write_exclusive(artifact(args.output, ordinal, role), payload[role])
                timings["storage_seconds"] = time.perf_counter() - begun
                timings["total_artifact_ready_seconds"] = time.perf_counter() - started
                safe = {
                    key: value for key, value in payload.items() if key not in {"master", "depth"}
                }
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "ok",
                        "payload": safe,
                        "timings": timings,
                    },
                )
                completed[operation["request_id"]] = safe
            except BaseException:
                timings = client.last_timings or dict.fromkeys(cold.TIMINGS, 0.0)
                timings["total_artifact_ready_seconds"] = time.perf_counter() - started
                legacy.append(
                    journal,
                    {
                        "kind": "result",
                        "ordinal": ordinal,
                        "status": "failed",
                        "code": client.last_failure
                        if client.last_failure in cold.FAILURES
                        else "local_operation_failed",
                        "timings": timings,
                    },
                )
                raise
        legacy.append(journal, {"kind": "complete"})
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
    path = args.output / "journal.jsonl"
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1000000)
    rows = [legacy.protocol.decode_json(line) for line in path.read_bytes().splitlines()]
    require(rows and preparation.encoded(rows[0]) == preparation.encoded(evidence_header))
    started, pending, failed, complete, cleanup = 0, None, False, False, None
    results, completed = [], {}
    for row in rows[1:]:
        require(cleanup is None)
        if row.get("kind") == "cleanup":
            require(set(row) == {"kind", "known_calls_cancelled", "external_app_stop_required"})
            require(
                type(row["known_calls_cancelled"]) is bool
                and type(row["external_app_stop_required"]) is bool
                and row["known_calls_cancelled"] is not row["external_app_stop_required"]
            )
            cleanup = row
            continue
        require(not complete)
        if row.get("kind") == "start":
            require(not failed and pending is None and started < MAX_OPERATIONS)
            operation = manifest["operations"][started]
            reference_sha = (
                completed[operation["reference_request_id"]]["metrics"]["master_sha256"]
                if operation["reference_request_id"]
                else None
            )
            require(
                preparation.encoded(row)
                == preparation.encoded(
                    {
                        "kind": "start",
                        "ordinal": started,
                        "request_sha256": legacy.protocol.digest(operation),
                        "reference_sha256": reference_sha,
                    }
                )
            )
            pending = started
            started += 1
        elif row.get("kind") == "result":
            require(
                pending is not None
                and type(row.get("ordinal")) is int
                and row["ordinal"] == pending
            )
            require(row.get("status") in {"ok", "failed"})
            require(
                set(row)
                == (
                    {"kind", "ordinal", "status", "payload", "timings"}
                    if row["status"] == "ok"
                    else {"kind", "ordinal", "status", "code", "timings"}
                )
            )
            require(
                isinstance(row["timings"], dict)
                and set(row["timings"]) == set(cold.TIMINGS)
                and all(cold.finite(v) for v in row["timings"].values())
            )
            operation = manifest["operations"][pending]
            if row["status"] == "ok":
                safe = row["payload"]
                require(set(safe) == {"request_id", "identity", "location", "metrics"})
                payload = {
                    **safe,
                    **{
                        role: read_artifact(
                            artifact(args.output, pending, role), safe["metrics"][f"{role}_sha256"]
                        )
                        for role in ("master", "depth")
                    },
                }
                validate_payload(payload, operation, manifest, reference_sha)
                validate_tokens(payload, operation, evidence_header)
                if reference_sha:
                    read_artifact(
                        artifact(args.output, pending - pending % 3, "master"),
                        reference_sha,
                        PROFILE["max_reference_bytes"],
                    )
                completed[operation["request_id"]] = safe
            else:
                require(row["code"] in cold.FAILURES)
                failed = True
            results.append(row)
            pending = None
        elif row == {"kind": "complete"}:
            require(not failed and pending is None and len(results) == MAX_OPERATIONS)
            complete = True
        else:
            raise ValueError("simple_scene_journal_invalid")
    successful = [r for r in results if r["status"] == "ok"]
    all_ok = (
        complete
        and not failed
        and pending is None
        and cleanup is not None
        and cleanup["known_calls_cancelled"]
    )
    containers = {r["payload"]["location"]["container_sha256"] for r in successful}
    stable = len(containers) == 1
    variants = {}
    for variant in ("base", "text_next", "reference_next"):
        selected = [
            r for r in successful if manifest["operations"][r["ordinal"]]["variant"] == variant
        ]
        warm = [r for r in selected if r["ordinal"] not in (0, 2)]
        variants[variant] = {
            "all_client_seconds": legacy.distribution(
                [r["timings"]["total_artifact_ready_seconds"] for r in selected]
            ),
            "eligible_client_seconds": legacy.distribution(
                [r["timings"]["total_artifact_ready_seconds"] for r in warm]
            ),
            "server_image_seconds": legacy.distribution(
                [r["payload"]["metrics"]["image_seconds"] for r in selected]
            ),
            "eligible_ordinals": [r["ordinal"] for r in warm],
        }
    timing_pass = (
        all_ok
        and stable
        and all(
            variants[v]["eligible_client_seconds"]["max"] is not None
            and variants[v]["eligible_client_seconds"]["max"] <= 2.5
            for v in ("text_next", "reference_next")
        )
    )
    return {
        "schema_version": 1,
        "kind": "simple-scene-comparison-summary",
        "header": evidence_header,
        "journal_sha256": legacy.file_hash(path),
        "started": started,
        "results": len(results),
        "failures": sum(r["status"] == "failed" for r in results),
        "unresolved_requests": int(pending is not None),
        "complete": all_ok,
        "artifacts_verified": 2 * len(successful),
        "container_count": len(containers),
        "single_container": stable,
        "variants": variants,
        "overall_client_seconds": legacy.distribution(
            [row["timings"]["total_artifact_ready_seconds"] for row in results]
        ),
        "initial_execution_client_seconds": [
            {"ordinal": row["ordinal"], "seconds": row["timings"]["total_artifact_ready_seconds"]}
            for row in results
            if row["ordinal"] in (0, 2)
        ],
        "warm_excluded_ordinals": [0, 2],
        "latency_pass": timing_pass,
        "cleanup": cleanup,
        "human_review_complete": False,
        "correctness_qualified": False,
        "decision": "ungraded" if all_ok else "reject",
        "platform_audit_pending": True,
        "physical_display_measured": False,
        "production_promotion_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--deadline-unix", type=float)
    modes = parser.add_mutually_exclusive_group()
    for mode in ("prepare", "preflight", "run", "aggregate-only"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            require(
                not any(
                    (args.authorization, args.authorization_sha256, args.deadline_unix, args.output)
                )
            )
            legacy.write_exclusive(args.manifest, preparation.encoded(prepare_manifest()))
            return 0
        manifest = read_manifest(args.manifest)
        require(args.output is not None)
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
            legacy.write_exclusive(
                args.output / "recomputed-summary.json",
                preparation.encoded(
                    aggregate(args, manifest, header(args, manifest, authorization))
                ),
            )
            return 0
        require(
            not args.output.exists()
            and not args.output.is_symlink()
            and args.output.parent.is_dir()
        )
        tokens = token_preflight()
        if not args.run:
            args.output.mkdir(mode=0o700)
            legacy.write_exclusive(
                args.output / "preflight.json",
                preparation.encoded(
                    {
                        "schema_version": 1,
                        "kind": "simple-scene-preflight",
                        "manifest_sha256": legacy.file_hash(args.manifest),
                        "status": manifest["status"],
                        "maximum_operations": MAX_OPERATIONS,
                        "cost_ceiling_usd": COST_CEILING_USD,
                        "generation_calls": 0,
                        "token_preflight": tokens,
                    }
                ),
            )
            return 0
        require(cold.finite(args.deadline_unix) and 0 < args.deadline_unix - time.time() <= 600)
        require(time.time() < manifest["expires_at"] <= time.time() + 7200)
        evidence_header = header(args, manifest, authorization, tokens)
        legacy.claim(args, evidence_header)
        args.output.mkdir(mode=0o700)
        client = SimpleClient(manifest, deadline=min(args.deadline_unix, manifest["expires_at"]))
        try:
            asyncio.run(execute(args, manifest, evidence_header, client))
        finally:
            legacy.write_exclusive(
                args.output / "summary.json",
                preparation.encoded(aggregate(args, manifest, evidence_header)),
            )
        return 0
    except BaseException:
        print("simple-scene comparison stopped; inspect sanitized evidence and supervised cleanup")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
