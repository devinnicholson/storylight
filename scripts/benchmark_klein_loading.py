#!/usr/bin/env python3
"""Four fresh-process shard-loading trials with a parent-enforced deadline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bookforge.klein_latency_client import TIMINGS, LatencyClient, finite  # noqa: E402
from scripts import benchmark_klein_cold_start as cold  # noqa: E402

APP = "bookforge-klein-loading"
EXPERIMENT = "klein-loading-20260905-a"
MANIFEST = ROOT / "benchmarks/renderer-loading-2026-09-05/manifest.json"
DEPLOYMENT = ROOT / "deploy/modal_klein_loading.py"
LEDGER = ROOT / ".bookforge/overnight-candidate/modal-ledger.json"
ATTEMPT = LEDGER.parent / "klein-loading-20260905-a-corrected-attempt.json"
WORK_SECONDS, CLEANUP_SECONDS, HOLD_USD = 260, 60, 1.20
SCHEDULE = ("baseline", "candidate", "candidate", "baseline")
METADATA_FIELDS = (
    "image_id",
    "resources",
    "autoscaler_settings",
    "max_inputs",
    "single_use_containers",
    "max_concurrent_inputs",
    "startup_timeout_secs",
    "timeout_secs",
    "cloud_provider_str",
    "routing_region",
    "scheduler_placement",
    "retry_policy",
    "checkpointing_enabled",
    "enable_gpu_snapshot",
    "_experimental_enable_gpu_snapshot",
    "is_class",
)
require = cold.require
sha = cold.legacy.file_hash
write = cold.legacy.write_exclusive


def save(path, value):
    write(path, cold.preparation.encoded(value))


def manifest():
    value = cold.legacy.protocol.decode_json(MANIFEST.read_bytes())
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "experiment_id",
            "expires_at",
            "image_id",
            "cache_id",
            "runtime_sha256",
            "deployment_sha256",
            "client_sha256",
            "expected_identity",
            "cases",
            "operations",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["status"] in {"draft", "authorized"})
    require(
        value["expires_at"] is None
        if value["status"] == "draft"
        else type(value["expires_at"]) is int and value["expires_at"] > 0
    )
    identity, cases = cold.frozen_cases()
    require(value["experiment_id"] == EXPERIMENT)
    require(
        cold.preparation.encoded(value["expected_identity"]) == cold.preparation.encoded(identity)
    )
    require(cold.preparation.encoded(value["cases"]) == cold.preparation.encoded(cases))
    require(value["image_id"] == cold.IMAGE_ID and value["cache_id"] == cold.CACHE_ID)
    for key, path in (
        ("runtime_sha256", ROOT / "deploy/klein_scene_runtime.py"),
        ("deployment_sha256", DEPLOYMENT),
        ("client_sha256", Path(__file__)),
    ):
        require(value[key] == sha(path))
    require(
        cold.preparation.encoded(value["operations"])
        == cold.preparation.encoded(
            [
                {
                    "ordinal": i,
                    "variant": variant,
                    "request_id": hashlib.sha256(f"{EXPERIMENT}:{i}".encode()).hexdigest()[:32],
                }
                for i, variant in enumerate(SCHEDULE)
            ]
        )
    )
    return value


def header(args):
    return {
        "kind": "header",
        "manifest_sha256": sha(MANIFEST),
        "ledger_sha256": args.ledger_sha256,
        "reserved_usd": HOLD_USD,
        "dependencies": {
            name: sha(ROOT / name)
            for name in (
                "scripts/benchmark_klein_cold_start.py",
                "scripts/benchmark_klein_latency.py",
                "scripts/prepare_klein_region_comparison.py",
                "deploy/klein_latency_protocol.py",
                "src/bookforge/klein_latency_client.py",
                "src/bookforge/finite_modal_provider.py",
            )
        },
    }


def validate_metadata(data):
    require(len(data["ranked_functions"]) == 1)
    function = data["ranked_functions"][0]["function"]
    require(function["image_id"] == cold.IMAGE_ID)
    require(
        function["resources"]
        == {
            "gpu_config": {"count": 1, "gpu_type": "L4"},
            "memory_mb": 65536,
            "memory_mb_max": 65536,
            "milli_cpu": 8000,
            "milli_cpu_max": 8000,
        }
    )
    scale = function["autoscaler_settings"]
    require(scale["max_containers"] == 1 and scale.get("min_containers", 0) == 0)
    require(scale.get("buffer_containers", 0) == 0 and scale["scaledown_window"] == 2)
    require(function["max_inputs"] == 1 and function["single_use_containers"] is True)
    require(function["max_concurrent_inputs"] == 1)
    require(function["startup_timeout_secs"] == 30 and function["timeout_secs"] == 120)
    require(function["cloud_provider_str"] == "aws" and function["routing_region"] == "us-east")
    require(function["scheduler_placement"]["regions"] == ["us-east"])
    require(function.get("retry_policy", {}).get("retries", 0) == 0)
    require(
        not any(
            function.get(key)
            for key in (
                "checkpointing_enabled",
                "enable_gpu_snapshot",
                "_experimental_enable_gpu_snapshot",
                "experimental_options",
                "is_class",
                "web_url",
            )
        )
    )
    keys = (
        "image_id",
        "resources",
        "autoscaler_settings",
        "max_inputs",
        "single_use_containers",
        "max_concurrent_inputs",
        "startup_timeout_secs",
        "timeout_secs",
        "cloud_provider_str",
        "routing_region",
        "scheduler_placement",
        "retry_policy",
    )
    return {key: function[key] for key in keys if key in function}


def authorize(args):
    value = manifest()
    prior = MANIFEST.parent / "predispatch"
    require(
        sha(prior / "supervisor.json")
        == "2a4021b31c4f098c140131229550edb23a2584001c002625f25b8f72c27d92e3"
    )
    require(
        sha(prior / "shutdown.json")
        == "866af4ae73bc70a40045bce4244a8e2c56051d45317b30701c4d6b349a57d8ba"
    )
    previous_seconds = json.loads((prior / "supervisor.json").read_bytes())["total_wall_seconds"]
    require(2 * 0.00082054 * (previous_seconds + WORK_SECONDS + CLEANUP_SECONDS) + 0.5 < HOLD_USD)
    require(value["status"] == "authorized")
    require(
        type(value["expires_at"]) is int and time.time() < value["expires_at"] <= time.time() + 7200
    )
    require(args.ledger == LEDGER and args.ledger.resolve() == LEDGER)
    require(args.ledger_sha256 and sha(args.ledger) == args.ledger_sha256)
    ledger = json.loads(args.ledger.read_bytes())
    require(ledger["reservations"][f"reservation:{EXPERIMENT}"] == HOLD_USD)
    envelope = ledger["envelope"]
    usage = ledger["estimated_usage_usd"]
    require(usage <= envelope["run_cap_usd"])
    require(
        usage + envelope["usage_before_lab_usd"]
        <= (
            envelope["monthly_credit_usd"]
            + envelope["authorized_paid_usd"]
            - envelope["reserve_usd"]
        )
    )
    return value


def validate(payload, operation, value):
    require(payload["ordinal"] == operation["ordinal"])
    require(
        payload["loading_environment"]
        == {
            "HF_ENABLE_PARALLEL_LOADING": "true"
            if operation["variant"] == "candidate"
            else "false",
            "HF_PARALLEL_LOADING_WORKERS": "4",
        }
    )
    require(finite(payload["stages"]["first_artifact_seconds"]))
    require(0 < payload["stages"]["first_artifact_seconds"] <= payload["stages"]["worker_seconds"])
    inventory = payload["shard_inventory"]
    require(set(inventory) == {"klein/transformer", "klein/text_encoder", "klein/vae", "depth"})
    for row in inventory.values():
        require(
            set(row) == {"index_count", "indexed_shards", "safetensors_files", "referenced_bytes"}
        )
        require(all(type(n) is int and n >= 0 for n in row.values()))
        require(row["index_count"] <= 1 and row["indexed_shards"] <= row["safetensors_files"])
    require(any(row["indexed_shards"] > 1 for row in inventory.values()))
    base = {
        key: item
        for key, item in payload.items()
        if key not in {"ordinal", "loading_environment", "shard_inventory"}
    }
    base["stages"] = {
        key: item for key, item in payload["stages"].items() if key != "first_artifact_seconds"
    }
    cold.validate_payload(base, operation, value)
    for sample in payload["samples"]:
        case = value["cases"][sample["case_index"]]
        for role in ("master", "depth"):
            require(sample["metrics"][f"{role}_sha256"] == case[f"{role}_sha256"])


async def worker(args):
    value = authorize(args)
    require(json.loads(ATTEMPT.read_bytes()) == {"output": str(args.output), **header(args)})
    save(args.output / "worker-attempt.json", header(args))
    import modal

    client = LatencyClient(value)
    subprocess.run(
        [sys.executable, "-m", "modal", "deploy", str(DEPLOYMENT)], check=True, timeout=120
    )
    rows = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "modal", "app", "list", "--json"], timeout=15
        )
    )
    matches = [r for r in rows if r["description"] == APP and r["state"] != "stopped"]
    require(len(matches) == 1 and int(matches[0]["tasks"]) == 0)
    app_id = matches[0]["app_id"]
    save(args.output / "app.json", {"app_id": app_id})
    subprocess.run(
        [sys.executable, str(Path(__file__)), "--metadata", "--output", str(args.output)],
        check=True,
        timeout=20,
    )
    metadata = json.loads((args.output / "deployment-check.json").read_bytes())
    function = modal.Function.from_name(APP, "loading_cycle")
    await asyncio.wait_for(function.hydrate.aio(), 15)
    require(function.object_id == metadata["function_id"])

    async def spawn(*, request):
        return await function.spawn.aio(request["ordinal"])

    client.instance = SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
    journal = args.output / "journal.jsonl"
    write(journal, (json.dumps(header(args), sort_keys=True) + "\n").encode())
    try:
        for operation in value["operations"]:
            require(time.time() < value["expires_at"])
            cold.legacy.append(journal, {"kind": "start", **operation})
            started = time.perf_counter()
            timings = dict.fromkeys(TIMINGS, 0.0)
            payload = await client.sdk(operation, timings)
            validate(payload, operation, value)
            ordinal = operation["ordinal"]
            for i, sample in enumerate(payload["samples"]):
                for role in ("master", "depth"):
                    write(args.output / f"{ordinal}-{i}-{role}.jpg", sample.pop(role))
            cold.legacy.append(
                journal,
                {
                    "kind": "result",
                    "ordinal": ordinal,
                    "payload": payload,
                    "client_cycle_seconds": time.perf_counter() - started,
                    "timings": timings,
                },
            )
    finally:
        save(args.output / "call-cleanup.json", {"known_calls_cancelled": await client.cleanup()})


async def inspect_deployment(output):
    # Raw RPC clients and the public SDK use different event loops; isolate their singletons.
    from google.protobuf.json_format import MessageToDict
    from modal.client import _Client
    from modal_proto import api_pb2

    app_id = json.loads((output / "app.json").read_bytes())["app_id"]
    connection = await _Client.from_env()
    info = await connection.stub.FunctionGet(
        api_pb2.FunctionGetRequest(
            app_name=APP, object_tag="loading_cycle", environment_name="main"
        )
    )
    layout = await connection.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    require(info.function_id in layout.app_layout.function_ids.values())
    data = MessageToDict(info.function, preserving_proto_field_name=True)
    save(
        output / "deployment-observed.json",
        {
            "functions": [
                {
                    **{
                        key: row["function"][key]
                        for key in METADATA_FIELDS
                        if key in row["function"]
                    },
                    "has_web_url": bool(row["function"].get("web_url")),
                    "has_experimental_options": bool(row["function"].get("experimental_options")),
                }
                for row in data.get("ranked_functions", [])
            ]
        },
    )
    metadata = validate_metadata(data)
    save(
        output / "deployment-check.json",
        {"app_id": app_id, "function_id": info.function_id, **metadata},
    )


def aggregate(output):
    value = manifest()
    rows = [
        cold.legacy.protocol.decode_json(line)
        for line in (output / "journal.jsonl").read_bytes().splitlines()
    ]
    require(len(rows) == 9)
    authorization = json.loads((output / "authorization.json").read_bytes())
    require(rows[0] == header(SimpleNamespace(ledger_sha256=authorization["ledger_sha256"])))
    require(
        authorization["manifest_sha256"] == sha(MANIFEST)
        and authorization["reserved_usd"] == HOLD_USD
    )
    cleanup = cold.legacy.protocol.decode_json((output / "call-cleanup.json").read_bytes())
    require(set(cleanup) == {"known_calls_cancelled"} and cleanup["known_calls_cancelled"] is True)
    supervisor = json.loads((output / "supervisor.json").read_bytes())
    require(supervisor["returncode"] == 0 and supervisor["external_app_shutdown_verified"] is True)
    shutdown = json.loads((output / "shutdown.json").read_bytes())
    require(
        shutdown["app"]["app_id"] == supervisor["app_id"] and shutdown["active_containers"] == 0
    )
    require(shutdown["app"]["state"] == "stopped" and int(shutdown["app"]["tasks"]) == 0)
    rows = rows[1:]
    results = []
    for ordinal, operation in enumerate(value["operations"]):
        require(rows[ordinal * 2] == {"kind": "start", **operation})
        row = rows[ordinal * 2 + 1]
        require(row["kind"] == "result" and row["ordinal"] == ordinal)
        require(finite(row["client_cycle_seconds"]))
        require(
            set(row["timings"]) == set(TIMINGS) and all(finite(v) for v in row["timings"].values())
        )
        payload = row["payload"]
        for i, sample in enumerate(payload["samples"]):
            for role in ("master", "depth"):
                sample[role] = (output / f"{ordinal}-{i}-{role}.jpg").read_bytes()
        validate(payload, operation, value)
        results.append(row)
    require(len({r["payload"]["location"]["container_sha256"] for r in results}) == 4)
    require(
        all(
            r["payload"]["shard_inventory"] == results[0]["payload"]["shard_inventory"]
            for r in results
        )
    )
    summaries = {}
    for variant in ("baseline", "candidate"):
        group = [r for r in results if r["payload"]["variant"] == variant]
        summaries[variant] = {
            key: statistics.median(r["payload"]["stages"][key] for r in group)
            for key in (*cold.STAGES, "first_artifact_seconds")
        }
        summaries[variant]["client_four_render_cycle_seconds"] = statistics.median(
            r["client_cycle_seconds"] for r in group
        )
    baseline, candidate = summaries["baseline"], summaries["candidate"]
    gain = 1 - candidate["first_artifact_seconds"] / baseline["first_artifact_seconds"]
    return {
        "manifest_sha256": sha(MANIFEST),
        "journal_sha256": sha(output / "journal.jsonl"),
        "samples_per_variant": 2,
        "artifacts_verified": 32,
        "exact_hash_match": True,
        "medians": summaries,
        "first_artifact_reduction_fraction": gain,
        "scope": "worker entry through first master/depth; excludes queue and client transport",
        "promotion": "not_promoted_small_sample",
        "generation_calls": 16,
    }


def supervise(args):
    authorize(args)
    require(args.output.is_absolute() and args.output.resolve() == args.output)
    args.output.mkdir(mode=0o700)
    save(
        args.output / "authorization.json",
        {
            "manifest_sha256": sha(MANIFEST),
            "ledger_sha256": args.ledger_sha256,
            "reserved_usd": HOLD_USD,
            "work_seconds": WORK_SECONDS,
            "cleanup_seconds": CLEANUP_SECONDS,
        },
    )
    command = [sys.executable, "-m", "modal"]

    def cli(*parts, timeout=15):
        return subprocess.check_output([*command, *parts], timeout=timeout)

    before = json.loads(cli("app", "list", "--json"))
    require(not any(r["description"] == APP and r["state"] != "stopped" for r in before))
    previous = {r["app_id"] for r in before}
    authorize(args)
    save(ATTEMPT, {"output": str(args.output), **header(args)})
    started = time.monotonic()
    record = {"external_app_shutdown_verified": False, "returncode": 1}
    process = None

    def interrupted(*_):
        raise InterruptedError("loading supervisor interrupted")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (args.output / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--worker",
                    "--output",
                    str(args.output),
                    "--ledger",
                    str(args.ledger),
                    "--ledger-sha256",
                    args.ledger_sha256,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["returncode"] = process.wait(timeout=WORK_SECONDS)
    finally:
        record["work_wall_seconds"] = time.monotonic() - started
        deadline = min(started + WORK_SECONDS + CLEANUP_SECONDS, time.monotonic() + CLEANUP_SECONDS)
        try:
            app_path = args.output / "app.json"
            if app_path.exists():
                app_id = json.loads(app_path.read_bytes())["app_id"]
            else:
                app_id = None
                while time.monotonic() + 31 < deadline:
                    try:
                        rows = json.loads(cli("app", "list", "--json", timeout=5))
                    except (OSError, ValueError, subprocess.SubprocessError):
                        record["recovery_read_failures"] = (
                            record.get("recovery_read_failures", 0) + 1
                        )
                        rows = []
                    matches = [
                        r for r in rows if r["description"] == APP and r["app_id"] not in previous
                    ]
                    require(len(matches) <= 1)
                    if matches:
                        app_id = matches[0]["app_id"]
                        break
                    time.sleep(0.5)
                require(app_id is not None)
            require(re.fullmatch(r"ap-[A-Za-z0-9]+", app_id))
            record["app_id"] = app_id
            require(time.monotonic() + 15 < deadline)
            cli("app", "stop", "--yes", app_id)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            while time.monotonic() + 10 < deadline:
                apps = json.loads(cli("app", "list", "--json", timeout=5))
                containers = json.loads(cli("container", "list", "--json", timeout=5))
                exact = [r for r in apps if r["app_id"] == app_id]
                require(len(exact) == 1 and all("app_id" in r for r in containers))
                if (
                    exact[0]["state"] == "stopped"
                    and int(exact[0]["tasks"]) == 0
                    and not any(r["app_id"] == app_id for r in containers)
                ):
                    record["external_app_shutdown_verified"] = True
                    save(args.output / "shutdown.json", {"app": exact[0], "active_containers": 0})
                    break
                time.sleep(0.5)
            require(record["external_app_shutdown_verified"])
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            record["total_wall_seconds"] = time.monotonic() - started
            save(args.output / "supervisor.json", record)
            signal.signal(signal.SIGTERM, previous_handler)
    require(record["returncode"] == 0)
    save(args.output / "summary.json", aggregate(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "run", "worker", "aggregate", "metadata"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--ledger-sha256")
    args = parser.parse_args()
    if args.preflight:
        manifest()
        print(json.dumps({"generation_calls": 0, "operations": 4, "reserved_usd": HOLD_USD}))
    elif args.worker:
        asyncio.run(worker(args))
    elif args.aggregate:
        print(json.dumps(aggregate(args.output), sort_keys=True, indent=2))
    elif args.metadata:
        asyncio.run(inspect_deployment(args.output))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
