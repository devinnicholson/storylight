#!/usr/bin/env python3
"""Preflight, supervise and verify one finite conditioning comparison."""

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
from scripts import benchmark_klein_loading as loading  # noqa: E402

APP = "bookforge-klein-conditioning"
EXPERIMENT = "klein-conditioning-20260906-a"
MANIFEST = ROOT / "benchmarks/renderer-conditioning-2026-09-06/manifest.json"
DEPLOYMENT = ROOT / "deploy/modal_klein_conditioning.py"
LEDGER = loading.LEDGER
ATTEMPT = LEDGER.parent / f"{EXPERIMENT}-attempt.json"
WORK_SECONDS, CLEANUP_SECONDS, HOLD_USD = 150, 30, 0.50
SCHEDULE = (
    (0, "baseline"),
    (1, "baseline"),
    (0, "baseline"),
    (0, "candidate"),
    (1, "candidate"),
    (1, "baseline"),
    (0, "candidate"),
    (0, "baseline"),
    (1, "baseline"),
    (1, "candidate"),
)
cold, require, sha, save = loading.cold, loading.require, loading.sha, loading.save


def read(path):
    return cold.legacy.protocol.decode_json(path.read_bytes())


def manifest():
    value = read(MANIFEST)
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "expires_at",
            "experiment_id",
            "image_id",
            "cache_id",
            "maximum_calls",
            "cases",
            "expected_identity",
            "sources",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["status"] in {"draft", "authorized"})
    require(
        value["expires_at"] is None
        if value["status"] == "draft"
        else type(value["expires_at"]) is int and value["expires_at"] > 0
    )
    require(value["experiment_id"] == EXPERIMENT and value["image_id"] == cold.IMAGE_ID)
    require(
        value["cache_id"] == cold.CACHE_ID
        and type(value["maximum_calls"]) is int
        and value["maximum_calls"] == 1
    )
    identity, cases = cold.frozen_cases()
    require(
        cold.preparation.encoded([value["expected_identity"], value["cases"]])
        == cold.preparation.encoded([identity, cases])
    )
    sources = {
        name: ROOT / "deploy" / name
        for name in (
            "klein_scene_runtime.py",
            "klein_conditioning.py",
            "klein_conditioning_probe.py",
            "modal_klein_conditioning.py",
        )
    }
    sources.update(
        {
            "client": Path(__file__),
            "loading_helpers": Path(loading.__file__),
            "cold_helpers": Path(cold.__file__),
            "transport": ROOT / "src/bookforge/klein_latency_client.py",
        }
    )
    require(value["sources"] == {name: sha(path) for name, path in sources.items()})
    return value


def authorize(args):
    value = manifest()
    require(
        value["status"] == "authorized" and time.time() < value["expires_at"] <= time.time() + 7200
    )
    require(args.ledger_sha256 == sha(LEDGER))
    ledger = read(LEDGER)
    require(ledger["reservations"][f"reservation:{EXPERIMENT}"] == HOLD_USD)
    env, usage = ledger["envelope"], ledger["estimated_usage_usd"]
    require(usage <= env["run_cap_usd"])
    require(
        usage + env["usage_before_lab_usd"]
        <= env["monthly_credit_usd"] + env["authorized_paid_usd"] - env["reserve_usd"]
    )
    return value


async def metadata(output):
    from google.protobuf.json_format import MessageToDict
    from modal.client import _Client
    from modal_proto import api_pb2

    app_id = read(output / "app.json")["app_id"]
    client = await _Client.from_env()
    info = await client.stub.FunctionGet(
        api_pb2.FunctionGetRequest(app_name=APP, object_tag="compare", environment_name="main")
    )
    layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    require(info.function_id in layout.app_layout.function_ids.values())
    data = MessageToDict(info.function, preserving_proto_field_name=True)
    save(
        output / "deployment-observed.json",
        {
            "functions": [
                {
                    **{
                        k: row["function"][k]
                        for k in loading.METADATA_FIELDS
                        if k in row["function"]
                    },
                    "has_web_url": bool(row["function"].get("web_url")),
                    "has_experimental_options": bool(row["function"].get("experimental_options")),
                }
                for row in data.get("ranked_functions", [])
            ]
        },
    )
    save(
        output / "deployment-check.json",
        {"app_id": app_id, "function_id": info.function_id, **loading.validate_metadata(data)},
    )


def validate(payload, value):
    require(payload["manifest_sha256"] == sha(MANIFEST))
    require(payload["identity"] == value["expected_identity"])
    require(
        payload["location"]["cloud"] == "CLOUD_PROVIDER_AWS"
        and payload["location"]["region"] == "us-east-1"
    )
    require(re.fullmatch(r"[a-f0-9]{64}", payload["location"]["container_sha256"]))
    require(
        all(
            finite(payload[k])
            for k in ("worker_seconds", "model_load_seconds", "cache_setup_seconds")
        )
    )
    require(len(payload["records"]) == len(SCHEDULE))
    for i, (row, (case_index, variant)) in enumerate(
        zip(payload["records"], SCHEDULE, strict=True)
    ):
        require(type(row["ordinal"]) is int and type(row["case_index"]) is int)
        require(
            row["ordinal"] == i and row["case_index"] == case_index and row["variant"] == variant
        )
        require(row["phase"] == ("warmup" if i < 2 else "measured"))
        case, metrics = value["cases"][case_index], row["metrics"]
        require(all(type(metrics[k]) is int for k in ("seed", "sequence_bucket", "token_count")))
        require(0 < metrics["token_count"] <= case["expected_bucket"])
        require(
            metrics["seed"] == case["seed"]
            and metrics["sequence_bucket"] == case["expected_bucket"]
        )
        require(all(finite(metrics[k]) for k in cold.METRICS))
        require(metrics["total_seconds"] > 0)
        for role in ("master", "depth"):
            content = row[role]
            require(type(content) is bytes and 0 < len(content) <= cold.MAX_ARTIFACT_BYTES)
            require(
                hashlib.sha256(content).hexdigest()
                == metrics[f"{role}_sha256"]
                == case[f"{role}_sha256"]
            )
            require(cold._jpeg_dimensions(content) == (1024, 576))
        require(set(row["stages"]) == {"conditioning", "transformer", "vae_decode"})
        for name, count in (("conditioning", 1), ("transformer", 4), ("vae_decode", 1)):
            stage = row["stages"][name]
            require(type(stage["calls"]) is int)
            require(
                stage["calls"] == count
                and finite(stage["host_seconds"])
                and finite(stage["cuda_elapsed_seconds"])
            )
        for report in (row["embedding"], row["embedding"]["text_ids"]):
            require(
                isinstance(report["shape"], list)
                and report["shape"]
                and all(type(n) is int and n > 0 for n in report["shape"])
            )
            require(
                report["matches_baseline"] is True
                and re.fullmatch(r"[a-f0-9]{64}", report["sha256"])
            )
        require(row["embedding"]["dtype"] == "torch.bfloat16")
        reference = payload["records"][case_index]["embedding"]
        require(row["embedding"] == reference)


async def worker(args):
    value = authorize(args)
    require(read(ATTEMPT) == read(args.output / "authorization.json"))
    save(args.output / "worker-attempt.json", {"manifest_sha256": sha(MANIFEST)})
    subprocess.run(
        [sys.executable, "-m", "modal", "deploy", str(DEPLOYMENT)], check=True, timeout=60
    )
    rows = json.loads(
        subprocess.check_output([sys.executable, "-m", "modal", "app", "list", "--json"], timeout=5)
    )
    rows = [r for r in rows if r["description"] == APP and r["state"] != "stopped"]
    require(len(rows) == 1 and int(rows[0]["tasks"]) == 0)
    save(args.output / "app.json", {"app_id": rows[0]["app_id"]})
    subprocess.run(
        [sys.executable, str(Path(__file__)), "--metadata", "--output", str(args.output)],
        check=True,
        timeout=15,
    )
    import modal

    function = modal.Function.from_name(APP, "compare")
    await asyncio.wait_for(function.hydrate.aio(), 10)
    require(function.object_id == read(args.output / "deployment-check.json")["function_id"])

    async def spawn(*, request):
        return await function.spawn.aio()

    client = LatencyClient(value)
    client.instance = SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
    save(args.output / "dispatch.json", {"manifest_sha256": sha(MANIFEST), "calls": 1})
    try:
        timings = dict.fromkeys(TIMINGS, 0.0)
        payload = await client.sdk({}, timings)
        validate(payload, value)
        for row in payload["records"]:
            for role in ("master", "depth"):
                cold.legacy.write_exclusive(
                    args.output / f"{row['ordinal']}-{role}.jpg", row.pop(role)
                )
        save(args.output / "result.json", payload)
    finally:
        save(args.output / "call-cleanup.json", {"known_calls_cancelled": await client.cleanup()})


def aggregate(output):
    value, payload = manifest(), read(output / "result.json")
    authority, supervisor = read(output / "authorization.json"), read(output / "supervisor.json")
    require(authority["manifest_sha256"] == sha(MANIFEST) and authority["reserved_usd"] == HOLD_USD)
    require(supervisor["returncode"] == 0 and supervisor["external_app_shutdown_verified"] is True)
    require(read(output / "call-cleanup.json")["known_calls_cancelled"] is True)
    require(finite(supervisor["total_wall_seconds"]) and supervisor["total_wall_seconds"] <= 180)
    require(finite(supervisor["work_wall_seconds"]) and supervisor["work_wall_seconds"] <= 150)
    shutdown, metadata = read(output / "shutdown.json"), read(output / "deployment-check.json")
    require(
        shutdown["app"]["app_id"]
        == supervisor["app_id"]
        == metadata["app_id"]
        == read(output / "app.json")["app_id"]
    )
    require(shutdown["app"]["state"] == "stopped" and int(shutdown["app"]["tasks"]) == 0)
    require(type(shutdown["active_containers"]) is int and shutdown["active_containers"] == 0)
    require(re.fullmatch(r"fu-[A-Za-z0-9]+", metadata["function_id"]))
    loading.validate_metadata({"ranked_functions": [{"function": metadata}]})
    require(read(output / "dispatch.json") == {"manifest_sha256": sha(MANIFEST), "calls": 1})
    for row in payload["records"]:
        for role in ("master", "depth"):
            row[role] = (output / f"{row['ordinal']}-{role}.jpg").read_bytes()
    validate(payload, value)
    pairs = []
    for index in (2, 4, 6, 8):
        a, b = payload["records"][index : index + 2]
        baseline, candidate = (a, b) if a["variant"] == "baseline" else (b, a)
        pairs.append(
            {
                "case_index": a["case_index"],
                "order": a["variant"],
                "baseline_seconds": baseline["metrics"]["total_seconds"],
                "candidate_seconds": candidate["metrics"]["total_seconds"],
                "conditioning_baseline_seconds": baseline["stages"]["conditioning"][
                    "cuda_elapsed_seconds"
                ],
                "conditioning_candidate_seconds": candidate["stages"]["conditioning"][
                    "cuda_elapsed_seconds"
                ],
            }
        )
    return {
        "manifest_sha256": sha(MANIFEST),
        "result_sha256": sha(output / "result.json"),
        "renders": 10,
        "verified_jpegs": 20,
        "all_embeddings_exact": True,
        "pairs": pairs,
        "median_runtime_reduction_fraction": statistics.median(
            1 - p["candidate_seconds"] / p["baseline_seconds"] for p in pairs
        ),
        "scope": "instrumented server runtime; excludes planning, transport and display",
        "production_promoted": False,
    }


def supervise(args):
    authorize(args)
    require(args.output.is_absolute() and args.output.resolve() == args.output)
    args.output.mkdir(mode=0o700)
    command = [sys.executable, "-m", "modal"]

    def cli(*parts, timeout=5):
        return subprocess.check_output([*command, *parts], timeout=timeout)

    before = json.loads(cli("app", "list", "--json"))
    require(not any(r["description"] == APP and r["state"] != "stopped" for r in before))
    previous = {r["app_id"] for r in before}
    authority = {
        "manifest_sha256": sha(MANIFEST),
        "ledger_sha256": args.ledger_sha256,
        "output": str(args.output),
        "reserved_usd": HOLD_USD,
        "work_seconds": WORK_SECONDS,
        "cleanup_seconds": CLEANUP_SECONDS,
    }
    authorize(args)
    save(ATTEMPT, authority)
    save(args.output / "authorization.json", authority)
    started = time.monotonic()
    record, process = {"returncode": 1, "external_app_shutdown_verified": False}, None

    def interrupted(*_):
        raise InterruptedError("conditioning supervisor interrupted")

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
            app_id = (
                read(args.output / "app.json")["app_id"]
                if (args.output / "app.json").exists()
                else None
            )
            while app_id is None and time.monotonic() + 21 < deadline:
                try:
                    rows = json.loads(cli("app", "list", "--json"))
                except (OSError, ValueError, subprocess.SubprocessError):
                    rows = []
                matches = [
                    r for r in rows if r["description"] == APP and r["app_id"] not in previous
                ]
                require(len(matches) <= 1)
                if matches:
                    app_id = matches[0]["app_id"]
                else:
                    time.sleep(0.2)
            require(app_id and re.fullmatch(r"ap-[A-Za-z0-9]+", app_id))
            record["app_id"] = app_id
            require(time.monotonic() + 10 < deadline)
            cli("app", "stop", "--yes", app_id, timeout=10)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            while time.monotonic() + 10 < deadline:
                apps = json.loads(cli("app", "list", "--json"))
                containers = json.loads(cli("container", "list", "--json"))
                exact = [r for r in apps if r["app_id"] == app_id]
                require(len(exact) == 1 and all("app_id" in r for r in containers))
                if (
                    exact[0]["state"] == "stopped"
                    and int(exact[0]["tasks"]) == 0
                    and not any(r["app_id"] == app_id for r in containers)
                ):
                    save(args.output / "shutdown.json", {"app": exact[0], "active_containers": 0})
                    record["external_app_shutdown_verified"] = True
                    break
                time.sleep(0.2)
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
    for name in ("preflight", "run", "worker", "metadata", "aggregate"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-sha256")
    args = parser.parse_args()
    if args.preflight:
        manifest()
        print(
            json.dumps(
                {"gpu_calls": 0, "planned_calls": 1, "planned_renders": 10, "hold_usd": HOLD_USD}
            )
        )
    elif args.metadata:
        asyncio.run(metadata(args.output))
    elif args.worker:
        asyncio.run(worker(args))
    elif args.aggregate:
        print(json.dumps(aggregate(args.output), sort_keys=True, indent=2))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
