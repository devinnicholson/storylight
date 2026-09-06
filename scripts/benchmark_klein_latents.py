#!/usr/bin/env python3
"""Qualify identical initial Klein noise before a separate cross-GPU replay."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from bookforge.klein_latency_client import TIMINGS, LatencyClient, finite  # noqa: E402
from deploy import klein_latent_probe as probe  # noqa: E402
from scripts import benchmark_klein_hardware as hardware  # noqa: E402
from scripts import klein_trial_lifecycle as life  # noqa: E402

EVIDENCE = ROOT / "benchmarks/renderer-latents-2026-09-06"
DEPLOYMENT = ROOT / "deploy/modal_klein_latents.py"
LEDGER = hardware.LEDGER
require, read, sha, save = life.require, hardware.read, life.sha, life.save


def settings(phase):
    require(phase in ("capture", "replay"))
    return SimpleNamespace(
        phase=phase,
        app=f"bookforge-klein-latents-{phase}",
        experiment=f"klein-latents-{phase}-20260906-a",
        manifest=EVIDENCE / phase / "manifest.json",
        work=180 if phase == "capture" else 210,
        cleanup=30,
        hold=0.25 if phase == "capture" else 0.46,
        gpu="L4" if phase == "capture" else "L40S",
    )


def artifacts(metadata, cases):
    require(isinstance(metadata, list) and len(metadata) == 2)
    result = []
    for index, item in enumerate(metadata):
        path = EVIDENCE / "capture" / f"latent-{index}.bin"
        require(path.stat().st_size == 589824)
        artifact = {**item, "data": path.read_bytes()}
        probe.validate_artifact(artifact, cases[index], index)
        result.append(artifact)
    return result


def manifest(config):
    require(config.manifest.stat().st_size <= 32768)
    value = read(config.manifest)
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "expires_at",
            "phase",
            "experiment_id",
            "image_id",
            "cache_id",
            "maximum_calls",
            "sources",
            "expected_identity",
            "cases",
            "artifacts",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(type(value["maximum_calls"]) is int and value["maximum_calls"] == 1)
    require(value["phase"] == config.phase and value["experiment_id"] == config.experiment)
    require(value["image_id"] == hardware.cold.IMAGE_ID)
    require(value["cache_id"] == (hardware.cold.CACHE_ID if config.phase == "capture" else None))
    require(value["status"] in {"draft", "authorized"})
    require(
        value["expires_at"] is None
        if value["status"] == "draft"
        else type(value["expires_at"]) is int and value["expires_at"] > 0
    )
    identity, cases = hardware.cold.frozen_cases()
    identity = {**identity, "gpu": f"NVIDIA {config.gpu}"}
    encode = hardware.cold.preparation.encoded
    require(encode([value["expected_identity"], value["cases"]]) == encode([identity, cases]))
    paths = {
        name: ROOT / "deploy" / name
        for name in ("klein_scene_runtime.py", "klein_latent_probe.py", "modal_klein_latents.py")
    }
    paths.update(
        client=Path(__file__),
        lifecycle=Path(life.__file__),
        hardware_helpers=Path(hardware.__file__),
        loading_helpers=Path(hardware.loading.__file__),
        cold_helpers=Path(hardware.cold.__file__),
        transport=ROOT / "src/bookforge/klein_latency_client.py",
    )
    if config.phase == "capture":
        require(value["artifacts"] == [])
    else:
        proof_path = EVIDENCE / "capture/capture-proof.json"
        paths["capture_proof"] = proof_path
        proof = read(proof_path)
        require(
            proof["schema_version"] == 1
            and proof["phase"] == "capture"
            and proof["historical_images_exact"] is True
            and proof["renders"] == 4
        )
        require(proof["artifacts"] == value["artifacts"])
        require(proof["manifest_sha256"] == sha(settings("capture").manifest))
        captured = qualified(EVIDENCE / "capture", settings("capture"))
        require(proof["result_sha256"] == sha(EVIDENCE / "capture/result.json"))
        require(
            proof["artifacts"]
            == [{k: v for k, v in item.items() if k != "data"} for item in captured["artifacts"]]
        )
        artifacts(value["artifacts"], cases)
    require(value["sources"] == {key: sha(path) for key, path in paths.items()})
    return value


def authorize(args, config):
    value = manifest(config)
    require(
        value["status"] == "authorized" and time.time() < value["expires_at"] <= time.time() + 7200
    )
    require(args.ledger_sha256 == sha(LEDGER))
    ledger = read(LEDGER)
    require(ledger["reservations"][f"reservation:{config.experiment}"] == config.hold)
    env, usage = ledger["envelope"], ledger["estimated_usage_usd"]
    require(
        usage <= env["run_cap_usd"]
        and usage + env["usage_before_lab_usd"]
        <= env["monthly_credit_usd"] + env["authorized_paid_usd"] - env["reserve_usd"]
    )
    return value


def validate_metadata(data, config):
    require(len(data["ranked_functions"]) == 1)
    function = data["ranked_functions"][0]["function"]
    require(function["resources"]["gpu_config"] == {"count": 1, "gpu_type": config.gpu})
    # All remaining resource/lifecycle constraints are the established hardware contract.
    common = copy.deepcopy(data)
    common["ranked_functions"][0]["function"]["resources"]["gpu_config"]["gpu_type"] = "L40S"
    shared = hardware.validate_metadata(common)
    require(
        not any(
            function.get(k) for k in ("shared_volume_mounts", "s3_mounts", "cloud_bucket_mounts")
        )
    )
    mounts = function.get("volume_mounts", [])
    if config.phase == "capture":
        require(len(mounts) == 1 and mounts[0]["mount_path"] == "/compiled")
        require(re.fullmatch(r"vo-[A-Za-z0-9]+", mounts[0]["volume_id"]))
        require(not mounts[0].get("sub_path"))
    else:
        require(not mounts)
    return {
        **shared,
        "resources": function["resources"],
        "volume_mounts": mounts,
    }


async def metadata(output, config):
    from google.protobuf.json_format import MessageToDict
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    app_id = read(output / "app.json")["app_id"]
    info = await client.stub.FunctionGet(
        api_pb2.FunctionGetRequest(
            app_name=config.app, object_tag="compare", environment_name="main"
        )
    )
    layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    require(info.function_id in layout.app_layout.function_ids.values())
    data = MessageToDict(info.function, preserving_proto_field_name=True)
    keys = {
        *hardware.loading.METADATA_FIELDS,
        "volume_mounts",
        "shared_volume_mounts",
        "s3_mounts",
        "cloud_bucket_mounts",
        "experimental_options",
    }
    save(
        output / "deployment-observed.json",
        {
            "functions": [
                {key: val for key, val in row["function"].items() if key in keys}
                for row in data.get("ranked_functions", [])
            ]
        },
    )
    save(
        output / "deployment-check.json",
        {"app_id": app_id, "function_id": info.function_id, **validate_metadata(data, config)},
    )


def validate(payload, value, config):
    require(payload["status"] == "complete" and payload["failure_stage"] is None)
    require(
        payload["identity"] == value["expected_identity"]
        and payload["manifest_sha256"] == sha(config.manifest)
    )
    require(
        all(
            finite(payload[k])
            for k in ("worker_seconds", "model_load_seconds", "cache_setup_seconds")
        )
    )
    location = payload["location"]
    require(
        re.fullmatch(r"CLOUD_PROVIDER_[A-Z]+", location["cloud"])
        and re.fullmatch(r"us-[a-z0-9-]+", location["region"])
        and re.fullmatch(r"[a-f0-9]{64}", location["container_sha256"])
    )
    require(len(payload["artifacts"]) == 2)
    for index, artifact in enumerate(payload["artifacts"]):
        probe.validate_artifact(artifact, value["cases"][index], index)
    if config.phase == "replay":
        require(
            [{k: v for k, v in item.items() if k != "data"} for item in payload["artifacts"]]
            == value["artifacts"]
        )
    schedule = (0, 1) * (2 if config.phase == "capture" else 5)
    require(len(payload["records"]) == len(schedule))
    for ordinal, (row, index) in enumerate(zip(payload["records"], schedule, strict=True)):
        require(type(row["ordinal"]) is int and type(row["case_index"]) is int)
        require(row["ordinal"] == ordinal and row["case_index"] == index)
        require(row["phase"] == ("warmup" if ordinal < 2 else "measured"))
        require(row["initial_noise_exact"] is True)
        artifact, case, metrics = payload["artifacts"][index], value["cases"][index], row["metrics"]
        require(
            row["latent_sha256"] == artifact["sha256"]
            and row["latent_ids_sha256"] == artifact["latent_ids_sha256"]
        )
        require(all(type(metrics[k]) is int for k in ("seed", "sequence_bucket", "token_count")))
        require(
            metrics["seed"] == case["seed"]
            and metrics["sequence_bucket"] == case["expected_bucket"]
            and 0 < metrics["token_count"] <= case["expected_bucket"]
        )
        require(
            all(finite(metrics[k]) for k in hardware.cold.METRICS) and metrics["total_seconds"] > 0
        )
        for role in ("master", "depth"):
            data = row[role]
            require(type(data) is bytes and 0 < len(data) <= hardware.cold.MAX_ARTIFACT_BYTES)
            require(hashlib.sha256(data).hexdigest() == metrics[f"{role}_sha256"])
            require(hardware.cold._jpeg_dimensions(data) == (1024, 576))
        exact = all(
            metrics[f"{role}_sha256"] == case[f"{role}_sha256"] for role in ("master", "depth")
        )
        require(
            type(row["historical_images_exact"]) is bool and row["historical_images_exact"] == exact
        )
        require(config.phase != "capture" or exact)


def cli(*parts, timeout=5):
    return subprocess.check_output([sys.executable, "-m", "modal", *parts], timeout=timeout)


async def worker(args, config):
    value = authorize(args, config)
    require(
        read(LEDGER.parent / f"{config.experiment}-attempt.json")
        == read(args.output / "authorization.json")
    )
    save(args.output / "worker-attempt.json", {"manifest_sha256": sha(config.manifest)})
    os.environ["BOOKFORGE_LATENT_PHASE"] = config.phase
    cli("deploy", str(DEPLOYMENT), timeout=60)
    rows = [
        r
        for r in json.loads(cli("app", "list", "--json"))
        if r["description"] == config.app and r["state"] != "stopped"
    ]
    require(len(rows) == 1 and int(rows[0]["tasks"]) == 0)
    save(args.output / "app.json", {"app_id": rows[0]["app_id"]})
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--metadata",
            "--phase",
            config.phase,
            "--output",
            str(args.output),
        ],
        check=True,
        timeout=15,
    )
    import modal

    function = modal.Function.from_name(config.app, "compare")
    await asyncio.wait_for(function.hydrate.aio(), 10)
    require(function.object_id == read(args.output / "deployment-check.json")["function_id"])

    async def spawn(*, request):
        return await function.spawn.aio()

    client = LatencyClient(value)
    client.instance = SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
    save(args.output / "dispatch.json", {"manifest_sha256": sha(config.manifest), "calls": 1})
    try:
        payload = await client.sdk({}, dict.fromkeys(TIMINGS, 0.0))
        require(len(payload["records"]) <= (4 if config.phase == "capture" else 10))
        require(len(payload["artifacts"]) <= 2)
        for index, artifact in enumerate(payload["artifacts"]):
            require(type(artifact["data"]) is bytes and len(artifact["data"]) == 589824)
            life.write(args.output / f"latent-{index}.bin", artifact["data"])
        for ordinal, row in enumerate(payload["records"]):
            for role in ("master", "depth"):
                require(
                    type(row[role]) is bytes and len(row[role]) <= hardware.cold.MAX_ARTIFACT_BYTES
                )
                life.write(args.output / f"{ordinal}-{role}.jpg", row[role])
        saved = {
            **payload,
            "artifacts": [
                {k: v for k, v in a.items() if k != "data"} for a in payload["artifacts"]
            ],
            "records": [
                {k: v for k, v in r.items() if k not in {"master", "depth"}}
                for r in payload["records"]
            ],
        }
        save(args.output / "result.json", saved)
        validate(payload, value, config)
    finally:
        save(args.output / "call-cleanup.json", {"known_calls_cancelled": await client.cleanup()})


def qualified(output, config):
    value, payload = manifest(config), read(output / "result.json")
    supervisor, shutdown = read(output / "supervisor.json"), read(output / "shutdown.json")
    require(supervisor["returncode"] == 0 and supervisor["external_app_shutdown_verified"] is True)
    require(
        finite(supervisor["work_wall_seconds"])
        and finite(supervisor["total_wall_seconds"])
        and supervisor["work_wall_seconds"] <= config.work
        and supervisor["total_wall_seconds"] <= config.work + config.cleanup
    )
    require(
        shutdown["app"]["state"] == "stopped"
        and int(shutdown["app"]["tasks"]) == 0
        and type(shutdown["active_containers"]) is int
        and shutdown["active_containers"] == 0
    )
    metadata_row = read(output / "deployment-check.json")
    require(re.fullmatch(r"fu-[A-Za-z0-9]+", metadata_row["function_id"]))
    require(
        supervisor["app_id"]
        == shutdown["app"]["app_id"]
        == metadata_row["app_id"]
        == read(output / "app.json")["app_id"]
    )
    validate_metadata({"ranked_functions": [{"function": metadata_row}]}, config)
    require(read(output / "dispatch.json") == {"manifest_sha256": sha(config.manifest), "calls": 1})
    authority = read(output / "authorization.json")
    require(
        authority["manifest_sha256"] == sha(config.manifest)
        and authority["reserved_usd"] == config.hold
    )
    require(read(output / "call-cleanup.json")["known_calls_cancelled"] is True)
    for index, artifact in enumerate(payload["artifacts"]):
        artifact["data"] = (output / f"latent-{index}.bin").read_bytes()
    for ordinal, row in enumerate(payload["records"]):
        for role in ("master", "depth"):
            row[role] = (output / f"{ordinal}-{role}.jpg").read_bytes()
    validate(payload, value, config)
    return payload


def export_capture(output, config):
    require(config.phase == "capture")
    payload = qualified(output, config)
    names = [
        "result.json",
        "supervisor.json",
        "shutdown.json",
        "deployment-check.json",
        "app.json",
        "dispatch.json",
        "authorization.json",
        "call-cleanup.json",
        "latent-0.bin",
        "latent-1.bin",
        *(f"{index}-{role}.jpg" for index in range(4) for role in ("master", "depth")),
    ]
    require(
        not any((EVIDENCE / "capture" / name).exists() for name in [*names, "capture-proof.json"])
    )
    for name in names:
        life.write(EVIDENCE / "capture" / name, (output / name).read_bytes())
    save(
        EVIDENCE / "capture/capture-proof.json",
        {
            "schema_version": 1,
            "phase": "capture",
            "renders": 4,
            "historical_images_exact": True,
            "manifest_sha256": sha(config.manifest),
            "result_sha256": sha(output / "result.json"),
            "artifacts": [
                {k: v for k, v in a.items() if k != "data"} for a in payload["artifacts"]
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "run", "worker", "metadata", "verify", "export-capture"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--phase", required=True, choices=("capture", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-sha256")
    args = parser.parse_args()
    config = settings(args.phase)
    if args.preflight:
        manifest(config)
        print(json.dumps({"gpu_calls": 0, "phase": config.phase, "planned_calls": 1}))
    elif args.metadata:
        asyncio.run(metadata(args.output, config))
    elif args.worker:
        asyncio.run(worker(args, config))
    elif args.export_capture:
        export_capture(args.output, config)
    elif args.verify:
        qualified(args.output, config)
        print(json.dumps({"phase": config.phase, "verified": True, "production_promoted": False}))
    else:
        authority = {
            "manifest_sha256": sha(config.manifest),
            "ledger_sha256": args.ledger_sha256,
            "output": str(args.output),
            "reserved_usd": config.hold,
            "work_seconds": config.work,
            "cleanup_seconds": config.cleanup,
        }
        life.supervise(
            args.output,
            [
                sys.executable,
                __file__,
                "--worker",
                "--phase",
                config.phase,
                "--output",
                str(args.output),
                "--ledger-sha256",
                args.ledger_sha256,
            ],
            app=config.app,
            attempt_path=LEDGER.parent / f"{config.experiment}-attempt.json",
            authority=authority,
            work_seconds=config.work,
            cleanup_seconds=config.cleanup,
            authorize=lambda: authorize(args, config),
            cli=cli,
        )
        qualified(args.output, config)


if __name__ == "__main__":
    main()
