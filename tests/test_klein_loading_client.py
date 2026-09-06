import asyncio
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_cold_start import payload, synthetic_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_loading as benchmark  # noqa: E402


def inputs(tmp_path, monkeypatch):
    value, content = synthetic_inputs(tmp_path, monkeypatch)
    value.update(
        experiment_id=benchmark.EXPERIMENT,
        deployment_sha256=benchmark.sha(benchmark.DEPLOYMENT),
        client_sha256=benchmark.sha(Path(benchmark.__file__)),
        operations=[
            {
                "ordinal": index,
                "variant": variant,
                "request_id": hashlib.sha256(
                    f"{benchmark.EXPERIMENT}:{index}".encode()
                ).hexdigest()[:32],
            }
            for index, variant in enumerate(benchmark.SCHEDULE)
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(benchmark.cold.preparation.encoded(value))
    monkeypatch.setattr(benchmark, "MANIFEST", path)
    return value, content


def test_loading_manifest_and_live_metadata_fail_before_dispatch(tmp_path, monkeypatch):
    value, _ = inputs(tmp_path, monkeypatch)
    assert benchmark.manifest() == value
    for mutation in ("ordinal", "seed", "unknown", "duplicate"):
        changed = copy.deepcopy(value)
        if mutation == "ordinal":
            changed["operations"][0]["ordinal"] = 0.0
        elif mutation == "seed":
            changed["cases"][0]["seed"] = float(changed["cases"][0]["seed"])
        elif mutation == "unknown":
            changed["extra"] = "unapproved"
        else:
            changed["operations"][1]["request_id"] = changed["operations"][0]["request_id"]
        benchmark.MANIFEST.write_bytes(benchmark.cold.preparation.encoded(changed))
        with pytest.raises(ValueError):
            benchmark.manifest()
    function = {
        "image_id": benchmark.cold.IMAGE_ID,
        "resources": {
            "gpu_config": {"count": 1, "gpu_type": "L4"},
            "memory_mb": 65536,
            "memory_mb_max": 65536,
            "milli_cpu": 8000,
            "milli_cpu_max": 8000,
        },
        "autoscaler_settings": {"max_containers": 1, "scaledown_window": 2},
        "max_inputs": 1,
        "single_use_containers": True,
        "max_concurrent_inputs": 1,
        "startup_timeout_secs": 30,
        "timeout_secs": 120,
        "cloud_provider_str": "aws",
        "routing_region": "us-east",
        "scheduler_placement": {"regions": ["us-east"]},
    }
    data = {"ranked_functions": [{"function": function}]}
    assert benchmark.validate_metadata(data)["single_use_containers"] is True
    for change in (
        {"max_concurrent_inputs": 2},
        {"max_inputs": 0},
        {"single_use_containers": False},
        {"timeout_secs": 180},
        {"startup_timeout_secs": 120},
        {"cloud_provider_str": "gcp"},
        {"image_id": "wrong"},
        {"checkpointing_enabled": True},
        {"retry_policy": {"retries": 1}},
        {"autoscaler_settings": {"max_containers": 2, "scaledown_window": 2}},
    ):
        invalid = copy.deepcopy(data)
        invalid["ranked_functions"][0]["function"].update(change)
        with pytest.raises(ValueError):
            benchmark.validate_metadata(invalid)


def test_loading_aggregate_requires_all_artifacts_provenance_and_cleanup(tmp_path, monkeypatch):
    value, content = inputs(tmp_path, monkeypatch)
    output = tmp_path / "results"
    output.mkdir()
    header = benchmark.header(SimpleNamespace(ledger_sha256="a" * 64))
    rows = [header]
    for operation in value["operations"]:
        result = payload(value, operation, content)
        result.update(
            ordinal=operation["ordinal"],
            loading_environment={
                "HF_ENABLE_PARALLEL_LOADING": "true"
                if operation["variant"] == "candidate"
                else "false",
                "HF_PARALLEL_LOADING_WORKERS": "4",
            },
            shard_inventory={
                component: dict(
                    index_count=1, indexed_shards=2, safetensors_files=2, referenced_bytes=200
                )
                for component in ("klein/transformer", "klein/text_encoder", "klein/vae", "depth")
            },
        )
        result["stages"].update(
            first_artifact_seconds=10 if operation["variant"] == "candidate" else 20,
            worker_seconds=30,
        )
        for index, sample in enumerate(result["samples"]):
            for role in ("master", "depth"):
                (output / f"{operation['ordinal']}-{index}-{role}.jpg").write_bytes(
                    sample.pop(role)
                )
        rows.extend(
            [
                {"kind": "start", **operation},
                {
                    "kind": "result",
                    "ordinal": operation["ordinal"],
                    "payload": result,
                    "client_cycle_seconds": 35,
                    "timings": dict.fromkeys(benchmark.TIMINGS, 0.0),
                },
            ]
        )
    files = {
        "authorization.json": {
            "manifest_sha256": benchmark.sha(benchmark.MANIFEST),
            "ledger_sha256": "a" * 64,
            "reserved_usd": 1.20,
        },
        "call-cleanup.json": {"known_calls_cancelled": True},
        "supervisor.json": {
            "returncode": 0,
            "external_app_shutdown_verified": True,
            "app_id": "ap-Synthetic",
        },
        "shutdown.json": {
            "app": {"app_id": "ap-Synthetic", "state": "stopped", "tasks": "0"},
            "active_containers": 0,
        },
    }
    for name, data in files.items():
        benchmark.save(output / name, data)

    def write_journal(entries):
        (output / "journal.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))

    write_journal(rows)
    result = benchmark.aggregate(output)
    assert result["artifacts_verified"] == 32 and result["generation_calls"] == 16
    assert result["first_artifact_reduction_fraction"] == 0.5
    assert result["promotion"] == "not_promoted_small_sample"
    for mutation in ("header", "container", "inventory", "timing"):
        changed = copy.deepcopy(rows)
        if mutation == "header":
            changed[0]["manifest_sha256"] = "b" * 64
        elif mutation == "container":
            changed[4]["payload"]["location"] = changed[2]["payload"]["location"]
        elif mutation == "inventory":
            changed[4]["payload"]["shard_inventory"]["depth"]["referenced_bytes"] += 1
        else:
            changed[2]["client_cycle_seconds"] = float("nan")
        write_journal(changed)
        with pytest.raises(ValueError):
            benchmark.aggregate(output)
    write_journal(rows)
    for name in ("call-cleanup.json", "supervisor.json"):
        original = (output / name).read_bytes()
        (output / name).unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            benchmark.aggregate(output)
        (output / name).write_bytes(original)
    (output / "3-3-depth.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(output)


def test_worker_isolates_raw_metadata_before_public_sdk_hydration(tmp_path, monkeypatch):
    value, _ = inputs(tmp_path, monkeypatch)
    value.update(status="authorized", expires_at=int(time.time()) + 600)
    args = SimpleNamespace(output=tmp_path / "output", ledger_sha256="a" * 64)
    args.output.mkdir()
    monkeypatch.setattr(benchmark, "authorize", lambda _: value)
    monkeypatch.setattr(benchmark, "ATTEMPT", tmp_path / "attempt.json")
    benchmark.save(benchmark.ATTEMPT, {"output": str(args.output), **benchmark.header(args)})
    state = {"metadata_child": False, "hydrated": False, "dispatches": 0}

    class StopBeforeGPU(Exception):
        pass

    class RawClient:
        @classmethod
        async def from_env(cls):
            pytest.fail("Raw metadata client must not enter the public SDK worker loop")

    def subprocess_run(command, *, check, timeout):
        assert check
        if "deploy" in command:
            assert timeout == 120
            return
        assert "--metadata" in command and timeout == 20
        assert not state["hydrated"]
        state["metadata_child"] = True
        benchmark.save(args.output / "deployment-check.json", {"function_id": "fu-Synthetic"})

    async def hydrate():
        assert state["metadata_child"]
        state["hydrated"] = True

    async def spawn(ordinal):
        assert state["hydrated"] and ordinal == 0
        state["dispatches"] += 1
        raise StopBeforeGPU

    function = SimpleNamespace(
        object_id="fu-Synthetic",
        hydrate=SimpleNamespace(aio=hydrate),
        spawn=SimpleNamespace(aio=spawn),
    )
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Function=SimpleNamespace(from_name=lambda app, name: function),
        ),
    )
    monkeypatch.setitem(sys.modules, "modal.client", SimpleNamespace(_Client=RawClient))
    monkeypatch.setattr(benchmark.subprocess, "run", subprocess_run)
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps(
            [
                {
                    "description": benchmark.APP,
                    "state": "deployed",
                    "tasks": "0",
                    "app_id": "ap-Synthetic",
                },
            ]
        ).encode(),
    )

    class PublicClient:
        def __init__(self, _):
            pass

        async def sdk(self, operation, timings):
            return await self.instance.invoke.spawn.aio(request=operation)

        async def cleanup(self):
            return True

    monkeypatch.setattr(benchmark, "LatencyClient", PublicClient)
    with pytest.raises(StopBeforeGPU):
        asyncio.run(benchmark.worker(args))
    assert state == {"metadata_child": True, "hydrated": True, "dispatches": 1}
    assert json.loads((args.output / "call-cleanup.json").read_bytes()) == {
        "known_calls_cancelled": True,
    }
