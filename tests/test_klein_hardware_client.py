import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_cold_start import synthetic_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_hardware as benchmark  # noqa: E402


def test_hardware_protocol_and_closed_summary_separate_speed_from_image_equivalence(
    tmp_path, monkeypatch
):
    original, content = synthetic_inputs(tmp_path, monkeypatch)
    sources = {
        name: benchmark.ROOT / "deploy" / name
        for name in ("klein_scene_runtime.py", "klein_hardware_probe.py", "modal_klein_hardware.py")
    }
    sources.update(
        client=Path(benchmark.__file__),
        loading_helpers=Path(benchmark.loading.__file__),
        cold_helpers=Path(benchmark.cold.__file__),
        transport=benchmark.ROOT / "src/bookforge/klein_latency_client.py",
    )
    value = {
        key: original[key]
        for key in (
            "schema_version",
            "status",
            "expires_at",
            "image_id",
            "cases",
            "expected_identity",
        )
    }
    value.update(
        experiment_id=benchmark.EXPERIMENT,
        maximum_calls=1,
        cache_id=None,
        expected_identity={**value["expected_identity"], "gpu": "NVIDIA L40S"},
        sources={name: benchmark.sha(path) for name, path in sources.items()},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(benchmark, "MANIFEST", manifest)
    assert benchmark.manifest() == value
    invalid = {**value, "cache_id": benchmark.cold.CACHE_ID}
    manifest.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        benchmark.manifest()
    manifest.write_text(json.dumps(value))
    result = {
        "status": "complete",
        "failure_stage": None,
        "manifest_sha256": benchmark.sha(manifest),
        "identity": value["expected_identity"],
        "location": {
            "cloud": "CLOUD_PROVIDER_GCP",
            "region": "us-central1",
            "container_sha256": "c" * 64,
        },
        "worker_seconds": 20,
        "model_load_seconds": 5,
        "cache_setup_seconds": 0.1,
        "records": [],
    }
    output = tmp_path / "result"
    output.mkdir()
    for ordinal, index in enumerate(benchmark.SCHEDULE):
        case = value["cases"][index]
        result["records"].append(
            {
                "ordinal": ordinal,
                "case_index": index,
                "phase": "warmup" if ordinal < 2 else "measured",
                "historical_images_exact": True,
                "metrics": {
                    **dict.fromkeys(benchmark.cold.METRICS, 0.1),
                    "seed": case["seed"],
                    "sequence_bucket": case["expected_bucket"],
                    "token_count": case["expected_bucket"],
                    "master_sha256": case["master_sha256"],
                    "depth_sha256": case["depth_sha256"],
                },
            }
        )
        for role in ("master", "depth"):
            (output / f"{ordinal}-{role}.jpg").write_bytes(content)
    function = {
        "image_id": benchmark.cold.IMAGE_ID,
        "resources": {
            "gpu_config": {"count": 1, "gpu_type": "L40S"},
            "memory_mb": 32768,
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
        "routing_region": "us-east",
        "scheduler_placement": {"regions": ["us"]},
    }
    files = {
        "result.json": result,
        "authorization.json": {"manifest_sha256": benchmark.sha(manifest), "reserved_usd": 0.46},
        "dispatch.json": {"manifest_sha256": benchmark.sha(manifest), "calls": 1},
        "app.json": {"app_id": "ap-Synthetic"},
        "deployment-check.json": {
            "app_id": "ap-Synthetic",
            "function_id": "fu-Synthetic",
            **function,
        },
        "supervisor.json": {
            "returncode": 0,
            "external_app_shutdown_verified": True,
            "app_id": "ap-Synthetic",
            "work_wall_seconds": 25,
            "total_wall_seconds": 30,
        },
        "shutdown.json": {
            "app": {"app_id": "ap-Synthetic", "state": "stopped", "tasks": "0"},
            "active_containers": 0,
        },
        "call-cleanup.json": {"known_calls_cancelled": True},
    }
    for name, data in files.items():
        (output / name).write_text(json.dumps(data))
    summary = benchmark.aggregate(output)
    assert summary["renders"] == 10 and summary["verified_jpegs"] == 20
    assert summary["all_images_exact"] and summary["exact_image_speed_screen_passed"]
    assert not summary["production_promoted"]
    # A valid JPEG with extra trailing bytes differs historically while retaining pixel data.
    changed = content + b"different encoding"
    alternate = copy.deepcopy(result)
    alternate["records"][0]["metrics"]["master_sha256"] = hashlib.sha256(changed).hexdigest()
    alternate["records"][0]["historical_images_exact"] = False
    (output / "0-master.jpg").write_bytes(changed)
    (output / "result.json").write_text(json.dumps(alternate))
    summary = benchmark.aggregate(output)
    assert summary["speed_screen_passed"] and not summary["all_images_exact"]
    assert not summary["exact_image_speed_screen_passed"] and not summary["production_promoted"]
    (output / "0-master.jpg").write_bytes(content)
    (output / "result.json").write_text(json.dumps(result))
    for name, mutate in (
        ("result.json", lambda v: v.update(status="failed", failure_stage="render")),
        ("result.json", lambda v: v["records"][0].update(historical_images_exact=False)),
        ("result.json", lambda v: v["records"][2]["metrics"].update(seed=999)),
        ("result.json", lambda v: v["records"][2]["metrics"].update(total_seconds=float("nan"))),
        ("result.json", lambda v: v["location"].update(region="eu-west-1")),
        ("deployment-check.json", lambda v: v["resources"]["gpu_config"].update(gpu_type="L4")),
        ("deployment-check.json", lambda v: v["resources"].update(memory_mb_max=131072)),
        ("shutdown.json", lambda v: v["app"].update(tasks="1")),
        ("supervisor.json", lambda v: v.update(total_wall_seconds=241)),
        ("dispatch.json", lambda v: v.update(calls=2)),
    ):
        invalid = copy.deepcopy(files[name])
        mutate(invalid)
        (output / name).write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            benchmark.aggregate(output)
        (output / name).write_text(json.dumps(files[name]))
    (output / "9-depth.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(output)


def test_stop_nonzero_requires_authoritative_empty_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "authorize", lambda args: {})
    monkeypatch.setattr(benchmark, "sha", lambda path: "a" * 64)
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)

    for label, app_state, tasks, containers, closed in (
        ("already-stopped", "stopped", "0", [], True),
        ("still-running", "deployed", "1", [], False),
        ("remaining-container", "stopped", "0", [{"app_id": "ap-Synthetic"}], False),
    ):
        output = tmp_path / label
        state = SimpleNamespace(clock=0.0, app_reads=0, stops=0)

        def clock(state=state):
            state.clock += 0.25
            return state.clock

        monkeypatch.setattr(
            benchmark, "time", SimpleNamespace(monotonic=clock, sleep=lambda _: None)
        )
        monkeypatch.setattr(benchmark, "ATTEMPT", tmp_path / f"{label}-attempt.json")

        def cli(
            command, state=state, app_state=app_state, tasks=tasks, containers=containers, **kwargs
        ):
            operation = command[3:]
            if operation[:2] == ["app", "stop"]:
                assert operation == ["app", "stop", "--yes", "ap-Synthetic"]
                state.stops += 1
                raise benchmark.subprocess.CalledProcessError(1, command, output=b"untrusted error")
            if operation == ["app", "list", "--json"]:
                state.app_reads += 1
                rows = (
                    []
                    if state.app_reads == 1
                    else [
                        {
                            "app_id": "ap-Synthetic",
                            "description": benchmark.APP,
                            "state": app_state,
                            "tasks": tasks,
                        }
                    ]
                )
                return json.dumps(rows).encode()
            assert operation == ["container", "list", "--json"]
            return json.dumps(containers).encode()

        def popen(*args, output=output, **kwargs):
            (output / "app.json").write_text(json.dumps({"app_id": "ap-Synthetic"}))
            return SimpleNamespace(wait=lambda **kw: 1, poll=lambda: 1)

        monkeypatch.setattr(benchmark.subprocess, "check_output", cli)
        monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
        # The deployment failure remains a failed run even when its cleanup succeeds.
        with pytest.raises(ValueError):
            benchmark.supervise(SimpleNamespace(output=output, ledger_sha256="b" * 64))
        evidence = json.loads((output / "supervisor.json").read_bytes())
        assert evidence["returncode"] == evidence["stop_returncode"] == 1
        assert evidence["external_app_shutdown_verified"] is closed and state.stops == 1
        assert (output / "shutdown.json").exists() is closed
        assert not (output / "summary.json").exists()
