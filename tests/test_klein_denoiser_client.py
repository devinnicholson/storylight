import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_klein_cold_start import synthetic_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_denoiser as benchmark  # noqa: E402


def test_denoiser_protocol_and_aggregate_require_exact_artifacts_and_closed_app(
    tmp_path, monkeypatch
):
    original, content = synthetic_inputs(tmp_path, monkeypatch)
    names = (
        "klein_scene_runtime.py",
        "klein_denoiser_graph.py",
        "klein_denoiser_probe.py",
        "modal_klein_denoiser.py",
    )
    sources = {name: benchmark.ROOT / "deploy" / name for name in names}
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
            "cache_id",
            "expected_identity",
            "cases",
        )
    }
    value.update(
        experiment_id=benchmark.EXPERIMENT,
        maximum_calls=1,
        sources={name: benchmark.sha(path) for name, path in sources.items()},
    )
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(benchmark, "MANIFEST", manifest)
    manifest.write_text(json.dumps(value))
    assert benchmark.manifest() == value
    invalid = copy.deepcopy(value)
    invalid["cases"][0]["seed"] = float(invalid["cases"][0]["seed"])
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
            "cloud": "CLOUD_PROVIDER_AWS",
            "region": "us-east-1",
            "container_sha256": "c" * 64,
        },
        "worker_seconds": 20,
        "model_load_seconds": 5,
        "cache_setup_seconds": 1,
        "records": [],
        "profiles": [],
        "graph_report": {
            "graph_count": 2,
            "capture_warmup_calls": 6,
            "captured_forward_calls": 2,
            "graphs": [
                {
                    "bucket": bucket,
                    "signature_sha256": str(index) * 64,
                    "capture_seconds": 1,
                    "replays": 12,
                }
                for index, bucket in enumerate((128, 256))
            ],
        },
    }
    output = tmp_path / "results"
    output.mkdir()
    for ordinal, (index, variant) in enumerate(benchmark.SCHEDULE):
        case = value["cases"][index]
        result["records"].append(
            {
                "ordinal": ordinal,
                "case_index": index,
                "variant": variant,
                "phase": (
                    "warmup"
                    if ordinal < 2
                    else "profiled"
                    if ordinal < 4
                    else "preparation"
                    if ordinal < 6
                    else "measured"
                ),
                "metrics": {
                    **dict.fromkeys(benchmark.cold.METRICS, 0.1),
                    "seed": case["seed"],
                    "sequence_bucket": case["expected_bucket"],
                    "token_count": case["expected_bucket"],
                    "master_sha256": case["master_sha256"],
                    "depth_sha256": case["depth_sha256"],
                    "total_seconds": 1 if variant == "candidate" else 2,
                },
            }
        )
        for role in ("master", "depth"):
            (output / f"{ordinal}-{role}.jpg").write_bytes(content)
    from deploy.klein_denoiser_probe import analyze_trace

    events = []
    for index in range(4):
        start = index * 100
        events.extend(
            [
                {
                    "ph": "X",
                    "name": f"bookforge.denoiser.{index}",
                    "cat": "user_annotation",
                    "ts": start,
                    "dur": 90,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "ph": "X",
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ts": start + 1,
                    "dur": 2,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlation": index},
                },
                {
                    "ph": "X",
                    "name": "synthetic_gemm",
                    "cat": "kernel",
                    "ts": start + 5,
                    "dur": 60,
                    "pid": 2,
                    "tid": 2,
                    "args": {"correlation": index},
                },
            ]
        )
    trace = json.dumps({"traceEvents": events}).encode()
    for index in range(2):
        result["profiles"].append(
            {
                "case_index": index,
                "trace_sha256": hashlib.sha256(trace).hexdigest(),
                "analysis": analyze_trace(trace),
            }
        )
        (output / f"{index}-trace.json.gz").write_bytes(gzip.compress(trace, mtime=0))
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
    files = {
        "result.json": result,
        "authorization.json": {
            "manifest_sha256": benchmark.sha(manifest),
            "reserved_usd": 0.27,
            "work_seconds": 120,
            "cleanup_seconds": 30,
            "ledger_sha256": "a" * 64,
        },
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
    assert summary["verified_jpegs"] == 28 and summary["all_images_exact"] is True
    assert summary["median_runtime_reduction_fraction"] == 0.5
    assert summary["production_promoted"] is False
    mutations = (
        ("result.json", lambda v: v.update(status="failed", failure_stage="capture")),
        ("result.json", lambda v: v["records"][3].update(case_index=0.0)),
        ("result.json", lambda v: v["graph_report"]["graphs"][0].update(replays=0)),
        ("result.json", lambda v: v["profiles"][0]["analysis"].update(launch_count=0)),
        ("result.json", lambda v: v["records"][3]["metrics"].update(total_seconds=float("nan"))),
        ("shutdown.json", lambda v: v["app"].update(state="deployed", tasks="1")),
        ("shutdown.json", lambda v: v.update(active_containers=1)),
        ("supervisor.json", lambda v: v.update(total_wall_seconds=151)),
        ("deployment-check.json", lambda v: v.update(max_concurrent_inputs=2)),
        ("dispatch.json", lambda v: v.update(calls=2)),
    )
    for name, mutate in mutations:
        changed = copy.deepcopy(files[name])
        mutate(changed)
        (output / name).write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            benchmark.aggregate(output)
        (output / name).write_text(json.dumps(files[name]))
    for name in ("call-cleanup.json", "deployment-check.json", "dispatch.json"):
        (output / name).unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            benchmark.aggregate(output)
        (output / name).write_text(json.dumps(files[name]))
    (output / "1-trace.json.gz").write_bytes(gzip.compress(b"{}"))
    with pytest.raises(ValueError):
        benchmark.aggregate(output)
    (output / "1-trace.json.gz").write_bytes(gzip.compress(trace, mtime=0))
    (output / "13-depth.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(output)
