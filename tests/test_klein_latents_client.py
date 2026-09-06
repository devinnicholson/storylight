import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_cold_start import synthetic_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_latents as benchmark  # noqa: E402


def artifact(case, index):
    probe = benchmark.probe
    data = bytes([index, 0]) * (probe.NOISE_BYTES // 2)
    return {
        "schema_version": 1,
        "case_index": index,
        "case_sha256": probe.case_sha256(case),
        "seed": case["seed"],
        "shape": list(probe.SHAPE),
        "dtype": "bfloat16",
        "byte_order": "little",
        "data": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "packed_shape": list(probe.PACKED_SHAPE),
        "packed_stride": list(probe.PACKED_STRIDE),
        "latent_ids_shape": list(probe.IDS_SHAPE),
        "latent_ids_dtype": "int64",
        "latent_ids_sha256": probe.IDS_SHA256,
    }


def test_capture_proof_noise_and_closed_lifecycle_gate_replay(tmp_path, monkeypatch):
    original, image = synthetic_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark, "EVIDENCE", tmp_path / "evidence")
    capture, replay = benchmark.settings("capture"), benchmark.settings("replay")
    for config in (capture, replay):
        config.manifest.parent.mkdir(parents=True)
    assert (capture.app, capture.experiment) != (replay.app, replay.experiment)
    assert (capture.work, capture.cleanup, capture.hold) == (180, 30, 0.25)
    assert (replay.work, replay.cleanup, replay.hold) == (210, 30, 0.46)
    paths = {
        name: benchmark.ROOT / "deploy" / name
        for name in ("klein_scene_runtime.py", "klein_latent_probe.py", "modal_klein_latents.py")
    }
    paths.update(
        client=Path(benchmark.__file__),
        lifecycle=Path(benchmark.life.__file__),
        hardware_helpers=Path(benchmark.hardware.__file__),
        loading_helpers=Path(benchmark.hardware.loading.__file__),
        cold_helpers=Path(benchmark.hardware.cold.__file__),
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
        phase="capture",
        experiment_id=capture.experiment,
        maximum_calls=1,
        cache_id=benchmark.hardware.cold.CACHE_ID,
        artifacts=[],
        sources={k: benchmark.sha(p) for k, p in paths.items()},
    )
    capture.manifest.write_text(json.dumps(value))
    assert benchmark.manifest(capture) == value
    noise = [artifact(case, i) for i, case in enumerate(value["cases"])]
    metadata = [{k: v for k, v in a.items() if k != "data"} for a in noise]
    function = {
        "image_id": value["image_id"],
        "resources": {
            "gpu_config": {"count": 1, "gpu_type": "L4"},
            "memory_mb": 32768,
            "memory_mb_max": 65536,
            "milli_cpu": 8000,
            "milli_cpu_max": 8000,
        },
        "autoscaler_settings": {"max_containers": 1, "scaledown_window": 2},
        "max_inputs": 1,
        "single_use_containers": True,
        "max_concurrent_inputs": 1,
        "timeout_secs": 120,
        "startup_timeout_secs": 30,
        "routing_region": "us-east",
        "scheduler_placement": {"regions": ["us"]},
        "volume_mounts": [{"mount_path": "/compiled", "volume_id": "vo-Synthetic"}],
    }
    benchmark.validate_metadata({"ranked_functions": [{"function": function}]}, capture)
    with pytest.raises(ValueError):
        benchmark.validate_metadata({"ranked_functions": [{"function": function}]}, replay)
    result = {
        "status": "complete",
        "failure_stage": None,
        "records": [],
        "artifacts": metadata,
        "identity": value["expected_identity"],
        "manifest_sha256": benchmark.sha(capture.manifest),
        "model_load_seconds": 1,
        "cache_setup_seconds": 0.1,
        "worker_seconds": 10,
        "location": {
            "cloud": "CLOUD_PROVIDER_GCP",
            "region": "us-east1",
            "container_sha256": "a" * 64,
        },
    }
    output = tmp_path / "capture-output"
    output.mkdir()
    for ordinal, index in enumerate((0, 1) * 2):
        case = value["cases"][index]
        result["records"].append(
            {
                "ordinal": ordinal,
                "case_index": index,
                "phase": "warmup" if ordinal < 2 else "measured",
                "historical_images_exact": True,
                "initial_noise_exact": True,
                "latent_sha256": noise[index]["sha256"],
                "latent_ids_sha256": noise[index]["latent_ids_sha256"],
                "metrics": {
                    **dict.fromkeys(benchmark.hardware.cold.METRICS, 0.1),
                    "seed": case["seed"],
                    "sequence_bucket": case["expected_bucket"],
                    "token_count": 5,
                    "master_sha256": case["master_sha256"],
                    "depth_sha256": case["depth_sha256"],
                },
            }
        )
        for role in ("master", "depth"):
            (output / f"{ordinal}-{role}.jpg").write_bytes(image)
    for index, a in enumerate(noise):
        (output / f"latent-{index}.bin").write_bytes(a["data"])
    files = {
        "result.json": result,
        "supervisor.json": {
            "returncode": 0,
            "external_app_shutdown_verified": True,
            "work_wall_seconds": 20,
            "total_wall_seconds": 25,
            "app_id": "ap-Synthetic",
        },
        "shutdown.json": {
            "app": {"app_id": "ap-Synthetic", "state": "stopped", "tasks": "0"},
            "active_containers": 0,
        },
        "deployment-check.json": {
            **function,
            "app_id": "ap-Synthetic",
            "function_id": "fu-Synthetic",
        },
        "app.json": {"app_id": "ap-Synthetic"},
        "dispatch.json": {"manifest_sha256": benchmark.sha(capture.manifest), "calls": 1},
        "authorization.json": {
            "manifest_sha256": benchmark.sha(capture.manifest),
            "reserved_usd": 0.25,
        },
        "call-cleanup.json": {"known_calls_cancelled": True},
    }
    for name, data in files.items():
        (output / name).write_text(json.dumps(data))
    (output / "0-master.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.export_capture(output, capture)
    assert not (capture.manifest.parent / "capture-proof.json").exists()
    (output / "0-master.jpg").write_bytes(image)
    benchmark.export_capture(output, capture)
    assert (capture.manifest.parent / "latent-0.bin").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        benchmark.export_capture(output, capture)
    proof = capture.manifest.parent / "capture-proof.json"
    replay_value = {
        **value,
        "phase": "replay",
        "experiment_id": replay.experiment,
        "cache_id": None,
        "expected_identity": {**value["expected_identity"], "gpu": "NVIDIA L40S"},
        "artifacts": metadata,
        "sources": {**value["sources"], "capture_proof": benchmark.sha(proof)},
    }
    replay.manifest.write_text(json.dumps(replay_value))
    assert benchmark.manifest(replay) == replay_value
    archived = capture.manifest.parent / "supervisor.json"
    old = archived.read_bytes()
    for bad in (-1, False):
        modified = json.loads(old)
        modified["work_wall_seconds"] = bad
        archived.write_text(json.dumps(modified))
        with pytest.raises(ValueError):
            benchmark.manifest(replay)
    archived.write_bytes(old)
    bad_image = capture.manifest.parent / "0-master.jpg"
    bad_image.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.manifest(replay)
    bad_image.write_bytes(image)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"same":1,"same":2}')
    with pytest.raises(ValueError):
        benchmark.read(duplicate)
    (capture.manifest.parent / "latent-0.bin").write_bytes(b"x" * benchmark.probe.NOISE_BYTES)
    with pytest.raises(ValueError):
        benchmark.manifest(replay)
    (capture.manifest.parent / "latent-0.bin").write_bytes(noise[0]["data"])
    invalid = copy.deepcopy(replay_value)
    invalid["artifacts"][0]["seed"] += 1
    replay.manifest.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        benchmark.manifest(replay)
    replay.manifest.write_text(json.dumps(replay_value))
    in_memory = benchmark.qualified(output, capture)
    in_memory["records"][0]["initial_noise_exact"] = False
    with pytest.raises(ValueError):
        benchmark.validate(in_memory, value, capture)

    life = benchmark.life
    supervised = tmp_path / "supervised"
    state = SimpleNamespace(clock=0.0, reads=0, stops=0)

    def clock():
        state.clock += 0.25
        return state.clock

    monkeypatch.setattr(life, "time", SimpleNamespace(monotonic=clock, sleep=lambda _: None))
    monkeypatch.setattr(life.signal, "signal", lambda *a: None)

    def cli(*parts, **kwargs):
        if parts[:2] == ("app", "stop"):
            assert parts[-1] == "ap-Synthetic"
            state.stops += 1
            raise life.subprocess.CalledProcessError(1, parts)
        if parts == ("container", "list", "--json"):
            return b"[]"
        state.reads += 1
        return json.dumps(
            []
            if state.reads == 1
            else [
                {
                    "description": capture.app,
                    "app_id": "ap-Synthetic",
                    "state": "stopped",
                    "tasks": "0",
                }
            ]
        ).encode()

    def popen(*args, **kwargs):
        (supervised / "app.json").write_text(json.dumps({"app_id": "ap-Synthetic"}))
        return SimpleNamespace(wait=lambda **kw: 1, poll=lambda: 1)

    monkeypatch.setattr(life.subprocess, "Popen", popen)
    with pytest.raises(ValueError):
        life.supervise(
            supervised,
            ["no-network"],
            app=capture.app,
            attempt_path=tmp_path / "attempt.json",
            authority={},
            work_seconds=180,
            cleanup_seconds=30,
            authorize=lambda: None,
            cli=cli,
        )
    assert life.read(supervised / "supervisor.json")["external_app_shutdown_verified"] is True
    assert state.stops == 1


def test_deployment_phases_pin_gpu_cache_and_refuse_before_initialization(tmp_path, monkeypatch):
    import builtins
    import runpy
    import time

    original_import = builtins.__import__
    frozen = json.loads(
        (benchmark.ROOT / "benchmarks/renderer-cold-start-2026-09-05/manifest.json").read_bytes()
    )
    for phase, gpu in (("capture", "L4"), ("replay", "L40S")):
        options, mounts, claimed, heavy = {}, [], set(), []

        class Image:
            def add_local_file(self, source, target, mounts=mounts):
                mounts.append((source, target))
                return self

        class App:
            def __init__(self, name, phase=phase):
                assert name == f"bookforge-klein-latents-{phase}"

            def function(self, options=options, **kwargs):
                options.update(kwargs)
                return lambda f: f

        def claim(key, value, *, skip_if_exists, claimed=claimed):
            assert key == "comparison" and value is skip_if_exists is True
            if key in claimed:
                return False
            claimed.add(key)
            return True

        monkeypatch.setitem(
            sys.modules,
            "modal",
            SimpleNamespace(
                is_local=lambda: True,
                App=App,
                Image=SimpleNamespace(from_id=lambda _: Image()),
                Dict=SimpleNamespace(from_name=lambda *a, **kw: SimpleNamespace(put=claim)),
                Volume=SimpleNamespace(from_name=lambda name: name),
                concurrent=lambda **kw: lambda f: f,
            ),
        )
        monkeypatch.setenv("BOOKFORGE_LATENT_PHASE", phase)
        monkeypatch.setenv("MODAL_REGION", "us-east1")
        monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_GCP")
        monkeypatch.setenv("MODAL_TASK_ID", "synthetic")
        compare = runpy.run_path(str(benchmark.DEPLOYMENT))["compare"]
        g = compare.__globals__
        assert options["gpu"] == gpu and options["memory"] == (32768, 65536)
        assert options["cpu"] == (8, 8) and options["max_containers"] == 1
        assert options["timeout"] == 120 and options["startup_timeout"] == 30
        assert options["retries"] == 0 and options["single_use_containers"] is True
        assert options["volumes"] == (
            {"/compiled": "bookforge-klein-compile-cache-v1"} if phase == "capture" else {}
        )
        assert not options.get("enable_memory_snapshot")
        assert sum(target.endswith(".bin") for _, target in mounts) == (
            0 if phase == "capture" else 2
        )
        noise = [artifact(case, i) for i, case in enumerate(frozen["cases"])]
        files = {f"/root/latent-{i}.bin": tmp_path / f"{phase}-{i}.bin" for i in range(2)}
        for i, item in enumerate(noise):
            files[f"/root/latent-{i}.bin"].write_bytes(item["data"])
        g["Path"] = lambda p, files=files: files.get(str(p), Path(p))
        path = tmp_path / f"{phase}-manifest.json"
        g["MANIFEST"] = path
        value = {
            "schema_version": 1,
            "status": "authorized",
            "phase": phase,
            "experiment_id": g["EXPERIMENT"],
            "image_id": g["IMAGE_ID"],
            "maximum_calls": 1,
            "cache_id": g["CACHE_ID"] if phase == "capture" else None,
            "expires_at": int(time.time()) + 600,
            "cases": frozen["cases"],
            "expected_identity": {**frozen["expected_identity"], "gpu": f"NVIDIA {gpu}"},
            "artifacts": []
            if phase == "capture"
            else [{k: v for k, v in a.items() if k != "data"} for a in noise],
            "sources": {
                name: benchmark.sha(benchmark.ROOT / "deploy" / name) for name in g["SOURCES"]
            },
        }

        def runtime(model_root, claimed=claimed, heavy=heavy):
            assert claimed == {"comparison"} and model_root == Path("/models")
            heavy.append(True)
            raise RuntimeError("synthetic initialization")

        def imports(name, *args, runtime=runtime, **kwargs):
            if name == "klein_latent_probe":
                return SimpleNamespace(
                    validate_artifact=benchmark.probe.validate_artifact,
                    run_capture=None,
                    run_replay=None,
                )
            if name == "klein_scene_runtime":
                return SimpleNamespace(KleinSceneRuntime=runtime)
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", imports)
        for changes in (
            {"expires_at": 1},
            {"cases": []},
            {"expected_identity": {}},
            {"artifacts": noise[:1] if phase == "capture" else []},
        ):
            # Only JSON metadata belongs in the manifest, never binary tensors.
            invalid = {**value, **changes}
            if phase == "capture" and changes.get("artifacts"):
                invalid["artifacts"] = [{}]
            path.write_text(json.dumps(invalid))
            with pytest.raises(ValueError):
                compare()
        path.write_text(json.dumps(value))
        if phase == "replay":
            files["/root/latent-0.bin"].write_bytes(b"x" * benchmark.probe.NOISE_BYTES)
            with pytest.raises(ValueError):
                compare()
            files["/root/latent-0.bin"].write_bytes(noise[0]["data"])
        assert not claimed and not heavy
        with pytest.raises(RuntimeError, match="synthetic initialization"):
            compare()
        with pytest.raises(ValueError):
            compare()
        assert heavy == [True]
        monkeypatch.setattr(builtins, "__import__", original_import)
