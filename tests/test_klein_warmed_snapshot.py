import builtins
import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / "deploy/modal_klein_warmed_snapshot.py"
    options, mounts, claims, calls = {}, [], {}, []
    clock = {"wall": 1000, "mono": 5}

    class Image:
        @staticmethod
        def from_id(value):
            assert value == "im-WtXer8GjRPdgMqWAAUSMwJ"
            return Image()

        def add_local_file(self, source, destination):
            mounts.append((str(source), destination))
            return self

    class App:
        def __init__(self, name):
            assert name == "bookforge-klein-warmed-snapshot"

        def cls(self, **kwargs):
            options.update(kwargs)
            return lambda cls: cls

    def claim(key, value, *, skip_if_exists):
        assert skip_if_exists is True
        if key in claims:
            return False
        claims[key] = value
        return True

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Image=Image,
            App=App,
            is_local=lambda: True,
            enter=lambda **kwargs: lambda function: function,
            method=lambda: lambda function: function,
            Dict=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(put=claim)),
            Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
        ),
    )
    module = runpy.run_path(str(source))
    namespace = module["_manifest"].__globals__
    path = tmp_path / "manifest.json"
    namespace["MANIFEST"] = path
    namespace["time"] = SimpleNamespace(
        time=lambda: clock["wall"], perf_counter=lambda: clock["mono"]
    )
    manifest = json.loads(
        (root / "benchmarks/renderer-cold-start-2026-09-05/manifest.json").read_text()
    )
    manifest.update(
        experiment_id=module["EXPERIMENT"],
        expires_at=2000,
        deployment_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        max_operations=3,
        max_captures=1,
        operations=[
            {"ordinal": i, "variant": "warmed_snapshot", "request_id": f"{i:032x}"}
            for i in range(3)
        ],
    )
    # Fake renders use small deterministic bytes while preserving the real source/case pin check.
    assert (
        hashlib.sha256(
            json.dumps(manifest["cases"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == module["CASES_SHA256"]
    )
    for case in manifest["cases"]:
        for kind in ("master", "depth"):
            case[kind + "_sha256"] = hashlib.sha256(
                f"{kind}-{case['expected_bucket']}".encode()
            ).hexdigest()
    namespace["CASES_SHA256"] = hashlib.sha256(
        json.dumps(manifest["cases"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    for name, value in {
        "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_AWS",
        "MODAL_REGION": "us-east-1",
        "MODAL_TASK_ID": "capture-task",
    }.items():
        monkeypatch.setenv(name, value)
    packages = {
        name: SimpleNamespace(__version__=version) for name, version in module["VERSIONS"].items()
    }
    packages["torch"].cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _: "NVIDIA L4",
        get_device_capability=lambda _: (8, 9),
    )
    packages["torch"].version = SimpleNamespace(cuda="12.8")
    for name in ("AutoImageProcessor", "AutoModelForDepthEstimation", "pipeline"):
        setattr(packages["transformers"], name, None)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name not in packages:
            return original_import(name, *args, **kwargs)
        assert any(key.startswith("capture:") for key in claims)
        calls.append(name)
        return packages[name]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    class Runtime:
        def __init__(self, model_root):
            assert "capture:0" in claims
            assert model_root == Path("/models")
            calls.append("runtime")
            self.identity = manifest["expected_identity"]
            self.load_seconds = 1.0

        def compile(self, cache_root):
            assert cache_root == Path("/compiled") / module["CACHE_ID"]
            calls.append("compile")
            return 0.5

        def render(self, prompt, seed):
            case = next(row for row in manifest["cases"] if row["prompt"] == prompt)
            assert seed == case["seed"]
            calls.append("render")
            clock["mono"] += 1
            bucket = case["expected_bucket"]
            return (
                {
                    "sequence_bucket": bucket,
                    "total_seconds": 1.0,
                    "master_sha256": case["master_sha256"],
                    "depth_sha256": case["depth_sha256"],
                },
                f"master-{bucket}".encode(),
                f"depth-{bucket}".encode(),
            )

    monkeypatch.setitem(
        sys.modules, "klein_scene_runtime", SimpleNamespace(KleinSceneRuntime=Runtime)
    )
    return SimpleNamespace(
        module=module,
        cls=module["WarmedSnapshot"],
        options=options,
        mounts=mounts,
        claims=claims,
        calls=calls,
        clock=clock,
        path=path,
        manifest=manifest,
        packages=packages,
    )


def test_warmed_snapshot_verifies_capture_then_restores_without_model_reload(
    audit, monkeypatch, capsys
):
    a = audit
    assert {target for _, target in a.mounts} == {
        "/root/klein_scene_runtime.py",
        "/root/warmed-snapshot-manifest.json",
    }
    assert a.options["gpu"] == "L4" and a.options["cpu"] == (8, 8)
    assert a.options["memory"] == (65536, 65536)
    assert a.options["enable_memory_snapshot"] and a.options["single_use_containers"]
    assert a.options["experimental_options"] == {"enable_gpu_snapshot": True}
    assert "cloud" not in a.options
    assert tuple(
        a.options[k]
        for k in (
            "startup_timeout",
            "timeout",
            "min_containers",
            "max_containers",
            "buffer_containers",
            "scaledown_window",
            "retries",
            "region",
            "routing_region",
        )
    ) == (120, 60, 0, 1, 0, 2, 0, "us", "us-east")
    captured = a.cls()
    captured.capture()
    assert a.calls[-6:] == ["runtime", "compile", "render", "render", "render", "render"]
    assert len(a.claims) == 1 and a.claims["capture:0"] == captured.capture_id
    assert len(captured.warmups) == 4 and all(row["hashes_match"] for row in captured.warmups)
    assert not any(isinstance(value, bytes) for row in captured.warmups for value in row.values())
    assert captured.initialization == {
        "model_load_seconds": 1.0,
        "cache_setup_seconds": 0.5,
        "warmup_seconds": 4,
    }
    first, second = copy.deepcopy(captured), copy.deepcopy(captured)
    monkeypatch.setenv("MODAL_TASK_ID", "restored-one")
    first.activate()
    response = first.cycle("0" * 32)
    monkeypatch.setenv("MODAL_TASK_ID", "restored-two")
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_GCP")
    monkeypatch.setenv("MODAL_REGION", "us-central1")
    second.activate()
    other = second.cycle(f"{1:032x}")
    assert a.calls.count("runtime") == a.calls.count("compile") == 1
    assert a.calls.count("render") == 12
    assert response["snapshot"]["capture_id"] == other["snapshot"]["capture_id"]
    assert response["snapshot"]["activation_id"] != other["snapshot"]["activation_id"]
    assert response["location"]["container_sha256"] != other["location"]["container_sha256"]
    assert response["snapshot"]["warmups"] == other["snapshot"]["warmups"]
    assert response["stages"]["worker_seconds"] == other["stages"]["worker_seconds"] == 4
    assert response["variant"] == "warmed_snapshot" and len(response["samples"]) == 4
    before = a.calls.copy()
    with pytest.raises(ValueError, match="already claimed"):
        second.cycle(f"{1:032x}")
    assert a.calls == before
    logs = capsys.readouterr().out
    assert all(case["prompt"] not in logs for case in a.manifest["cases"])
    assert "restored-one" not in logs and "restored-two" not in logs


def test_warmed_snapshot_refuses_expiry_bad_gpu_and_wrong_hash_without_retry(audit):
    a = audit
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        a.cls().capture()
    assert not a.claims and not a.calls
    a.clock["wall"] = 1000
    captured = a.cls()
    captured.capture()
    before = a.calls.copy()
    with pytest.raises(ValueError, match="allowance exhausted"):
        a.cls().capture()
    assert a.calls == before
    a.packages["torch"].cuda.get_device_capability = lambda _: (8, 0)
    with pytest.raises(ValueError, match="restored GPU differs"):
        captured.activate()
    assert not hasattr(captured, "activation_id")
    a.packages["torch"].cuda.get_device_capability = lambda _: (8, 9)
    captured.activate()
    with pytest.raises(ValueError, match="not authorized"):
        captured.cycle("f" * 32)
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        captured.cycle("0" * 32)
    assert not any(key.startswith("request:") for key in a.claims)
    a.clock["wall"] = 1000
    original_render = captured.runtime.render

    def corrupt(*args):
        metrics, master, depth = original_render(*args)
        return metrics, master + b"wrong", depth

    captured.runtime.render = corrupt
    with pytest.raises(RuntimeError, match="cycle failed"):
        captured.cycle("0" * 32)
    assert a.claims["request:" + "0" * 32] is True
    before = a.calls.copy()
    with pytest.raises(ValueError, match="already claimed"):
        captured.cycle("0" * 32)
    assert a.calls == before
