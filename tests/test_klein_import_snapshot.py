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
    source = root / "deploy/modal_klein_import_snapshot.py"
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
            assert name == "bookforge-klein-import-snapshot"

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
        max_operations=5,
        max_captures=3,
        operations=[
            {"ordinal": i, "variant": "snapshot", "request_id": f"{i:032x}"} for i in range(5)
        ],
    )
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
    packages["torch"].cuda = SimpleNamespace(is_available=lambda: True)
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
            assert any(key.startswith("request:") for key in claims)
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
            return {"sequence_bucket": case["expected_bucket"]}, b"master", b"depth"

    monkeypatch.setitem(
        sys.modules, "klein_scene_runtime", SimpleNamespace(KleinSceneRuntime=Runtime)
    )
    return SimpleNamespace(
        module=module,
        cls=module["ImportSnapshot"],
        options=options,
        mounts=mounts,
        claims=claims,
        calls=calls,
        clock=clock,
        path=path,
        manifest=manifest,
        packages=packages,
    )


def test_import_snapshot_separates_capture_restore_and_claimed_model_work(
    audit, monkeypatch, capsys
):
    a = audit
    assert {target for _, target in a.mounts} == {
        "/root/klein_scene_runtime.py",
        "/root/import-snapshot-manifest.json",
    }
    assert a.options["gpu"] == "L4" and a.options["cpu"] == (8, 8)
    assert a.options["memory"] == (65536, 65536)
    assert a.options["enable_memory_snapshot"] and a.options["single_use_containers"]
    assert a.options["experimental_options"] == {"enable_gpu_snapshot": True}
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
            "cloud",
            "region",
            "routing_region",
        )
    ) == (
        120,
        180,
        0,
        1,
        0,
        2,
        0,
        "aws",
        "us-east",
        "us-east",
    )
    captured = a.cls()
    captured.capture()
    assert "runtime" not in a.calls and len(a.claims) == 1
    first, second = copy.deepcopy(captured), copy.deepcopy(captured)
    monkeypatch.setenv("MODAL_TASK_ID", "restored-task-one")
    first.activate()
    assert "runtime" not in a.calls
    response = first.cycle("0" * 32)
    assert a.calls[-6:] == ["runtime", "compile", "render", "render", "render", "render"]
    assert [row["metrics"]["sequence_bucket"] for row in response["samples"]] == [
        128,
        256,
        128,
        256,
    ]
    assert response["stages"]["worker_seconds"] == 4
    before = a.calls.copy()
    with pytest.raises(ValueError, match="already claimed"):
        first.cycle("0" * 32)
    assert a.calls == before
    monkeypatch.setenv("MODAL_TASK_ID", "restored-task-two")
    second.activate()
    other = second.cycle(f"{1:032x}")
    assert response["snapshot"]["capture_id"] == other["snapshot"]["capture_id"]
    assert response["snapshot"]["activation_id"] != other["snapshot"]["activation_id"]
    assert response["location"]["container_sha256"] != other["location"]["container_sha256"]
    assert (
        response["snapshot"]["capture_container_sha256"] != response["location"]["container_sha256"]
    )
    assert response["snapshot"]["versions"] == a.module["VERSIONS"]
    logs = capsys.readouterr().out
    assert all(case["prompt"] not in logs for case in a.manifest["cases"])
    assert "capture-task" not in logs and "restored-task" not in logs


def test_import_snapshot_expiry_capture_exhaustion_and_failed_claims_are_terminal(audit):
    a = audit
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        a.cls().capture()
    assert not a.claims and not a.calls
    a.clock["wall"] = 1000
    a.packages["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="capture failed"):
        a.cls().capture()
    assert len(a.claims) == 1
    a.packages["torch"].cuda.is_available = lambda: True
    a.cls().capture()
    captured = a.cls()
    captured.capture()
    before = a.calls.copy()
    with pytest.raises(ValueError, match="allowance exhausted"):
        a.cls().capture()
    assert len(a.claims) == 3 and a.calls == before
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        captured.activate()
    a.clock["wall"] = 1000
    captured.activate()
    with pytest.raises(ValueError, match="not authorized"):
        captured.cycle("f" * 32)
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        captured.cycle("0" * 32)
    assert not any(key.startswith("request:") for key in a.claims)
    assert "runtime" not in a.calls
