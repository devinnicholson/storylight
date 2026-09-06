import builtins
import copy
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    options, mounts, stored, concurrency = {}, [], {}, {}

    def concurrent(**kwargs):
        concurrency.update(kwargs)
        return lambda function: function

    class Image:
        @staticmethod
        def from_id(value):
            assert value == "im-WtXer8GjRPdgMqWAAUSMwJ"
            return Image()

        def add_local_file(self, source, target):
            mounts.append((source, target))
            return self

    class App:
        def __init__(self, name):
            assert name == "bookforge-klein-loading"

        def function(self, **kwargs):
            options.update(kwargs)
            return lambda function: function

    def put(key, value, *, skip_if_exists):
        assert skip_if_exists and value is True
        if key in stored:
            return False
        stored[key] = value
        return True

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=App,
            Image=Image,
            is_local=lambda: True,
            concurrent=concurrent,
            Dict=SimpleNamespace(
                from_name=lambda *a, **kw: SimpleNamespace(put=put, get=stored.get)
            ),
            Volume=SimpleNamespace(from_name=lambda name: name),
        ),
    )
    source = Path(__file__).resolve().parents[1] / "deploy/modal_klein_loading.py"
    module = runpy.run_path(str(source))
    call = module["loading_cycle"]
    g = call.__globals__
    manifest = json.loads(
        (source.parents[1] / "benchmarks/renderer-cold-start-2026-09-05/manifest.json").read_bytes()
    )
    manifest.update(
        experiment_id=g["EXPERIMENT"],
        expires_at=int(time.time()) + 600,
        deployment_sha256=g["file_hash"](source),
        operations=[
            {
                "ordinal": i,
                "request_id": hashlib.sha256(f"{g['EXPERIMENT']}:{i}".encode()).hexdigest()[:32],
                "variant": variant,
            }
            for i, variant in enumerate(g["SCHEDULE"])
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    g["MANIFEST"], g["MODELS"], g["COMPILED"] = path, tmp_path / "models", tmp_path / "compiled"
    g["COMPILED"].mkdir()
    artifact = b"frozen compiler artifact"
    (g["COMPILED"] / "artifacts.bin").write_bytes(artifact)
    (g["COMPILED"] / "manifest.json").write_text(
        json.dumps(
            {
                "identity": manifest["expected_identity"],
                "sha256": hashlib.sha256(artifact).hexdigest(),
            }
        )
    )
    shards = g["MODELS"] / "klein/text_encoder"
    shards.mkdir(parents=True)
    (shards / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"weight_a": "first.safetensors", "weight_b": "second.safetensors"},
            }
        )
    )
    for name in ("first", "second"):
        (shards / f"{name}.safetensors").write_bytes(b"shard")
    for name in ("torch", "diffusers", "transformers", "triton"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_AWS")
    monkeypatch.setenv("MODAL_REGION", "us-east-1")
    monkeypatch.setenv("MODAL_TASK_ID", "task-test")
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", "false")
    monkeypatch.setenv("HF_PARALLEL_LOADING_WORKERS", "4")
    return SimpleNamespace(
        call=call,
        g=g,
        manifest=manifest,
        path=path,
        options=options,
        mounts=mounts,
        stored=stored,
        concurrency=concurrency,
    )


def test_loading_admission_pins_resources_cache_and_fresh_process(deployment, monkeypatch):
    d = deployment
    assert d.concurrency == {"max_inputs": 1}
    assert {
        key: d.options[key]
        for key in (
            "gpu",
            "cpu",
            "memory",
            "timeout",
            "startup_timeout",
            "retries",
            "min_containers",
            "buffer_containers",
            "max_containers",
            "scaledown_window",
            "single_use_containers",
            "cloud",
            "region",
            "routing_region",
            "include_source",
        )
    } == dict(
        gpu="L4",
        cpu=(8, 8),
        memory=(65536, 65536),
        timeout=120,
        startup_timeout=30,
        retries=0,
        min_containers=0,
        buffer_containers=0,
        max_containers=1,
        scaledown_window=2,
        single_use_containers=True,
        cloud="aws",
        region="us-east",
        routing_region="us-east",
        include_source=True,
    )
    assert [target for _, target in d.mounts] == [
        "/root/klein_scene_runtime.py",
        "/root/loading-manifest.json",
    ]
    assert d.g["configuration"](0) == d.manifest
    for key in ("cases", "expected_identity", "operations"):
        invalid = copy.deepcopy(d.manifest)
        invalid[key] = []
        d.path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            d.call(0)
    for changes in (
        {"expires_at": 1},
        {"image_id": "wrong"},
        {"runtime_sha256": "wrong"},
        {"deployment_sha256": "wrong"},
        {"cache_id": "wrong"},
    ):
        d.path.write_text(json.dumps({**d.manifest, **changes}))
        with pytest.raises(ValueError):
            d.call(0)
    d.path.write_text(json.dumps(d.manifest))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError):
        d.call(0)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setenv("MODAL_REGION", "us-west-2")
    with pytest.raises(ValueError):
        d.call(0)
    monkeypatch.setenv("MODAL_REGION", "us-east-1")
    (d.g["COMPILED"] / "artifacts.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        d.call(0)
    assert not d.stored


@pytest.mark.parametrize("fail", [False, True])
def test_loading_sets_environment_before_imports_and_never_retries(deployment, monkeypatch, fail):
    d = deployment
    master, depth = b"reference master", b"reference depth"
    for case in d.manifest["cases"]:
        case.update(
            master_sha256=hashlib.sha256(master).hexdigest(),
            depth_sha256=hashlib.sha256(depth).hexdigest(),
        )
    d.g["CASES_SHA256"] = d.g["digest"](d.manifest["cases"])
    d.path.write_text(json.dumps(d.manifest))
    imports, renders = [], []
    ordinal = 0

    class Runtime:
        def __init__(self, model_root):
            assert model_root == d.g["MODELS"]
            self.identity, self.load_seconds = d.manifest["expected_identity"], 0.01

        def compile(self, directory):
            assert directory == d.g["COMPILED"]
            return 0.02

        def render(self, prompt, seed):
            renders.append((prompt, seed))
            if fail:
                raise RuntimeError("partial render failed")
            case = next(row for row in d.manifest["cases"] if row["seed"] == seed)
            return (
                {
                    "sequence_bucket": case["expected_bucket"],
                    "master_sha256": case["master_sha256"],
                    "depth_sha256": case["depth_sha256"],
                },
                master,
                depth,
            )

    original = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("diffusers", "torch", "transformers", "triton", "klein_scene_runtime"):
            assert d.manifest["operations"][ordinal]["request_id"] in d.stored
            assert d.g["os"].environ["HF_ENABLE_PARALLEL_LOADING"] == (
                "true" if ordinal in (1, 2) else "false"
            )
            assert d.g["os"].environ["HF_PARALLEL_LOADING_WORKERS"] == "4"
            imports.append(name)
            return SimpleNamespace(
                KleinSceneRuntime=Runtime,
                AutoImageProcessor=None,
                AutoModelForDepthEstimation=None,
                pipeline=None,
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ValueError):
        d.call(1)
    for ordinal in range(4):
        if fail:
            with pytest.raises(RuntimeError):
                d.call(ordinal)
        else:
            result = d.call(ordinal)
            assert len(result["samples"]) == 4
            assert result["shard_inventory"]["klein/text_encoder"]["indexed_shards"] == 2
            assert (
                0
                <= result["stages"]["first_artifact_seconds"]
                <= result["stages"]["worker_seconds"]
            )
        before = len(imports)
        with pytest.raises(ValueError):
            d.call(ordinal)
        assert len(imports) == before
        if fail:
            with pytest.raises(ValueError):
                d.call(ordinal + 1)
            break
    assert len(renders) == (1 if fail else 16)
    with pytest.raises(ValueError):
        d.call(4)
