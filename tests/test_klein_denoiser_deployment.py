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


def test_denoiser_deployment_admits_one_pinned_comparison(tmp_path, monkeypatch):
    options, concurrency, mounts, claimed, imports = {}, {}, [], set(), []

    class Image:
        @staticmethod
        def from_id(image_id):
            assert image_id == "im-WtXer8GjRPdgMqWAAUSMwJ"
            return Image()

        def add_local_file(self, source, target):
            mounts.append((source, target))
            return self

    class App:
        def __init__(self, name):
            assert name == "bookforge-klein-denoiser"

        def function(self, **kwargs):
            options.update(kwargs)
            return lambda function: function

    def concurrent(**kwargs):
        concurrency.update(kwargs)
        return lambda function: function

    def claim(key, value, *, skip_if_exists):
        assert key == "comparison" and value is skip_if_exists is True
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=App,
            Image=Image,
            concurrent=concurrent,
            is_local=lambda: True,
            Dict=SimpleNamespace(from_name=lambda *a, **kw: SimpleNamespace(put=claim)),
            Volume=SimpleNamespace(from_name=lambda name: name),
        ),
    )
    root = Path(__file__).resolve().parents[1]
    source = root / "deploy/modal_klein_denoiser.py"
    compare = runpy.run_path(str(source))["compare"]
    g = compare.__globals__
    assert concurrency == {"max_inputs": 1}
    assert options == dict(
        image=options["image"],
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
        volumes={"/compiled": "bookforge-klein-compile-cache-v1"},
    )
    assert mounts == [
        *((root / "deploy" / name, f"/root/{name}") for name in g["SOURCES"][:-1]),
        (
            root / "benchmarks/renderer-denoiser-2026-09-06/manifest.json",
            "/root/denoiser-manifest.json",
        ),
    ]
    frozen = json.loads(
        (root / "benchmarks/renderer-cold-start-2026-09-05/manifest.json").read_bytes()
    )
    manifest = dict(
        schema_version=1,
        status="authorized",
        experiment_id=g["EXPERIMENT"],
        image_id=g["IMAGE_ID"],
        cache_id=g["CACHE_ID"],
        maximum_calls=1,
        expires_at=int(time.time()) + 600,
        cases=frozen["cases"],
        expected_identity=frozen["expected_identity"],
        sources={
            name: hashlib.sha256((root / "deploy" / name).read_bytes()).hexdigest()
            for name in g["SOURCES"]
        },
    )
    path = tmp_path / "manifest.json"
    g["MANIFEST"] = path
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_AWS")
    monkeypatch.setenv("MODAL_REGION", "us-east-1")
    monkeypatch.setenv("MODAL_TASK_ID", "synthetic-task")
    original_import = builtins.__import__

    def fail_runtime(model_root):
        assert model_root == Path("/models") and claimed == {"comparison"}
        raise RuntimeError("synthetic initialization failure")

    def fake_import(name, *args, **kwargs):
        if name in ("klein_denoiser_probe", "klein_scene_runtime"):
            assert claimed == {"comparison"}
            imports.append(name)
            return SimpleNamespace(run_comparison=None, KleinSceneRuntime=fail_runtime)
        assert name not in ("torch", "diffusers", "transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    invalids = [
        {**manifest, **change}
        for change in (
            {"expires_at": 1},
            {"expires_at": int(time.time()) + 8000},
            {"status": "draft"},
            {"maximum_calls": 2},
            {"image_id": "other"},
            {"cache_id": "other"},
            {"cases": []},
            {"expected_identity": {}},
        )
    ]
    for name in g["SOURCES"]:
        invalid = copy.deepcopy(manifest)
        invalid["sources"][name] = "0" * 64
        invalids.append(invalid)
    for invalid in invalids:
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            compare()
    path.write_text(json.dumps(manifest))
    for name, value in (
        ("MODAL_REGION", "us-west-2"),
        ("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_GCP"),
    ):
        previous = g["os"].environ[name]
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            compare()
        monkeypatch.setenv(name, previous)
    assert not claimed and not imports
    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        compare()
    assert imports == ["klein_denoiser_probe", "klein_scene_runtime"]
    with pytest.raises(ValueError):
        compare()
    assert len(imports) == 2 and claimed == {"comparison"}
