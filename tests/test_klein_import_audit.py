import builtins
import hashlib
import json
import math
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cpu_import_audit_claims_once_and_restores_swallowed_cuda_guards(tmp_path, monkeypatch):
    options, mounts, claimed = {}, [], set()

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
            assert name == "bookforge-klein-import-audit"

        def function(self, **kwargs):
            options.update(kwargs)
            return lambda function: function

    def claim(key, value, *, skip_if_exists):
        assert value is skip_if_exists is True
        if key in claimed:
            return False
        claimed.add(key)
        return True

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Image=Image,
            App=App,
            is_local=lambda: True,
            Dict=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(put=claim)),
        ),
    )
    source = Path(__file__).resolve().parents[1] / "deploy/modal_klein_import_audit.py"
    module = runpy.run_path(str(source))
    assert len(mounts) == 1 and mounts[0][1] == "/root/import-audit-manifest.json"
    assert "gpu" not in options and options["cpu"] == (8, 8)
    assert options["memory"] == (8192, 16384) and options["single_use_containers"]
    assert tuple(
        options[k]
        for k in (
            "startup_timeout",
            "timeout",
            "min_containers",
            "max_containers",
            "scaledown_window",
            "retries",
            "cloud",
            "region",
            "include_source",
        )
    ) == (120, 120, 0, 1, 2, 0, "aws", "us-east", True)
    invoke = module["audit_imports"]
    manifest_path = tmp_path / "manifest.json"
    invoke.__globals__["MANIFEST"] = manifest_path
    manifest = dict(
        schema_version=1,
        status="authorized",
        experiment_id=module["EXPERIMENT"],
        expires_at=int(time.time()) + 600,
        image_id=module["IMAGE_ID"],
        deployment_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        request_id="0" * 32,
    )
    originals = {
        name: (lambda: None)
        for name in (
            "is_available",
            "_lazy_init",
            "device_count",
            "get_device_capability",
            "get_device_properties",
            "current_device",
            "init",
        )
    }
    cuda = SimpleNamespace(**originals)
    core = SimpleNamespace(_cuda_getDeviceCount=lambda: None, _cuda_init=lambda: None)
    core_originals = vars(core).copy()
    packages = {
        name: SimpleNamespace(__version__=version) for name, version in module["VERSIONS"].items()
    }
    packages["torch"].cuda, packages["torch"]._C = cuda, core
    for name in ("AutoImageProcessor", "AutoModelForDepthEstimation", "pipeline"):
        setattr(packages["transformers"], name, None)
    imports, fail = [], False
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name not in packages:
            return original_import(name, *args, **kwargs)
        assert claimed == {manifest["request_id"]}
        imports.append(name)
        if name == "diffusers":
            for callback in (cuda.is_available, cuda.is_available, core._cuda_getDeviceCount):
                with pytest.raises(RuntimeError, match="CUDA access forbidden"):
                    callback()
            if fail:
                raise RuntimeError("private framework failure details")
        return packages[name]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for invalid in (
        {**manifest, "expires_at": 1},
        {**manifest, "image_id": "wrong"},
        {**manifest, "deployment_sha256": "wrong"},
    ):
        manifest_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="authorization differs"):
            invoke(manifest["request_id"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="authorization differs"):
        invoke("f" * 32)
    assert not imports and not claimed
    report = invoke(manifest["request_id"])
    assert report["import_completed"] and report["guards_installed"]
    assert report["attempts"] == ["torch.cuda.is_available", "torch._C._cuda_getDeviceCount"]
    assert report["pre_torch_cuda_audit"] == "unknown"
    assert report["packages"] == module["VERSIONS"]
    assert all(
        math.isfinite(report[key]) and report[key] >= 0
        for key in (
            "imports_seconds",
            "torch_import_seconds",
            "process_peak_rss_gib",
        )
    )
    assert vars(cuda) == originals and vars(core) == core_originals
    before = imports.copy()
    with pytest.raises(ValueError, match="already claimed"):
        invoke(manifest["request_id"])
    assert imports == before
    fail = True
    failed = module["_audit_imports"]()
    assert not failed["import_completed"] and "private" not in json.dumps(failed)
    assert vars(cuda) == originals and vars(core) == core_originals
