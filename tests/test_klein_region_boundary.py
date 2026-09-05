from __future__ import annotations

import asyncio
import copy
import json
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_latency_client import response

from bookforge.klein_latency_client import LatencyError, validate_payload
from bookforge.klein_region_client import RegionClient
from deploy import klein_latency_protocol as protocol
from scripts import freeze_klein_latency as freezer
from scripts import prepare_klein_region_comparison as preparation


def test_explicit_runtime_mounts_import_in_a_fresh_container_layout(tmp_path):
    deployment = preparation.ROOT / "deploy/modal_klein_region.py"
    stub = r"""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

mounts = {}
candidate = None
registered_images = []
class Image:
    def __init__(self, cached=False):
        self.cached = cached
    @staticmethod
    def from_id(image_id):
        global candidate
        assert image_id == "im-WtXer8GjRPdgMqWAAUSMwJ"
        assert candidate is None
        candidate = Image(cached=True)
        return candidate
    def add_local_file(self, source, destination, **kwargs):
        if self.cached:
            assert not kwargs.get("copy", False)
            mounts[destination] = str(Path(source).resolve())
        return self
    def run_function(self, *args, **kwargs):
        assert not self.cached, "candidate image must not build"
        return self
    def env(self, *args, **kwargs):
        assert not self.cached, "candidate image must not build"
        return self
    def uv_pip_install(self, *args, **kwargs):
        assert not self.cached, "candidate image must not build"
        return self

def decorator(**kwargs):
    return lambda target: target

class App:
    def __init__(self, name):
        pass
    @staticmethod
    def register(**kwargs):
        registered_images.append(kwargs.get("image"))
        return decorator(**kwargs)
    cls = server = register

sys.modules["modal"] = SimpleNamespace(
    App=App, Image=Image, is_local=lambda: sys.argv[1] == "build",
    concurrent=decorator, enter=decorator, method=decorator,
    Volume=SimpleNamespace(from_name=lambda *args, **kwargs: object()),
    Dict=SimpleNamespace(from_name=lambda *args, **kwargs: object()),
)
"""
    build = (
        stub
        + r"""
sys.modules["klein_restart"] = SimpleNamespace(WEIGHTS={})
sys.modules["klein_weights"] = SimpleNamespace(bake_weights=lambda *args: None)
sys.modules["modal_compare"] = SimpleNamespace(image=Image())
module = runpy.run_path(sys.argv[2])
assert candidate is not None and module["image"] is candidate
assert registered_images[-2:] == [candidate, candidate]
assert "torch" not in sys.modules and "diffusers" not in sys.modules
print(json.dumps(mounts))
"""
    )
    built = subprocess.run(
        [sys.executable, "-I", "-S", "-c", build, "build", str(deployment)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    runtime = tmp_path / "container-root"
    runtime.mkdir()
    for destination, source in json.loads(built.stdout).items():
        relative = Path(destination).relative_to("/root")
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if destination == "/root/region-manifest.json":
            assert Path(source) == (
                preparation.ROOT / "benchmarks/renderer-region-2026-09-05-c/manifest.json"
            )
            target.write_bytes(preparation.encoded(preparation.prepare("isolated-layout")))
        else:
            shutil.copyfile(source, target)
    # Modal supplies the service entry module; all its sibling imports must be explicit mounts.
    shutil.copyfile(deployment, runtime / deployment.name)
    remote = (
        stub
        + r"""
root = Path(sys.argv[2]).resolve()
assert not any(Path(entry).is_relative_to(Path(sys.argv[3])) for entry in sys.path if entry)
assert "modal_klein_latency" not in sys.modules
assert "klein_latency_runtime" not in sys.modules
sys.path.insert(0, str(root))
try:
    module = runpy.run_path(str(root / "modal_klein_region.py"))
except ModuleNotFoundError as error:
    print(json.dumps({"missing": error.name}))
    raise SystemExit(2)
assert "RegionRuntime" in module and "RegionStudio" in module and "RegionServer" in module
for name in (
    "modal_klein_latency", "klein_latency_runtime", "klein_scene_runtime", "klein_latency_protocol"
):
    assert Path(sys.modules[name].__file__).parent == root
heavy_modules = {"torch", "diffusers", "transformers", "klein_restart", "modal_compare"}
assert not heavy_modules & set(sys.modules)
print(json.dumps({"imported": True, "model_initialized": False}))
"""
    )

    def import_remote():
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                remote,
                "remote",
                str(runtime),
                str(preparation.ROOT),
            ],
            cwd=runtime,
            capture_output=True,
            text=True,
            timeout=15,
        )

    imported = import_remote()
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert json.loads(imported.stdout) == {"imported": True, "model_initialized": False}
    (runtime / "modal_klein_latency.py").unlink()
    missing = import_remote()
    assert missing.returncode == 2
    assert json.loads(missing.stdout) == {"missing": "modal_klein_latency"}


def test_deployment_pins_placement_and_rejects_wrong_host_before_model_initialization(
    monkeypatch, tmp_path
):
    registrations, model_starts = {}, []
    manifest_path = tmp_path / "manifest.json"
    # Authorization-file validation has its own test; exercise the real CLI's emitted status here.
    monkeypatch.setattr(preparation, "read_authorization", lambda *args: {})
    assert (
        preparation.main(
            [
                "--experiment-id",
                "boundary-runtime",
                "--output",
                str(manifest_path),
                "--activate-until",
                str(int(time.time()) + 1200),
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--authorization-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())

    def decorator(**kwargs):
        return lambda target: target

    class App:
        def __init__(self, name):
            self.name = name

        def register(self, **kwargs):
            registrations[self.name] = kwargs
            return lambda target: target

        cls = server = register

    def model(*args):
        model_starts.append(args)
        return SimpleNamespace(
            identity=manifest["expected_identity"],
            instrumentation_sha256=manifest["instrumentation_sha256"],
            compile=lambda path: 0.0,
        )

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=App,
            is_local=lambda: False,
            concurrent=decorator,
            enter=decorator,
            method=decorator,
            Dict=SimpleNamespace(from_name=lambda *args, **kwargs: object()),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "klein_latency_runtime",
        SimpleNamespace(
            LatencySceneRuntime=model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "modal_klein_latency",
        SimpleNamespace(
            RESOURCES={"gpu": "L4"},
            Runtime=object,
            image=None,
        ),
    )
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "deploy/modal_klein_region.py")
    )
    sdk = registrations["bookforge-klein-region-sdk"]
    http = registrations["bookforge-klein-region-http"]
    assert (sdk["cloud"], sdk["region"], sdk["routing_region"]) == ("aws", "us-east", "us-east")
    assert (http["cloud"], http["compute_region"], http["routing_region"]) == (
        "aws",
        "us-east",
        "us-east",
    )
    assert sdk["retries"] == 0 and http["unauthenticated"] is False
    for cloud, region in (("CLOUD_PROVIDER_GCP", "us-east-1"), ("CLOUD_PROVIDER_AWS", "us-west-2")):
        monkeypatch.setenv("MODAL_CLOUD_PROVIDER", cloud)
        monkeypatch.setenv("MODAL_REGION", region)
        with pytest.raises(RuntimeError, match="compute placement"):
            module["RegionRuntime"]("sdk")
    monkeypatch.setenv("MODAL_CLOUD_PROVIDER", "CLOUD_PROVIDER_AWS")
    monkeypatch.setenv("MODAL_REGION", "us-east-1")
    module["require_placement"]()
    assert not model_starts
    files = {**preparation.SUPPORT, "region_deployment_sha256": "modal_klein_region.py"}
    for name in files.values():
        (tmp_path / name).write_bytes((preparation.ROOT / "deploy" / name).read_bytes())
    (tmp_path / "klein_region_client.py").write_bytes(preparation.REGION_CLIENT.read_bytes())
    runtime_globals = module["RegionRuntime"].__init__.__globals__
    monkeypatch.setitem(runtime_globals, "DEPLOY", tmp_path)
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            read_text(manifest_path)
            if str(path) == "/root/region-manifest.json"
            else read_text(path, *args, **kwargs)
        ),
    )
    manifest_path.write_text(json.dumps({**manifest, "baked_image_id": "im-another-image"}))
    with pytest.raises(RuntimeError, match="image"):
        module["RegionRuntime"]("sdk")
    assert not model_starts
    manifest_path.write_text(json.dumps(manifest))
    worker = module["RegionRuntime"]("sdk")
    assert len(model_starts) == 1 and len(worker.allowed) == 7


def test_client_binds_new_origins_and_region_evidence_for_render_and_prewarm():
    manifest = freezer.freeze("region-boundary", int(time.time()) + 1200)
    manifest["region_deployment_sha256"] = "f" * 64
    original_pin = manifest["deployment_sha256"]
    lookups = []
    selected_url = "https://isolated.us-east.modal.direct/"
    selected_payload = None

    async def hydrate_remote():
        return None

    async def get_url():
        return selected_url

    async def get():
        return protocol.pack_response(selected_payload)

    async def spawn(**kwargs):
        return SimpleNamespace(get=SimpleNamespace(aio=get))

    class Remote:
        hydrate = SimpleNamespace(aio=hydrate_remote)

        def __call__(self):
            return SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))

    def from_name(app, name):
        lookups.append((app, name))
        return (
            Remote()
            if name == "RegionStudio"
            else SimpleNamespace(
                get_url=SimpleNamespace(aio=get_url),
            )
        )

    modal = SimpleNamespace(
        Cls=SimpleNamespace(from_name=from_name),
        Server=SimpleNamespace(from_name=from_name),
    )

    async def scenario():
        nonlocal selected_url, selected_payload
        client = RegionClient(manifest, modal_module=modal)
        await client.lookup("sdk")
        await client.lookup("http")
        assert lookups == [
            ("bookforge-klein-region-sdk", "RegionStudio"),
            ("bookforge-klein-region-http", "RegionServer"),
        ]
        assert client.url == selected_url.rstrip("/")
        assert client.expected["deployment_sha256"] == "f" * 64
        assert manifest["deployment_sha256"] == original_pin
        for invalid_url in (
            "http://isolated.us-east.modal.direct",
            "https://isolated.modal.run",
            "https://isolated.us-east.modal.direct.evil.test",
            "https://user@isolated.us-east.modal.direct",
        ):
            selected_url = invalid_url
            client.url = None
            with pytest.raises(LatencyError, match="region_server_url_invalid"):
                await client.lookup("http")
            assert client.url is None
        for operation in (manifest["operations"][0], manifest["operations"][2]):
            request, bucket = operation["request"], operation["expected_bucket"]
            selected_payload = response(client.expected, request, bucket)
            selected_payload["location"].update(
                cloud="CLOUD_PROVIDER_AWS", compute_region="us-east-1"
            )
            payload, _ = await client.invoke("sdk", request, bucket)
            assert payload["deployment_sha256"] == "f" * 64
            with pytest.raises(LatencyError, match="response_identity_mismatch"):
                validate_payload(payload, request, manifest, bucket)
            valid = copy.deepcopy(selected_payload)
            for field, wrong in (("cloud", "CLOUD_PROVIDER_GCP"), ("compute_region", "us-west-2")):
                selected_payload = copy.deepcopy(valid)
                selected_payload["location"][field] = wrong
                with pytest.raises(LatencyError, match="region_response_provenance_mismatch"):
                    await client.invoke("sdk", request, bucket)
            selected_payload = copy.deepcopy(valid)
            selected_payload["deployment_sha256"] = original_pin
            with pytest.raises(LatencyError, match="operation_failed"):
                await client.invoke("sdk", request, bucket)

    asyncio.run(scenario())
