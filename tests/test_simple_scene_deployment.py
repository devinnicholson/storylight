import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_simple_scenes as benchmark  # noqa: E402


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    options, mounts, claims, calls = {}, [], {}, []
    clock = {"wall": 1000}

    class Image:
        @staticmethod
        def from_id(value):
            assert value == benchmark.cold.IMAGE_ID
            return Image()

        def add_local_file(self, source, destination):
            mounts.append((str(source), destination))
            return self

    class App:
        def __init__(self, name):
            assert name == "bookforge-klein-simple-scenes"

        def cls(self, **kwargs):
            options.update(kwargs)
            return lambda cls: cls

    def concurrent(**kwargs):
        assert kwargs == {"max_inputs": 1}
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
            concurrent=concurrent,
            enter=lambda: lambda function: function,
            method=lambda: lambda function: function,
            Dict=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(put=claim)),
            Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
        ),
    )
    module = runpy.run_path(str(root / "deploy/modal_klein_simple_scenes.py"))
    namespace = module["_configuration"].__globals__
    manifest_path = tmp_path / "manifest.json"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(benchmark.FIXTURE.read_bytes())
    manifest = benchmark.prepare_manifest()
    manifest.update(status="authorized", expires_at=2000)
    manifest_path.write_text(json.dumps(manifest))
    namespace.update(MANIFEST=manifest_path, FIXTURE=fixture_path)
    namespace["time"] = SimpleNamespace(time=lambda: clock["wall"], perf_counter=lambda: 1.0)
    for name, value in {
        "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_AWS",
        "MODAL_REGION": "us-east-1",
        "MODAL_TASK_ID": "private-task-name",
    }.items():
        monkeypatch.setenv(name, value)

    class Tokenizer:
        count = 100

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return messages[0]["content"]

        def __call__(self, text):
            return {"input_ids": list(range(self.count))}

    class Runtime:
        def __init__(self, model_root):
            assert "initialization" in claims and model_root == Path("/models")
            calls.append("initialize")
            self.identity = copy.deepcopy(manifest["expected_identity"])
            self.load_seconds = 1.0
            self.pipe = SimpleNamespace(tokenizer=Tokenizer())
            self.fail = False

        def compile(self, path):
            assert path == Path("/compiled") / module["CACHE_ID"]
            calls.append("compile")
            return 0.5

        def render(self, prompt, seed, **kwargs):
            calls.append((prompt, seed, kwargs))
            if self.fail:
                raise RuntimeError("private failure details")
            master = f"master-{seed}".encode()
            return {"master_sha256": hashlib.sha256(master).hexdigest()}, master, b"depth"

    monkeypatch.setitem(
        sys.modules,
        "klein_reference_runtime",
        SimpleNamespace(
            KleinReferenceRuntime=Runtime,
        ),
    )
    return SimpleNamespace(
        cls=module["SimpleSceneRenderer"],
        module=module,
        options=options,
        mounts=mounts,
        claims=claims,
        calls=calls,
        clock=clock,
        manifest=manifest,
        manifest_path=manifest_path,
        fixture_path=fixture_path,
    )


def test_exact_deployment_runs_twelve_ordered_operations_with_bound_first_state(deployment, capsys):
    a = deployment
    assert {destination for _, destination in a.mounts} == {
        "/root/klein_scene_runtime.py",
        "/root/klein_reference_runtime.py",
        "/root/simple-scene-controls.json",
        "/root/simple-scenes-manifest.json",
    }
    assert a.options["gpu"] == "L4" and a.options["cpu"] == (8, 8)
    assert a.options["memory"] == (65536, 65536)
    assert not a.options.get("enable_memory_snapshot", False)
    assert not a.options.get("experimental_options")
    assert not a.options.get("single_use_containers", False)
    assert "cloud" not in a.options
    assert tuple(
        a.options[key]
        for key in (
            "startup_timeout",
            "timeout",
            "retries",
            "min_containers",
            "max_containers",
            "buffer_containers",
            "scaledown_window",
            "region",
            "routing_region",
        )
    ) == (120, 60, 0, 0, 1, 0, 90, "us", "us-east")
    renderer = a.cls()
    renderer.load()
    assert a.calls == ["initialize", "compile"]
    with pytest.raises(ValueError, match="initialization allowance exhausted"):
        a.cls().load()
    operations = a.manifest["operations"]
    assert [row["variant"] for row in operations] == [
        "base",
        "text_next",
        "reference_next",
        "base",
        "reference_next",
        "text_next",
        "base",
        "text_next",
        "reference_next",
        "base",
        "reference_next",
        "text_next",
    ]
    bases = {}
    for ordinal, row in enumerate(operations):
        kwargs = {}
        if row["reference_request_id"] is not None:
            master = bases[row["reference_request_id"]]
            kwargs = dict(
                reference_jpeg=master, reference_sha256=hashlib.sha256(master).hexdigest()
            )
        result = renderer.render(row["request_id"], **kwargs)
        assert result["request_id"] == row["request_id"]
        assert renderer.next_ordinal == ordinal + 1
        if row["variant"] == "base":
            bases[row["request_id"]] = result["master"]
        assert a.calls[-1][2] == {
            "reference_jpeg": kwargs.get("reference_jpeg"),
            "reference_sha256": kwargs.get("reference_sha256"),
        }
        before = len(a.calls)
        with pytest.raises(ValueError):
            renderer.render(row["request_id"], **kwargs)
        assert len(a.calls) == before
    assert len(a.claims) == 13 and len(a.calls) == 14
    logs = capsys.readouterr().out
    assert "private-task-name" not in logs
    assert all(row["before_prompt"] not in logs for row in benchmark.fixture()["controls"])


def test_expiry_and_counterfactual_manifest_refuse_before_initialization(deployment):
    a = deployment
    for changed in (
        {"expires_at": 999},
        {"status": "draft"},
        {"maximum_operations": 13},
        {"runtime_sha256": "0" * 64},
        {"reference_runtime_sha256": "0" * 64},
        {"deployment_sha256": "0" * 64},
        {"operations": list(reversed(a.manifest["operations"]))},
    ):
        a.manifest_path.write_text(json.dumps({**a.manifest, **changed}))
        with pytest.raises(ValueError):
            a.cls().load()
    a.manifest_path.write_text(json.dumps(a.manifest))
    a.fixture_path.write_bytes(a.fixture_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest differs"):
        a.cls().load()
    assert not a.calls and not a.claims


def test_reference_binding_expiry_and_partial_failure_cannot_reroll(deployment, capsys):
    a = deployment
    renderer = a.cls()
    renderer.load()
    rows = a.manifest["operations"]
    with pytest.raises(ValueError, match="request order"):
        renderer.render(rows[1]["request_id"])
    first = renderer.render(rows[0]["request_id"])
    renderer.render(rows[1]["request_id"])
    before = len(a.calls)
    wrong = b"different-first-state"
    with pytest.raises(ValueError, match="first state"):
        renderer.render(rows[2]["request_id"], wrong, hashlib.sha256(wrong).hexdigest())
    assert len(a.calls) == before and renderer.next_ordinal == 2
    assert f"request:{rows[2]['request_id']}" not in a.claims
    a.clock["wall"] = 2001
    with pytest.raises(ValueError, match="expired"):
        renderer.render(rows[2]["request_id"])
    assert len(a.calls) == before
    a.clock["wall"] = 1000
    renderer.runtime.fail = True
    with pytest.raises(RuntimeError, match="render failed"):
        renderer.render(rows[2]["request_id"], first["master"], first["metrics"]["master_sha256"])
    assert a.claims[f"request:{rows[2]['request_id']}"] is True
    assert renderer.next_ordinal == 3 and len(a.calls) == before + 1
    with pytest.raises(ValueError, match="request order"):
        renderer.render(rows[2]["request_id"], first["master"], first["metrics"]["master_sha256"])
    assert len(a.calls) == before + 1
    assert "private failure details" not in capsys.readouterr().out
