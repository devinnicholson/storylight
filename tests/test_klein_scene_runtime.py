import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("klein_runtime_test", "deploy/klein_scene_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.mark.parametrize(
    "tokens,expected", [(1, 128), (128, 128), (129, 256), (256, 256), (257, 512), (512, 512)]
)
def test_sequence_bucket_never_truncates(tokens, expected):
    assert runtime.sequence_bucket(tokens) == expected


@pytest.mark.parametrize("tokens", [0, -1, 513, 4000])
def test_out_of_range_tokens_fail(tokens):
    with pytest.raises(ValueError):
        runtime.sequence_bucket(tokens)


def test_cache_requires_matching_runtime_and_bytes():
    artifact = b"trusted test artifact"
    identity = {**runtime.PROFILE, "gpu": "NVIDIA L4", "torch": "2.8.0"}
    manifest = {"identity": identity, "sha256": hashlib.sha256(artifact).hexdigest()}
    runtime.validate_cache(manifest, artifact, identity)
    with pytest.raises(ValueError, match="checksum"):
        runtime.validate_cache(manifest, b"changed", identity)
    for key in ("gpu", "torch", "model_revision", "compile_mode", "buckets"):
        with pytest.raises(ValueError, match="match"):
            runtime.validate_cache(manifest, artifact, {**identity, key: "changed"})


@pytest.mark.parametrize(
    "prompt,seed", [("", 1), ("x" * 4001, 1), ("a boat", -1), ("a boat", 2**32)]
)
def test_invalid_requests_do_not_reach_the_gpu(prompt, seed, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    obj = runtime.KleinSceneRuntime.__new__(runtime.KleinSceneRuntime)
    with pytest.raises(ValueError):
        obj.render(prompt, seed)


def test_overlong_tokenized_prompt_is_rejected_before_gpu_inference(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt with chat-template tokens"

        def __call__(self, text):
            return {"input_ids": list(range(513))}

    obj = runtime.KleinSceneRuntime.__new__(runtime.KleinSceneRuntime)
    obj.pipe = SimpleNamespace(tokenizer=Tokenizer())
    with pytest.raises(ValueError, match="not truncating"):
        obj.render("a synthetically long story brief", 2**32 - 1)


def test_compile_checks_cache_before_deserializing(tmp_path, monkeypatch):
    import sys

    loaded = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            compiler=SimpleNamespace(load_cache_artifacts=loaded.append),
        ),
    )
    obj = runtime.KleinSceneRuntime.__new__(runtime.KleinSceneRuntime)
    obj.identity = {"gpu": "L4"}
    obj.pipe = SimpleNamespace(
        transformer=SimpleNamespace(compile_repeated_blocks=lambda **kw: None)
    )
    artifact = b"private artifact"
    (tmp_path / "artifacts.bin").write_bytes(artifact)
    manifest = {"identity": {"gpu": "other"}, "sha256": hashlib.sha256(artifact).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="match"):
        obj.compile(tmp_path)
    assert loaded == []
    manifest["identity"] = obj.identity
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    obj.compile(tmp_path)
    assert loaded == [artifact]


def test_baked_models_are_loaded_without_network(tmp_path, monkeypatch):
    import sys

    options = []
    model = SimpleNamespace(to=lambda device: model, set_progress_bar_config=lambda **kw: None)

    def load(path, **kwargs):
        options.append((path, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            __version__="0.39.0",
            Flux2KleinPipeline=SimpleNamespace(from_pretrained=load),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="4.57.1",
            pipeline=lambda **kw: model,
            AutoImageProcessor=SimpleNamespace(from_pretrained=load),
            AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=load),
        ),
    )
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.8.0",
            version=SimpleNamespace(cuda="test"),
            bfloat16="bf16",
            float16="fp16",
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                get_device_name=lambda i: "NVIDIA L4",
                get_device_capability=lambda i: (8, 9),
            ),
        ),
    )
    (tmp_path / "identities.json").write_text(
        json.dumps(
            {
                "klein": [runtime.MODEL, runtime.MODEL_REVISION],
                "depth": [runtime.DEPTH_MODEL, runtime.DEPTH_REVISION],
            }
        )
    )
    obj = runtime.KleinSceneRuntime(tmp_path)
    assert len(options) == 3
    assert all(kw["local_files_only"] for _, kw in options)
    assert all(Path(path).is_relative_to(tmp_path) for path, _ in options)
    assert obj.identity["model_revision"] == runtime.MODEL_REVISION
    assert obj.identity["runtime_sha256"]
