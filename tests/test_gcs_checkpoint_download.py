import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "deploy/jetson/download-gcs-tensorrt-checkpoint.py"
SPEC = importlib.util.spec_from_file_location("bookforge_gcs_checkpoint_download", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_direct_gcs_download_is_atomic_and_manifest_verified(tmp_path, monkeypatch) -> None:
    objects = {
        "llm/model.onnx": b"onnx",
        "llm/weights.safetensors": b"weights",
    }
    manifest = {
        "result": "complete",
        "gcs_prefix": "gs://private-bucket/pinned/export",
        "export_run_id": "run-1",
        "file_count": len(objects),
        "total_bytes": sum(len(value) for value in objects.values()),
        "files": [
            {"path": path, "bytes": len(value), "sha256": _sha(value)}
            for path, value in objects.items()
        ],
    }

    def fake_fetch(token, bucket, object_name, destination):
        assert token == "short-lived-token"
        assert bucket == "private-bucket"
        if object_name.endswith("/export.manifest.json"):
            value = json.dumps(manifest).encode()
        else:
            relative = object_name.split("/onnx/", maxsplit=1)[1]
            value = objects[relative]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        return _sha(value), len(value)

    monkeypatch.setattr(MODULE, "_fetch", fake_fetch)
    destination = tmp_path / "installed"
    result = MODULE.download_bundle(
        "short-lived-token",
        "private-bucket",
        "pinned/export",
        destination,
        2,
    )

    assert result["result"] == "downloaded_and_verified"
    assert result["total_bytes"] == manifest["total_bytes"]
    assert json.loads((destination / "export.manifest.json").read_text()) == manifest
    assert (destination / "onnx/llm/model.onnx").read_bytes() == b"onnx"
    assert not list(tmp_path.glob(".installed.partial-*"))


@pytest.mark.parametrize("value", ["../secret", "/absolute", "nested/../secret", ""])
def test_direct_gcs_download_rejects_unsafe_manifest_paths(value) -> None:
    with pytest.raises(ValueError, match="invalid|unsafe"):
        MODULE._safe_relative_path(value)
