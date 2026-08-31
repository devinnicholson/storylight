from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("deploy/jetson/install-qwen15-tensorrt-checkpoint.py")
SPEC = importlib.util.spec_from_file_location("qwen15_checkpoint_installer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    files = {
        "llm/config.json": b"{}\n",
        "llm/model.onnx": b"onnx-graph",
        "llm/model.onnx.data": b"onnx-weights",
        "llm/embedding.safetensors": b"embedding",
    }
    entries = []
    for relative, content in files.items():
        path = root / "onnx" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {"path": relative, "bytes": len(content), "sha256": _sha256(content)}
        )
    manifest = {
        "schema_version": 1,
        "model_id": INSTALLER.MODEL_ID,
        "model_revision": INSTALLER.MODEL_REVISION,
        "export_id": INSTALLER.EXPORT_ID,
        "tensorrt_edge_llm_version": INSTALLER.EDGELLM_VERSION,
        "tensorrt_edge_llm_revision": INSTALLER.EDGELLM_REVISION,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (root / "export.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_qwen15_checkpoint_verifies_and_installs_atomically(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    destination = tmp_path / "models" / "onnx"

    result = INSTALLER.install_bundle(bundle, destination)

    assert result["result"] == "installed"
    assert result["file_count"] == 4
    assert (destination / "llm/model.onnx.data").read_bytes() == b"onnx-weights"
    assert not list(destination.parent.glob(".onnx.partial-*"))


def test_qwen15_checkpoint_rejects_tampering_and_overwrite(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "onnx/llm/model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        INSTALLER.verify_bundle(bundle)

    clean_bundle = _bundle(tmp_path / "clean")
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        INSTALLER.install_bundle(clean_bundle, destination)
