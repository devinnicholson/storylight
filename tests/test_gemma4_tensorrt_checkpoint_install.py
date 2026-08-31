from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("deploy/jetson/install-gemma4-tensorrt-checkpoint.py")
SPEC = importlib.util.spec_from_file_location("checkpoint_installer", SCRIPT)
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
        "llm/weights.safetensors": b"int4-weights",
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
        "result": "complete",
        "model_id": INSTALLER.MODEL_ID,
        "model_revision": INSTALLER.MODEL_REVISION,
        "export_id": INSTALLER.EXPORT_ID,
        "export_run_id": "test-export",
        "quantization": "int4_awq",
        "components": ["thinker"],
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": INSTALLER.EDGELLM_VERSION,
        "tensorrt_edge_llm_revision": INSTALLER.EDGELLM_REVISION,
        "mtp_included": False,
        "engine_built_in_cloud": False,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (root / "export.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_checkpoint_installer_verifies_and_atomically_installs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    destination = tmp_path / "models" / "onnx"

    result = INSTALLER.install_bundle(bundle, destination)

    assert result["result"] == "installed"
    assert result["file_count"] == 3
    assert (destination / "llm/model.onnx").read_bytes() == b"onnx-graph"
    assert not list(destination.parent.glob(".onnx.partial-*"))


def test_checkpoint_installer_rejects_unmanifested_or_modified_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "onnx/llm/model.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        INSTALLER.verify_bundle(bundle)


def test_checkpoint_installer_refuses_to_overwrite_existing_destination(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    destination = tmp_path / "models" / "onnx"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        INSTALLER.install_bundle(bundle, destination)


def test_checkpoint_installer_rejects_path_traversal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "export.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../config.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe file path"):
        INSTALLER.verify_bundle(bundle)
