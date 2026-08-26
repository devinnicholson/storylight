"""Finite, pinned TensorRT Edge-LLM checkpoint export on Modal.

This utility performs CPU-only ONNX export for an officially supported,
pre-quantized checkpoint.  It deliberately does not build TensorRT engines:
engines are hardware-specific and must be built on the target Jetson.

Run the bounded export, then download the resulting directory::

    modal run deploy/modal_tensorrt_edge_export.py::export_cli
    modal volume get bookforge-tensorrt-edge-llm \
      qwen2.5-0.5b-instruct-awq-v010/onnx artifacts/tensorrt-edge-llm/onnx
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "bookforge-tensorrt-edge-export"
EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
EXPORT_ID = "qwen2.5-0.5b-instruct-awq-v010"
EXPORT_ROOT = Path("/exports")
TIMEOUT_SECONDS = 1_800

export_volume = modal.Volume.from_name("bookforge-tensorrt-edge-llm", create_if_missing=True)

export_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 --branch v0.10.0 "
        "https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/tensorrt-edge-llm",
        f'test "$(git -C /opt/tensorrt-edge-llm rev-parse HEAD)" = "{EDGELLM_REVISION}"',
        "python -m pip install --no-cache-dir -e /opt/tensorrt-edge-llm",
        "python -m pip install --no-cache-dir torchvision==0.28.0",
        "python -c 'from transformers import Gemma4AudioConfig; print(Gemma4AudioConfig.__name__)'",
    )
    .env(
        {
            "HF_HOME": "/exports/huggingface",
            "HF_HUB_CACHE": "/exports/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return files


@app.function(
    image=export_image,
    cpu=8,
    memory=32_768,
    timeout=TIMEOUT_SECONDS,
    volumes={str(EXPORT_ROOT): export_volume},
)
def export_checkpoint() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    destination = EXPORT_ROOT / EXPORT_ID / "onnx"
    manifest_path = EXPORT_ROOT / EXPORT_ID / "export.manifest.json"

    if not manifest_path.is_file():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("partial export exists without a manifest; refusing to overwrite it")
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
        subprocess.run(
            [
                "tensorrt-edgellm-export",
                checkpoint,
                str(destination),
            ],
            check=True,
            cwd="/opt/tensorrt-edge-llm",
            timeout=TIMEOUT_SECONDS - 30,
        )
        files = _file_manifest(destination)
        manifest = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "export_id": EXPORT_ID,
            "tensorrt_edge_llm_version": EDGELLM_VERSION,
            "tensorrt_edge_llm_revision": EDGELLM_REVISION,
            "elapsed_seconds": time.perf_counter() - started,
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
            "files": files,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        export_volume.commit()

    return json.loads(manifest_path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def export_cli() -> None:
    print(json.dumps(export_checkpoint.remote(), indent=2, sort_keys=True))
