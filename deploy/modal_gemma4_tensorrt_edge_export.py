"""Quantize and export the pinned Gemma 4 E2B planner for Jetson.

This is a finite, target-only TensorRT Edge-LLM preparation job.  It writes a
unified INT4-AWQ checkpoint and a text-only ONNX checkpoint to the existing
Bookforge Modal volume.  TensorRT engines remain device-specific and are built
on the Jetson, never on Modal.

Run exactly one guarded attempt::

    modal run deploy/modal_gemma4_tensorrt_edge_export.py::export_cli

The local entrypoint checks authoritative workspace billing before it invokes
the GPU function.  The remote timeout is the second, independent cost bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "bookforge-gemma4-tensorrt-edge-export"
EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
EXPORT_ID = "gemma4-e2b-it-int4-awq-v010"
EXPORT_ROOT = Path("/exports")
REMOTE_TIMEOUT_SECONDS = 1_500
CALIBRATION_SAMPLES = 128
WORKSPACE_HARD_STOP_USD = 28.0
FULL_COMMAND_CEILING_USD = 1.30

export_volume = modal.Volume.from_name("bookforge-tensorrt-edge-llm", create_if_missing=True)

export_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 --branch v0.10.0 "
        "https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/tensorrt-edge-llm",
        f'test "$(git -C /opt/tensorrt-edge-llm rev-parse HEAD)" = "{EDGELLM_REVISION}"',
        "python -m pip install --no-cache-dir -e '/opt/tensorrt-edge-llm[tools]'",
        "tensorrt-edgellm-quantize --help >/dev/null",
        "tensorrt-edgellm-export --help >/dev/null",
    )
    .env(
        {
            "HF_HOME": "/exports/huggingface",
            "HF_HUB_CACHE": "/exports/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
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


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run(command: list[str], *, timeout: float) -> float:
    started = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        cwd="/opt/tensorrt-edge-llm",
        timeout=max(1, int(timeout)),
    )
    return time.perf_counter() - started


@app.function(
    image=export_image,
    gpu="L40S",
    cpu=8,
    memory=65_536,
    timeout=REMOTE_TIMEOUT_SECONDS,
    volumes={str(EXPORT_ROOT): export_volume},
)
def quantize_and_export() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    root = EXPORT_ROOT / EXPORT_ID
    source_dir = root / "source"
    quantized_dir = root / "quantized"
    onnx_dir = root / "onnx"
    manifest_path = root / "export.manifest.json"
    root.mkdir(parents=True, exist_ok=True)

    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    stage_seconds: dict[str, float] = {}
    if not (source_dir / "model.safetensors").is_file():
        if source_dir.exists():
            shutil.rmtree(source_dir)
        download_started = time.perf_counter()
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=source_dir,
        )
        stage_seconds["download"] = time.perf_counter() - download_started
        export_volume.commit()

    if not (quantized_dir / "config.json").is_file():
        temporary_quantized = root / "quantized.partial"
        if temporary_quantized.exists():
            shutil.rmtree(temporary_quantized)
        remaining = REMOTE_TIMEOUT_SECONDS - (time.perf_counter() - started) - 90
        stage_seconds["quantize"] = _run(
            [
                "tensorrt-edgellm-quantize",
                "llm",
                "--model_dir",
                str(source_dir),
                "--output_dir",
                str(temporary_quantized),
                "--quantization",
                "int4_awq",
                "--text_dataset",
                "wikitext",
                "--num_samples",
                str(CALIBRATION_SAMPLES),
            ],
            timeout=remaining,
        )
        if quantized_dir.exists():
            shutil.rmtree(quantized_dir)
        os.replace(temporary_quantized, quantized_dir)
        export_volume.commit()

    if not (onnx_dir / "llm" / "model.onnx").is_file():
        temporary_onnx = root / "onnx.partial"
        if temporary_onnx.exists():
            shutil.rmtree(temporary_onnx)
        remaining = REMOTE_TIMEOUT_SECONDS - (time.perf_counter() - started) - 30
        stage_seconds["export"] = _run(
            [
                "tensorrt-edgellm-export",
                str(quantized_dir),
                str(temporary_onnx),
                "--components",
                "thinker",
                "--skip-visual",
                "--skip-audio",
                "--externalize-weights",
                "int4_ffn",
            ],
            timeout=remaining,
        )
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir)
        os.replace(temporary_onnx, onnx_dir)

    files = _file_manifest(onnx_dir)
    if not files or not (onnx_dir / "llm" / "config.json").is_file():
        raise RuntimeError("Gemma 4 ONNX export is incomplete")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "export_id": EXPORT_ID,
        "quantization": "int4_awq",
        "calibration_dataset": "wikitext",
        "calibration_samples": CALIBRATION_SAMPLES,
        "components": ["thinker"],
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": EDGELLM_VERSION,
        "tensorrt_edge_llm_revision": EDGELLM_REVISION,
        "gpu": "L40S",
        "remote_timeout_seconds": REMOTE_TIMEOUT_SECONDS,
        "stage_seconds": stage_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "mtp_included": False,
        "engine_built_on_modal": False,
    }
    _write_manifest(manifest_path, manifest)
    export_volume.commit()
    return manifest


def _authoritative_workspace_total() -> float:
    completed = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Modal billing report was not a JSON list")
    return sum(float(row["Cost"]) for row in rows)


@app.local_entrypoint()
def export_cli() -> None:
    workspace_total = _authoritative_workspace_total()
    projected_total = workspace_total + FULL_COMMAND_CEILING_USD
    if projected_total > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError(
            "Gemma 4 export refused by budget gate: "
            f"${workspace_total:.8f} + ${FULL_COMMAND_CEILING_USD:.2f} "
            f"> ${WORKSPACE_HARD_STOP_USD:.2f}"
        )
    result = quantize_and_export.remote()
    print(
        json.dumps(
            {
                "billing_preflight_workspace_total_usd": workspace_total,
                "full_command_ceiling_usd": FULL_COMMAND_CEILING_USD,
                "projected_hard_ceiling_usd": projected_total,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
