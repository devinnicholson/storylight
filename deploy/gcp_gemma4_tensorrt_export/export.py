"""Finite Gemma 4 E2B INT4-AWQ export for a private Cloud Run GPU Job.

The job downloads one immutable public model revision, performs target-only
quantization and text-only TensorRT Edge-LLM export, and uploads the ONNX
checkpoint to a private GCS prefix.  The completion manifest is uploaded last.
TensorRT engines remain device-specific and are built on the Jetson.
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

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
EXPORT_ID = "gemma4-e2b-it-int4-awq-v010"
CALIBRATION_SAMPLES = 128
WORK_ROOT = Path("/tmp/bookforge-export")
JOB_TIMEOUT_SECONDS = 1_200
COMPLETION_OBJECT = "export.manifest.json"


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _remaining_seconds(started: float, reserve: int) -> int:
    remaining = JOB_TIMEOUT_SECONDS - (time.perf_counter() - started) - reserve
    if remaining < 1:
        raise TimeoutError("The bounded export no longer has enough time for another stage")
    return int(remaining)


def _run(command: list[str], *, started: float, reserve: int) -> float:
    stage_started = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        cwd="/opt/tensorrt-edge-llm",
        timeout=_remaining_seconds(started, reserve),
    )
    return time.perf_counter() - stage_started


def _upload_export(
    *,
    bucket_name: str,
    object_prefix: str,
    onnx_dir: Path,
    manifest: dict[str, Any],
) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    completion = bucket.blob(f"{object_prefix}/{COMPLETION_OBJECT}")
    if completion.exists(client):
        raise RuntimeError("The immutable export prefix already has a completion manifest")

    for item in manifest["files"]:
        relative = item["path"]
        blob = bucket.blob(f"{object_prefix}/onnx/{relative}")
        blob.metadata = {
            "sha256": item["sha256"],
            "export-id": EXPORT_ID,
        }
        blob.upload_from_filename(onnx_dir / relative, if_generation_match=0)

    completion.upload_from_string(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
    )


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download

    bucket_name = _required_environment("BOOKFORGE_EXPORT_BUCKET")
    run_id = _required_environment("BOOKFORGE_EXPORT_RUN_ID")
    if not run_id.replace("-", "").isalnum() or len(run_id) > 96:
        raise RuntimeError("BOOKFORGE_EXPORT_RUN_ID must be a bounded slug")
    object_prefix = f"tensorrt-edge-llm/{EXPORT_ID}/{run_id}"

    started = time.perf_counter()
    root = WORK_ROOT / run_id
    source_dir = root / "source"
    quantized_dir = root / "quantized"
    onnx_dir = root / "onnx"
    root.mkdir(parents=True, exist_ok=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Gemma 4 export")
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in gpu_name.upper():
        raise RuntimeError(f"Expected RTX PRO 6000, got {gpu_name}")

    stage_seconds: dict[str, float] = {}
    download_started = time.perf_counter()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=source_dir,
    )
    stage_seconds["download"] = time.perf_counter() - download_started

    temporary_quantized = root / "quantized.partial"
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
        started=started,
        reserve=180,
    )
    os.replace(temporary_quantized, quantized_dir)

    temporary_onnx = root / "onnx.partial"
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
        started=started,
        reserve=90,
    )
    os.replace(temporary_onnx, onnx_dir)

    files = _file_manifest(onnx_dir)
    if not files or not (onnx_dir / "llm" / "config.json").is_file():
        raise RuntimeError("Gemma 4 ONNX export is incomplete")
    if not (onnx_dir / "llm" / "model.onnx").is_file():
        raise RuntimeError("Gemma 4 ONNX graph is missing")
    if not list((onnx_dir / "llm").glob("*.safetensors")):
        raise RuntimeError("Externalized Gemma 4 INT4 weights are missing")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "result": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "export_id": EXPORT_ID,
        "export_run_id": run_id,
        "gcs_prefix": f"gs://{bucket_name}/{object_prefix}",
        "quantization": "int4_awq",
        "calibration_dataset": "wikitext",
        "calibration_samples": CALIBRATION_SAMPLES,
        "components": ["thinker"],
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": EDGELLM_VERSION,
        "tensorrt_edge_llm_revision": EDGELLM_REVISION,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION", "unknown"),
        "cloud_run_task_index": os.environ.get("CLOUD_RUN_TASK_INDEX", "unknown"),
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "stage_seconds": stage_seconds,
        "elapsed_before_upload_seconds": time.perf_counter() - started,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "mtp_included": False,
        "engine_built_in_cloud": False,
    }
    upload_started = time.perf_counter()
    _upload_export(
        bucket_name=bucket_name,
        object_prefix=object_prefix,
        onnx_dir=onnx_dir,
        manifest=manifest,
    )
    manifest["upload_seconds_observed_client_side"] = time.perf_counter() - upload_started
    manifest["elapsed_seconds_observed_client_side"] = time.perf_counter() - started
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
