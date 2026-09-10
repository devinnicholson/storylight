"""Finite, checksum-bound TensorRT export for one tuned Gemma 4 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from scripts.validate_fidelity_release import (
    BASE_EXPORT_ID,
    MODEL_ID,
    MODEL_REVISION,
    ValidatedRelease,
    sha256_file,
    validate_release,
)

EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
QUANTIZATION = "int4_awq"
CALIBRATION_DATASET = "wikitext"
CALIBRATION_SAMPLES = 128
CALIBRATION_DECISION = "preserve-pinned-wikitext"
CALIBRATION_REASON = (
    "TensorRT Edge-LLM v0.10.0 selects registered dataset names with --text_dataset but "
    "does not expose a supported local file or URI argument; the pinned package is not modified"
)
CALIBRATION_PROVENANCE = {
    "resolver": "TensorRT-Edge-LLM registered text dataset",
    "dataset": CALIBRATION_DATASET,
    "samples": CALIBRATION_SAMPLES,
    "tensorrt_edge_llm_revision": EDGELLM_REVISION,
}
CALIBRATION_PROVENANCE_SHA256 = hashlib.sha256(
    json.dumps(CALIBRATION_PROVENANCE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
JOB_TIMEOUT_SECONDS = 1_200
WORK_ROOT = Path("/tmp/storylight-fidelity-export")
COMPLETION_OBJECT = "export.manifest.json"
SOURCE_MANIFEST_OBJECT = "release.manifest.json"
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")

RunCommand = Callable[[Sequence[str], Path, int], None]


class Blob(Protocol):
    metadata: dict[str, str]

    def download_to_filename(self, path: Path) -> None: ...

    def upload_from_filename(self, path: Path, *, if_generation_match: int) -> None: ...

    def upload_from_string(
        self,
        value: str,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None: ...


class IamConfiguration(Protocol):
    public_access_prevention: str
    uniform_bucket_level_access_enabled: bool


class Bucket(Protocol):
    iam_configuration: IamConfiguration

    def blob(self, name: str) -> Blob: ...

    def reload(self) -> None: ...


class StorageClient(Protocol):
    def bucket(self, name: str) -> Bucket: ...

    def list_blobs(self, bucket: Bucket, *, prefix: str) -> Iterable[Blob]: ...


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _remaining_seconds(started: float, reserve: int) -> int:
    remaining = JOB_TIMEOUT_SECONDS - (time.perf_counter() - started) - reserve
    if remaining < 1:
        raise TimeoutError("candidate export has no bounded time remaining")
    return int(remaining)


def _run_command(command: Sequence[str], cwd: Path, timeout: int) -> None:
    subprocess.run(list(command), cwd=cwd, timeout=timeout, check=True)


def build_quantize_command(source_dir: Path, output_dir: Path) -> list[str]:
    return [
        "tensorrt-edgellm-quantize",
        "llm",
        "--model_dir",
        str(source_dir),
        "--output_dir",
        str(output_dir),
        "--quantization",
        QUANTIZATION,
        "--text_dataset",
        CALIBRATION_DATASET,
        "--num_samples",
        str(CALIBRATION_SAMPLES),
    ]


def build_export_command(quantized_dir: Path, output_dir: Path) -> list[str]:
    return [
        "tensorrt-edgellm-export",
        str(quantized_dir),
        str(output_dir),
        "--components",
        "thinker",
        "--skip-visual",
        "--skip-audio",
        "--externalize-weights",
        "int4_ffn",
    ]


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError("export output may not contain symbolic links")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _write_completion(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_local_candidate(
    release: ValidatedRelease,
    destination_root: Path | str,
    *,
    private_output_prefix: str,
    run_command: RunCommand = _run_command,
    tool_root: Path | str = Path("/opt/tensorrt-edge-llm"),
    execution_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export once into a content-addressed directory; completion is written last."""

    if not private_output_prefix or "://" not in private_output_prefix:
        raise ValueError("private output prefix must be an explicit non-local URI")
    destination = (
        Path(destination_root).resolve() / release.candidate_id / release.manifest_sha256[:20]
    )
    if destination.exists():
        raise RuntimeError("candidate release already has export state and cannot be repeated")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    quantized_partial = destination / "quantized.partial"
    quantized = destination / "quantized"
    stage_started = time.perf_counter()
    run_command(
        build_quantize_command(release.release_root, quantized_partial),
        Path(tool_root),
        _remaining_seconds(started, 180),
    )
    os.replace(quantized_partial, quantized)
    stage_seconds["quantize"] = time.perf_counter() - stage_started

    onnx_partial = destination / "onnx.partial"
    onnx = destination / "onnx"
    stage_started = time.perf_counter()
    run_command(
        build_export_command(quantized, onnx_partial),
        Path(tool_root),
        _remaining_seconds(started, 60),
    )
    os.replace(onnx_partial, onnx)
    stage_seconds["export"] = time.perf_counter() - stage_started
    required = (onnx / "llm" / "config.json", onnx / "llm" / "model.onnx")
    if not all(path.is_file() for path in required):
        raise RuntimeError("candidate ONNX export lacks its graph or configuration")
    if not list((onnx / "llm").glob("*.safetensors")):
        raise RuntimeError("candidate export lacks externalized INT4 weights")
    files = _file_manifest(onnx)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "candidate_id": release.candidate_id,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source_release_manifest_sha256": release.manifest_sha256,
        "source_files_content_sha256": release.files_content_sha256,
        "config_sha256": release.config_sha256,
        "dataset_manifest_sha256": release.dataset_manifest_sha256,
        "training_run_id": release.training_run_id,
        "private_output_prefix": private_output_prefix.rstrip("/"),
        "quantization": QUANTIZATION,
        "calibration_dataset": CALIBRATION_DATASET,
        "calibration_samples": CALIBRATION_SAMPLES,
        "calibration_decision": CALIBRATION_DECISION,
        "calibration_reason": CALIBRATION_REASON,
        "calibration_provenance": CALIBRATION_PROVENANCE,
        "calibration_provenance_sha256": CALIBRATION_PROVENANCE_SHA256,
        "components": ["thinker"],
        "skip_visual": True,
        "skip_audio": True,
        "externalized_weights": ["int4_ffn"],
        "tensorrt_edge_llm_version": EDGELLM_VERSION,
        "tensorrt_edge_llm_revision": EDGELLM_REVISION,
        "base_export_reused": False,
        "base_export_id": BASE_EXPORT_ID,
        "engine_built_in_cloud": False,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "stage_seconds": stage_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "execution": dict(execution_metadata or {}),
    }
    _write_completion(destination / COMPLETION_OBJECT, payload)
    return payload


def _download_release(
    client: StorageClient,
    *,
    bucket_name: str,
    object_prefix: str,
    destination: Path,
    expected_manifest_sha256: str,
) -> tuple[Path, Path]:
    bucket = client.bucket(bucket_name)
    manifest_path = destination / SOURCE_MANIFEST_OBJECT
    source = destination / "source"
    source.mkdir(parents=True, exist_ok=False)
    manifest_blob = bucket.blob(f"{object_prefix.rstrip('/')}/{SOURCE_MANIFEST_OBJECT}")
    manifest_blob.download_to_filename(manifest_path)
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("private merged-HF release manifest hash changed")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise RuntimeError("private merged-HF release has no file manifest")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError("private merged-HF file declaration is invalid")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("private merged-HF file path escaped its prefix")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(f"{object_prefix.rstrip('/')}/{relative.as_posix()}").download_to_filename(
            target
        )
    return manifest_path, source


def _require_private_bucket(bucket: Bucket) -> None:
    bucket.reload()
    iam = bucket.iam_configuration
    if iam.public_access_prevention != "enforced":
        raise RuntimeError("candidate bucket must enforce public-access prevention")
    if iam.uniform_bucket_level_access_enabled is not True:
        raise RuntimeError("candidate bucket must use uniform bucket-level access")


def _create_export_intent(
    client: StorageClient,
    *,
    bucket_name: str,
    intent_object: str,
    payload: Mapping[str, Any],
) -> None:
    bucket = client.bucket(bucket_name)
    _require_private_bucket(bucket)
    bucket.blob(intent_object).upload_from_string(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
    )


def _upload_export(
    client: StorageClient,
    *,
    bucket_name: str,
    object_prefix: str,
    local_root: Path,
    payload: dict[str, Any],
) -> None:
    bucket = client.bucket(bucket_name)
    _require_private_bucket(bucket)
    existing = client.list_blobs(
        bucket,
        prefix=f"{object_prefix.rstrip('/')}/",
    )
    if list(existing):
        raise RuntimeError("private candidate output prefix already contains state")
    onnx = (
        local_root
        / payload["candidate_id"]
        / payload["source_release_manifest_sha256"][:20]
        / "onnx"
    )
    for item in payload["files"]:
        blob = bucket.blob(f"{object_prefix.rstrip('/')}/onnx/{item['path']}")
        blob.metadata = {
            "sha256": item["sha256"],
            "candidate-id": payload["candidate_id"],
        }
        blob.upload_from_filename(onnx / item["path"], if_generation_match=0)
    completion = bucket.blob(f"{object_prefix.rstrip('/')}/{COMPLETION_OBJECT}")
    completion.upload_from_string(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
    )


def main() -> None:
    import torch
    from google.cloud import storage

    release_bucket = _required_environment("STORYLIGHT_FIDELITY_RELEASE_BUCKET")
    release_prefix = _required_environment("STORYLIGHT_FIDELITY_RELEASE_PREFIX")
    release_manifest_sha = _required_environment("STORYLIGHT_FIDELITY_RELEASE_MANIFEST_SHA256")
    expected_config_sha = _required_environment("STORYLIGHT_FIDELITY_CONFIG_SHA256")
    expected_dataset_sha = _required_environment("STORYLIGHT_FIDELITY_DATASET_SHA256")
    expected_candidate = _required_environment("STORYLIGHT_FIDELITY_CANDIDATE_ID")
    export_bucket = _required_environment("STORYLIGHT_FIDELITY_EXPORT_BUCKET")
    run_id = _required_environment("STORYLIGHT_FIDELITY_EXPORT_RUN_ID")
    export_image = _required_environment("STORYLIGHT_FIDELITY_EXPORT_IMAGE")
    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", export_image) is None:
        raise RuntimeError("STORYLIGHT_FIDELITY_EXPORT_IMAGE must be digest-pinned")
    if not _RUN_ID.fullmatch(run_id):
        raise RuntimeError("STORYLIGHT_FIDELITY_EXPORT_RUN_ID must be a bounded slug")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fidelity candidate export")
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in gpu_name.upper():
        raise RuntimeError(f"Expected RTX PRO 6000, got {gpu_name}")

    client = storage.Client()
    _require_private_bucket(client.bucket(release_bucket))
    run_root = WORK_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest, source = _download_release(
        client,
        bucket_name=release_bucket,
        object_prefix=release_prefix,
        destination=run_root,
        expected_manifest_sha256=release_manifest_sha,
    )
    release = validate_release(
        manifest,
        source,
        expected_manifest_sha256=release_manifest_sha,
        expected_config_sha256=expected_config_sha,
        expected_dataset_manifest_sha256=expected_dataset_sha,
        expected_candidate=expected_candidate,
    )
    object_prefix = (
        f"tensorrt-edge-llm/fidelity/{release.candidate_id}/{release.manifest_sha256[:20]}/{run_id}"
    )
    private_uri = f"gs://{export_bucket}/{object_prefix}"
    intent_object = (
        f"tensorrt-edge-llm/fidelity-intents/{release.candidate_id}/"
        f"{release.manifest_sha256[:20]}/{run_id}.json"
    )
    _create_export_intent(
        client,
        bucket_name=export_bucket,
        intent_object=intent_object,
        payload={
            "schema_version": "1.0",
            "status": "export-intent-recorded",
            "retry_allowed": False,
            "candidate_id": release.candidate_id,
            "source_release_manifest_sha256": release.manifest_sha256,
            "config_sha256": release.config_sha256,
            "dataset_manifest_sha256": release.dataset_manifest_sha256,
            "output_prefix": private_uri,
        },
    )
    local_exports = run_root / "exports"
    payload = export_local_candidate(
        release,
        local_exports,
        private_output_prefix=private_uri,
        execution_metadata={
            "backend": "gcp-cloud-run-job",
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "intent_object": f"gs://{export_bucket}/{intent_object}",
            "container_image": export_image,
            "calibration_provenance_sha256": CALIBRATION_PROVENANCE_SHA256,
        },
    )
    _upload_export(
        client,
        bucket_name=export_bucket,
        object_prefix=object_prefix,
        local_root=local_exports,
        payload=payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    shutil.rmtree(run_root)


if __name__ == "__main__":
    main()
