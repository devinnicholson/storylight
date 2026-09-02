#!/usr/bin/env python3
"""Finite Vertex worker: verify staged inputs, train once, publish completion last."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit


class Blob(Protocol):
    name: str
    generation: int
    metadata: dict[str, str]

    def download_to_filename(self, path: Path) -> None: ...

    def upload_from_filename(self, path: Path, *, if_generation_match: int) -> None: ...

    def upload_from_string(
        self,
        value: str | bytes,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None: ...


class Bucket(Protocol):
    def blob(self, name: str) -> Blob: ...


class StorageClient(Protocol):
    def bucket(self, name: str) -> Bucket: ...

    def list_blobs(self, bucket: Bucket, *, prefix: str) -> Iterable[Blob]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gcs_location(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("expected a non-root gs:// URI")
    return parsed.netloc, parsed.path.strip("/")


def _download_prefix(client: StorageClient, uri: str, destination: Path) -> None:
    bucket_name, prefix = _gcs_location(uri)
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=f"{prefix}/"))
    if not blobs:
        raise RuntimeError("staged input prefix is empty")
    for blob in blobs:
        relative = blob.name.removeprefix(f"{prefix}/")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("staged input object escaped its run prefix")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def verify_input_population(
    root: Path, *, run_id: str, expected_manifest_sha256: str
) -> dict[str, object]:
    manifest_path = root / "inputs.manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("staged input manifest hash changed")
    document = _json_object(manifest_path)
    if (
        document.get("producer") != "bookforge-gcp-jax-input-stager"
        or document.get("run_id") != run_id
        or document.get("status") != "complete"
    ):
        raise RuntimeError("staged input manifest identity changed")
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("staged input manifest has no files")
    expected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("staged input declaration is invalid")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("staged input declaration escaped its prefix")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"staged input is missing: {relative.as_posix()}")
        if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha256(path):
            raise RuntimeError(f"staged input checksum changed: {relative.as_posix()}")
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise RuntimeError("staged input prefix contains undeclared or missing files")
    return document


def verify_base_orbax(
    root: Path,
    *,
    input_manifest: dict[str, object],
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> Path:
    """Resolve the only approved step-0 base leaf beneath a staged population."""

    from training.jax_fidelity.integrity import verify_artifact_manifest
    from training.jax_fidelity.orbax_receipt import verify_orbax_leaf_receipt

    checkpoint = root / "checkpoint"
    manifest_path = root / "checkpoint.manifest.json"
    receipt_path = root / "checkpoint.receipt.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("base Orbax manifest checksum changed")
    if _sha256(receipt_path) != expected_receipt_sha256:
        raise RuntimeError("base Orbax receipt checksum changed")
    manifest = _json_object(manifest_path)
    receipt = _json_object(receipt_path)
    leaf = verify_orbax_leaf_receipt(
        checkpoint,
        receipt,
        expected_step=0,
        role="base-maxtext",
    )
    verify_artifact_manifest(leaf, manifest)
    if input_manifest.get("base_orbax") != {
        "role": "base-maxtext",
        "expected_step": 0,
        "relative_path": receipt.get("relative_path"),
        "receipt_sha256": expected_receipt_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "content_sha256": manifest.get("content_sha256"),
    }:
        raise RuntimeError("staged input manifest base Orbax binding changed")
    return leaf


def _upload_release(
    client: StorageClient, source: Path, release_uri: str
) -> list[dict[str, object]]:
    bucket_name, prefix = _gcs_location(release_uri)
    bucket = client.bucket(bucket_name)
    files: list[dict[str, object]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        if relative == "completion.json":
            continue
        if path.is_symlink():
            raise RuntimeError("release artifacts may not be symbolic links")
        digest = _sha256(path)
        blob = bucket.blob(f"{prefix}/{relative}")
        blob.upload_from_filename(path, if_generation_match=0)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    return files


def _write_completion(
    client: StorageClient, release_uri: str, payload: dict[str, object]
) -> dict[str, object]:
    bucket_name, prefix = _gcs_location(release_uri)
    blob = client.bucket(bucket_name).blob(f"{prefix}/completion.json")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    completion_sha256 = hashlib.sha256(encoded).hexdigest()
    blob.metadata = {"bookforge-completion-sha256": completion_sha256}
    blob.upload_from_string(
        encoded,
        content_type="application/json",
        if_generation_match=0,
    )
    return {
        "schema_version": "1.0",
        "status": "release-complete",
        "run_id": payload["run_id"],
        "completion_uri": f"{release_uri.rstrip('/')}/completion.json",
        "completion_sha256": completion_sha256,
        "completion_generation": blob.generation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--release-prefix", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--prepared-train-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-receipt-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    from google.cloud import storage

    args = _parser().parse_args()
    root = Path("/tmp/bookforge-jax")
    inputs = root / "inputs"
    outputs = root / "outputs"
    runs = root / "runs"
    inputs.mkdir(parents=True, exist_ok=False)
    outputs.mkdir(parents=True, exist_ok=False)
    runs.mkdir(parents=True, exist_ok=False)

    storage_client = storage.Client(project="your-gcp-project")
    _download_prefix(storage_client, args.input_prefix, inputs)
    input_manifest = verify_input_population(
        inputs,
        run_id=args.run_id,
        expected_manifest_sha256=args.input_manifest_sha256,
    )
    config = inputs / "config.json"
    manifest = inputs / "dataset" / "manifest.json"
    prepared = inputs / "prepared" / "train.jsonl"
    tokenizer_checkpoint = inputs / "tokenizer"
    tokenizer_manifest_path = inputs / "tokenizer.manifest.json"
    if _sha256(config) != args.config_sha256:
        raise RuntimeError("staged config hash changed")
    if _sha256(manifest) != args.dataset_manifest_sha256:
        raise RuntimeError("staged dataset manifest hash changed")
    prepared_sha256 = _sha256(prepared)
    if prepared_sha256 != args.prepared_train_sha256:
        raise RuntimeError("staged prepared training data hash changed")
    if _sha256(tokenizer_manifest_path) != args.tokenizer_manifest_sha256:
        raise RuntimeError("tokenizer manifest hash changed")

    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import validate_dataset_manifest, verify_artifact_manifest
    from training.jax_fidelity.manifests import stable_run_id
    from training.jax_fidelity.prepared_staging import verify_v2_prepared_training_input
    from training.jax_fidelity.recovery_staging import verify_recovery_training_input
    from training.jax_fidelity.remote_release import package_training_release
    from training.jax_fidelity.runtime import approval_token

    experiment = load_config(config)
    validated = validate_dataset_manifest(
        manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
        required_split_records=experiment.dataset["required_split_records"],
    )
    base_checkpoint = verify_base_orbax(
        inputs,
        input_manifest=input_manifest,
        expected_manifest_sha256=args.base_checkpoint_manifest_sha256,
        expected_receipt_sha256=args.base_checkpoint_receipt_sha256,
    )
    verify_artifact_manifest(tokenizer_checkpoint, _json_object(tokenizer_manifest_path))
    verify_v2_prepared_training_input(
        inputs,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=args.config_sha256,
        prepared_sha256=prepared_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
    )
    verify_recovery_training_input(
        inputs,
        experiment=experiment,
        input_manifest=input_manifest,
        config_sha256=args.config_sha256,
        prepared_sha256=prepared_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
    )
    stage = "lora-smoke" if args.smoke else "lora-train"
    training_run_id = stable_run_id(
        stage=stage,
        config_sha256=experiment.sha256,
        dataset_manifest_sha256=validated.manifest_sha256,
    )
    environment = os.environ.copy()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage=stage,
        run_id=training_run_id,
        config_sha256=experiment.sha256,
        input_sha256=prepared_sha256,
    )
    command = [
        "python3",
        "-m",
        "training.jax_fidelity.train",
        "--config",
        str(config),
        "--dataset-manifest",
        str(manifest),
        "--dataset-manifest-sha256",
        args.dataset_manifest_sha256,
        "--prepared-train-jsonl",
        str(prepared),
        "--prepared-train-sha256",
        prepared_sha256,
        "--base-checkpoint",
        str(base_checkpoint),
        "--hf-tokenizer-checkpoint",
        str(tokenizer_checkpoint),
        "--output-directory",
        str(outputs),
        "--run-directory",
        str(runs),
        "--maxtext-root",
        "/opt/MaxText",
        "--execute",
    ]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=environment, timeout=2_640)

    training_completion_path = runs / training_run_id / "completion.json"
    training_completion = _json_object(training_completion_path)
    if (
        training_completion.get("status") != "succeeded"
        or training_completion.get("run_id") != training_run_id
        or not training_completion.get("artifacts")
        or not training_completion.get("evidence")
    ):
        raise RuntimeError("training has no nonempty successful terminal evidence")
    package = root / "release-package"
    package_evidence = package_training_release(
        output_directory=outputs,
        run_directory=runs,
        training_run_id=training_run_id,
        runtime_lock="/opt/bookforge/runtime.lock.json",
        destination=package,
    )
    files = _upload_release(storage_client, package, args.release_prefix)
    completion = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "training_run_id": training_run_id,
        "status": "succeeded",
        "backend": "vertex-tpu-v6e",
        "config_sha256": args.config_sha256,
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "input_manifest_sha256": args.input_manifest_sha256,
        "base_checkpoint_manifest_sha256": args.base_checkpoint_manifest_sha256,
        "base_checkpoint_receipt_sha256": args.base_checkpoint_receipt_sha256,
        "tokenizer_manifest_sha256": args.tokenizer_manifest_sha256,
        "source_training_completion_sha256": _sha256(training_completion_path),
        "portable_package": package_evidence,
        "files": files,
    }
    terminal_evidence = _write_completion(storage_client, args.release_prefix, completion)
    print(json.dumps(terminal_evidence, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
