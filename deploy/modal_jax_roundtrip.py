"""One finite HF -> MaxText -> five-step LoRA -> HF roundtrip on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from deploy.modal_jax_image import JAX_IMAGE, offline_environment

REPOSITORY_ROOT = Path(__file__).parents[1]
PLAN_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-roundtrip-plan-2026-09.json"
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
APP_NAME = "bookforge-jax-roundtrip-smoke"
GPU = "L4"
TIMEOUT_SECONDS = 2_700
WORKSPACE_HARD_STOP_USD = 28.0
BUDGET_MONTH = "2026-09"
_INPUT_ROOT = Path("/inputs")
_SCRATCH_ROOT = Path("/scratch/roundtrip")
_RELEASE_ROOT = Path("/releases/roundtrip")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")

input_volume = modal.Volume.from_name("bookforge-jax-fidelity-inputs", create_if_missing=False)
scratch_volume = modal.Volume.from_name("bookforge-jax-fidelity-scratch", create_if_missing=False)
release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _approval_token(
    run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    hf_snapshot_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
) -> str:
    return (
        f"APPROVE_MODAL_JAX_ROUNDTRIP:{run_id}:{config_sha256}:"
        f"{dataset_manifest_sha256}:{prepared_train_sha256}:{input_manifest_sha256}:"
        f"{hf_snapshot_manifest_sha256}:{tokenizer_manifest_sha256}"
    )


def _base_cache_approval_token(
    source_run_id: str,
    receipt_sha256: str,
    completion_sha256: str,
) -> str:
    return (
        f"APPROVE_MODAL_JAX_BASE_CACHE:{source_run_id}:"
        f"{receipt_sha256}:{completion_sha256}"
    )


def _validate_base_cache_request(
    request: dict[str, object], *, target_run_id: str
) -> tuple[str, str, str] | None:
    names = (
        "base_cache_run_id",
        "base_cache_receipt_sha256",
        "base_cache_completion_sha256",
        "base_cache_approval_token",
    )
    values = [request.get(name) for name in names]
    if all(value in (None, "") for value in values):
        return None
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("base checkpoint cache request must be complete")
    source_run_id, receipt_sha, completion_sha, token = values
    assert isinstance(source_run_id, str)
    assert isinstance(receipt_sha, str)
    assert isinstance(completion_sha, str)
    assert isinstance(token, str)
    if _RUN_ID.fullmatch(source_run_id) is None or source_run_id == target_run_id:
        raise ValueError("base checkpoint cache source run ID is invalid")
    if _SHA256.fullmatch(receipt_sha) is None or _SHA256.fullmatch(completion_sha) is None:
        raise ValueError("base checkpoint cache SHA-256 is invalid")
    expected = _base_cache_approval_token(source_run_id, receipt_sha, completion_sha)
    if token != expected:
        raise ValueError("base checkpoint cache approval token is not exact")
    return source_run_id, receipt_sha, completion_sha


def _validate_request(request: dict[str, object]) -> tuple[str, ...]:
    names = (
        "config_sha256",
        "dataset_manifest_sha256",
        "prepared_train_sha256",
        "input_manifest_sha256",
        "hf_snapshot_manifest_sha256",
        "tokenizer_manifest_sha256",
    )
    run_id = request.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid roundtrip run ID")
    values: list[str] = []
    for name in names:
        value = request.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
        values.append(value)
    expected = _approval_token(run_id, *values)
    if request.get("approval_token") != expected:
        raise ValueError("roundtrip approval token is not exact")
    return (run_id, *values)


def _write_once(path: Path, document: dict[str, Any], mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _remaining_seconds(started: float, *, reserve: int) -> int:
    remaining = TIMEOUT_SECONDS - (time.monotonic() - started) - reserve
    if remaining < 1:
        raise TimeoutError("roundtrip worker has no bounded time remaining")
    return int(remaining)


def _run_stage(command: list[str], *, environment: dict[str, str], started: float) -> None:
    subprocess.run(
        command,
        cwd="/opt/bookforge",
        env=environment,
        check=True,
        timeout=_remaining_seconds(started, reserve=180),
    )


def _artifact_contract(path: Path, declarations: list[str]) -> str:
    from training.jax_fidelity.artifact_contract import create_artifact_contract

    document = create_artifact_contract(declarations)
    _write_once(path, document)
    return _sha256(path)


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"release source is not a safe directory: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"release source contains a symbolic link: {path}")
    shutil.copytree(source, destination, symlinks=False)


def _completion_path(runs: Path, run_id: str) -> Path:
    completion = runs / run_id / "completion.json"
    document = _json_object(completion)
    if document.get("run_id") != run_id or document.get("status") != "succeeded":
        raise RuntimeError(f"stage {run_id} has no successful terminal completion")
    if not document.get("artifacts") or not document.get("evidence"):
        raise RuntimeError(f"stage {run_id} has empty terminal evidence")
    return completion


def _release_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError("roundtrip release may not contain symbolic links")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError("roundtrip release contains no artifacts")
    return files


@app.function(
    image=JAX_IMAGE,
    cpu=1,
    memory=2_048,
    timeout=60,
    retries=0,
)
def hydration_preflight() -> dict[str, object]:
    """Verify worker hydration and the immutable runtime without allocating a GPU."""

    from training.jax_fidelity.verify_runtime import validate_runtime, validate_runtime_lock

    lock_path = Path("/opt/bookforge/runtime.lock.json")
    validate_runtime()
    validate_runtime_lock(lock_path)
    return {
        "schema_version": "1.0",
        "ready": True,
        "backend": "modal-cpu-preflight",
        "runtime_lock_sha256": _sha256(lock_path),
    }


@app.function(
    image=JAX_IMAGE,
    gpu=GPU,
    cpu=2,
    memory=8_192,
    timeout=300,
    retries=0,
    max_containers=1,
)
def gpu_configuration_preflight(approval_token_value: str) -> dict[str, object]:
    """Prove the single-GPU MaxText configuration before checkpoint conversion."""

    from training.jax_fidelity.configuration import load_config

    config_path = Path("/opt/bookforge/experiments/jax-fidelity-lab/config.json")
    config_sha = _sha256(config_path)
    expected = f"APPROVE_MODAL_JAX_GPU_PREFLIGHT:{config_sha}"
    if approval_token_value != expected:
        raise ValueError("exact Modal JAX GPU preflight approval token is required")
    experiment = load_config(config_path)
    script = "\n".join(
        [
            "import json, sys",
            "from maxtext.configs import pyconfig",
            "config = pyconfig.initialize(sys.argv)",
            "from transformer_engine.jax.sharding import global_shard_guard, MeshResource",
            "import jax",
            "import jax.numpy as jnp",
            "devices = jax.devices()",
            "if len(devices) != 1 or devices[0].platform != 'gpu':",
            "    raise RuntimeError(f'expected one GPU, found {devices!r}')",
            "value = jax.device_get(jnp.arange(1024, dtype=jnp.bfloat16).sum())",
            "print(json.dumps({'hardware': config.hardware, 'devices': len(devices), "
            "'platform': devices[0].platform, 'probe_sum': float(value)}), flush=True)",
        ]
    )
    command = [
        "python3",
        "-c",
        script,
        "/opt/MaxText/src/maxtext/configs/base.yml",
        f"model_name={experiment.production['maxtext_model_name']}",
        "hardware=gpu",
        "skip_jax_distributed_system=true",
        f"use_multimodal={str(experiment.production['use_multimodal']).lower()}",
        f"scan_layers={str(experiment.production['scan_layers']).lower()}",
        "enable_checkpointing=false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd="/opt/MaxText",
            env=offline_environment(os.environ),
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "no subprocess output").strip()
        raise RuntimeError(f"GPU configuration preflight failed:\n{detail}") from error
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    print(json.dumps(payload, sort_keys=True), flush=True)
    return {"schema_version": "1.0", "ready": True, **payload}


@app.function(
    image=JAX_IMAGE,
    gpu=GPU,
    cpu=8,
    memory=65_536,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={
        str(_INPUT_ROOT): input_volume,
        "/scratch": scratch_volume,
        "/releases": release_volume,
    },
)
def run_roundtrip_finite(request: dict[str, object]) -> dict[str, object]:
    """Execute the compatibility seam once and publish completion last."""

    (
        run_id,
        config_sha,
        dataset_sha,
        prepared_sha,
        input_manifest_sha,
        hf_manifest_sha,
        tokenizer_manifest_sha,
    ) = _validate_request(request)
    base_cache = _validate_base_cache_request(request, target_run_id=run_id)
    started = time.monotonic()
    input_directory = _INPUT_ROOT / run_id
    scratch = _SCRATCH_ROOT / run_id
    release = _RELEASE_ROOT / run_id
    if scratch.exists() or release.exists():
        raise RuntimeError("roundtrip run ID already has terminal or partial state")

    from infra.gcp.jax.vertex_entrypoint import verify_input_population
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import (
        artifact_manifest,
        validate_dataset_manifest,
        verify_artifact_manifest,
    )
    from training.jax_fidelity.manifests import stable_run_id
    from training.jax_fidelity.orbax_receipt import (
        discover_orbax_items,
        orbax_leaf_receipt,
        verify_orbax_leaf_receipt,
        write_orbax_leaf_receipt,
    )
    from training.jax_fidelity.roundtrip_evidence import build_roundtrip_evidence
    from training.jax_fidelity.roundtrip_smoke import validate_roundtrip_evidence
    from training.jax_fidelity.runtime import approval_token

    verify_input_population(
        input_directory,
        run_id=run_id,
        expected_manifest_sha256=input_manifest_sha,
    )
    config_path = input_directory / "config.json"
    dataset_manifest = input_directory / "dataset/manifest.json"
    prepared_train = input_directory / "prepared/train.jsonl"
    hf_snapshot = input_directory / "checkpoint"
    tokenizer = input_directory / "tokenizer"
    hf_manifest_path = input_directory / "checkpoint.manifest.json"
    tokenizer_manifest_path = input_directory / "tokenizer.manifest.json"
    if _sha256(config_path) != config_sha or _sha256(dataset_manifest) != dataset_sha:
        raise RuntimeError("roundtrip config or dataset hash changed")
    if _sha256(prepared_train) != prepared_sha:
        raise RuntimeError("roundtrip prepared training hash changed")
    if _sha256(hf_manifest_path) != hf_manifest_sha:
        raise RuntimeError("roundtrip HF snapshot manifest hash changed")
    if _sha256(tokenizer_manifest_path) != tokenizer_manifest_sha:
        raise RuntimeError("roundtrip tokenizer manifest hash changed")
    verify_artifact_manifest(hf_snapshot, _json_object(hf_manifest_path))
    verify_artifact_manifest(tokenizer, _json_object(tokenizer_manifest_path))
    experiment = load_config(config_path)
    validated_dataset = validate_dataset_manifest(
        dataset_manifest,
        expected_manifest_sha256=dataset_sha,
        required_split_records=experiment.dataset["required_split_records"],
    )
    if experiment.training["smoke_steps"] != 5:
        raise RuntimeError("roundtrip contract requires exactly five smoke steps")

    scratch.mkdir(parents=True, exist_ok=False)
    runs = scratch / "runs"
    evidence = scratch / "evidence"
    evidence.mkdir()
    environment = offline_environment(os.environ.copy())
    environment["JAX_PLATFORMS"] = "cuda"

    hf_to_maxtext_input = evidence / "hf-to-maxtext.inputs.json"
    hf_to_maxtext_input_sha = _artifact_contract(
        hf_to_maxtext_input,
        [f"hf_checkpoint={hf_snapshot}"],
    )
    hf_to_maxtext_id = stable_run_id(
        stage="hf-to-maxtext",
        config_sha256=config_sha,
        dataset_manifest_sha256=hf_to_maxtext_input_sha,
    )
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage="hf-to-maxtext",
        run_id=hf_to_maxtext_id,
        config_sha256=config_sha,
        input_sha256=hf_to_maxtext_input_sha,
    )
    if base_cache is None:
        base_output = scratch / "base-maxtext"
        _run_stage(
            [
                "python3",
                "-m",
                "training.jax_fidelity.convert",
                "hf-to-maxtext",
                "--config",
                str(config_path),
                "--input-manifest",
                str(hf_to_maxtext_input),
                "--input-manifest-sha256",
                hf_to_maxtext_input_sha,
                "--hf-checkpoint",
                str(hf_snapshot),
                "--output-directory",
                str(base_output),
                "--run-directory",
                str(runs),
                "--maxtext-root",
                "/opt/MaxText",
                "--execute",
            ],
            environment=environment,
            started=started,
        )
        hf_to_maxtext_completion = _completion_path(runs, hf_to_maxtext_id)
        base_leaf = discover_orbax_items(base_output, expected_step=0)
        base_receipt = orbax_leaf_receipt(
            base_output,
            base_leaf,
            expected_step=0,
            role="base-maxtext",
        )
        base_receipt_path = evidence / "base-orbax.receipt.json"
        write_orbax_leaf_receipt(base_receipt_path, base_receipt)
    else:
        cache_run_id, expected_receipt_sha, expected_completion_sha = base_cache
        cache_root = _SCRATCH_ROOT / cache_run_id
        base_output = cache_root / "base-maxtext"
        base_receipt_path = cache_root / "evidence/base-orbax.receipt.json"
        hf_to_maxtext_completion = (
            cache_root / "runs" / hf_to_maxtext_id / "completion.json"
        )
        if _sha256(base_receipt_path) != expected_receipt_sha:
            raise RuntimeError("cached base Orbax receipt hash changed")
        if _sha256(hf_to_maxtext_completion) != expected_completion_sha:
            raise RuntimeError("cached HF-to-MaxText completion hash changed")
        completion = _json_object(hf_to_maxtext_completion)
        if (
            completion.get("run_id") != hf_to_maxtext_id
            or completion.get("status") != "succeeded"
            or completion.get("config_sha256") != config_sha
            or completion.get("evidence", {}).get("input_manifest_sha256")
            != hf_to_maxtext_input_sha
        ):
            raise RuntimeError("cached HF-to-MaxText completion identity changed")
        base_receipt = _json_object(base_receipt_path)
        base_leaf = verify_orbax_leaf_receipt(
            base_output,
            base_receipt,
            expected_step=0,
            role="base-maxtext",
        )

    smoke_id = stable_run_id(
        stage="lora-smoke",
        config_sha256=config_sha,
        dataset_manifest_sha256=validated_dataset.manifest_sha256,
    )
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage="lora-smoke",
        run_id=smoke_id,
        config_sha256=config_sha,
        input_sha256=prepared_sha,
    )
    smoke_output = scratch / "smoke-output"
    _run_stage(
        [
            "python3",
            "-m",
            "training.jax_fidelity.train",
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(dataset_manifest),
            "--dataset-manifest-sha256",
            dataset_sha,
            "--prepared-train-jsonl",
            str(prepared_train),
            "--prepared-train-sha256",
            prepared_sha,
            "--base-checkpoint",
            str(base_leaf),
            "--hf-tokenizer-checkpoint",
            str(tokenizer),
            "--output-directory",
            str(smoke_output),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            "/opt/MaxText",
            "--smoke",
            "--execute",
        ],
        environment=environment,
        started=started,
    )
    smoke_completion = _completion_path(runs, smoke_id)
    smoke_leaf = discover_orbax_items(
        smoke_output,
        expected_step=experiment.training["smoke_steps"],
    )
    smoke_receipt = orbax_leaf_receipt(
        smoke_output,
        smoke_leaf,
        expected_step=experiment.training["smoke_steps"],
        role="smoke-lora",
    )
    smoke_receipt_path = evidence / "smoke-orbax.receipt.json"
    write_orbax_leaf_receipt(smoke_receipt_path, smoke_receipt)

    maxtext_to_hf_input = evidence / "maxtext-to-hf.inputs.json"
    maxtext_to_hf_input_sha = _artifact_contract(
        maxtext_to_hf_input,
        [
            f"base_checkpoint={base_leaf}",
            f"adapter_checkpoint={smoke_leaf}",
            f"hf_checkpoint={hf_snapshot}",
        ],
    )
    maxtext_to_hf_id = stable_run_id(
        stage="maxtext-to-hf",
        config_sha256=config_sha,
        dataset_manifest_sha256=maxtext_to_hf_input_sha,
    )
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage="maxtext-to-hf",
        run_id=maxtext_to_hf_id,
        config_sha256=config_sha,
        input_sha256=maxtext_to_hf_input_sha,
    )
    merged_hf = scratch / "merged-hf"
    _run_stage(
        [
            "python3",
            "-m",
            "training.jax_fidelity.convert",
            "maxtext-to-hf",
            "--config",
            str(config_path),
            "--input-manifest",
            str(maxtext_to_hf_input),
            "--input-manifest-sha256",
            maxtext_to_hf_input_sha,
            "--base-checkpoint",
            str(base_leaf),
            "--adapter-checkpoint",
            str(smoke_leaf),
            "--hf-checkpoint",
            str(hf_snapshot),
            "--output-directory",
            str(merged_hf),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            "/opt/MaxText",
            "--execute",
        ],
        environment=environment,
        started=started,
    )
    maxtext_to_hf_completion = _completion_path(runs, maxtext_to_hf_id)

    logit_input = evidence / "logit-check.inputs.json"
    logit_input_sha = _artifact_contract(
        logit_input,
        [
            f"maxtext_checkpoint={base_leaf}",
            f"adapter_checkpoint={smoke_leaf}",
            f"hf_checkpoint={merged_hf}",
        ],
    )
    logit_id = stable_run_id(
        stage="logit-check",
        config_sha256=config_sha,
        dataset_manifest_sha256=logit_input_sha,
    )
    environment["BOOKFORGE_JAX_EXECUTION_APPROVAL"] = approval_token(
        stage="logit-check",
        run_id=logit_id,
        config_sha256=config_sha,
        input_sha256=logit_input_sha,
    )
    _run_stage(
        [
            "python3",
            "-m",
            "training.jax_fidelity.convert",
            "logit-check",
            "--config",
            str(config_path),
            "--input-manifest",
            str(logit_input),
            "--input-manifest-sha256",
            logit_input_sha,
            "--maxtext-checkpoint",
            str(base_leaf),
            "--adapter-checkpoint",
            str(smoke_leaf),
            "--hf-checkpoint",
            str(merged_hf),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            "/opt/MaxText",
            "--execute",
        ],
        environment=environment,
        started=started,
    )
    logit_completion = _completion_path(runs, logit_id)
    roundtrip_document = build_roundtrip_evidence(
        config_path=config_path,
        base_checkpoint=hf_snapshot,
        exported_checkpoint=merged_hf,
        hf_to_maxtext_completion=hf_to_maxtext_completion,
        hf_to_maxtext_completion_sha256=_sha256(hf_to_maxtext_completion),
        smoke_completion=smoke_completion,
        smoke_completion_sha256=_sha256(smoke_completion),
        maxtext_to_hf_completion=maxtext_to_hf_completion,
        maxtext_to_hf_completion_sha256=_sha256(maxtext_to_hf_completion),
        logit_completion=logit_completion,
        logit_completion_sha256=_sha256(logit_completion),
    )
    validate_roundtrip_evidence(experiment, roundtrip_document, exported_checkpoint=merged_hf)
    roundtrip_path = evidence / "roundtrip.json"
    _write_once(roundtrip_path, roundtrip_document)

    release.mkdir(parents=True, exist_ok=False)
    base_release_root = release / "base-orbax"
    base_release_leaf = base_release_root / base_receipt["relative_path"]
    base_release_leaf.parent.mkdir(parents=True)
    _copy_tree(base_leaf, base_release_leaf)
    verify_orbax_leaf_receipt(
        base_release_root,
        base_receipt,
        expected_step=0,
        role="base-maxtext",
    )
    smoke_release_root = release / "smoke-adapter"
    smoke_release_leaf = smoke_release_root / smoke_receipt["relative_path"]
    smoke_release_leaf.parent.mkdir(parents=True)
    _copy_tree(smoke_leaf, smoke_release_leaf)
    verify_orbax_leaf_receipt(
        smoke_release_root,
        smoke_receipt,
        expected_step=experiment.training["smoke_steps"],
        role="smoke-lora",
    )
    _copy_tree(merged_hf, release / "merged-hf")
    (release / "inputs").mkdir()
    shutil.copyfile(config_path, release / "inputs/config.json", follow_symlinks=False)
    shutil.copyfile(
        dataset_manifest,
        release / "inputs/dataset.manifest.json",
        follow_symlinks=False,
    )
    (release / "evidence").mkdir()
    evidence_files = {
        "base-orbax.receipt.json": base_receipt_path,
        "smoke-orbax.receipt.json": smoke_receipt_path,
        "roundtrip.json": roundtrip_path,
        "hf-to-maxtext.completion.json": hf_to_maxtext_completion,
        "smoke.completion.json": smoke_completion,
        "maxtext-to-hf.completion.json": maxtext_to_hf_completion,
        "logit-check.completion.json": logit_completion,
    }
    for name, source in evidence_files.items():
        shutil.copyfile(source, release / "evidence" / name, follow_symlinks=False)
    shutil.copyfile(
        "/opt/bookforge/runtime.lock.json",
        release / "runtime.lock.json",
        follow_symlinks=False,
    )
    base_manifest = artifact_manifest(base_release_leaf)
    smoke_manifest = artifact_manifest(smoke_release_leaf)
    merged_manifest = artifact_manifest(release / "merged-hf")
    _write_once(release / "base-orbax.manifest.json", base_manifest)
    _write_once(release / "smoke-adapter.manifest.json", smoke_manifest)
    _write_once(release / "merged-hf.manifest.json", merged_manifest)
    files = _release_files(release)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l4",
        "run_id": run_id,
        "base_cache_run_id": base_cache[0] if base_cache is not None else None,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "input_manifest_sha256": input_manifest_sha,
        "hf_snapshot_manifest_sha256": hf_manifest_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
        "hf_to_maxtext_run_id": hf_to_maxtext_id,
        "smoke_training_run_id": smoke_id,
        "maxtext_to_hf_run_id": maxtext_to_hf_id,
        "logit_run_id": logit_id,
        "base_orbax_receipt_sha256": _sha256(release / "evidence/base-orbax.receipt.json"),
        "smoke_orbax_receipt_sha256": _sha256(release / "evidence/smoke-orbax.receipt.json"),
        "roundtrip_evidence_sha256": _sha256(release / "evidence/roundtrip.json"),
        "base_orbax_manifest_sha256": _sha256(release / "base-orbax.manifest.json"),
        "smoke_adapter_manifest_sha256": _sha256(release / "smoke-adapter.manifest.json"),
        "merged_hf_manifest_sha256": _sha256(release / "merged-hf.manifest.json"),
        "elapsed_seconds": time.monotonic() - started,
        "files": files,
    }
    completion_path = release / "completion.json"
    _write_once(completion_path, payload)
    scratch_volume.commit()
    release_volume.commit()
    return {**payload, "completion_sha256": _sha256(completion_path)}


def _authoritative_workspace_total() -> float:
    completed = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Modal billing report was not a list")
    return sum(float(row["Cost"]) for row in rows)


@app.local_entrypoint()
def run_cli(
    run_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    prepared_train_sha256: str,
    input_manifest_sha256: str,
    hf_snapshot_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    approval_token_value: str,
    base_cache_run_id: str = "",
    base_cache_receipt_sha256: str = "",
    base_cache_completion_sha256: str = "",
    base_cache_approval_token_value: str = "",
) -> None:
    plan = _json_object(PLAN_PATH)
    if (
        plan.get("status") != "plan-only"
        or plan.get("automatic_retries") != 0
        or plan.get("function_calls") != 1
    ):
        raise RuntimeError("Modal roundtrip plan is not finite")
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("Modal roundtrip plan is outside its approved budget month")
    expected = _approval_token(
        run_id,
        config_sha256,
        dataset_manifest_sha256,
        prepared_train_sha256,
        input_manifest_sha256,
        hf_snapshot_manifest_sha256,
        tokenizer_manifest_sha256,
    )
    if approval_token_value != expected:
        raise RuntimeError("exact one-purpose Modal roundtrip approval token is required")
    cache_request = {
        "base_cache_run_id": base_cache_run_id,
        "base_cache_receipt_sha256": base_cache_receipt_sha256,
        "base_cache_completion_sha256": base_cache_completion_sha256,
        "base_cache_approval_token": base_cache_approval_token_value,
    }
    _validate_base_cache_request(cache_request, target_run_id=run_id)
    ceiling = plan.get("gross_ceiling_usd")
    if type(ceiling) not in (int, float) or float(ceiling) <= 0:
        raise RuntimeError("Modal roundtrip gross ceiling is invalid")
    workspace_total = _authoritative_workspace_total()
    if workspace_total + float(ceiling) > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace budget has insufficient roundtrip headroom")

    from infra.gcp.jax.modal_reconciliation import append_reconciliation, reserve_attempt

    attempt_id = f"jax-roundtrip:{run_id}:{input_manifest_sha256[:20]}"
    reserve_attempt(LEDGER_PATH, attempt_id=attempt_id, stage="jax-roundtrip-smoke")
    result: dict[str, object] | None = None
    status = "remote-error"
    postrun_total: float | None = None
    postrun_error: str | None = None
    try:
        result = run_roundtrip_finite.remote(
            {
                "run_id": run_id,
                "config_sha256": config_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "prepared_train_sha256": prepared_train_sha256,
                "input_manifest_sha256": input_manifest_sha256,
                "hf_snapshot_manifest_sha256": hf_snapshot_manifest_sha256,
                "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
                "approval_token": approval_token_value,
                **cache_request,
            }
        )
        status = "succeeded"
    finally:
        try:
            postrun_total = _authoritative_workspace_total()
        except Exception as error:
            postrun_error = f"{type(error).__name__}: {error}"
        append_reconciliation(
            LEDGER_PATH,
            attempt_id=attempt_id,
            stage="jax-roundtrip-smoke",
            workspace_before_usd=workspace_total,
            workspace_after_usd=postrun_total,
            declared_ceiling_usd=float(ceiling),
            status=status,
            result=result,
            postrun_report_error=postrun_error,
        )
    if result is None:
        raise RuntimeError("Modal roundtrip returned no terminal result")
    print(json.dumps(result, indent=2, sort_keys=True))
