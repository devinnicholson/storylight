"""One finite MaxText LoRA merge into a provisional HF development candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from deploy.modal_jax_image import JAX_IMAGE, offline_environment

REPOSITORY_ROOT = Path(__file__).parents[1]
PLAN_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-merge-plan-2026-09.json"
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
APP_NAME = "bookforge-jax-full-adapter-merge"
GPU = "L40S"
TIMEOUT_SECONDS = 1_800
BUDGET_MONTH = "2026-09"
WORKSPACE_HARD_STOP_USD = 28.0
_INPUT_ROOT = Path("/inputs")
_SCRATCH_ROOT = Path("/scratch/merge")
_RELEASE_ROOT = Path("/releases")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")

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
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_once(path: Path, document: Mapping[str, Any], mode: int = 0o400) -> None:
    from training.jax_fidelity.integrity import canonical_json_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


_BINDING_NAMES = (
    "input_manifest_sha256",
    "hf_snapshot_manifest_sha256",
    "roundtrip_completion_sha256",
    "base_orbax_receipt_sha256",
    "base_orbax_manifest_sha256",
    "training_release_completion_sha256",
    "adapter_manifest_sha256",
    "training_run_sha256",
    "training_completion_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
)


def _approval_token(
    *,
    merge_run_id: str,
    roundtrip_run_id: str,
    training_release_run_id: str,
    training_run_id: str,
    bindings: Mapping[str, str],
) -> str:
    values = ":".join(bindings[name] for name in _BINDING_NAMES)
    return (
        f"APPROVE_MODAL_JAX_FULL_ADAPTER_MERGE:{merge_run_id}:{roundtrip_run_id}:"
        f"{training_release_run_id}:{training_run_id}:{values}"
    )


def _validate_request(
    request: Mapping[str, object],
) -> tuple[str, str, str, str, dict[str, str]]:
    identifiers: list[str] = []
    for name in (
        "merge_run_id",
        "roundtrip_run_id",
        "training_release_run_id",
        "training_run_id",
    ):
        value = request.get(name)
        if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
            raise ValueError(f"{name} must be an immutable lowercase slug")
        identifiers.append(value)
    bindings: dict[str, str] = {}
    for name in _BINDING_NAMES:
        value = request.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
        bindings[name] = value
    expected = _approval_token(
        merge_run_id=identifiers[0],
        roundtrip_run_id=identifiers[1],
        training_release_run_id=identifiers[2],
        training_run_id=identifiers[3],
        bindings=bindings,
    )
    if request.get("approval_token") != expected:
        raise ValueError("full-adapter merge approval token is not exact")
    return (*identifiers, bindings)


def _verify_declared_files(root: Path, rows: object, *, excluded: set[str]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("release completion has no declared files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("release file declaration is malformed")
        relative = Path(row["path"])
        name = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in declared:
            raise ValueError("release file path is unsafe or duplicated")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"release file failed checksum verification: {name}")
        declared.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    if actual != declared:
        raise ValueError("release has undeclared or missing files")


def _artifact_binding(manifest: Mapping[str, Any]) -> dict[str, object]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact manifest contains no files")
    return {
        "content_sha256": manifest.get("content_sha256"),
        "files": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
    }


def _verify_training_release(
    root: Path,
    *,
    release_run_id: str,
    release_completion_sha256: str,
    adapter_manifest_sha256: str,
    training_run_id: str,
    training_run_sha256: str,
    training_completion_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    base_binding: Mapping[str, object],
) -> tuple[Path, dict[str, Any]]:
    """Verify a portable full-training release and return its exact adapter leaf."""

    from training.jax_fidelity.integrity import verify_artifact_manifest
    from training.jax_fidelity.manifests import stable_run_id

    completion_path = root / "completion.json"
    if _sha256(completion_path) != release_completion_sha256:
        raise ValueError("training release completion checksum changed")
    outer = _json_object(completion_path)
    if (
        outer.get("schema_version") != "1.0"
        or outer.get("status") != "succeeded"
        or outer.get("run_id") != release_run_id
        or outer.get("training_run_id") != training_run_id
        or outer.get("config_sha256") != config_sha256
        or outer.get("dataset_manifest_sha256") != dataset_manifest_sha256
    ):
        raise ValueError("training release identity or terminal status changed")
    _verify_declared_files(root, outer.get("files"), excluded={"completion.json"})

    paths = {
        "adapter_manifest": root / "adapter.manifest.json",
        "package_manifest": root / "package.manifest.json",
        "training_run": root / "training/run.json",
        "training_completion": root / "training/completion.json",
    }
    expected_hashes = {
        "adapter_manifest": adapter_manifest_sha256,
        "training_run": training_run_sha256,
        "training_completion": training_completion_sha256,
    }
    for name, expected in expected_hashes.items():
        if _sha256(paths[name]) != expected:
            raise ValueError(f"training {name.replace('_', ' ')} checksum changed")
    package_evidence = outer.get("portable_package")
    if not isinstance(package_evidence, dict) or any(
        package_evidence.get(name) != expected
        for name, expected in (
            ("adapter_manifest_sha256", adapter_manifest_sha256),
            ("training_run_sha256", training_run_sha256),
            ("training_completion_sha256", training_completion_sha256),
        )
    ):
        raise ValueError("portable training package evidence changed")
    package_manifest_sha = package_evidence.get("package_manifest_sha256")
    if (
        not isinstance(package_manifest_sha, str)
        or _sha256(paths["package_manifest"]) != package_manifest_sha
    ):
        raise ValueError("portable package manifest checksum changed")
    package_manifest = _json_object(paths["package_manifest"])
    _verify_declared_files(
        root,
        package_manifest.get("files"),
        excluded={
            "completion.json",
            "package.manifest.json",
        },
    )

    adapter_manifest = _json_object(paths["adapter_manifest"])
    adapter_root = root / "adapter"
    verify_artifact_manifest(adapter_root, adapter_manifest)
    run = _json_object(paths["training_run"])
    training_completion = _json_object(paths["training_completion"])
    expected_training_run_id = stable_run_id(
        stage="lora-train",
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    metadata = run.get("metadata")
    inputs = metadata.get("inputs") if isinstance(metadata, dict) else None
    evidence = training_completion.get("evidence")
    if (
        training_run_id != expected_training_run_id
        or run.get("run_id") != training_run_id
        or run.get("stage") != "lora-train"
        or run.get("status") != "planned"
        or run.get("config_sha256") != config_sha256
        or run.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or not isinstance(inputs, dict)
        or inputs.get("base_checkpoint") != dict(base_binding)
        or training_completion.get("run_id") != training_run_id
        or training_completion.get("status") != "succeeded"
        or training_completion.get("run_manifest_sha256") != training_run_sha256
        or not isinstance(evidence, dict)
        or evidence.get("inputs") != inputs
        or not training_completion.get("artifacts")
    ):
        raise ValueError("full training run is not bound to the verified base/config/dataset")
    completion_rows = training_completion.get("artifacts")
    if not isinstance(completion_rows, list):
        raise ValueError("portable training completion has no adapter artifacts")
    declared_adapter_rows: dict[str, dict[str, object]] = {}
    for row in completion_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("portable training artifact declaration is malformed")
        relative = Path(row["path"])
        name = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in declared_adapter_rows:
            raise ValueError("portable training artifact path is unsafe or duplicated")
        declared_adapter_rows[name] = row
    expected_adapter_rows = {row["path"]: row for row in adapter_manifest["files"]}
    if declared_adapter_rows != expected_adapter_rows:
        raise ValueError("training completion does not bind every approved adapter byte")
    if outer.get("source_training_completion_sha256") != training_completion.get(
        "source_training_completion_sha256"
    ):
        raise ValueError("portable completion lost its source-training binding")

    return adapter_root, {
        "outer": outer,
        "adapter_manifest": adapter_manifest,
        "run": run,
        "completion": training_completion,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir() or destination.exists():
        raise ValueError("merge copy source or destination is unsafe")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("merge source contains a symbolic link")
    shutil.copytree(source, destination, symlinks=False)


def _release_files(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError("merged release contains a symbolic link")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _remaining_seconds(started: float, *, reserve: int = 120) -> int:
    remaining = TIMEOUT_SECONDS - (time.monotonic() - started) - reserve
    if remaining < 1:
        raise TimeoutError("full-adapter merge has no bounded time remaining")
    return int(remaining)


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
def merge_finite(request: dict[str, object]) -> dict[str, object]:
    """Run MaxText -> HF exactly once and publish completion only after verification."""

    merge_run_id, roundtrip_run_id, training_release_run_id, training_run_id, bindings = (
        _validate_request(request)
    )
    started = time.monotonic()
    source_input = _INPUT_ROOT / roundtrip_run_id
    roundtrip_root = _RELEASE_ROOT / "roundtrip" / roundtrip_run_id
    training_root = _RELEASE_ROOT / training_release_run_id
    scratch = _SCRATCH_ROOT / merge_run_id
    release = _RELEASE_ROOT / "merged" / merge_run_id
    if scratch.exists() or release.exists():
        raise RuntimeError("merge run ID already has partial or terminal state")

    from infra.gcp.jax.vertex_entrypoint import verify_input_population
    from training.jax_fidelity.artifact_contract import create_artifact_contract
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import (
        artifact_manifest,
        canonical_sha256,
        verify_artifact_manifest,
    )
    from training.jax_fidelity.manifests import stable_run_id
    from training.jax_fidelity.merged_candidate import (
        build_merged_candidate_manifest,
        validate_merged_candidate_manifest,
    )
    from training.jax_fidelity.orbax_receipt import (
        discover_orbax_items,
        orbax_leaf_receipt,
        verify_orbax_leaf_receipt,
        write_orbax_leaf_receipt,
    )
    from training.jax_fidelity.runtime import approval_token

    verify_input_population(
        source_input,
        run_id=roundtrip_run_id,
        expected_manifest_sha256=bindings["input_manifest_sha256"],
    )
    config_path = source_input / "config.json"
    dataset_path = source_input / "dataset/manifest.json"
    hf_snapshot = source_input / "checkpoint"
    hf_manifest_path = source_input / "checkpoint.manifest.json"
    if _sha256(config_path) != bindings["config_sha256"]:
        raise RuntimeError("original staged config checksum changed")
    if _sha256(dataset_path) != bindings["dataset_manifest_sha256"]:
        raise RuntimeError("original staged dataset manifest checksum changed")
    if _sha256(hf_manifest_path) != bindings["hf_snapshot_manifest_sha256"]:
        raise RuntimeError("original staged HF manifest checksum changed")
    verify_artifact_manifest(hf_snapshot, _json_object(hf_manifest_path))
    config = load_config(config_path)

    roundtrip_completion_path = roundtrip_root / "completion.json"
    if _sha256(roundtrip_completion_path) != bindings["roundtrip_completion_sha256"]:
        raise RuntimeError("roundtrip completion checksum changed")
    roundtrip = _json_object(roundtrip_completion_path)
    if (
        roundtrip.get("status") != "succeeded"
        or roundtrip.get("backend") != "modal-l40s"
        or roundtrip.get("run_id") != roundtrip_run_id
        or roundtrip.get("config_sha256") != bindings["config_sha256"]
        or roundtrip.get("dataset_manifest_sha256") != bindings["dataset_manifest_sha256"]
        or roundtrip.get("input_manifest_sha256") != bindings["input_manifest_sha256"]
        or roundtrip.get("hf_snapshot_manifest_sha256") != bindings["hf_snapshot_manifest_sha256"]
        or roundtrip.get("base_orbax_receipt_sha256") != bindings["base_orbax_receipt_sha256"]
        or roundtrip.get("base_orbax_manifest_sha256") != bindings["base_orbax_manifest_sha256"]
    ):
        raise RuntimeError("roundtrip terminal lineage changed")
    _verify_declared_files(roundtrip_root, roundtrip.get("files"), excluded={"completion.json"})
    if (
        _sha256(roundtrip_root / "inputs/config.json") != bindings["config_sha256"]
        or _sha256(roundtrip_root / "inputs/dataset.manifest.json")
        != bindings["dataset_manifest_sha256"]
    ):
        raise RuntimeError("roundtrip release config or dataset bytes changed")

    base_receipt_path = roundtrip_root / "evidence/base-orbax.receipt.json"
    base_manifest_path = roundtrip_root / "base-orbax.manifest.json"
    if _sha256(base_receipt_path) != bindings["base_orbax_receipt_sha256"]:
        raise RuntimeError("base Orbax receipt checksum changed")
    if _sha256(base_manifest_path) != bindings["base_orbax_manifest_sha256"]:
        raise RuntimeError("base Orbax manifest checksum changed")
    base_leaf = verify_orbax_leaf_receipt(
        roundtrip_root / "base-orbax",
        _json_object(base_receipt_path),
        expected_step=0,
        role="base-maxtext",
    )
    base_manifest = _json_object(base_manifest_path)
    verify_artifact_manifest(base_leaf, base_manifest)
    base_binding = _artifact_binding(base_manifest)

    adapter_root, training = _verify_training_release(
        training_root,
        release_run_id=training_release_run_id,
        release_completion_sha256=bindings["training_release_completion_sha256"],
        adapter_manifest_sha256=bindings["adapter_manifest_sha256"],
        training_run_id=training_run_id,
        training_run_sha256=bindings["training_run_sha256"],
        training_completion_sha256=bindings["training_completion_sha256"],
        config_sha256=bindings["config_sha256"],
        dataset_manifest_sha256=bindings["dataset_manifest_sha256"],
        base_binding=base_binding,
    )
    adapter_leaf = discover_orbax_items(
        adapter_root,
        expected_step=config.training["steps"],
    )
    adapter_receipt = orbax_leaf_receipt(
        adapter_root,
        adapter_leaf,
        expected_step=config.training["steps"],
        role="full-lora",
    )
    adapter_rows = {row["path"]: row for row in training["adapter_manifest"]["files"]}
    for row in adapter_receipt["artifact_manifest"]["files"]:
        portable = Path(adapter_receipt["relative_path"]).joinpath(row["path"]).as_posix()
        if adapter_rows.get(portable) != {**row, "path": portable}:
            raise RuntimeError("selected adapter leaf is outside the approved adapter manifest")
    verify_orbax_leaf_receipt(
        adapter_root,
        adapter_receipt,
        expected_step=config.training["steps"],
        role="full-lora",
    )

    scratch.mkdir(parents=True, exist_ok=False)
    evidence = scratch / "evidence"
    evidence.mkdir()
    adapter_receipt_path = evidence / "adapter-orbax.receipt.json"
    write_orbax_leaf_receipt(adapter_receipt_path, adapter_receipt)
    contract_path = evidence / "maxtext-to-hf.inputs.json"
    contract = create_artifact_contract(
        [
            f"base_checkpoint={base_leaf}",
            f"adapter_checkpoint={adapter_leaf}",
            f"hf_checkpoint={hf_snapshot}",
        ]
    )
    _write_once(contract_path, contract)
    contract_sha = _sha256(contract_path)
    conversion_run_id = stable_run_id(
        stage="maxtext-to-hf",
        config_sha256=bindings["config_sha256"],
        dataset_manifest_sha256=contract_sha,
    )
    environment = offline_environment(os.environ.copy())
    environment.update(
        {
            "BOOKFORGE_JAX_EXECUTION_APPROVAL": approval_token(
                stage="maxtext-to-hf",
                run_id=conversion_run_id,
                config_sha256=bindings["config_sha256"],
                input_sha256=contract_sha,
            ),
            "JAX_PLATFORMS": "cuda",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    merged_hf = scratch / "merged-hf"
    runs = scratch / "runs"
    subprocess.run(
        [
            "python3",
            "-m",
            "training.jax_fidelity.convert",
            "maxtext-to-hf",
            "--config",
            str(config_path),
            "--input-manifest",
            str(contract_path),
            "--input-manifest-sha256",
            contract_sha,
            "--base-checkpoint",
            str(base_leaf),
            "--adapter-checkpoint",
            str(adapter_leaf),
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
        cwd="/opt/bookforge",
        env=environment,
        check=True,
        timeout=_remaining_seconds(started),
    )
    conversion_run_path = runs / conversion_run_id / "run.json"
    conversion_completion_path = runs / conversion_run_id / "completion.json"
    conversion_completion = _json_object(conversion_completion_path)
    if (
        conversion_completion.get("run_id") != conversion_run_id
        or conversion_completion.get("status") != "succeeded"
        or conversion_completion.get("run_manifest_sha256") != _sha256(conversion_run_path)
        or not conversion_completion.get("artifacts")
        or not conversion_completion.get("evidence")
    ):
        raise RuntimeError("MaxText-to-HF merge has no successful terminal evidence")
    merged_manifest = artifact_manifest(merged_hf)
    output_evidence = conversion_completion.get("evidence")
    if (
        not isinstance(output_evidence, dict)
        or output_evidence.get("output_manifest") != merged_manifest
    ):
        raise RuntimeError("conversion completion does not bind the merged HF bytes")
    candidate = build_merged_candidate_manifest(
        config_path=config_path,
        dataset_manifest_sha256=bindings["dataset_manifest_sha256"],
        training_run_id=training_run_id,
        merged_hf_checkpoint=merged_hf,
    )

    release.mkdir(parents=True, exist_ok=False)
    _copy_tree(merged_hf, release / "merged-hf")
    _write_once(release / "merged-hf.manifest.json", merged_manifest)
    _write_once(release / "candidate.manifest.json", candidate)
    (release / "evidence").mkdir()
    shutil.copyfile(adapter_receipt_path, release / "evidence/adapter-orbax.receipt.json")
    shutil.copyfile(contract_path, release / "evidence/maxtext-to-hf.inputs.json")
    shutil.copyfile(conversion_run_path, release / "evidence/maxtext-to-hf.run.json")
    shutil.copyfile(
        conversion_completion_path,
        release / "evidence/maxtext-to-hf.completion.json",
    )
    source_bindings: dict[str, object] = {
        **bindings,
        "roundtrip_run_id": roundtrip_run_id,
        "training_release_run_id": training_release_run_id,
        "training_run_id": training_run_id,
        "adapter_orbax_receipt_sha256": _sha256(adapter_receipt_path),
        "conversion_input_manifest_sha256": contract_sha,
        "conversion_run_id": conversion_run_id,
        "conversion_run_sha256": _sha256(conversion_run_path),
        "conversion_completion_sha256": _sha256(conversion_completion_path),
    }
    _write_once(release / "evidence/source-bindings.json", source_bindings)
    validate_merged_candidate_manifest(
        release / "candidate.manifest.json",
        release / "merged-hf",
        config_path=config_path,
        expected_manifest_sha256=_sha256(release / "candidate.manifest.json"),
        expected_config_sha256=bindings["config_sha256"],
        expected_dataset_manifest_sha256=bindings["dataset_manifest_sha256"],
        expected_candidate_id=str(candidate["candidate_id"]),
    )
    verify_artifact_manifest(
        release / "merged-hf",
        _json_object(release / "merged-hf.manifest.json"),
    )
    files = _release_files(release)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "backend": "modal-l40s",
        "release_type": "provisional-merged-hf-development-candidate",
        "merge_run_id": merge_run_id,
        "candidate_id": candidate["candidate_id"],
        "training_run_id": training_run_id,
        **source_bindings,
        "merged_hf_manifest_sha256": _sha256(release / "merged-hf.manifest.json"),
        "checkpoint_manifest_sha256": canonical_sha256(merged_manifest),
        "checkpoint_content_sha256": merged_manifest["content_sha256"],
        "candidate_manifest_sha256": _sha256(release / "candidate.manifest.json"),
        "elapsed_seconds": time.monotonic() - started,
        "development_evaluated": False,
        "release_authorized": False,
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
        raise RuntimeError("Modal billing report was not a JSON list")
    return sum(float(row["Cost"]) for row in rows)


@app.local_entrypoint()
def merge_cli(
    merge_run_id: str,
    roundtrip_run_id: str,
    training_release_run_id: str,
    training_run_id: str,
    input_manifest_sha256: str,
    hf_snapshot_manifest_sha256: str,
    roundtrip_completion_sha256: str,
    base_orbax_receipt_sha256: str,
    base_orbax_manifest_sha256: str,
    training_release_completion_sha256: str,
    adapter_manifest_sha256: str,
    training_run_sha256: str,
    training_completion_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    approval_token_value: str,
) -> None:
    plan = _json_object(PLAN_PATH)
    if (
        plan.get("status") != "plan-only"
        or plan.get("function_calls") != 1
        or plan.get("automatic_retries") != 0
        or plan.get("web_endpoint") is not False
        or plan.get("budget_month") != BUDGET_MONTH
        or plan.get("workspace_hard_stop_usd") != WORKSPACE_HARD_STOP_USD
        or plan.get("gross_ceiling_policy") != "declared-estimate-not-provider-enforced"
    ):
        raise RuntimeError("Modal full-adapter merge plan is not finite")
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("Modal full-adapter merge approval is outside its budget month")
    arguments = locals().copy()
    bindings = {name: arguments[name] for name in _BINDING_NAMES}
    request: dict[str, object] = {
        "merge_run_id": merge_run_id,
        "roundtrip_run_id": roundtrip_run_id,
        "training_release_run_id": training_release_run_id,
        "training_run_id": training_run_id,
        **bindings,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    ceiling = plan.get("gross_ceiling_usd")
    if type(ceiling) not in (int, float) or float(ceiling) <= 0:
        raise RuntimeError("Modal full-adapter merge gross ceiling is invalid")
    workspace_before = _authoritative_workspace_total()
    if workspace_before + float(ceiling) > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace budget has insufficient merge headroom")

    from infra.gcp.jax.modal_reconciliation import append_reconciliation, reserve_attempt

    attempt_id = f"jax-merge:{merge_run_id}"
    reserve_attempt(LEDGER_PATH, attempt_id=attempt_id, stage="jax-full-adapter-merge")
    result: dict[str, object] | None = None
    status = "remote-error"
    workspace_after: float | None = None
    postrun_error: str | None = None
    try:
        result = merge_finite.remote(request)
        status = "succeeded"
    finally:
        try:
            workspace_after = _authoritative_workspace_total()
        except Exception as error:
            postrun_error = f"{type(error).__name__}: {error}"
        append_reconciliation(
            LEDGER_PATH,
            attempt_id=attempt_id,
            stage="jax-full-adapter-merge",
            workspace_before_usd=workspace_before,
            workspace_after_usd=workspace_after,
            declared_ceiling_usd=float(ceiling),
            status=status,
            result=result,
            postrun_report_error=postrun_error,
        )
    if result is None:
        raise RuntimeError("Modal full-adapter merge returned no result")
    print(json.dumps(result, indent=2, sort_keys=True))
