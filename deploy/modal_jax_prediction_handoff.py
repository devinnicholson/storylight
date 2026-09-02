"""Publish one immutable no-reupload Modal prediction reference."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal

from infra.gcp.jax.prediction_handoff import (
    INPUT_VOLUME,
    PREDICTION_ROOT,
    RELEASE_VOLUME,
    approval_token,
    canonical_bytes,
    manifest_sha256,
    verify_reference_sources,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
APP_NAME = "bookforge-jax-prediction-handoff"
TIMEOUT_SECONDS = 1_200
MAX_CONTAINERS = 1
_INPUT_ROOT = Path("/inputs")
_RELEASE_ROOT = Path("/releases/merged")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,95}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        REPOSITORY_ROOT / "infra/gcp/jax",
        "/opt/bookforge/infra/gcp/jax",
        copy=True,
    )
    .env({"PYTHONPATH": "/opt/bookforge"})
)
input_volume = modal.Volume.from_name(INPUT_VOLUME, create_if_missing=False)
release_volume = modal.Volume.from_name(RELEASE_VOLUME, create_if_missing=False)
app = modal.App(APP_NAME)


def _validate_request(request: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in ("source_run_id", "merge_run_id", "prediction_run_id"):
        value = request.get(name)
        if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
            raise ValueError(f"{name} must be an immutable lowercase slug")
        values[name] = value
    if len({values[name] for name in ("source_run_id", "merge_run_id", "prediction_run_id")}) != 3:
        raise ValueError("source, merge, and prediction run IDs must be distinct")
    candidate_id = request.get("candidate_id")
    if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("candidate_id must be content-addressed")
    values["candidate_id"] = candidate_id
    for name in (
        "source_input_manifest_sha256",
        "merge_completion_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
        "target_manifest_sha256",
    ):
        value = request.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
        values[name] = value
    expected = approval_token(**values)
    if request.get("approval_token") != expected:
        raise ValueError("prediction handoff approval token is not exact")
    return values


def _write_manifest_last(path: Path, document: dict[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def stage_reference(request: dict[str, object]) -> dict[str, object]:
    """Verify both immutable sources and publish only their reference manifest."""

    values = _validate_request(request)
    target = _INPUT_ROOT / PREDICTION_ROOT / values["prediction_run_id"]
    if target.exists() or target.is_symlink():
        raise RuntimeError("prediction handoff prefix already contains state")
    manifest, _ = verify_reference_sources(
        source_input_root=_INPUT_ROOT / values["source_run_id"],
        source_input_manifest_sha256=values["source_input_manifest_sha256"],
        merge_release_root=_RELEASE_ROOT / values["merge_run_id"],
        merge_completion_sha256=values["merge_completion_sha256"],
        prediction_run_id=values["prediction_run_id"],
    )
    if (
        manifest_sha256(manifest) != values["target_manifest_sha256"]
        or manifest["bindings"]["candidate_id"] != values["candidate_id"]
        or manifest["bindings"]["candidate_manifest_sha256"]
        != values["candidate_manifest_sha256"]
        or manifest["bindings"]["checkpoint_manifest_sha256"]
        != values["checkpoint_manifest_sha256"]
        or manifest["bindings"]["checkpoint_content_sha256"]
        != values["checkpoint_content_sha256"]
    ):
        raise RuntimeError("derived prediction reference differs from the approved request")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)
    _write_manifest_last(target / "inputs.manifest.json", manifest)
    return {
        "schema_version": "1.0",
        "status": "staged",
        "producer": "bookforge-modal-jax-prediction-reference-stager",
        "source_run_id": values["source_run_id"],
        "source_input_manifest_sha256": values["source_input_manifest_sha256"],
        "merge_run_id": values["merge_run_id"],
        "merge_completion_sha256": values["merge_completion_sha256"],
        "prediction_run_id": values["prediction_run_id"],
        "candidate_id": values["candidate_id"],
        "input_manifest_sha256": values["target_manifest_sha256"],
        "prefix": f"{PREDICTION_ROOT}/{values['prediction_run_id']}",
        "references": len(manifest["references"]),
        "manifest_uploaded_last": True,
        "hidden_records_uploaded": False,
        "checkpoint_bytes_duplicated": 0,
        "overwrite_enabled": False,
    }


@app.function(
    image=image,
    cpu=4,
    memory=8_192,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=MAX_CONTAINERS,
    volumes={str(_INPUT_ROOT): input_volume, "/releases": release_volume},
)
def stage_finite(request: dict[str, object]) -> dict[str, object]:
    input_volume.reload()
    release_volume.reload()
    result = stage_reference(request)
    input_volume.commit()
    return result


@app.local_entrypoint()
def run_cli(
    source_run_id: str,
    source_input_manifest_sha256: str,
    merge_run_id: str,
    merge_completion_sha256: str,
    prediction_run_id: str,
    candidate_id: str,
    candidate_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_content_sha256: str,
    target_manifest_sha256: str,
    approval_token_value: str,
) -> None:
    request = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_input_manifest_sha256,
        "merge_run_id": merge_run_id,
        "merge_completion_sha256": merge_completion_sha256,
        "prediction_run_id": prediction_run_id,
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_content_sha256": checkpoint_content_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "approval_token": approval_token_value,
    }
    _validate_request(request)
    print(json.dumps(stage_finite.remote(request), indent=2, sort_keys=True))
