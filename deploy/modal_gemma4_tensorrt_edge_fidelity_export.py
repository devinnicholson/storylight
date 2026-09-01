"""One finite Modal L40S export for a checksum-bound tuned Gemma candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

from scripts.validate_fidelity_release import validate_release

REPOSITORY_ROOT = Path(__file__).parents[1]
APP_NAME = "bookforge-gemma4-fidelity-tensorrt-export"
GPU = "L40S"
MAX_CONTAINERS = 1
RETRIES = 0
REMOTE_TIMEOUT_SECONDS = 1_500
BUDGET_MONTH = "2026-09"
WORKSPACE_HARD_STOP_USD = 28.0
FULL_COMMAND_CEILING_USD = 1.30
FULL_COMMAND_CEILING_PROVIDER_ENFORCED = False
LEDGER_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"
_INPUT_ROOT = Path("/releases")
_EXPORT_ROOT = Path("/exports")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")
_RELATIVE_PREFIX = re.compile(r"[a-z0-9][a-z0-9._/-]{3,160}\Z")
_DIGEST_IMAGE = re.compile(r"nvcr\.io/nvidia/pytorch:[^@]+@sha256:[0-9a-f]{64}\Z")


def _pinned_nvidia_image() -> str:
    image = os.environ.get("BOOKFORGE_NVIDIA_PYTORCH_IMAGE", "")
    if _DIGEST_IMAGE.fullmatch(image) is None:
        raise RuntimeError(
            "BOOKFORGE_NVIDIA_PYTORCH_IMAGE must be an NVIDIA registry image pinned by digest"
        )
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


NVIDIA_PYTORCH_IMAGE = _pinned_nvidia_image()

release_volume = modal.Volume.from_name("bookforge-jax-fidelity-release", create_if_missing=False)
export_volume = modal.Volume.from_name(
    "bookforge-tensorrt-edge-llm-fidelity", create_if_missing=False
)

export_image = (
    modal.Image.from_registry(NVIDIA_PYTORCH_IMAGE)
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 --branch v0.10.0 "
        "https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/tensorrt-edge-llm",
        f'test "$(git -C /opt/tensorrt-edge-llm rev-parse HEAD)" = "{EDGELLM_REVISION}"',
        "python -m pip install --no-cache-dir -e '/opt/tensorrt-edge-llm[tools]'",
        "tensorrt-edgellm-quantize llm --help >/dev/null",
        "tensorrt-edgellm-export --help >/dev/null",
    )
    .add_local_file(
        REPOSITORY_ROOT / "deploy/gcp_gemma4_tensorrt_export/export_fidelity_candidate.py",
        "/opt/bookforge/deploy/gcp_gemma4_tensorrt_export/export_fidelity_candidate.py",
        copy=True,
    )
    .add_local_file(
        REPOSITORY_ROOT / "scripts/validate_fidelity_release.py",
        "/opt/bookforge/scripts/validate_fidelity_release.py",
        copy=True,
    )
    .env({"PYTHONPATH": "/opt/bookforge", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App(APP_NAME)


def _approval_token(
    candidate_id: str,
    release_manifest_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
) -> str:
    return (
        f"APPROVE_MODAL_FIDELITY_EXPORT:{candidate_id}:{release_manifest_sha256}:"
        f"{config_sha256}:{dataset_manifest_sha256}"
    )


def _validate_request(request: dict[str, object]) -> tuple[str, str, str, str, str]:
    release_prefix = request.get("release_prefix")
    candidate_id = request.get("candidate_id")
    release_sha = request.get("release_manifest_sha256")
    config_sha = request.get("config_sha256")
    dataset_sha = request.get("dataset_manifest_sha256")
    approval = request.get("approval_token")
    if (
        not isinstance(release_prefix, str)
        or not _RELATIVE_PREFIX.fullmatch(release_prefix)
        or release_prefix.startswith("/")
        or ".." in Path(release_prefix).parts
    ):
        raise ValueError("release_prefix must be a safe private-volume relative prefix")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("candidate_id is not a content-addressed fidelity ID")
    for name, value in (
        ("release_manifest_sha256", release_sha),
        ("config_sha256", config_sha),
        ("dataset_manifest_sha256", dataset_sha),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    expected = _approval_token(candidate_id, release_sha, config_sha, dataset_sha)
    if approval != expected:
        raise ValueError("Modal export approval token is not exact")
    return release_prefix, candidate_id, release_sha, config_sha, dataset_sha


@app.function(
    image=export_image,
    gpu=GPU,
    cpu=8,
    memory=65_536,
    timeout=REMOTE_TIMEOUT_SECONDS,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
    volumes={str(_INPUT_ROOT): release_volume, str(_EXPORT_ROOT): export_volume},
)
def export_finite(request: dict[str, object]) -> dict[str, object]:
    """Run exactly once; any existing candidate state makes the ID terminal."""

    release_prefix, candidate_id, release_sha, config_sha, dataset_sha = _validate_request(request)
    release_directory = _INPUT_ROOT / release_prefix
    manifest = release_directory / "release.manifest.json"
    source = release_directory / "merged-hf"
    release = validate_release(
        manifest,
        source,
        expected_manifest_sha256=release_sha,
        expected_config_sha256=config_sha,
        expected_dataset_manifest_sha256=dataset_sha,
        expected_candidate=candidate_id,
    )
    destination = _EXPORT_ROOT / candidate_id / release_sha[:20]
    intent = _EXPORT_ROOT / "intents" / f"{candidate_id}-{release_sha[:20]}.json"
    if destination.exists() or intent.exists():
        raise RuntimeError(
            "candidate export state already exists; retries and overwrites are forbidden"
        )
    intent.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(intent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "1.0",
                "status": "export-intent-recorded",
                "retry_allowed": False,
                "candidate_id": candidate_id,
                "release_manifest_sha256": release_sha,
                "config_sha256": config_sha,
                "dataset_manifest_sha256": dataset_sha,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    export_volume.commit()

    from deploy.gcp_gemma4_tensorrt_export.export_fidelity_candidate import (
        export_local_candidate,
    )

    payload = export_local_candidate(
        release,
        _EXPORT_ROOT,
        private_output_prefix=(
            f"modal-private://bookforge-tensorrt-edge-llm-fidelity/"
            f"{candidate_id}/{release_sha[:20]}"
        ),
        execution_metadata={
            "backend": "modal",
            "gpu": GPU,
            "intent_path": str(intent),
            "container_image": NVIDIA_PYTORCH_IMAGE,
        },
    )
    export_volume.commit()
    manifest_path = destination / "export.manifest.json"
    return {
        **payload,
        "export_manifest_sha256": _sha256(manifest_path),
        "export_volume_prefix": f"{candidate_id}/{release_sha[:20]}",
    }


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
def export_cli(
    release_prefix: str,
    candidate_id: str,
    release_manifest_sha256: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    approval_token_value: str,
) -> None:
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("this finite export plan is restricted to its declared billing month")
    expected = _approval_token(
        candidate_id,
        release_manifest_sha256,
        config_sha256,
        dataset_manifest_sha256,
    )
    if approval_token_value != expected:
        raise RuntimeError("exact one-purpose Modal approval token is required")
    workspace_total = _authoritative_workspace_total()
    projected_total = workspace_total + FULL_COMMAND_CEILING_USD
    if projected_total > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError(
            "fidelity export refused by current-month budget gate: "
            f"${workspace_total:.8f} + ${FULL_COMMAND_CEILING_USD:.2f} "
            f"> ${WORKSPACE_HARD_STOP_USD:.2f}"
        )
    from infra.gcp.jax.modal_reconciliation import append_reconciliation

    result: dict[str, object] | None = None
    status = "remote-error"
    postrun_total: float | None = None
    postrun_error: str | None = None
    try:
        result = export_finite.remote(
            {
                "release_prefix": release_prefix,
                "candidate_id": candidate_id,
                "release_manifest_sha256": release_manifest_sha256,
                "config_sha256": config_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "approval_token": approval_token_value,
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
            attempt_id=f"tensorrt-export:{candidate_id}:{release_manifest_sha256[:20]}",
            stage="tensorrt-export",
            workspace_before_usd=workspace_total,
            workspace_after_usd=postrun_total,
            declared_ceiling_usd=FULL_COMMAND_CEILING_USD,
            status=status,
            result=result,
            postrun_report_error=postrun_error,
        )
    if result is None:
        raise RuntimeError("Modal TensorRT export returned no result")
    print(
        json.dumps(
            {
                "billing_month": BUDGET_MONTH,
                "authoritative_workspace_total_usd": workspace_total,
                "full_command_ceiling_usd": FULL_COMMAND_CEILING_USD,
                "full_command_ceiling_provider_enforced": (FULL_COMMAND_CEILING_PROVIDER_ENFORCED),
                "projected_workspace_total_usd": projected_total,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
