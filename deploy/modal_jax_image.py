"""Shared immutable CUDA JAX/MaxText image for finite Modal jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/config.json"
_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def _versions() -> dict[str, object]:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    versions = document.get("versions") if isinstance(document, dict) else None
    if not isinstance(versions, dict):
        raise ValueError("JAX config has no versions object")
    return versions


def pinned_image_uri() -> str:
    image = _versions().get("container_image")
    if not isinstance(image, str) or _DIGEST_IMAGE.fullmatch(image) is None:
        raise ValueError("JAX container image must be pinned by sha256 digest")
    return image


def maxtext_revision() -> str:
    revision = _versions().get("maxtext_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("MaxText revision must be a full Git commit")
    return revision


MAXTEXT_REVISION = maxtext_revision()
JAX_IMAGE = (
    modal.Image.from_registry(pinned_image_uri())
    .apt_install("git", "ca-certificates")
    .run_commands(
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /opt/MaxText",
        f"git -C /opt/MaxText checkout {MAXTEXT_REVISION}",
        f'test "$(git -C /opt/MaxText rev-parse HEAD)" = "{MAXTEXT_REVISION}"',
        'test -z "$(git -C /opt/MaxText status --porcelain)"',
    )
    .pip_install_from_requirements(REPOSITORY_ROOT / "training/jax_fidelity/requirements.lock")
    .pip_install("jax[cuda12]==0.11.0")
    .add_local_dir(REPOSITORY_ROOT / "training", "/opt/bookforge/training", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "src", "/opt/bookforge/src", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "infra/gcp/jax", "/opt/bookforge/infra/gcp/jax", copy=True)
    .run_commands(
        "PYTHONPATH=/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --write-lock /opt/bookforge/runtime.lock.json",
        "PYTHONPATH=/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --lock /opt/bookforge/runtime.lock.json",
    )
    .env(
        {
            "PYTHONPATH": "/opt/bookforge:/opt/bookforge/src",
            "JAX_PLATFORMS": "cuda",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
)


def offline_environment(environment: dict[str, str]) -> dict[str, str]:
    """Remove inherited model credentials and force local-only checkpoint access."""

    sanitized = dict(environment)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        sanitized.pop(name, None)
    sanitized.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return sanitized
