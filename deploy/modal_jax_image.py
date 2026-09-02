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
CUDA_BUILD_REQUIREMENTS = (
    "setuptools==84.0.0",
    "pybind11[global]==3.0.2",
    "jax[cuda12]==0.11.0",
    "flax==0.12.8",
    "nvidia-cudnn-frontend==1.28.0",
    "nvidia-curand-cu12==10.3.10.19",
    "nvidia-nvtx-cu12==12.9.79",
)
NCCL_LIBRARY_DIR = "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib"
CUDA_WHEEL_LIBRARY_DIRS = (
    "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib",
    "/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib",
    NCCL_LIBRARY_DIR,
    "/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
)
CUDA_WHEEL_LIBRARY_PATH = ":".join(CUDA_WHEEL_LIBRARY_DIRS)
NVIDIA_PYTHON_ROOT = "/usr/local/lib/python3.12/site-packages/nvidia"
TRANSFORMER_ENGINE_BUILD_ENV = {
    "NVTE_BUILD_USE_NVIDIA_WHEELS": "1",
    "LIBRARY_PATH": NCCL_LIBRARY_DIR,
    "LDFLAGS": f"-L{NCCL_LIBRARY_DIR} -Wl,-rpath,{NCCL_LIBRARY_DIR}",
}
JAX_IMAGE = (
    modal.Image.from_registry(pinned_image_uri())
    .apt_install("git", "ca-certificates", "build-essential")
    .run_commands(
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /opt/MaxText",
        f"git -C /opt/MaxText checkout {MAXTEXT_REVISION}",
        f'test "$(git -C /opt/MaxText rev-parse HEAD)" = "{MAXTEXT_REVISION}"',
        'test -z "$(git -C /opt/MaxText status --porcelain)"',
    )
    # Transformer Engine 2.18's isolated build requirements omit the NVTX
    # wheel even though its compiler searches nvidia/nvtx/include. Seed the
    # exact CUDA build environment, then compile that extension against it.
    .pip_install(*CUDA_BUILD_REQUIREMENTS)
    .run_commands(
        f"test -f {NCCL_LIBRARY_DIR}/libnccl.so.2",
        f"ln -sfn libnccl.so.2 {NCCL_LIBRARY_DIR}/libnccl.so",
    )
    .pip_install(
        "transformer-engine-jax==2.18.0",
        extra_options="--no-build-isolation",
        env=TRANSFORMER_ENGINE_BUILD_ENV,
    )
    .pip_install_from_requirements(REPOSITORY_ROOT / "training/jax_fidelity/requirements.lock")
    # Transformer Engine 2.18 searches the NVIDIA Python namespace for
    # ``cudart``, while the CUDA 12 runtime wheel exposes ``cuda_runtime``.
    # Keep the vendor wheel intact and provide the compatibility name its
    # loader expects.
    .run_commands(
        f"test -f {NVIDIA_PYTHON_ROOT}/cuda_runtime/lib/libcudart.so.12",
        f"ln -sfnT cuda_runtime {NVIDIA_PYTHON_ROOT}/cudart",
        "python -c \"import glob; assert glob.glob("
        f"'{NVIDIA_PYTHON_ROOT}/cudart/lib/lib*.so.*[0-9]')\"",
    )
    .add_local_dir(REPOSITORY_ROOT / "training", "/opt/bookforge/training", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "src", "/opt/bookforge/src", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "infra/gcp/jax", "/opt/bookforge/infra/gcp/jax", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "deploy", "/opt/bookforge/deploy", copy=True)
    .add_local_file(
        CONFIG_PATH,
        "/opt/bookforge/experiments/jax-fidelity-lab/config.json",
        copy=True,
    )
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
            "LD_LIBRARY_PATH": CUDA_WHEEL_LIBRARY_PATH,
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
