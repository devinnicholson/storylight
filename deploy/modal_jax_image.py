"""Shared immutable CUDA JAX/MaxText image for finite Modal jobs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import modal

from infra.gcp.jax.packaged_source_manifest import (
    BOOKFORGE_CONTAINER_ROOT,
    LOCAL_SOURCE_IGNORE,
    PACKAGED_BOOKFORGE_DIRECTORIES,
    PACKAGED_BOOKFORGE_FILES,
    packaged_bookforge_source_manifest,
    source_manifest_bytes,
    source_manifest_sha256,
)

__all__ = (
    "BOOKFORGE_CONTAINER_ROOT",
    "LOCAL_SOURCE_IGNORE",
    "PACKAGED_BOOKFORGE_DIRECTORIES",
    "PACKAGED_BOOKFORGE_FILES",
    "packaged_bookforge_source_manifest",
)

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/jax-fidelity-lab/config.json"
MAXTEXT_NATIVE_LORA_PATCH = (
    REPOSITORY_ROOT
    / "training/jax_fidelity/patches/maxtext-native-lora-materialization.patch"
)
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
BOOKFORGE_SOURCE_MANIFEST_CONTAINER_PATH = (
    BOOKFORGE_CONTAINER_ROOT / "source.manifest.json"
)
def _materialize_source_manifest(document: dict[str, object]) -> tuple[Path, str]:
    payload = source_manifest_bytes(document)
    digest = source_manifest_sha256(document)
    directory = Path(tempfile.mkdtemp(prefix="bookforge-jax-source-manifest-"))
    path = directory / "source.manifest.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return path, digest


BOOKFORGE_SOURCE_MANIFEST = packaged_bookforge_source_manifest(REPOSITORY_ROOT)
BOOKFORGE_SOURCE_MANIFEST_PATH, BOOKFORGE_SOURCE_MANIFEST_SHA256 = (
    _materialize_source_manifest(BOOKFORGE_SOURCE_MANIFEST)
)
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
    # Keep the stable, multi-gigabyte dependency layers above the mutable
    # Bookforge patch so a patch revision only rebuilds the final image layers.
    .add_local_file(
        MAXTEXT_NATIVE_LORA_PATCH,
        "/opt/bookforge/patches/maxtext-native-lora-materialization.patch",
        copy=True,
    )
    .run_commands(
        "git -C /opt/MaxText apply --check --unidiff-zero "
        "/opt/bookforge/patches/maxtext-native-lora-materialization.patch",
        "git -C /opt/MaxText apply --unidiff-zero "
        "/opt/bookforge/patches/maxtext-native-lora-materialization.patch",
        "git -C /opt/MaxText diff --check",
        "test \"$(git -C /opt/MaxText status --short)\" = "
        "\"$(printf '%s\\n%s\\n%s' ' M src/maxtext/common/checkpointing.py' "
        "' M src/maxtext/trainers/pre_train/train.py' "
        "' M src/maxtext/utils/train_utils.py')\"",
        "git -C /opt/MaxText diff --no-ext-diff --binary --abbrev=8 --unified=0 -- "
        "src/maxtext/common/checkpointing.py "
        "src/maxtext/trainers/pre_train/train.py src/maxtext/utils/train_utils.py "
        "| cmp -s - /opt/bookforge/patches/maxtext-native-lora-materialization.patch",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "training",
        "/opt/bookforge/training",
        copy=True,
        ignore=LOCAL_SOURCE_IGNORE,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "src",
        "/opt/bookforge/src",
        copy=True,
        ignore=LOCAL_SOURCE_IGNORE,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "infra/gcp/jax",
        "/opt/bookforge/infra/gcp/jax",
        copy=True,
        ignore=LOCAL_SOURCE_IGNORE,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "deploy",
        "/opt/bookforge/deploy",
        copy=True,
        ignore=LOCAL_SOURCE_IGNORE,
    )
    .add_local_file(
        CONFIG_PATH,
        "/opt/bookforge/experiments/jax-fidelity-lab/config.json",
        copy=True,
    )
    .add_local_file(
        BOOKFORGE_SOURCE_MANIFEST_PATH,
        str(BOOKFORGE_SOURCE_MANIFEST_CONTAINER_PATH),
        copy=True,
    )
    .run_commands(
        "PYTHONPATH=/opt/MaxText/src:/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --maxtext-root /opt/MaxText "
        "--write-lock /opt/bookforge/runtime.lock.json",
        "PYTHONPATH=/opt/MaxText/src:/opt/bookforge:/opt/bookforge/src python -m "
        "training.jax_fidelity.verify_runtime --maxtext-root /opt/MaxText "
        "--lock /opt/bookforge/runtime.lock.json",
    )
    .env(
        {
            "PYTHONPATH": "/opt/MaxText/src:/opt/bookforge:/opt/bookforge/src",
            "JAX_PLATFORMS": "cuda",
            # Gemma 4 E2B LoRA compiles to a ~19.6 GiB L4 graph. JAX defaults
            # to a 75% (16.5 GiB) pool, so expose a bounded 95% pool while
            # leaving device headroom for the CUDA runtime.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
            "LD_LIBRARY_PATH": CUDA_WHEEL_LIBRARY_PATH,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "BOOKFORGE_MAXTEXT_APPROVED_PATCH": (
                "/opt/bookforge/patches/maxtext-native-lora-materialization.patch"
            ),
            "BOOKFORGE_SOURCE_MANIFEST": str(
                BOOKFORGE_SOURCE_MANIFEST_CONTAINER_PATH
            ),
            "BOOKFORGE_SOURCE_MANIFEST_SHA256": BOOKFORGE_SOURCE_MANIFEST_SHA256,
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
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    return sanitized
