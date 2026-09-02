"""Shared immutable CUDA JAX/MaxText image for finite Modal jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import modal

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
LOCAL_SOURCE_IGNORE = (
    "**/__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
)
BOOKFORGE_CONTAINER_ROOT = Path("/opt/bookforge")
BOOKFORGE_SOURCE_MANIFEST_CONTAINER_PATH = (
    BOOKFORGE_CONTAINER_ROOT / "source.manifest.json"
)
PACKAGED_BOOKFORGE_DIRECTORIES = (
    ("training", "training"),
    ("src", "src"),
    ("infra/gcp/jax", "infra/gcp/jax"),
    ("deploy", "deploy"),
)
PACKAGED_BOOKFORGE_FILES = (
    (
        "training/jax_fidelity/patches/maxtext-native-lora-materialization.patch",
        "patches/maxtext-native-lora-materialization.patch",
    ),
    (
        "experiments/jax-fidelity-lab/config.json",
        "experiments/jax-fidelity-lab/config.json",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ignored_local_source(relative: Path) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}


def _regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _ignored_local_source(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"packaged Bookforge source may not be a symlink: {path}")
        if path.is_file():
            yield path


def packaged_bookforge_source_manifest(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Describe every regular repository file copied into the JAX image."""

    rows: list[dict[str, object]] = []
    for local_relative, container_relative in PACKAGED_BOOKFORGE_DIRECTORIES:
        local_root = repository_root / local_relative
        if not local_root.is_dir() or local_root.is_symlink():
            raise ValueError(f"packaged Bookforge directory is missing or unsafe: {local_root}")
        for path in _regular_files(local_root):
            relative = path.relative_to(local_root)
            rows.append(
                {
                    "path": (Path(container_relative) / relative).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    for local_relative, container_relative in PACKAGED_BOOKFORGE_FILES:
        path = repository_root / local_relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"packaged Bookforge file is missing or unsafe: {path}")
        rows.append(
            {
                "path": container_relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    paths = [str(row["path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("packaged Bookforge source paths are not unique")
    files_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "bookforge-jax-packaged-source-v1",
        "producer": "bookforge-modal-jax-image",
        "container_root": str(BOOKFORGE_CONTAINER_ROOT),
        "ignore_patterns": list(LOCAL_SOURCE_IGNORE),
        "file_count": len(rows),
        "files_sha256": files_sha256,
        "files": rows,
    }


def _materialize_source_manifest(document: dict[str, object]) -> tuple[Path, str]:
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(tempfile.mkdtemp(prefix="bookforge-jax-source-manifest-"))
    path = directory / "source.manifest.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return path, digest


BOOKFORGE_SOURCE_MANIFEST = packaged_bookforge_source_manifest()
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
        "\"$(printf '%s\\n%s' ' M src/maxtext/trainers/pre_train/train.py' "
        "' M src/maxtext/utils/train_utils.py')\"",
        "git -C /opt/MaxText diff --no-ext-diff --binary --abbrev=8 --unified=0 -- "
        "src/maxtext/trainers/pre_train/train.py src/maxtext/utils/train_utils.py "
        "| cmp -s - /opt/bookforge/patches/maxtext-native-lora-materialization.patch",
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
