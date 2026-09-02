"""Pure helpers for the Bookforge source bytes shared by JAX runtimes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

LOCAL_SOURCE_IGNORE = (
    "**/__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
)
BOOKFORGE_CONTAINER_ROOT = Path("/opt/bookforge")
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


def _ignored_local_source(relative: PurePosixPath) -> bool:
    return "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}


def _regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _ignored_local_source(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"packaged Bookforge source may not be a symlink: {path}")
        if path.is_file():
            yield path


def packaged_bookforge_local_rows(repository_root: Path) -> list[dict[str, object]]:
    """Describe unique repository files copied into the Bookforge runtime."""

    repository_root = repository_root.resolve()
    rows: dict[str, dict[str, object]] = {}
    for local_relative, _container_relative in PACKAGED_BOOKFORGE_DIRECTORIES:
        local_root = repository_root / local_relative
        if not local_root.is_dir() or local_root.is_symlink():
            raise ValueError(f"packaged Bookforge directory is missing or unsafe: {local_root}")
        for path in _regular_files(local_root):
            relative = path.relative_to(repository_root).as_posix()
            rows[relative] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    for local_relative, _container_relative in PACKAGED_BOOKFORGE_FILES:
        path = repository_root / local_relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"packaged Bookforge file is missing or unsafe: {path}")
        rows[local_relative] = {
            "path": local_relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return [rows[path] for path in sorted(rows)]


def _validated_source_rows(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for source in source_rows:
        relative = source.get("path")
        byte_count = source.get("bytes")
        sha256 = source.get("sha256")
        if (
            not isinstance(relative, str)
            or type(byte_count) is not int
            or int(byte_count) < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("packaged Bookforge source row is malformed")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or relative in {"", "."}
            or ".." in path.parts
            or path.as_posix() != relative
            or relative in rows
        ):
            raise ValueError("packaged Bookforge source row has an unsafe or duplicate path")
        rows[relative] = dict(source)
    return rows


def packaged_bookforge_source_manifest_from_rows(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Derive the runtime manifest from repository-relative source rows."""

    sources = _validated_source_rows(source_rows)
    packaged: list[dict[str, object]] = []
    for local_relative, container_relative in PACKAGED_BOOKFORGE_DIRECTORIES:
        prefix = PurePosixPath(local_relative)
        for relative, row in sources.items():
            path = PurePosixPath(relative)
            try:
                suffix = path.relative_to(prefix)
            except ValueError:
                continue
            if _ignored_local_source(suffix):
                continue
            packaged.append(
                {
                    "path": (PurePosixPath(container_relative) / suffix).as_posix(),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
            )
    for local_relative, container_relative in PACKAGED_BOOKFORGE_FILES:
        row = sources.get(local_relative)
        if row is None:
            raise ValueError(f"packaged Bookforge file has no source row: {local_relative}")
        packaged.append(
            {
                "path": container_relative,
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
        )
    packaged.sort(key=lambda row: str(row["path"]))
    paths = [str(row["path"]) for row in packaged]
    if len(paths) != len(set(paths)):
        raise ValueError("packaged Bookforge source paths are not unique")
    files_sha256 = hashlib.sha256(
        json.dumps(packaged, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "bookforge-jax-packaged-source-v1",
        "producer": "bookforge-modal-jax-image",
        "container_root": str(BOOKFORGE_CONTAINER_ROOT),
        "ignore_patterns": list(LOCAL_SOURCE_IGNORE),
        "file_count": len(packaged),
        "files_sha256": files_sha256,
        "files": packaged,
    }


def packaged_bookforge_source_manifest(
    repository_root: Path,
) -> dict[str, object]:
    return packaged_bookforge_source_manifest_from_rows(
        packaged_bookforge_local_rows(repository_root)
    )


def source_manifest_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def source_manifest_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(source_manifest_bytes(document)).hexdigest()
