"""Configure JAX's persistent compilation cache before backend initialization."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CACHE_RECEIPT_SCHEMA = "bookforge-jax-cache-runtime-v1"
CACHE_DIRECTORY_ENV = "JAX_COMPILATION_CACHE_DIR"
CACHE_RECEIPT_ENV = "BOOKFORGE_JAX_CACHE_RECEIPT_PATH"

_REQUIRED_ENVIRONMENT = {
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
}

_CONFIG_VALUES: tuple[tuple[str, str, object], ...] = (
    ("jax_enable_compilation_cache", "JAX_ENABLE_COMPILATION_CACHE", True),
    (
        "jax_persistent_cache_min_compile_time_secs",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        0.0,
    ),
    (
        "jax_persistent_cache_min_entry_size_bytes",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        -1,
    ),
    (
        "jax_persistent_cache_enable_xla_caches",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "all",
    ),
    (
        "jax_raise_persistent_cache_errors",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
        True,
    ),
)


def cache_inventory(directory: Path) -> dict[str, int]:
    """Return a symlink-safe count of cache files and bytes."""

    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("JAX compilation cache directory is missing or unsafe")
    files = 0
    total_bytes = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("JAX compilation cache contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("JAX compilation cache contains a non-regular entry")
        files += 1
        total_bytes += path.stat().st_size
    return {"files": files, "bytes": total_bytes}


def configure_persistent_compilation_cache(
    jax_module: Any,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Apply and verify Bookforge's exact persistent-cache policy."""

    values = os.environ if environment is None else environment
    raw_directory = values.get(CACHE_DIRECTORY_ENV)
    if not raw_directory:
        return {"schema_version": CACHE_RECEIPT_SCHEMA, "configured": False}
    directory = Path(raw_directory)
    if not directory.is_absolute():
        raise RuntimeError("JAX compilation cache directory must be absolute")
    cache_inventory(directory)
    for name, expected in _REQUIRED_ENVIRONMENT.items():
        if values.get(name) != expected:
            raise RuntimeError(f"JAX compilation cache environment changed: {name}")

    jax_module.config.update("jax_compilation_cache_dir", str(directory))
    for config_name, _environment_name, expected in _CONFIG_VALUES:
        jax_module.config.update(config_name, expected)

    actual = {
        "jax_compilation_cache_dir": jax_module.config.values.get("jax_compilation_cache_dir")
    }
    for config_name, _environment_name, _expected in _CONFIG_VALUES:
        actual[config_name] = jax_module.config.values.get(config_name)
    expected_config = {
        "jax_compilation_cache_dir": str(directory),
        **{name: expected for name, _environment_name, expected in _CONFIG_VALUES},
    }
    if actual != expected_config:
        raise RuntimeError("JAX persistent compilation cache configuration changed")
    return {
        "schema_version": CACHE_RECEIPT_SCHEMA,
        "configured": True,
        "cache_directory": str(directory),
        "config": actual,
    }


def validate_cache_receipt(
    document: Mapping[str, object], *, expected_directory: Path
) -> dict[str, object]:
    expected_config = {
        "jax_compilation_cache_dir": str(expected_directory),
        **{name: expected for name, _environment_name, expected in _CONFIG_VALUES},
    }
    if (
        document.get("schema_version") != CACHE_RECEIPT_SCHEMA
        or document.get("configured") is not True
        or document.get("cache_directory") != str(expected_directory)
        or document.get("config") != expected_config
    ):
        raise RuntimeError("JAX persistent compilation cache receipt changed")
    return dict(document)


def write_cache_receipt(document: Mapping[str, object]) -> Path | None:
    """Write the configured child-process receipt when the parent requests one."""

    raw_path = os.environ.get(CACHE_RECEIPT_ENV)
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("JAX compilation cache receipt path is unsafe")
    payload = json.dumps(dict(document), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path
