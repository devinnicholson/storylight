# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity import maxtext_entrypoint
from training.jax_fidelity.compilation_cache import (
    CACHE_RECEIPT_ENV,
    configure_persistent_compilation_cache,
    write_cache_receipt,
)


class _Config:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, name: str, value: object) -> None:
        self.values[name] = value


def _environment(cache: Path) -> dict[str, str]:
    return {
        "JAX_COMPILATION_CACHE_DIR": str(cache),
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
    }


def test_cache_receipt_is_write_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = configure_persistent_compilation_cache(
        SimpleNamespace(config=_Config()), _environment(cache)
    )
    path = tmp_path / "receipt.json"
    monkeypatch.setenv(CACHE_RECEIPT_ENV, str(path))

    assert write_cache_receipt(receipt) == path
    assert json.loads(path.read_text()) == receipt
    with pytest.raises(FileExistsError):
        write_cache_receipt(receipt)


def test_maxtext_entrypoint_configures_cache_before_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt_path = tmp_path / "entrypoint-receipt.json"
    for name, value in _environment(cache).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(CACHE_RECEIPT_ENV, str(receipt_path))
    fake_jax = SimpleNamespace(config=_Config())
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    observed: dict[str, object] = {}

    def fake_run_module(name: str, *, run_name: str, alter_sys: bool) -> None:
        observed.update(
            {
                "name": name,
                "run_name": run_name,
                "alter_sys": alter_sys,
                "argv": list(sys.argv),
                "config": dict(fake_jax.config.values),
            }
        )

    monkeypatch.setattr(maxtext_entrypoint.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entrypoint",
            "maxtext.example",
            "config.yml",
            "steps=1",
            f"jax_cache_dir={cache}",
            "dump_hlo=false",
        ],
    )

    maxtext_entrypoint.main()

    assert observed["name"] == "maxtext.example"
    assert observed["argv"] == [
        "maxtext.example",
        "config.yml",
        "steps=1",
        f"jax_cache_dir={cache}",
        "dump_hlo=false",
    ]
    assert observed["config"] == json.loads(receipt_path.read_text())["config"]
