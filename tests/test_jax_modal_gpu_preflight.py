from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from training.jax_fidelity import modal_gpu_preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"


def _result(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["python3"],
        0,
        stdout="MaxText log\n" + json.dumps(payload) + "\n",
        stderr="",
    )


def test_two_gpu_preflight_returns_measured_fsdp_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "hardware": "gpu",
        "devices": 2,
        "platform": "gpu",
        "memory_fraction": "0.95",
        "ici_fsdp_parallelism": -1,
        "mesh_shape": {"fsdp": 2},
        "probe_sum": 523776.0,
        "compilation_cache": {
            "schema_version": "bookforge-jax-cache-runtime-v1",
            "configured": False,
        },
    }
    monkeypatch.setattr(
        modal_gpu_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: _result(payload),
    )

    result = modal_gpu_preflight.run_two_gpu_fsdp_preflight(
        config_path=CONFIG,
        maxtext_root=tmp_path,
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"},
    )

    assert result == payload


def test_two_gpu_preflight_rejects_non_fsdp_mesh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "hardware": "gpu",
        "devices": 2,
        "platform": "gpu",
        "memory_fraction": "0.95",
        "ici_fsdp_parallelism": -1,
        "mesh_shape": {"fsdp": 1, "tensor": 2},
        "probe_sum": 523776.0,
        "compilation_cache": {
            "schema_version": "bookforge-jax-cache-runtime-v1",
            "configured": False,
        },
    }
    monkeypatch.setattr(
        modal_gpu_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: _result(payload),
    )

    with pytest.raises(RuntimeError, match="evidence changed"):
        modal_gpu_preflight.run_two_gpu_fsdp_preflight(
            config_path=CONFIG,
            maxtext_root=tmp_path,
            environment={},
        )
