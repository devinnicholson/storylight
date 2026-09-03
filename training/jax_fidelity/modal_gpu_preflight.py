"""Verify the exact two-GPU MaxText FSDP mesh used by Modal JAX jobs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import load_config


def run_two_gpu_fsdp_preflight(
    *,
    config_path: Path,
    maxtext_root: Path,
    environment: Mapping[str, str],
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    """Return measured device and mesh evidence or fail before training."""

    experiment = load_config(config_path)
    script = "\n".join(
        [
            "import json, os, sys",
            "import jax",
            "from pathlib import Path",
            "from training.jax_fidelity.compilation_cache import (",
            "    cache_inventory, configure_persistent_compilation_cache)",
            "cache_config = configure_persistent_compilation_cache(jax, os.environ)",
            "from maxtext.configs import pyconfig",
            "config = pyconfig.initialize(sys.argv)",
            "import jax.numpy as jnp",
            "post_config = configure_persistent_compilation_cache(jax, os.environ)",
            "if post_config != cache_config:",
            "    raise RuntimeError('MaxText initialization changed the JAX cache config')",
            "memory_fraction = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')",
            "if memory_fraction != '0.95':",
            "    raise RuntimeError(f'unexpected JAX memory fraction: {memory_fraction!r}')",
            "devices = jax.devices()",
            "if len(devices) != 2 or any(device.platform != 'gpu' for device in devices):",
            "    raise RuntimeError(f'expected two GPUs, found {devices!r}')",
            "if config.ici_fsdp_parallelism != -1:",
            "    raise RuntimeError('MaxText FSDP auto-sharding is disabled')",
            "from maxtext.utils import maxtext_utils",
            "mesh = maxtext_utils.create_device_mesh(config, devices)",
            "mesh_shape = dict(zip(config.mesh_axes, mesh.shape, strict=True))",
            "if mesh_shape.get('fsdp') != 2:",
            "    raise RuntimeError(f'expected a two-way FSDP mesh, found {mesh_shape!r}')",
            "value = jax.device_get(jnp.arange(1024, dtype=jnp.bfloat16).sum())",
            "cache_probe = jax.jit(lambda operand: jnp.sin(operand) + jnp.cos(operand))(",
            "    jnp.ones((4096, 4096), dtype=jnp.float32))",
            "cache_probe.block_until_ready()",
            "if cache_config['configured']:",
            "    cache_config['inventory'] = cache_inventory(",
            "        Path(cache_config['cache_directory']))",
            "    if cache_config['inventory']['files'] <= 0:",
            "        raise RuntimeError('JAX preflight wrote no persistent cache entries')",
            "print(json.dumps({'hardware': config.hardware, 'devices': len(devices), "
            "'platform': devices[0].platform, 'memory_fraction': memory_fraction, "
            "'ici_fsdp_parallelism': config.ici_fsdp_parallelism, 'mesh_shape': mesh_shape, "
            "'probe_sum': float(value), 'compilation_cache': cache_config}), flush=True)",
        ]
    )
    command = [
        "python3",
        "-c",
        script,
        str(maxtext_root / "src/maxtext/configs/base.yml"),
        f"model_name={experiment.production['maxtext_model_name']}",
        "hardware=gpu",
        "skip_jax_distributed_system=true",
        f"use_multimodal={str(experiment.production['use_multimodal']).lower()}",
        f"scan_layers={str(experiment.production['scan_layers']).lower()}",
        "enable_checkpointing=false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=maxtext_root,
            env=dict(environment),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "no subprocess output").strip()
        raise RuntimeError(f"GPU configuration preflight failed:\n{detail}") from error
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("GPU configuration preflight returned invalid evidence") from error
    cache = payload.get("compilation_cache") if isinstance(payload, dict) else None
    expected_cache_directory = environment.get("JAX_COMPILATION_CACHE_DIR")
    if (
        not isinstance(payload, dict)
        or payload.get("hardware") != "gpu"
        or payload.get("devices") != 2
        or payload.get("platform") != "gpu"
        or payload.get("memory_fraction") != "0.95"
        or payload.get("ici_fsdp_parallelism") != -1
        or not isinstance(payload.get("mesh_shape"), dict)
        or payload["mesh_shape"].get("fsdp") != 2
        or not isinstance(cache, dict)
        or cache.get("configured") is not bool(expected_cache_directory)
    ):
        raise RuntimeError("GPU configuration preflight evidence changed")
    if expected_cache_directory:
        inventory = cache.get("inventory")
        if (
            cache.get("cache_directory") != expected_cache_directory
            or not isinstance(inventory, dict)
            or type(inventory.get("files")) is not int
            or int(inventory["files"]) <= 0
            or type(inventory.get("bytes")) is not int
            or int(inventory["bytes"]) <= 0
        ):
            raise RuntimeError("GPU preflight did not prove the persistent cache")
    return payload
