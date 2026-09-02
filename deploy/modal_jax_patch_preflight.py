"""Finite CPU regression for MaxText's native NNX LoRA state factory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import modal

from deploy.modal_jax_image import (
    BOOKFORGE_SOURCE_MANIFEST,
    BOOKFORGE_SOURCE_MANIFEST_SHA256,
    JAX_IMAGE,
    MAXTEXT_NATIVE_LORA_PATCH,
    REPOSITORY_ROOT,
)

APP_NAME = "bookforge-jax-native-lora-cpu-preflight"
MAXTEXT_ROOT = Path("/opt/MaxText")
MAXTEXT_PATCH = Path(
    "/opt/bookforge/patches/maxtext-native-lora-materialization.patch"
)
UPSTREAM_TINY_LORA_TEST = (
    MAXTEXT_ROOT / "tests/integration/lora_e2e_nnx_test.py"
)
EXPECTED_CPU_DEVICES = 2
TIMEOUT_SECONDS = 600
BUDGET_MONTH = "2026-09"
GROSS_CEILING_USD = 0.25
WORKSPACE_HARD_STOP_USD = 28.0
LEDGER_PATH = (
    REPOSITORY_ROOT
    / "experiments/jax-fidelity-lab/modal-ledger-2026-09.json"
)
_ATTEMPT_ID = re.compile(r"jax-native-lora-cpu-preflight-20\d{6}-v\d+\Z")
_RECEIPT_SCHEMA = "bookforge-jax-native-lora-cpu-preflight-receipt-v1"

app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _write_preflight_receipt(
    ledger_path: Path,
    *,
    attempt_id: str,
    document: dict[str, object],
) -> dict[str, object]:
    directory = ledger_path.parent / f"{ledger_path.name}.receipts"
    if directory.is_symlink():
        raise RuntimeError("CPU preflight receipt directory may not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / f"{attempt_id}.json"
    payload = _canonical_json_bytes(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o400)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return {
        "path": path.relative_to(ledger_path.parent).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _preflight_receipt_document(
    *,
    attempt_id: str,
    ledger_attempt_id: str,
    started_at: str,
    finished_at: str,
    result: dict[str, object] | None,
    error: dict[str, object] | None,
    patch_sha256: str,
    workspace_before_usd: float,
    workspace_after_usd: float | None,
    postrun_report_error: str | None,
) -> dict[str, object]:
    return {
        "schema_version": _RECEIPT_SCHEMA,
        "producer": "bookforge-modal-jax-native-lora-cpu-preflight",
        "status": "succeeded" if error is None else "failed",
        "attempt_id": attempt_id,
        "ledger_attempt_id": ledger_attempt_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "source_manifest": {
            "sha256": BOOKFORGE_SOURCE_MANIFEST_SHA256,
            "schema_version": BOOKFORGE_SOURCE_MANIFEST["schema_version"],
            "file_count": BOOKFORGE_SOURCE_MANIFEST["file_count"],
            "files_sha256": BOOKFORGE_SOURCE_MANIFEST["files_sha256"],
        },
        "maxtext_patch": {
            "path": MAXTEXT_NATIVE_LORA_PATCH.relative_to(REPOSITORY_ROOT).as_posix(),
            "bytes": MAXTEXT_NATIVE_LORA_PATCH.stat().st_size,
            "sha256": patch_sha256,
        },
        "result": result,
        "error": error,
        "workspace_billing_observation": {
            "before_usd": workspace_before_usd,
            "after_usd": workspace_after_usd,
            "postrun_report_error": postrun_report_error,
        },
        "provider_settlement": {
            "status": "pending-provider-app-cost",
            "ledger_attempt_id": ledger_attempt_id,
            "ledger_field": "billing.settlements[0]",
        },
    }


def preflight_approval_token(attempt_id: str, patch_sha256: str) -> str:
    return f"APPROVE_MODAL_JAX_NATIVE_LORA_CPU_PREFLIGHT:{attempt_id}:{patch_sha256}"


def _authoritative_workspace_total() -> float:
    completed = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Modal billing report was not a list")
    costs: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Modal billing report entry was not an object")
        current = row.get("cost")
        legacy = row.get("Cost")
        if current is None and legacy is None:
            raise RuntimeError("Modal billing report entry had no cost")
        if current is not None and legacy is not None and str(current) != str(legacy):
            raise RuntimeError("Modal billing report entry had conflicting costs")
        cost = float(current if current is not None else legacy)
        if not math.isfinite(cost) or cost < 0:
            raise RuntimeError("Modal billing report entry had an invalid cost")
        costs.append(cost)
    return math.fsum(costs)


def _path_text(path: object) -> str:
    if isinstance(path, tuple):
        return "/".join(str(part) for part in path)
    return str(path)


def _canonical_state_entries(
    entries: list[tuple[object, object]], *, label: str
) -> tuple[dict[tuple[str, ...], object], dict[tuple[str, ...], tuple[tuple[str, str, str], ...]]]:
    values: dict[tuple[str, ...], object] = {}
    typed_paths: dict[tuple[str, ...], tuple[tuple[str, str, str], ...]] = {}
    for path, variable in entries:
        components = path if isinstance(path, tuple) else (path,)
        canonical = tuple(str(component) for component in components)
        signature = tuple(
            (
                type(component).__module__,
                type(component).__qualname__,
                repr(component),
            )
            for component in components
        )
        if canonical in typed_paths and typed_paths[canonical] != signature:
            raise RuntimeError(f"{label} has a typed path collision: {canonical}")
        if canonical in values:
            raise RuntimeError(f"{label} has a duplicate path: {canonical}")
        values[canonical] = variable.get_value()
        typed_paths[canonical] = signature
    return values, typed_paths


def _optimizer_lora_moment_count(
    lora_entries: list[tuple[object, object]],
    optimizer_entries: list[tuple[object, object]],
    *,
    jax_module: object,
    jnp_module: object,
) -> tuple[int, int]:
    arrays = jax_module.Array
    named_sharding = jax_module.sharding.NamedSharding
    lora_values, lora_typed_paths = _canonical_state_entries(
        lora_entries, label="LoRA state"
    )
    if not lora_values:
        raise RuntimeError("LoRA state is empty")
    for path, value in lora_values.items():
        if (
            not isinstance(value, arrays)
            or not jnp_module.issubdtype(value.dtype, jnp_module.floating)
            or not bool(jax_module.device_get(jnp_module.all(jnp_module.isfinite(value))))
            or not isinstance(value.sharding, named_sharding)
        ):
            raise RuntimeError(f"LoRA tensor is not a finite sharded float array: {path}")

    optimizer_values, optimizer_typed_paths = _canonical_state_entries(
        optimizer_entries, label="optimizer state"
    )
    optimizer_array_count = sum(hasattr(value, "shape") for value in optimizer_values.values())
    observed: dict[tuple[str, tuple[str, ...]], object] = {}
    for parts, value in optimizer_values.items():
        if not (parts and parts[0] == "opt_state" and any(part in ("mu", "nu") for part in parts)):
            continue
        if (
            len(parts) < 4
            or parts[:2] != ("opt_state", "0")
            or parts[2] not in ("mu", "nu")
            or parts[3:] not in lora_values
            or optimizer_typed_paths[parts][3:] != lora_typed_paths[parts[3:]]
        ):
            raise RuntimeError(f"optimizer has an unexpected Adam moment path: {parts}")
        key = (parts[2], parts[3:])
        if key in observed:
            raise RuntimeError(f"optimizer has a duplicate Adam moment: {key}")
        adapter = lora_values[parts[3:]]
        if (
            not isinstance(value, arrays)
            or not jnp_module.issubdtype(value.dtype, jnp_module.floating)
            or not bool(jax_module.device_get(jnp_module.all(jnp_module.isfinite(value))))
            or value.shape != adapter.shape
            or not isinstance(value.sharding, named_sharding)
            or value.sharding.mesh != adapter.sharding.mesh
            or value.sharding.spec != adapter.sharding.spec
            or value.sharding.memory_kind != adapter.sharding.memory_kind
        ):
            raise RuntimeError(f"optimizer moment is incompatible with its adapter: {key}")
        observed[key] = value
    expected = {
        (moment, path) for path in lora_values for moment in ("mu", "nu")
    }
    if set(observed) != expected:
        raise RuntimeError("optimizer moments do not map one-to-one to every LoRA tensor")
    return len(observed), optimizer_array_count


def _prepare_cpu_runtime() -> None:
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before the CPU topology was fixed")
    os.environ.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": (
                f"--xla_force_host_platform_device_count={EXPECTED_CPU_DEVICES}"
            ),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    os.environ.pop("BOOKFORGE_EXPECTED_LORA_PAIR_COUNT", None)


def _validate_upstream_fixture() -> None:
    source = UPSTREAM_TINY_LORA_TEST.read_text(encoding="utf-8")
    required = (
        "def _tiny_lora_pyconfig(",
        '"pure_nnx": True',
        '"dataset_type": "synthetic"',
        'self._run_e2e_flow("gemma4-26b", use_sft=False)',
    )
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError(f"pinned MaxText tiny LoRA fixture drifted: {missing}")


@app.function(
    image=JAX_IMAGE,
    cpu=2,
    memory=8_192,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
)
def native_lora_trace_preflight() -> dict[str, object]:
    """Trace and initialize a real tiny Gemma4 LoRA state on two CPU devices."""

    started = time.monotonic()
    _prepare_cpu_runtime()
    _validate_upstream_fixture()

    sys.path.insert(0, str(MAXTEXT_ROOT))

    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from maxtext.common import checkpointing, train_state_nnx
    from maxtext.utils import train_utils
    from tests.integration.lora_e2e_nnx_test import (
        _tiny_lora_pyconfig,
    )

    from training.jax_fidelity.orbax_receipt import (
        lora_checkpoint_storage_evidence,
    )
    from training.jax_fidelity.runtime import (
        validate_maxtext_import_provenance,
    )

    devices = jax.devices("cpu")
    if len(devices) != EXPECTED_CPU_DEVICES:
        raise RuntimeError(
            f"expected {EXPECTED_CPU_DEVICES} virtual CPU devices, got {len(devices)}"
        )
    if any(device.platform != "cpu" for device in devices):
        raise RuntimeError("native LoRA trace preflight allocated a non-CPU device")

    imports = validate_maxtext_import_provenance(MAXTEXT_ROOT)
    with tempfile.TemporaryDirectory(prefix="bookforge-native-lora-preflight-") as output:
        config = _tiny_lora_pyconfig(
            run_name="bookforge_native_lora_cpu_preflight",
            checkpoint_dir=output,
            model_name="gemma4-26b",
            hardware="cpu",
            dtype="bfloat16",
            weight_dtype="bfloat16",
            scan_layers=False,
            steps=1,
            enable_checkpointing=False,
            skip_jax_distributed_system=True,
            ici_fsdp_parallelism=-1,
            ici_data_parallelism=1,
            sharding_tolerance=1.0,
            lora={
                "enable_lora": True,
                "lora_rank": 4,
                "lora_alpha": 8.0,
                "lora_module_path": (
                    r"decoder/layers_[0-9]+/self_attention/(query|key|value|out)"
                ),
            },
        )
        setup = train_utils.setup_train_loop(config, recorder=None, devices=devices)

    state_mesh_shardings = setup[2]
    mesh = setup[4]
    state = setup[-1]
    mesh_shape = {str(name): int(size) for name, size in mesh.shape.items()}
    if mesh_shape.get("fsdp") != EXPECTED_CPU_DEVICES or mesh_shape.get("data") != 1:
        raise RuntimeError(f"tiny Gemma4 did not use the production FSDP mesh: {mesh_shape}")

    lora_entries = list(nnx.state(state.model, nnx.LoRAParam).flat_state())
    planned_lora_entries = list(
        nnx.filter_state(state_mesh_shardings.model, nnx.LoRAParam).flat_state()
    )
    lora_paths = [_path_text(path) for path, _ in lora_entries]
    planned_paths = [_path_text(path) for path, _ in planned_lora_entries]
    if not lora_entries or lora_paths != planned_paths:
        raise RuntimeError("concrete and planned LoRA topologies do not match")

    sharding_specs: set[str] = set()
    trainable_elements = 0
    for (path, variable), (planned_path, planned_variable) in zip(
        lora_entries, planned_lora_entries, strict=True
    ):
        if path != planned_path:
            raise RuntimeError("LoRA path order changed between planned and concrete state")
        value = variable.get_value()
        planned = planned_variable.get_value()
        if not isinstance(value, jax.Array):
            raise RuntimeError(f"LoRA value is not concrete at {_path_text(path)}")
        if not isinstance(value.sharding, jax.sharding.NamedSharding):
            raise RuntimeError(f"LoRA value has no NamedSharding at {_path_text(path)}")
        if not isinstance(planned, jax.sharding.NamedSharding):
            raise RuntimeError(f"LoRA plan has no NamedSharding at {_path_text(path)}")
        if value.sharding.spec != planned.spec:
            raise RuntimeError(f"LoRA sharding differs from its plan at {_path_text(path)}")
        trainable_elements += value.size
        sharding_specs.add(str(planned.spec))

    optimizer_entries = list(nnx.state(state.optimizer).flat_state())
    optimizer_lora_tensor_count, optimizer_array_count = _optimizer_lora_moment_count(
        lora_entries,
        optimizer_entries,
        jax_module=jax,
        jnp_module=jax.numpy,
    )

    checkpoint_state = train_state_nnx.to_checkpoint_dict(nnx.state(state))
    filtered_checkpoint = checkpointing._filter_lora_trainable_state(checkpoint_state)
    if not isinstance(filtered_checkpoint, dict):
        raise RuntimeError("MaxText produced no LoRA checkpoint state")
    with tempfile.TemporaryDirectory(
        prefix="bookforge-native-lora-orbax-preflight-"
    ) as checkpoint_root:
        checkpoint_items = Path(checkpoint_root) / "items"
        ocp.PyTreeCheckpointer().save(str(checkpoint_items), filtered_checkpoint)
        checkpoint_evidence = lora_checkpoint_storage_evidence(
            checkpoint_items,
            expected_pair_count=len(lora_entries) // 2,
        )
    if (
        checkpoint_evidence["lora_tensor_count"] != len(lora_entries)
        or checkpoint_evidence["optimizer_lora_tensor_count"]
        != optimizer_lora_tensor_count
    ):
        raise RuntimeError("saved Orbax LoRA census differs from native state")

    payload: dict[str, object] = {
        "schema_version": "bookforge-jax-native-lora-cpu-preflight-v1",
        "status": "passed",
        "backend": "modal-cpu",
        "accelerator_count": 0,
        "cpu_device_count": len(devices),
        "mesh_shape": mesh_shape,
        "model_name": "gemma4-26b",
        "dtype": "bfloat16",
        "scan_layers": False,
        "lora_tensor_count": len(lora_entries),
        "trainable_elements": trainable_elements,
        "optimizer_array_count": optimizer_array_count,
        "optimizer_lora_tensor_count": optimizer_lora_tensor_count,
        "optimizer_overhead_array_count": (
            optimizer_array_count - optimizer_lora_tensor_count
        ),
        "checkpoint_filter_roundtrip_passed": True,
        "checkpoint_tree_leaf_count": checkpoint_evidence["tree_leaf_count"],
        "checkpoint_restored_lora_array_count": checkpoint_evidence[
            "restored_lora_array_count"
        ],
        "lora_sharding_specs": sorted(sharding_specs),
        "maxtext_patch_sha256": _sha256(MAXTEXT_PATCH),
        "maxtext_imports": imports,
        "nested_eval_shape_trace_passed": True,
        "jitted_concrete_initialization_passed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


@app.local_entrypoint()
def run_cli(attempt_id: str, approval_token_value: str) -> None:
    from infra.gcp.jax.modal_reconciliation import (
        append_reconciliation,
        assert_attempt_available,
        reserve_attempt,
    )

    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise RuntimeError("CPU preflight attempt ID is invalid")
    patch_sha256 = _sha256(MAXTEXT_NATIVE_LORA_PATCH)
    expected = preflight_approval_token(attempt_id, patch_sha256)
    if approval_token_value != expected:
        raise RuntimeError("exact CPU preflight approval token is required")
    if datetime.now(UTC).strftime("%Y-%m") != BUDGET_MONTH:
        raise RuntimeError("CPU preflight is outside its approved budget month")
    workspace_before = _authoritative_workspace_total()
    if workspace_before + GROSS_CEILING_USD > WORKSPACE_HARD_STOP_USD:
        raise RuntimeError("Modal workspace budget has insufficient CPU-preflight headroom")

    ledger_attempt_id = f"jax-cpu-preflight:{attempt_id}"
    assert_attempt_available(LEDGER_PATH, attempt_id=ledger_attempt_id)
    reserve_attempt(
        LEDGER_PATH,
        attempt_id=ledger_attempt_id,
        stage="jax-native-lora-cpu-preflight",
    )
    result: dict[str, object] | None = None
    remote_error: BaseException | None = None
    error_evidence: dict[str, object] | None = None
    status = "remote-error"
    workspace_after: float | None = None
    report_error: str | None = None
    started_at = _utc_now()
    try:
        candidate = native_lora_trace_preflight.remote()
        if not isinstance(candidate, dict):
            raise RuntimeError("Modal CPU preflight returned no result")
        result = candidate
        if result.get("status") != "passed":
            raise RuntimeError("Modal CPU preflight did not return a passing result")
        if result.get("maxtext_patch_sha256") != patch_sha256:
            raise RuntimeError("Modal CPU preflight patch identity changed")
        status = "succeeded"
    except BaseException as error:
        remote_error = error
        error_evidence = {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    try:
        workspace_after = _authoritative_workspace_total()
    except Exception as error:
        report_error = f"{type(error).__name__}: {error}"
    finished_at = _utc_now()
    try:
        receipt = _preflight_receipt_document(
            attempt_id=attempt_id,
            ledger_attempt_id=ledger_attempt_id,
            started_at=started_at,
            finished_at=finished_at,
            result=result,
            error=error_evidence,
            patch_sha256=patch_sha256,
            workspace_before_usd=workspace_before,
            workspace_after_usd=workspace_after,
            postrun_report_error=report_error,
        )
        receipt_binding = _write_preflight_receipt(
            LEDGER_PATH,
            attempt_id=attempt_id,
            document=receipt,
        )
        append_reconciliation(
            LEDGER_PATH,
            attempt_id=ledger_attempt_id,
            stage="jax-native-lora-cpu-preflight",
            workspace_before_usd=workspace_before,
            workspace_after_usd=workspace_after,
            declared_ceiling_usd=GROSS_CEILING_USD,
            status=status,
            result=result,
            postrun_report_error=report_error,
            evidence={"cpu_preflight_receipt": receipt_binding},
        )
    except BaseException as evidence_error:
        if remote_error is None:
            raise
        remote_error.add_note(
            "CPU preflight receipt or reconciliation also failed: "
            f"{type(evidence_error).__name__}: {evidence_error}"
        )
    if remote_error is not None:
        raise remote_error
    assert result is not None
    print(json.dumps(result, indent=2, sort_keys=True))
