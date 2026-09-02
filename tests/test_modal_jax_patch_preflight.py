from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "deploy/modal_jax_patch_preflight.py"
IMAGE_DEFINITION = ROOT / "deploy/modal_jax_image.py"
MAXTEXT_PATCH = (
    ROOT
    / "training/jax_fidelity/patches/maxtext-native-lora-materialization.patch"
)
sys.path.insert(0, str(ROOT))

from infra.gcp.jax import modal_reconciliation  # noqa: E402


def _load():
    name = "bookforge_modal_jax_patch_preflight"
    spec = importlib.util.spec_from_file_location(name, PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_native_lora_patch_preflight_is_finite_cpu_only_and_real() -> None:
    preflight = _load()
    source = PREFLIGHT.read_text(encoding="utf-8")

    assert preflight.EXPECTED_CPU_DEVICES == 2
    assert preflight.TIMEOUT_SECONDS == 600
    assert 'XLA_FLAGS": (' in source
    assert "--xla_force_host_platform_device_count=" in source
    assert '"JAX_PLATFORMS": "cpu"' in source
    assert "cpu=2" in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "gpu=" not in source
    assert "@modal.web_endpoint" not in source

    assert "tests/integration/lora_e2e_nnx_test.py" in source
    assert "_tiny_lora_pyconfig" in source
    assert "train_utils.setup_train_loop" in source
    assert "model_name=\"gemma4-26b\"" in source
    assert "nnx.state(state.model, nnx.LoRAParam)" in source
    assert "nnx.filter_state(state_mesh_shardings.model, nnx.LoRAParam)" in source
    assert "value.sharding.spec != planned.spec" in source
    assert "_optimizer_lora_moment_count" in source
    assert 'parts[:2] != ("opt_state", "0")' in source
    assert 'parts[2] not in ("mu", "nu")' in source
    assert '"nested_eval_shape_trace_passed": True' in source
    assert '"jitted_concrete_initialization_passed": True' in source
    assert "GROSS_CEILING_USD = 0.25" in source
    assert "WORKSPACE_HARD_STOP_USD = 28.0" in source
    assert "reserve_attempt(" in source
    assert "append_reconciliation(" in source
    assert "_write_preflight_receipt(" in source
    assert 'evidence={"cpu_preflight_receipt": receipt_binding}' in source


def test_native_lora_patch_refuses_gradient_accumulation() -> None:
    source = MAXTEXT_PATCH.read_text(encoding="utf-8")

    assert 'getattr(config, "gradient_accumulation_steps", 1) != 1' in source
    assert "native LoRA evidence requires gradient_accumulation_steps=1" in source


def test_native_lora_patch_materializes_after_model_construction() -> None:
    source = MAXTEXT_PATCH.read_text(encoding="utf-8")

    assert "@@ -269,0 +272,7 @@ def setup_train_loop" in source
    assert "@@ -268,0 +271,7 @@ def setup_train_loop" not in source


def test_cpu_runtime_is_fixed_before_jax_import(monkeypatch) -> None:
    preflight = _load()
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    for name in (
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOOKFORGE_EXPECTED_LORA_PAIR_COUNT", "205")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")

    preflight._prepare_cpu_runtime()

    assert preflight.os.environ["JAX_PLATFORMS"] == "cpu"
    assert preflight.os.environ["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=2"
    )
    assert "BOOKFORGE_EXPECTED_LORA_PAIR_COUNT" not in preflight.os.environ


def test_cpu_preflight_billing_total_accepts_one_unambiguous_cost_alias(
    monkeypatch,
) -> None:
    preflight = _load()
    report = '[{"cost":"0.125"},{"Cost":"0.375"}]'
    observed: list[tuple[object, object]] = []

    class Completed:
        stdout = report

    def fake_run(argv, **kwargs):
        observed.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    assert preflight._authoritative_workspace_total() == 0.5
    assert observed == [
        (
            ["modal", "billing", "report", "--for", "this month", "--json"],
            {"check": True, "capture_output": True, "text": True, "timeout": 60},
        )
    ]
    with pytest.raises(RuntimeError, match="conflicting"):
        Completed.stdout = '[{"cost":"0.1","Cost":"0.2"}]'
        preflight._authoritative_workspace_total()


def test_shared_image_excludes_mutable_python_cache_artifacts() -> None:
    source = IMAGE_DEFINITION.read_text(encoding="utf-8")

    assert '"**/__pycache__/**"' in source
    assert '"**/*.pyc"' in source
    assert '"**/*.pyo"' in source
    assert source.count("ignore=LOCAL_SOURCE_IGNORE") == 4


def test_cpu_preflight_receipt_is_immutable_complete_and_ledger_bound(
    tmp_path: Path,
) -> None:
    preflight = _load()
    ledger = tmp_path / "ledger.json"
    attempt_id = "jax-native-lora-cpu-preflight-20260902-v99"
    ledger_attempt_id = f"jax-cpu-preflight:{attempt_id}"
    error = {
        "type": "RuntimeError",
        "message": "trace failed",
        "repr": "RuntimeError('trace failed')",
        "traceback": "Traceback fixture",
    }
    receipt = preflight._preflight_receipt_document(
        attempt_id=attempt_id,
        ledger_attempt_id=ledger_attempt_id,
        started_at="2026-09-02T01:02:03Z",
        finished_at="2026-09-02T01:02:04Z",
        result={"partial": True},
        error=error,
        patch_sha256=hashlib.sha256(preflight.MAXTEXT_NATIVE_LORA_PATCH.read_bytes()).hexdigest(),
        workspace_before_usd=1.0,
        workspace_after_usd=None,
        postrun_report_error="billing unavailable",
    )
    binding = preflight._write_preflight_receipt(
        ledger, attempt_id=attempt_id, document=receipt
    )
    receipt_path = ledger.parent / binding["path"]

    assert receipt["status"] == "failed"
    assert receipt["result"] == {"partial": True}
    assert receipt["error"] == error
    assert receipt["workspace_billing_observation"] == {
        "before_usd": 1.0,
        "after_usd": None,
        "postrun_report_error": "billing unavailable",
    }
    assert receipt["source_manifest"]["sha256"] == (
        preflight.BOOKFORGE_SOURCE_MANIFEST_SHA256
    )
    assert receipt["maxtext_patch"]["sha256"] == hashlib.sha256(
        preflight.MAXTEXT_NATIVE_LORA_PATCH.read_bytes()
    ).hexdigest()
    assert receipt["provider_settlement"] == {
        "status": "pending-provider-app-cost",
        "ledger_attempt_id": ledger_attempt_id,
        "ledger_field": "billing.settlements[0]",
    }
    assert receipt_path.stat().st_mode & 0o777 == 0o400
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == binding["sha256"]
    assert len(receipt_path.read_bytes()) == binding["bytes"]
    with pytest.raises(FileExistsError):
        preflight._write_preflight_receipt(
            ledger, attempt_id=attempt_id, document=receipt
        )

    modal_reconciliation.append_reconciliation(
        ledger,
        attempt_id=ledger_attempt_id,
        stage="jax-native-lora-cpu-preflight",
        workspace_before_usd=0.0,
        workspace_after_usd=0.0,
        declared_ceiling_usd=0.25,
        status="remote-error",
        result=None,
        evidence={"cpu_preflight_receipt": binding},
    )
    entry = json.loads(ledger.read_text())["entries"][0]
    assert entry["evidence"]["cpu_preflight_receipt"] == binding


def test_optimizer_gate_requires_exact_mu_and_nu_for_each_lora_path() -> None:
    preflight = _load()

    class FakeSharding:
        def __init__(
            self,
            *,
            mesh: str = "mesh",
            spec: tuple[str, ...] = ("data",),
            memory_kind: str = "device",
        ) -> None:
            self.mesh = mesh
            self.spec = spec
            self.memory_kind = memory_kind

    class FakeArray:
        def __init__(
            self,
            *,
            shape: tuple[int, ...] = (1,),
            kind: str = "f",
            finite: bool = True,
            sharding: FakeSharding | None = None,
        ) -> None:
            self.shape = shape
            self.dtype = SimpleNamespace(kind=kind)
            self.finite = finite
            self.sharding = sharding or FakeSharding()

    fake_jax = SimpleNamespace(
        Array=FakeArray,
        sharding=SimpleNamespace(NamedSharding=FakeSharding),
        device_get=lambda value: value,
    )
    fake_jnp = SimpleNamespace(
        floating=object(),
        issubdtype=lambda dtype, _expected: dtype.kind == "f",
        isfinite=lambda value: value.finite,
        all=lambda value: value,
    )

    def variable(value: object | None = None) -> SimpleNamespace:
        return SimpleNamespace(get_value=lambda: value or FakeArray())

    def census(
        lora: list[tuple[object, object]],
        optimizer: list[tuple[object, object]],
    ) -> tuple[int, int]:
        return preflight._optimizer_lora_moment_count(
            lora,
            optimizer,
            jax_module=fake_jax,
            jnp_module=fake_jnp,
        )

    lora = [(('decoder', 'query', 'kernel_lora_a'), variable())]
    complete = [
        (('opt_state', 0, moment, 'decoder', 'query', 'kernel_lora_a'), variable())
        for moment in ('mu', 'nu')
    ]
    complete.append((('opt_state', 0, 'count'), variable()))

    assert census(lora, complete) == (2, 3)
    for malformed in (
        complete[:1],
        [complete[0], complete[0]],
        [complete[0], (('opt_state', 1, 'nu', 'decoder', 'query', 'kernel_lora_a'), variable())],
        [
            complete[0],
            complete[1],
            (('opt_state', 0, 'mu', 'extra'), variable()),
        ],
    ):
        with pytest.raises(RuntimeError, match="moment|one-to-one|duplicate"):
            census(lora, malformed)

    for bad_value in (
        FakeArray(shape=(99,)),
        FakeArray(kind="i"),
        FakeArray(finite=False),
        FakeArray(sharding=FakeSharding(spec=("model",))),
        SimpleNamespace(shape=(1,)),
    ):
        malformed = [complete[0], (complete[1][0], variable(bad_value)), complete[2]]
        with pytest.raises(RuntimeError, match="incompatible"):
            census(lora, malformed)

    typed_collision = [
        (('decoder', 1, 'kernel_lora_a'), variable()),
        (('decoder', '1', 'kernel_lora_a'), variable()),
    ]
    with pytest.raises(RuntimeError, match="typed path collision"):
        census(typed_collision, complete)
