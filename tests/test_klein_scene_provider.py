import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_finite_modal_provider import _jpeg

from storylight import klein_scene_provider as klein
from storylight.finite_modal_provider import (
    DEPTH_MODEL,
    DEPTH_MODEL_REVISION,
    FastSceneRequest,
    FiniteModalBudgetError,
    FiniteModalProviderError,
)
from storylight.klein_scene_provider import (
    MODEL,
    REVISION,
    KleinInvoker,
    KleinSceneProvider,
    write_bundle,
)


def request(**kw):
    return FastSceneRequest(
        scene_id="scene-test", prompt="a synthetic rabbit", steps=4, guidance_scale=1.0, **kw
    )


def payload():
    content = _jpeg(1024, 576)
    return {
        "identity": {
            "model": MODEL,
            "model_revision": REVISION,
            "depth_model": DEPTH_MODEL,
            "depth_revision": DEPTH_MODEL_REVISION,
            "width": 1024,
            "height": 576,
            "steps": 4,
            "guidance": 1.0,
            "gpu": "NVIDIA L4",
        },
        "metrics": {
            "seed": 42,
            "sequence_bucket": 128,
            "token_count": 50,
            "image_seconds": 1.6,
            "depth_seconds": 0.03,
            "encoding_seconds": 0.02,
            "total_seconds": 1.65,
            "master_sha256": hashlib.sha256(content).hexdigest(),
            "depth_sha256": hashlib.sha256(content).hexdigest(),
        },
        "master": content,
        "depth": content,
        "warm_state": "warm",
        "model_load_seconds": 7,
        "startup_seconds": 8,
        "cache_setup_seconds": 0.1,
    }


def test_klein_bundle_checks_bytes_and_records_real_models(tmp_path):
    bundle = write_bundle(request(), tmp_path / "scene", payload(), 2, "reservation")
    assert bundle.master.path.read_bytes() == payload()["master"]
    assert bundle.manifest["stages"]["fast"]["model"] == MODEL
    assert bundle.manifest["policy"]["visual_acceptance"] == "human-review-required"


@pytest.mark.parametrize("corrupt", ["hash", "revision"])
def test_corrupt_response_never_publishes_assets(tmp_path, corrupt):
    data = payload()
    if corrupt == "hash":
        data["master"] += b"changed"
    if corrupt == "dimensions":
        data["master"] = _jpeg(512, 288)
        data["metrics"]["master_sha256"] = hashlib.sha256(data["master"]).hexdigest()
    if corrupt == "revision":
        data["identity"]["model_revision"] = "other"
    if corrupt == "seed":
        data["metrics"]["seed"] = 3
    if corrupt == "nan":
        data["metrics"]["image_seconds"] = float("nan")
    if corrupt == "bucket":
        data["metrics"]["sequence_bucket"] = 512
    with pytest.raises(FiniteModalProviderError):
        write_bundle(request(), tmp_path / "scene", data, 2, "r")
    assert not (tmp_path / "scene").exists()


def provider(tmp_path, calls, cap=0.5):
    plan = tmp_path / "budget.json"
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30,
                "workspace_usage_before_live_scenes_usd": 0,
                "billing_delay_reserve_usd": 1,
                "hard_stop_workspace_total_usd": 29,
                "maximum_new_spend_usd": 10,
            }
        )
    )

    async def invoke(operation, **kwargs):
        calls.append(operation)
        if operation == "prewarm":
            return {"model_load_seconds": 7, "warmup_seconds": 10}
        return payload()

    async def probe():
        return True, "metadata only"

    async def billing(*args):
        return 0

    return KleinSceneProvider(
        invoker=SimpleNamespace(invoke=invoke, probe=probe),
        plan_file=plan,
        ledger_path=tmp_path / "ledger.json",
        session_gpu_cap_usd=cap,
        billing_reader=billing,
    )


def test_call_budget_is_retained_across_retries(tmp_path):
    async def run():
        calls = []
        obj = provider(tmp_path, calls, cap=0.25)
        await obj.generate_fast(request(), output_dir=tmp_path / "first")
        with pytest.raises(FiniteModalBudgetError):
            await obj.generate_fast(request(), output_dir=tmp_path / "second")
        assert calls == ["generate"]

    asyncio.run(run())


def test_inline_gate_cannot_be_silently_bypassed(tmp_path):
    async def run():
        calls = []
        obj = provider(tmp_path, calls)
        with pytest.raises(ValueError):
            await obj.generate_fast(request(fidelity_label="rabbit"), output_dir=tmp_path / "first")
        assert not calls

    asyncio.run(run())


def test_session_authorization_runs_once_and_failure_keeps_reservation(tmp_path, monkeypatch):
    now = 1000.0
    monkeypatch.setattr(
        klein,
        "time",
        SimpleNamespace(
            monotonic=lambda: now,
            perf_counter=klein.time.perf_counter,
        ),
    )

    async def run():
        nonlocal now
        calls, reservations = [], []
        obj = provider(tmp_path, calls, cap=1)

        async def reserve(**kwargs):
            reservations.append(kwargs)
            return "session-reservation"

        obj._reserve_against_current_billing = reserve
        await obj.prewarm(prewarm_id="first")
        now += 30
        await obj.generate_fast(request(), output_dir=tmp_path / "first")
        assert obj.warm_deadline == 1090
        assert len(reservations) == 1
        assert reservations[0]["full_call_ceiling_usd"] == 1
        now = 1090
        assert (await obj.warm_status()).state == "idle"
        assert calls == ["prewarm", "generate"]

        async def fail(operation, **kwargs):
            calls.append(operation)
            raise RuntimeError("remote failure")

        obj.invoker.invoke = fail
        with pytest.raises(RuntimeError, match="remote failure"):
            await obj.prewarm(prewarm_id="failed")
        assert obj.reserved_usd == 0.75
        with pytest.raises(FiniteModalBudgetError, match="already attempted"):
            await obj.prewarm(prewarm_id="failed")
        assert (await obj.warm_status()).state == "idle"
        assert obj.prewarm_id is None and obj.warm_deadline == 0
        assert calls == ["prewarm", "generate", "prewarm"]
        assert len(reservations) == 1

    asyncio.run(run())


def test_concurrent_explicit_prewarm_reuses_original_receipt_and_remaining_time(
    tmp_path, monkeypatch
):
    now = 0.0
    monkeypatch.setattr(
        klein,
        "time",
        SimpleNamespace(
            monotonic=lambda: now,
            perf_counter=klein.time.perf_counter,
        ),
    )

    async def run():
        nonlocal now
        calls = []
        obj = provider(tmp_path, calls, cap=1)
        await obj.generate_fast(request(), output_dir=tmp_path / "first")
        assert (await obj.warm_status()).state == "idle"  # One image does not warm both buckets.
        started, release = asyncio.Event(), asyncio.Event()
        original = obj.invoker.invoke

        async def invoke(operation, **kwargs):
            started.set()
            await release.wait()
            return await original(operation, **kwargs)

        obj.invoker.invoke = invoke
        first = asyncio.create_task(obj.prewarm(prewarm_id="first"))
        await started.wait()
        second = asyncio.create_task(obj.prewarm(prewarm_id="second"))
        now = 20
        release.set()
        initial, reused = await asyncio.gather(first, second)
        assert initial == reused and reused.prewarm_id == "first"
        assert obj.warm_deadline == 110
        now = 30
        later = await obj.prewarm(prewarm_id="third")
        assert later.expires_in_seconds == 80 and obj.warm_deadline == 110
        assert later.reservation_id == initial.reservation_id
        assert calls == ["generate", "prewarm"] and obj.reserved_usd == 0.5
        assert obj.operations == {("generate", "scene-test"), ("prewarm", "first")}

        async def cancelled(operation, **kwargs):
            calls.append(operation)
            raise asyncio.CancelledError

        obj.invoker.invoke = cancelled
        with pytest.raises(asyncio.CancelledError):
            await obj.generate_fast(
                replace(request(), scene_id="cancelled"), output_dir=tmp_path / "cancelled"
            )
        assert (await obj.warm_status()).state == "idle"
        assert obj.reserved_usd == 0.75 and obj._prewarm_report is None
        obj.invoker.invoke = original
        recovered = await obj.prewarm(prewarm_id="explicit-recovery")
        assert recovered.prewarm_id == "explicit-recovery"
        assert calls == ["generate", "prewarm", "generate", "prewarm"]
        assert obj.reserved_usd == 1

    asyncio.run(run())


def test_cancelled_invocation_terminates_remote_container():
    async def run():
        started, cancelled = asyncio.Event(), []

        async def get():
            started.set()
            await asyncio.Event().wait()

        async def cancel(**kwargs):
            cancelled.append(kwargs)

        async def spawn(**kwargs):
            return SimpleNamespace(get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel))

        obj = KleinInvoker()
        obj.instance = SimpleNamespace(generate=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
        task = asyncio.create_task(obj.invoke("generate", prompt="synthetic", seed=1))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == [{"terminate_containers": True}]

    asyncio.run(run())
