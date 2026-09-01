import asyncio
import hashlib
import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import bookforge.finite_modal_provider as finite_modal_provider_module
from bookforge.asset_cache import AssetCache
from bookforge.domain import ModelMetrics
from bookforge.finite_modal_provider import (
    FAST_GPU,
    FAST_MODEL,
    FAST_MODEL_REVISION,
    FAST_STAGE_POLICY,
    MOTION_GPU,
    MOTION_MODEL,
    MOTION_MODEL_REVISION,
    WARM_FAST_GPU,
    WARM_FAST_PRESENTATION_SESSION_CEILING_USD,
    WARM_FAST_SESSION_CEILING_USD,
    WARM_FAST_STAGE_POLICY,
    WARM_FULL_SESSION_CEILING_USD,
    FastSceneRequest,
    FiniteModalBudgetError,
    FiniteModalLiveSceneProvider,
    FiniteModalProviderError,
    FiniteModalSceneProvider,
    FiniteSceneBundle,
    ModalSdkWarmInvoker,
    ModalStagePolicy,
    MotionTechnicalEvidence,
    MotionTechnicalGate,
    MotionUpgradeRequest,
    WarmModalSceneProvider,
    load_finite_scene_bundle,
)
from bookforge.live_scene import (
    LiveSceneArtifactKind,
    LiveSceneCostSource,
    LiveSceneCreateRequest,
    LiveSceneProviderUnavailableError,
    LiveSceneStage,
    LiveSceneUpdate,
    LiveSceneWarmState,
    build_live_scene_story_pack,
)
from bookforge.live_scene_planner import (
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlannerError,
    LiveScenePlanningResult,
)
from bookforge.modal_budget import budget_envelope_from_plan
from bookforge.visual_lab import VisualLabLedger


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _artifact(path: Path, content: bytes, mime_type: str) -> dict:
    path.write_bytes(content)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "mime_type": mime_type,
        "width": 1024 if mime_type == "image/png" else 800,
        "height": 576 if mime_type == "image/png" else 448,
        "duration_ms": 0,
        "frames": 1,
        "fps": 0,
    }


def _png(width: int, height: int, *, value: int = 96) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanline = b"\x00" + bytes((value, value, value, 255)) * width
    pixels = scanline * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels, level=1))
        + chunk(b"IEND", b"")
    )


def _jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x0b\x08"
        + struct.pack(">HH", height, width)
        + b"\x01\x01\x11\x00"
        + b"\xff\xd9"
    )


def _gemma_live_plan() -> LiveScenePlan:
    return LiveScenePlan(
        scene_summary="A moonlit reader releases a bridge of paper birds.",
        art_direction=(
            "Gemma-authored low-angle composition, cyan paper birds sweeping through indigo "
            "clouds, warm book light, watercolor fibers"
        ),
        camera_motion="float",
        background_prompt="[0.5, 0.5, 0.8, 0.8], cobalt sky over luminous clouds",
        focus_label="child",
        focus=LiveScenePlacedLayerPlan(
            kind="character",
            prompt="A child holding a luminous open book",
            anchor=(0.5, 0.5, 0.8, 0.8),
            depth=0.2,
            motion="float",
        ),
        accent=LiveScenePlacedLayerPlan(
            kind="effect",
            prompt="Cyan paper birds sweeping toward the floating school",
            anchor=(0.5, 0.5, 0.8, 0.8),
            depth=0.2,
            motion="float",
        ),
        ambience=["stars"],
    )


def _provider(
    tmp_path: Path,
    runner,
    *,
    session_cap: float = 1.0,
    workspace_total: float = 1.0,
):
    app = tmp_path / "modal_fast_scene.py"
    plan = tmp_path / "plan.json"
    app.write_text("# fixture\n")

    async def billing_reader(executable: str, timeout: float) -> float:
        assert executable == "true"
        assert timeout == 60
        return workspace_total

    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 13.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 16.0,
            }
        )
    )
    return FiniteModalSceneProvider(
        modal_executable="true",
        modal_app=app,
        plan_file=plan,
        ledger_path=tmp_path / "ledger.json",
        session_gpu_cap_usd=session_cap,
        runner=runner,
        billing_reader=billing_reader,
    )


class StubWarmInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.autoscaler_calls: list[tuple[str, int]] = []

    async def probe(self) -> tuple[bool, str]:
        return True, "fixture deployed classes ready"

    async def configure_scaledown_window(self, class_name: str, seconds: int) -> None:
        self.autoscaler_calls.append((class_name, seconds))

    async def invoke(self, class_name: str, method_name: str, arguments: dict) -> dict:
        self.calls.append((class_name, method_name, arguments))
        if method_name == "prewarm":
            return {
                "model_load_seconds": 2.5,
                "container_age_seconds": 0.01,
                "gpu": "L40S" if class_name == "FastSceneStudio" else "L4",
            }
        if method_name == "generate_preview":
            return {
                "master": _jpeg(arguments["width"], arguments["height"]),
                "master_media_type": "image/jpeg",
                "negative_prompt_supported": False,
                "image_seconds": 0.61,
                "packaging_seconds": 0.02,
                "master_jpeg_quality": 95,
                "model_load_seconds": 2.5,
                "container_age_seconds": 8.0,
                "gpu": "L40S",
            }
        if class_name == "FastSceneStudio":
            result = {
                "master": _jpeg(arguments["width"], arguments["height"]),
                "master_media_type": "image/jpeg",
                "depth": _jpeg(arguments["width"], arguments["height"]),
                "depth_media_type": "image/jpeg",
                "depth_dtype": "float16",
                "negative_prompt_supported": False,
                "image_seconds": 3.0,
                "depth_seconds": 0.2,
                "packaging_seconds": 0.04,
                "master_jpeg_quality": 95,
                "depth_jpeg_quality": 85,
                "model_load_seconds": 2.5,
                "container_age_seconds": 3.4,
                "gpu": "L40S",
                "selected_seed": arguments["seed"],
            }
            if arguments.get("fidelity_label"):
                result.update(
                    {
                        "quality_seconds": 0.3,
                        "quality_attempts": 1,
                        "quality_label": arguments["fidelity_label"],
                        "quality_object_label": arguments["fidelity_object_label"],
                        "quality_expected_count": arguments["expected_subject_count"],
                        "quality_subject_count": arguments["expected_subject_count"],
                        "quality_object_count": (
                            1 if arguments["fidelity_object_label"] else None
                        ),
                        "quality_subject_object_overlap": (
                            True if arguments["fidelity_object_label"] else None
                        ),
                        "quality_scores": [0.9],
                        "quality_passed": True,
                    }
                )
            return result
        return {
            "content": b"fixture-mp4",
            "generation_seconds": 14.0,
            "frames": arguments["generated_frames"] * 2 - 1,
            "fps": arguments["fps"],
            "duration_ms": round((arguments["generated_frames"] * 2 - 1) / arguments["fps"] * 1000),
            "model_load_seconds": 6.0,
            "container_age_seconds": 14.5,
            "gpu": "L4",
        }


class BlockingWarmInvoker(StubWarmInvoker):
    def __init__(self, block: tuple[str, str]) -> None:
        super().__init__()
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def invoke(self, class_name: str, method_name: str, arguments: dict) -> dict:
        if (class_name, method_name) == self.block:
            self.calls.append((class_name, method_name, arguments))
            self.started.set()
            await self.release.wait()
        return await super().invoke(class_name, method_name, arguments)


def _warm_provider(tmp_path: Path, invoker: StubWarmInvoker) -> WarmModalSceneProvider:
    app = tmp_path / "modal_fast_scene.py"
    plan = tmp_path / "plan.json"
    app.write_text("# fixture\n")
    plan.write_text(
        json.dumps(
            {
                "monthly_credit_usd": 30.0,
                "workspace_usage_before_live_scenes_usd": 13.0,
                "billing_delay_reserve_usd": 1.0,
                "hard_stop_workspace_total_usd": 29.0,
                "maximum_new_spend_usd": 16.0,
            }
        )
    )

    async def billing_reader(executable: str, timeout: float) -> float:
        assert executable == "true"
        assert timeout == 60
        return 13.9

    async def forbidden_runner(command: list[str], timeout: float) -> None:
        del command, timeout
        raise AssertionError("warm provider must not shell out to modal run")

    return WarmModalSceneProvider(
        modal_executable="true",
        modal_app=app,
        plan_file=plan,
        ledger_path=tmp_path / "ledger.json",
        runner=forbidden_runner,
        billing_reader=billing_reader,
        invoker=invoker,
    )


def test_provider_runs_finite_fast_then_motion_and_verifies_artifacts(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    async def runner(command: list[str], timeout: float) -> None:
        commands.append(command)
        assert timeout in {360, 480}
        if command[3].endswith("::fast_scene_cli"):
            destination = Path(_argument(command, "--output-dir"))
            destination.mkdir(parents=True)
            master = _artifact(destination / "master.png", b"master", "image/png")
            depth = _artifact(destination / "depth.png", b"depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": _argument(command, "--scene-id"),
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.012,
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            (destination / "scene.manifest.json").write_text(json.dumps(payload))
            return
        manifest_path = Path(_argument(command, "--scene-manifest-path"))
        payload = json.loads(manifest_path.read_text())
        motion = _artifact(manifest_path.parent / "motion.mp4", b"motion", "video/mp4")
        motion.update({"duration_ms": 2708, "frames": 65, "fps": 24})
        payload["stages"]["motion"] = {
            "model": MOTION_MODEL,
            "model_revision": MOTION_MODEL_REVISION,
            "gpu": "L4",
            "finite_call": True,
            "estimated_gpu_usd": 0.023,
        }
        payload["artifacts"]["motion"] = motion
        manifest_path.write_text(json.dumps(payload))

    async def run():
        provider = _provider(tmp_path, runner)
        fast = await provider.generate_fast(
            FastSceneRequest(scene_id="live-001", prompt="A luminous paper forest"),
            output_dir=tmp_path / "live-001",
        )
        return fast, await provider.upgrade_motion(fast, MotionUpgradeRequest())

    fast, complete = asyncio.run(run())

    assert fast.motion is None
    assert complete.master.sha256 == fast.master.sha256
    assert complete.motion is not None
    assert complete.motion.duration_ms == 2708
    assert complete.estimated_gpu_usd == pytest.approx(0.035)
    assert len(commands) == 2
    assert commands[0][:3] == ["true", "run", "--quiet"]
    assert commands[0][3].endswith("::fast_scene_cli")
    assert commands[1][3].endswith("::motion_upgrade_cli")


def test_checksum_failure_rejects_scene_bundle(tmp_path: Path) -> None:
    manifest = tmp_path / "scene.manifest.json"
    master = _artifact(tmp_path / "master.png", b"master", "image/png")
    depth = _artifact(tmp_path / "depth.png", b"depth", "image/png")
    master["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": "bad-checksum",
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
        )
    )

    with pytest.raises(FiniteModalProviderError, match="master checksum mismatch"):
        load_finite_scene_bundle(manifest)


def test_artifact_path_cannot_escape_scene_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    scene = tmp_path / "scene"
    scene.mkdir()
    depth = _artifact(scene / "depth.png", b"depth", "image/png")
    master = {
        **_artifact(scene / "master.png", b"master", "image/png"),
        "path": "../outside.png",
        "sha256": hashlib.sha256(b"outside").hexdigest(),
    }
    manifest = scene / "scene.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": "escape",
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
        )
    )

    with pytest.raises(FiniteModalProviderError, match="escapes"):
        load_finite_scene_bundle(manifest)


def test_stage_policy_prices_full_command_lifetime() -> None:
    assert FAST_GPU == "L4"
    assert FAST_STAGE_POLICY.worst_case_gpu_usd == pytest.approx(0.07992)
    assert WARM_FAST_GPU == "L40S"
    assert WARM_FAST_STAGE_POLICY.worst_case_gpu_usd == pytest.approx(0.19512)
    assert MOTION_GPU == "L4"

    policy = ModalStagePolicy(
        gpu="L4",
        remote_timeout_seconds=10,
        command_timeout_seconds=100,
        maximum_gpu_usd=0.03,
    )

    assert policy.worst_case_gpu_usd == pytest.approx(0.0222)

    with pytest.raises(ValueError, match="worst-case cost"):
        ModalStagePolicy(
            gpu="L4",
            remote_timeout_seconds=10,
            command_timeout_seconds=100,
            maximum_gpu_usd=0.02,
        )


def test_failed_call_charges_full_reservation_and_blocks_retry(tmp_path: Path) -> None:
    async def runner(command: list[str], timeout: float) -> None:
        del command, timeout
        raise TimeoutError

    async def run() -> None:
        provider = _provider(tmp_path, runner, session_cap=0.08)
        with pytest.raises(TimeoutError):
            await provider.generate_fast(
                FastSceneRequest(scene_id="failed-001", prompt="A forest"),
                output_dir=tmp_path / "failed-001",
            )
        with pytest.raises(FiniteModalBudgetError, match="session cap"):
            await provider.generate_fast(
                FastSceneRequest(scene_id="failed-002", prompt="Another forest"),
                output_dir=tmp_path / "failed-002",
            )

    asyncio.run(run())


def test_authoritative_workspace_total_blocks_paid_call_before_runner(tmp_path: Path) -> None:
    called = False

    async def runner(command: list[str], timeout: float) -> None:
        nonlocal called
        del command, timeout
        called = True

    async def run() -> None:
        provider = _provider(tmp_path, runner, workspace_total=28.95)
        with pytest.raises(FiniteModalBudgetError, match="hard stop"):
            await provider.generate_fast(
                FastSceneRequest(scene_id="blocked", prompt="A forest"),
                output_dir=tmp_path / "blocked",
            )

    asyncio.run(run())

    assert called is False


def test_modal_billing_parser_accepts_current_and_legacy_cli_fields() -> None:
    parse = finite_modal_provider_module._parse_modal_billing_total

    current = '[{"description":"scene","cost":"0.125"},{"cost":"1.25"}]'
    legacy = '[{"Description":"scene","Cost":"0.125"},{"Cost":"1.25"}]'

    assert parse(current) == pytest.approx(1.375)
    assert parse(legacy) == pytest.approx(1.375)
    assert parse("[]") == 0


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[null]",
        '[{"description":"missing"}]',
        '[{"cost":"not-a-number"}]',
        '[{"cost":"NaN"}]',
        '[{"cost":"Infinity"}]',
        '[{"cost":"-0.01"}]',
        '[{"cost":"1.0","Cost":"2.0"}]',
    ],
)
def test_modal_billing_parser_rejects_ambiguous_or_unsafe_reports(payload: str) -> None:
    with pytest.raises((TypeError, ValueError, json.JSONDecodeError)):
        finite_modal_provider_module._parse_modal_billing_total(payload)


def test_request_profiles_are_projection_native_and_bounded() -> None:
    fast = FastSceneRequest(scene_id="scene", prompt="A fox")
    motion = MotionUpgradeRequest()

    assert (fast.width, fast.height, fast.steps) == (1024, 576, 2)
    assert (motion.width, motion.height) == (800, 448)
    assert motion.generated_frames == 41
    assert (motion.generated_frames * 2 - 1) / motion.fps == pytest.approx(3.375)

    with pytest.raises(ValueError, match=r"8n\+1"):
        MotionUpgradeRequest(generated_frames=32)
    with pytest.raises(ValueError, match="between 1 and 4"):
        FastSceneRequest(scene_id="scene", prompt="A fox", steps=5)
    with pytest.raises(ValueError, match="requires an object label"):
        FastSceneRequest(
            scene_id="scene",
            prompt="A fox",
            fidelity_label="fox",
            require_subject_object_overlap=True,
        )


def test_fidelity_contract_extracts_action_object_and_overlap_requirement() -> None:
    page = _gemma_live_plan().model_copy(
        update={
            "focus": LiveScenePlacedLayerPlan(
                kind="character",
                prompt="a complete visible young otter, shown steering walnut boat",
                anchor=(0.5, 0.5, 0.4, 0.6),
                depth=2.5,
                motion="breathe",
            )
        }
    ).to_page(
        source_text="An animal crosses water.",
        visual_style="luminous paper theater",
        seed=27,
    )

    assert (
        finite_modal_provider_module._fidelity_action_object(page.layers)
        == "walnut shell boat"
    )
    assert finite_modal_provider_module._fidelity_requires_overlap(page.layers) is True
    assert (
        finite_modal_provider_module._fidelity_focus_prompt(page.layers)
        == "a complete visible young otter, shown steering walnut boat"
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("one child, shown holding brass key", "brass key"),
        ("one rabbit, shown carrying lantern", "lantern"),
        ("one rider, shown riding red bicycle", "red bicycle"),
        ("one otter, shown paddling coconut boat", "coconut shell boat"),
    ],
)
def test_fidelity_action_object_keeps_material_identity(
    prompt: str,
    expected: str,
) -> None:
    layers = [SimpleNamespace(layer_id="scene-focus", prompt=prompt)]

    assert finite_modal_provider_module._fidelity_action_object(layers) == expected


def test_deployed_warm_state_uses_measured_container_reuse() -> None:
    classify = finite_modal_provider_module._deployed_warm_state

    assert classify({}, remote_seconds=1, uses_session=False) == "unknown"
    assert classify({"container_age_seconds": 1.1}, remote_seconds=1, uses_session=False) == "cold"
    assert classify({"container_age_seconds": 12}, remote_seconds=1, uses_session=False) == "warm"
    assert classify({}, remote_seconds=30, uses_session=True) == "prewarmed"


def test_warm_fast_only_prewarm_never_loads_motion_and_settles_one_envelope(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        report = await provider.prewarm(prewarm_id="fast-only")
        status = await provider.warm_status()
        bundle = await provider.generate_fast(
            FastSceneRequest(scene_id="warm-fast-001", prompt="A brighter winged library"),
            output_dir=tmp_path / "warm-fast-001",
        )
        final_status = await provider.warm_status()
        return report, status, bundle, final_status

    report, status, bundle, final_status = asyncio.run(run())

    assert report.include_motion is False
    assert report.full_session_ceiling_usd == WARM_FAST_SESSION_CEILING_USD
    assert report.motion_remote_seconds == 0
    assert status.state == "prewarmed"
    assert status.expires_in_seconds > 0
    assert final_status.state == "idle"
    assert bundle.manifest["stages"]["fast"]["warm_state"] == "prewarmed"
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "prewarm"),
        ("FastSceneStudio", "generate"),
    ]
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-prewarm-master"]
    assert ledger.records[0].estimated_gpu_usd >= 30 * 0.000222


def test_presentation_prewarm_extends_only_fast_idle_window_with_bounded_cost(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        report = await provider.prewarm(
            prewarm_id="presentation",
            scaledown_window_seconds=600,
        )
        status = await provider.warm_status()
        bundle = await provider.generate_fast(
            FastSceneRequest(scene_id="presentation-001", prompt="A bright paper library"),
            output_dir=tmp_path / "presentation-001",
        )
        final_status = await provider.warm_status()
        return report, status, bundle, final_status

    report, status, bundle, final_status = asyncio.run(run())

    assert report.full_session_ceiling_usd == WARM_FAST_PRESENTATION_SESSION_CEILING_USD
    assert report.scaledown_window_seconds == 600
    assert report.expires_in_seconds == 600
    assert status.scaledown_window_seconds == 600
    assert status.expires_in_seconds > 599
    assert final_status.state == "idle"
    assert final_status.scaledown_window_seconds == 600
    assert invoker.autoscaler_calls == [("FastSceneStudio", 600)]
    assert bundle.manifest["stages"]["fast"]["warm_state"] == "prewarmed"
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "prewarm"),
        ("FastSceneStudio", "generate"),
    ]
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-prewarm-master"]
    assert ledger.records[0].estimated_gpu_usd >= 600 * 0.000222

    with pytest.raises(ValueError, match="master/depth only"):
        asyncio.run(
            provider.prewarm(
                prewarm_id="invalid-motion-presentation",
                include_motion=True,
                scaledown_window_seconds=600,
            )
        )


def test_auto_prewarm_overlaps_local_planning_before_generation(tmp_path: Path) -> None:
    invoker = BlockingWarmInvoker(("FastSceneStudio", "prewarm"))
    warm = _warm_provider(tmp_path, invoker)

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await self.release.wait()
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=4_000,
                    input_tokens=300,
                    output_tokens=108,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=4_050,
            )

    async def run():
        cache = AssetCache(tmp_path / "auto-prewarm-cache")
        await cache.initialize()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "auto-prewarm-output",
            planner=planner,
            auto_prewarm_on_submit=True,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A child opens a book and birds fill the sky.", seed=37),
            job_id="scene_000000000000000000000037",
        )
        draft = await anext(iterator)
        master_task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        await asyncio.wait_for(invoker.started.wait(), timeout=1)
        assert not master_task.done()
        planner.release.set()
        invoker.release.set()
        master = await asyncio.wait_for(master_task, timeout=2)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return draft, master

    draft, master = asyncio.run(run())

    assert draft.stage is LiveSceneStage.DRAFT_READY
    assert master.stage is LiveSceneStage.MASTER_READY
    assert master.complete is True
    assert master.metrics is not None
    assert master.metrics.preparation_ms > 0
    assert master.metrics.elapsed_ms == pytest.approx(
        max(master.metrics.planning_ms, master.metrics.preparation_ms)
        + master.metrics.provider_ms
        + master.metrics.cache_ms
    )
    assert [(class_name, method) for class_name, method, _ in invoker.calls].count(
        ("FastSceneStudio", "prewarm")
    ) >= 1
    assert ("FastSceneStudio", "generate") in [
        (class_name, method) for class_name, method, _ in invoker.calls
    ]


def test_capability_based_auto_prewarm_overlaps_gcp_planning(tmp_path: Path) -> None:
    class GcpLikeProvider:
        def __init__(self) -> None:
            self.prewarm_started = asyncio.Event()
            self.prewarm_release = asyncio.Event()
            self.prewarmed = False
            self.generate_calls = 0

        async def is_prewarmed(self) -> bool:
            return self.prewarmed

        async def is_renderer_likely_warm(self) -> bool:
            return self.prewarmed

        async def prewarm(self, *, prewarm_id: str, include_motion: bool) -> SimpleNamespace:
            assert prewarm_id.startswith("auto-scene_")
            assert include_motion is False
            self.prewarm_started.set()
            await self.prewarm_release.wait()
            self.prewarmed = True
            return SimpleNamespace(prewarm_id=prewarm_id)

        async def generate_fast(
            self,
            request: FastSceneRequest,
            *,
            output_dir: Path,
        ) -> FiniteSceneBundle:
            assert self.prewarmed is True
            self.generate_calls += 1
            output_dir.mkdir(parents=True)
            master = _artifact(output_dir / "master.png", b"master", "image/png")
            depth = _artifact(output_dir / "depth.png", b"depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": request.scene_id,
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                        "remote_seconds": 0.5,
                        "inference_seconds": 0.22,
                        "provider_overhead_seconds": 0.28,
                        "packaging_seconds": 0.05,
                        "image_seconds": 0.19,
                        "depth_seconds": 0.03,
                        "warm_state": "prewarmed",
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            manifest_path = output_dir / "scene.manifest.json"
            manifest_path.write_text(json.dumps(payload))
            return load_finite_scene_bundle(manifest_path)

        async def upgrade_motion(
            self,
            bundle: FiniteSceneBundle,
            request: MotionUpgradeRequest,
        ) -> FiniteSceneBundle:
            del bundle, request
            raise AssertionError("motion is disabled")

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await self.release.wait()
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=2_800,
                    input_tokens=300,
                    output_tokens=80,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=2_850,
            )

    async def run():
        cache = AssetCache(tmp_path / "gcp-auto-prewarm-cache")
        await cache.initialize()
        remote = GcpLikeProvider()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            remote,  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "gcp-auto-prewarm-output",
            planner=planner,
            auto_prewarm_on_submit=True,
            provider_name="gcp-cloud-run",
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A child opens a book and birds fill the sky.", seed=41),
            job_id="scene_000000000000000000000041",
        )
        draft = await anext(iterator)
        master_task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        await asyncio.wait_for(remote.prewarm_started.wait(), timeout=1)
        assert remote.generate_calls == 0
        assert not master_task.done()
        planner.release.set()
        remote.prewarm_release.set()
        master = await asyncio.wait_for(master_task, timeout=2)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return draft, master, remote

    draft, master, remote = asyncio.run(run())

    assert draft.stage is LiveSceneStage.DRAFT_READY
    assert master.stage is LiveSceneStage.MASTER_READY
    assert master.complete is True
    assert master.metrics.preparation_ms > 0
    assert master.metrics.warm_state is LiveSceneWarmState.WARM
    assert remote.generate_calls == 1


def test_safe_route_probe_overlaps_local_planning(tmp_path: Path) -> None:
    class BlockingProbeProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def probe(self) -> tuple[bool, str]:
            self.started.set()
            await self.release.wait()
            return True, "fixture route warm"

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await self.release.wait()
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=2_800,
                    input_tokens=300,
                    output_tokens=80,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=2_850,
            )

    async def run():
        cache = AssetCache(tmp_path / "safe-probe-cache")
        await cache.initialize()
        remote = BlockingProbeProvider()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            remote,  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "safe-probe-output",
            planner=planner,
            provider_name="resilient-cloud",
        )
        request = LiveSceneCreateRequest(
            text="A child opens a book and birds fill the sky.",
            seed=43,
        )
        draft = build_live_scene_story_pack(
            request,
            job_id="scene_000000000000000000000043",
            seed=43,
            assets=[],
            compiler_model="deterministic-live-scene-planner-v3",
        )
        task = asyncio.create_task(
            adapter._resolve_plan_while_preparing_renderer(
                request,
                job_id="scene_000000000000000000000043",
                seed=43,
                draft=draft,
            )
        )
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        await asyncio.wait_for(remote.started.wait(), timeout=1)
        assert not task.done()
        planner.release.set()
        remote.release.set()
        resolved = await asyncio.wait_for(task, timeout=2)
        return resolved

    resolved = asyncio.run(run())

    assert resolved.planning_ms == 2_850
    assert resolved.preparation_ms > 0


def test_billing_authorization_overlaps_planning_without_starting_gpu_early(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)
    billing_started = asyncio.Event()
    billing_release = asyncio.Event()

    async def billing_reader(executable: str, timeout: float) -> float:
        assert executable == "true"
        assert timeout == 60
        billing_started.set()
        await billing_release.wait()
        return 13.9

    warm._billing_reader = billing_reader

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await self.release.wait()
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=4_000,
                    input_tokens=300,
                    output_tokens=108,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=4_050,
            )

    async def run():
        cache = AssetCache(tmp_path / "billing-overlap-cache")
        await cache.initialize()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "billing-overlap-output",
            planner=planner,
            auto_prewarm_on_submit=False,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A child opens a book and birds fill the sky.", seed=39),
            job_id="scene_000000000000000000000039",
        )
        await anext(iterator)
        master_task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        await asyncio.wait_for(billing_started.wait(), timeout=1)
        assert invoker.calls == []
        planner.release.set()
        billing_release.set()
        master = await asyncio.wait_for(master_task, timeout=2)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return master

    master = asyncio.run(run())

    assert master.metrics is not None
    assert master.metrics.preparation_ms > 0
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "generate")
    ]
    envelope, _ = budget_envelope_from_plan(warm.plan_file)
    ledger = VisualLabLedger.read(warm.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-fast-scene"]


def test_uncached_model_plan_emits_privacy_safe_preview_before_final_master(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)

    class UncachedPlanner:
        def __init__(self, preview_complete: asyncio.Event) -> None:
            self.preview_complete = preview_complete

        async def has_cached_plan(self, *, text: str) -> bool:
            assert "Quenlora" in text
            return False

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            assert "Quenlora" in kwargs["text"]
            await self.preview_complete.wait()
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=4_000,
                    input_tokens=300,
                    output_tokens=108,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=4_050,
            )

    class PreviewFirstProvider(FiniteModalLiveSceneProvider):
        def __init__(self, *args, preview_complete: asyncio.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.preview_complete = preview_complete

        async def _generate_preview(self, *args, **kwargs):
            preview = await super()._generate_preview(*args, **kwargs)
            self.preview_complete.set()
            return preview

    async def run() -> list[LiveSceneUpdate]:
        cache = AssetCache(tmp_path / "preview-cache")
        await cache.initialize()
        await warm.prewarm(
            prewarm_id="preview-integration-prewarm",
            include_motion=False,
        )
        preview_complete = asyncio.Event()
        adapter = PreviewFirstProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "preview-output",
            planner=UncachedPlanner(preview_complete),
            enable_preview=True,
            preview_complete=preview_complete,
        )
        return [
            update
            async for update in adapter.generate(
                LiveSceneCreateRequest(
                    text=(
                        "Quenlora watches a silver whale cross a flooded library "
                        "while folded books become fish."
                    ),
                    seed=52,
                ),
                job_id="scene_000000000000000000000052",
            )
        ]

    updates = asyncio.run(run())

    assert [update.stage for update in updates] == [
        LiveSceneStage.DRAFT_READY,
        LiveSceneStage.PREVIEW_READY,
        LiveSceneStage.MASTER_READY,
    ]
    preview, master = updates[1:]
    assert preview.complete is False
    assert [artifact.kind for artifact in preview.artifacts] == [
        LiveSceneArtifactKind.PREVIEW
    ]
    assert [asset.role.value for asset in preview.story_pack.assets] == ["preview"]
    assert "Quenlora" not in preview.story_pack.assets[0].prompt
    assert "silver whale cross" not in preview.story_pack.assets[0].prompt
    assert "one graceful storybook whale" in preview.story_pack.assets[0].prompt
    assert preview.metrics.models[0].role == "preview"
    assert master.complete is True
    assert master.metrics.provider_ms > preview.metrics.provider_ms
    assert [model.role for model in master.metrics.models] == [
        "scene_plan",
            "preview",
            "master",
            "depth",
        "visual_fidelity",
    ]
    generate_call = next(
        arguments
        for class_name, method_name, arguments in invoker.calls
        if class_name == "FastSceneStudio" and method_name == "generate"
    )
    assert generate_call["fidelity_label"] == "child"
    assert generate_call["expected_subject_count"] == 1
    methods = [(class_name, method) for class_name, method, _ in invoker.calls]
    assert methods == [
        ("FastSceneStudio", "prewarm"),
        ("FastSceneStudio", "generate_preview"),
        ("FastSceneStudio", "generate"),
    ]
    envelope, _ = budget_envelope_from_plan(warm.plan_file)
    ledger = VisualLabLedger.read(warm.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-prewarm-master"]


@pytest.mark.parametrize(
    ("preview_failure", "cached_plan"),
    [(False, True), (False, False), (True, False)],
)
def test_cached_or_failed_preview_never_blocks_final_master(
    tmp_path: Path,
    preview_failure: bool,
    cached_plan: bool,
) -> None:
    class PreviewInvoker(StubWarmInvoker):
        async def invoke(self, class_name: str, method_name: str, arguments: dict) -> dict:
            if preview_failure and method_name == "generate_preview":
                self.calls.append((class_name, method_name, arguments))
                raise RuntimeError("optional preview fixture failed")
            return await super().invoke(class_name, method_name, arguments)

    class Planner:
        async def has_cached_plan(self, *, text: str) -> bool:
            return cached_plan

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=500,
                    input_tokens=100,
                    output_tokens=80,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=510,
            )

    invoker = PreviewInvoker()
    warm = _warm_provider(tmp_path, invoker)

    async def run() -> list[LiveSceneUpdate]:
        cache = AssetCache(tmp_path / "optional-preview-cache")
        await cache.initialize()
        if preview_failure:
            await warm.prewarm(
                prewarm_id="failed-preview-prewarm",
                include_motion=False,
            )
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "optional-preview-output",
            planner=Planner(),
            enable_preview=True,
        )
        return [
            update
            async for update in adapter.generate(
                LiveSceneCreateRequest(text="A whale crosses a paper sea.", seed=53),
                job_id="scene_000000000000000000000053",
            )
        ]

    updates = asyncio.run(run())

    assert [update.stage for update in updates] == [
        LiveSceneStage.DRAFT_READY,
        LiveSceneStage.MASTER_READY,
    ]
    assert updates[-1].complete is True
    methods = [method for _, method, _ in invoker.calls]
    assert methods[-1] == "generate"
    assert ("generate_preview" in methods) is preview_failure


def test_completed_planner_preempts_blocked_optional_preview(tmp_path: Path) -> None:
    class BlockingPreviewInvoker(StubWarmInvoker):
        def __init__(self) -> None:
            super().__init__()
            self.preview_started = asyncio.Event()
            self.preview_cancelled = asyncio.Event()

        async def invoke(self, class_name: str, method_name: str, arguments: dict) -> dict:
            if method_name == "generate_preview":
                self.calls.append((class_name, method_name, arguments))
                self.preview_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.preview_cancelled.set()
                    raise
            return await super().invoke(class_name, method_name, arguments)

    class ImmediatePlanner:
        async def has_cached_plan(self, *, text: str) -> bool:
            return False

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            await asyncio.sleep(0)
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=5,
                    input_tokens=100,
                    output_tokens=80,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=5,
            )

    invoker = BlockingPreviewInvoker()
    warm = _warm_provider(tmp_path, invoker)

    async def run() -> list[LiveSceneUpdate]:
        cache = AssetCache(tmp_path / "blocked-preview-cache")
        await cache.initialize()
        await warm.prewarm(
            prewarm_id="blocked-preview-prewarm",
            include_motion=False,
        )
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "blocked-preview-output",
            planner=ImmediatePlanner(),
            enable_preview=True,
        )
        async def collect() -> list[LiveSceneUpdate]:
            return [
                update
                async for update in adapter.generate(
                    LiveSceneCreateRequest(text="A whale crosses a paper sea.", seed=54),
                    job_id="scene_000000000000000000000054",
                )
            ]

        updates = await asyncio.wait_for(collect(), timeout=1)
        await asyncio.wait_for(invoker.preview_started.wait(), timeout=1)
        await asyncio.wait_for(invoker.preview_cancelled.wait(), timeout=1)
        return updates

    updates = asyncio.run(run())

    assert [update.stage for update in updates] == [
        LiveSceneStage.DRAFT_READY,
        LiveSceneStage.MASTER_READY,
    ]
    assert updates[-1].complete is True
    methods = [method for _, method, _ in invoker.calls]
    assert methods == ["prewarm", "generate_preview", "generate"]


def test_auto_prewarm_reuses_recent_remote_container_without_second_prewarm(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)

    async def run() -> tuple[LiveSceneUpdate, LiveSceneUpdate]:
        cache = AssetCache(tmp_path / "remote-warm-hint-cache")
        await cache.initialize()
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            output_root=tmp_path / "remote-warm-hint-output",
            auto_prewarm_on_submit=True,
        )

        async def generate(scene_id: str, seed: int) -> LiveSceneUpdate:
            iterator = adapter.generate(
                LiveSceneCreateRequest(text="A paper lighthouse wakes at dusk.", seed=seed),
                job_id=scene_id,
            )
            await anext(iterator)
            master = await anext(iterator)
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
            return master

        first = await generate("scene_000000000000000000000041", 41)
        second = await generate("scene_000000000000000000000042", 42)
        return first, second

    first, second = asyncio.run(run())

    assert first.complete is True
    assert second.complete is True
    methods = [(class_name, method) for class_name, method, _ in invoker.calls]
    assert methods.count(("FastSceneStudio", "prewarm")) == 1
    assert methods.count(("FastSceneStudio", "generate")) == 2
    envelope, _ = budget_envelope_from_plan(warm.plan_file)
    ledger = VisualLabLedger.read(warm.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == [
        "warm-prewarm-master",
        "warm-fast-scene",
    ]


def test_unused_overlapped_authorization_is_released_on_scene_cancellation(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> None:
        cache = AssetCache(tmp_path / "cancel-authorization-cache")
        await cache.initialize()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            planner=planner,
            auto_prewarm_on_submit=False,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A child opens a book.", seed=40),
            job_id="scene_000000000000000000000040",
        )
        await anext(iterator)
        task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        while not warm._prepared_fast_authorizations:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await iterator.aclose()

    asyncio.run(run())

    assert invoker.calls == []
    envelope, _ = budget_envelope_from_plan(warm.plan_file)
    ledger = VisualLabLedger.read(warm.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert ledger.records == []


def test_auto_prewarm_is_cancelled_and_drained_with_scene_planning(tmp_path: Path) -> None:
    invoker = BlockingWarmInvoker(("FastSceneStudio", "prewarm"))
    warm = _warm_provider(tmp_path, invoker)

    class BlockingPlanner:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            del kwargs
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> None:
        cache = AssetCache(tmp_path / "cancel-auto-prewarm-cache")
        await cache.initialize()
        planner = BlockingPlanner()
        adapter = FiniteModalLiveSceneProvider(
            warm,
            cache=cache,
            planner=planner,
            auto_prewarm_on_submit=True,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A child opens a book.", seed=38),
            job_id="scene_000000000000000000000038",
        )
        await anext(iterator)
        task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner.started.wait(), timeout=1)
        await asyncio.wait_for(invoker.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await iterator.aclose()

    asyncio.run(run())

    methods = [(class_name, method) for class_name, method, _ in invoker.calls]
    assert ("FastSceneStudio", "generate") not in methods
    envelope, _ = budget_envelope_from_plan(warm.plan_file)
    ledger = VisualLabLedger.read(warm.ledger_path, envelope=envelope)
    assert len(ledger.reservations) == 1


def test_sdk_probe_hydrates_both_deployed_classes_without_invoking_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []

    class Handle:
        def __init__(self, class_name: str) -> None:
            async def hydrate() -> None:
                events.append(("hydrate", class_name))

            self.class_name = class_name
            self.hydrate = SimpleNamespace(aio=hydrate)

        def with_options(self, **arguments):
            events.append(("with_options", self.class_name, arguments))
            return Handle(self.class_name)

        def __call__(self):
            async def prewarm(**arguments):
                events.append(("invoke", self.class_name, arguments))
                return {"model": self.class_name}

            def update_autoscaler(**arguments):
                events.append(("autoscaler", self.class_name, arguments))

            return SimpleNamespace(
                prewarm=SimpleNamespace(remote=SimpleNamespace(aio=prewarm)),
                update_autoscaler=update_autoscaler,
            )

    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str, **kwargs):
            events.append(("lookup", app_name, class_name, kwargs))
            return Handle(class_name)

    real_import = finite_modal_provider_module.importlib.import_module
    monkeypatch.setattr(
        finite_modal_provider_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Cls=FakeCls) if name == "modal" else real_import(name),
    )

    async def run():
        invoker = ModalSdkWarmInvoker(app_name="fixture-app")
        readiness = await invoker.probe()
        await invoker.configure_scaledown_window("FastSceneStudio", 600)
        result = await invoker.invoke("FastSceneStudio", "prewarm", {})
        return readiness, result

    readiness, result = asyncio.run(run())

    assert readiness == (True, "Authenticated deployed Modal classes are reachable")
    assert result == {"model": "FastSceneStudio"}
    assert [event for event in events if event[0] == "hydrate"] == [
        ("hydrate", "FastSceneStudio"),
        ("hydrate", "FastSceneStudio"),
        ("hydrate", "MotionUpgradeStudio"),
    ]
    assert len([event for event in events if event[0] == "lookup"]) == 2
    assert len([event for event in events if event[0] == "invoke"]) == 1
    assert (
        "with_options",
        "FastSceneStudio",
        {
            "gpu": "L40S",
            "max_containers": 1,
            "scaledown_window": 90,
            "timeout": 180,
        },
    ) in events
    assert ("autoscaler", "FastSceneStudio", {"scaledown_window": 600}) in events


def test_warm_readiness_fails_before_billing_when_deployment_is_missing(
    tmp_path: Path,
) -> None:
    class MissingDeploymentInvoker(StubWarmInvoker):
        async def probe(self) -> tuple[bool, str]:
            return False, "deployed Modal classes are unreachable: not found"

    invoker = MissingDeploymentInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        readiness = await provider.probe()
        with pytest.raises(FiniteModalProviderError, match="unreachable"):
            await provider.prewarm(prewarm_id="not-deployed")
        return readiness

    readiness = asyncio.run(run())

    assert readiness == (False, "deployed Modal classes are unreachable: not found")
    assert not provider.ledger_path.exists()
    assert invoker.calls == []


def test_warm_provider_fails_closed_if_modal_does_not_honor_l40s_variant(
    tmp_path: Path,
) -> None:
    class WrongGpuInvoker(StubWarmInvoker):
        async def invoke(self, class_name: str, method_name: str, arguments: dict) -> dict:
            result = await super().invoke(class_name, method_name, arguments)
            if class_name == "FastSceneStudio" and method_name == "prewarm":
                result["gpu"] = "L4"
            return result

    provider = _warm_provider(tmp_path, WrongGpuInvoker())

    async def run() -> None:
        with pytest.raises(FiniteModalProviderError, match="unexpected GPU"):
            await provider.prewarm(prewarm_id="wrong-gpu")

    asyncio.run(run())

    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert len(ledger.reservations) == 1


def test_warm_full_prewarm_is_explicit_and_settles_after_motion(tmp_path: Path) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        report = await provider.prewarm(prewarm_id="full-scene", include_motion=True)
        fast = await provider.generate_fast(
            FastSceneRequest(scene_id="warm-full-001", prompt="A winged library"),
            output_dir=tmp_path / "warm-full-001",
        )
        envelope, _ = budget_envelope_from_plan(provider.plan_file)
        active = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
        complete = await provider.upgrade_motion(fast, MotionUpgradeRequest())
        return report, active, complete

    report, active, complete = asyncio.run(run())

    assert report.include_motion is True
    assert report.full_session_ceiling_usd == WARM_FULL_SESSION_CEILING_USD
    assert len(active.reservations) == 1
    assert complete.motion is not None
    assert complete.manifest["stages"]["fast"]["warm_state"] == "prewarmed"
    assert complete.manifest["stages"]["motion"]["warm_state"] == "prewarmed"
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "prewarm"),
        ("MotionUpgradeStudio", "prewarm"),
        ("FastSceneStudio", "generate"),
        ("MotionUpgradeStudio", "generate"),
    ]
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-prewarm-scene"]
    assert ledger.records[0].estimated_gpu_usd >= 60 * 0.000222


def test_expired_prewarm_is_settled_and_rejected_instead_of_falsely_reused(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="expires")
        assert provider._warm_session is not None
        provider._warm_session.deadline_monotonic = 0
        with pytest.raises(FiniteModalProviderError, match="prewarm expired"):
            await provider.generate_fast(
                FastSceneRequest(scene_id="expired-001", prompt="An expired library"),
                output_dir=tmp_path / "expired-001",
            )
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "prewarm")
    ]
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert ledger.reservations == {}
    assert [record.stage for record in ledger.records] == ["warm-prewarm-expired"]


def test_new_prewarm_replaces_and_settles_an_expired_session(tmp_path: Path) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="first-expired")
        assert provider._warm_session is not None
        provider._warm_session.deadline_monotonic = 0
        report = await provider.prewarm(prewarm_id="second-current")
        return report, await provider.warm_status()

    report, status = asyncio.run(run())

    assert report.prewarm_id == "second-current"
    assert status.state == "prewarmed"
    assert status.prewarm_id == "second-current"
    assert [(class_name, method) for class_name, method, _ in invoker.calls] == [
        ("FastSceneStudio", "prewarm"),
        ("FastSceneStudio", "prewarm"),
    ]
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert [record.stage for record in ledger.records] == ["warm-prewarm-expired"]
    assert set(ledger.reservations) == {"reservation:warm-session:second-current"}


def test_cancelling_prewarm_retains_fail_closed_reservation(tmp_path: Path) -> None:
    invoker = BlockingWarmInvoker(("FastSceneStudio", "prewarm"))
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        task = asyncio.create_task(provider.prewarm(prewarm_id="cancel-prewarm"))
        await invoker.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert set(ledger.reservations) == {"reservation:warm-session:cancel-prewarm"}
    assert ledger.records == []


def test_cancelling_warm_fast_rpc_clears_session_without_reusing_reservation(
    tmp_path: Path,
) -> None:
    invoker = BlockingWarmInvoker(("FastSceneStudio", "generate"))
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="cancel-fast", include_motion=True)
        task = asyncio.create_task(
            provider.generate_fast(
                FastSceneRequest(scene_id="cancel-fast-001", prompt="A library"),
                output_dir=tmp_path / "cancel-fast-001",
            )
        )
        await invoker.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert set(ledger.reservations) == {"reservation:warm-session:cancel-fast"}


def test_cancelling_adapter_during_cache_promotion_abandons_assigned_scene(
    tmp_path: Path,
) -> None:
    class BlockingCache(AssetCache):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def store_generated(self, **kwargs):
            self.started.set()
            await self.release.wait()
            return await super().store_generated(**kwargs)

    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="cancel-cache", include_motion=True)
        cache = BlockingCache(tmp_path / "blocking-cache")
        await cache.initialize()
        adapter = FiniteModalLiveSceneProvider(
            provider,
            cache=cache,
            output_root=tmp_path / "cancel-cache-output",
            enable_motion=True,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A library opens its wings", seed=3),
            job_id="scene_000000000000000000000077",
        )
        assert (await anext(iterator)).stage is LiveSceneStage.DRAFT_READY
        task = asyncio.create_task(anext(iterator))
        await cache.started.wait()
        assert provider._warm_session is not None
        assert provider._warm_session.scene_id == "scene_000000000000000000000077"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert set(ledger.reservations) == {"reservation:warm-session:cancel-cache"}


def test_closing_adapter_exactly_after_master_ready_abandons_pending_warm_motion(
    tmp_path: Path,
) -> None:
    invoker = StubWarmInvoker()
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="cancel-after-master", include_motion=True)
        cache = AssetCache(tmp_path / "after-master-cache")
        await cache.initialize()
        adapter = FiniteModalLiveSceneProvider(
            provider,
            cache=cache,
            output_root=tmp_path / "after-master-output",
            enable_motion=True,
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text="A library waits to move", seed=4),
            job_id="scene_000000000000000000000078",
        )
        assert (await anext(iterator)).stage is LiveSceneStage.DRAFT_READY
        master = await anext(iterator)
        assert master.stage is LiveSceneStage.MASTER_READY
        assert provider._warm_session is not None
        await iterator.aclose()
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert set(ledger.reservations) == {"reservation:warm-session:cancel-after-master"}
    assert not any(call[:2] == ("MotionUpgradeStudio", "generate") for call in invoker.calls)


def test_cancelling_warm_motion_rpc_clears_session_and_keeps_reservation(
    tmp_path: Path,
) -> None:
    invoker = BlockingWarmInvoker(("MotionUpgradeStudio", "generate"))
    provider = _warm_provider(tmp_path, invoker)

    async def run():
        await provider.prewarm(prewarm_id="cancel-motion", include_motion=True)
        fast = await provider.generate_fast(
            FastSceneRequest(scene_id="cancel-motion-001", prompt="A library"),
            output_dir=tmp_path / "cancel-motion-001",
        )
        task = asyncio.create_task(provider.upgrade_motion(fast, MotionUpgradeRequest()))
        await invoker.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await provider.warm_status()

    status = asyncio.run(run())

    assert status.state == "idle"
    envelope, _ = budget_envelope_from_plan(provider.plan_file)
    ledger = VisualLabLedger.read(provider.ledger_path, envelope=envelope)
    assert set(ledger.reservations) == {"reservation:warm-session:cancel-motion"}


def test_motion_technical_gate_rejects_before_cache_promotion(tmp_path: Path) -> None:
    class GateStub:
        async def generate_fast(
            self,
            request: FastSceneRequest,
            *,
            output_dir: Path,
        ) -> FiniteSceneBundle:
            output_dir.mkdir(parents=True)
            master = _artifact(output_dir / "master.png", b"master", "image/png")
            depth = _artifact(output_dir / "depth.png", b"depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": request.scene_id,
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            path = output_dir / "scene.manifest.json"
            path.write_text(json.dumps(payload))
            return load_finite_scene_bundle(path)

        async def upgrade_motion(
            self,
            bundle: FiniteSceneBundle,
            request: MotionUpgradeRequest,
        ) -> FiniteSceneBundle:
            del request
            payload = json.loads(bundle.manifest_path.read_text())
            motion = _artifact(
                bundle.manifest_path.parent / "motion.mp4",
                b"unstable-motion",
                "video/mp4",
            )
            motion.update({"duration_ms": 3375, "frames": 81, "fps": 24})
            payload["stages"]["motion"] = {
                "model": MOTION_MODEL,
                "model_revision": MOTION_MODEL_REVISION,
                "gpu": "L4",
                "finite_call": True,
                "estimated_gpu_usd": 0.02,
            }
            payload["artifacts"]["motion"] = motion
            bundle.manifest_path.write_text(json.dumps(payload))
            return load_finite_scene_bundle(bundle.manifest_path)

    async def run():
        cache = AssetCache(tmp_path / "gate-cache")
        await cache.initialize()
        provider = FiniteModalLiveSceneProvider(
            GateStub(),  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "gate-generated",
            enable_motion=True,
            motion_evaluator=lambda path: MotionTechnicalEvidence(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                duration_seconds=3.375,
                fps=24,
                endpoint_ssim=0.70,
                motion_stability=0.45,
                projection_legibility=0.10,
            ),
        )
        iterator = provider.generate(
            LiveSceneCreateRequest(text="A library flies", seed=12),
            job_id="scene_000000000000000000000099",
        )
        draft = await anext(iterator)
        master = await anext(iterator)
        with pytest.raises(FiniteModalProviderError, match="technical promotion gate rejected"):
            await anext(iterator)
        return cache, draft, master

    cache, draft, master = asyncio.run(run())

    assert draft.stage is LiveSceneStage.DRAFT_READY
    assert master.stage is LiveSceneStage.MASTER_READY
    assert {asset.role.value for asset in master.story_pack.assets} == {"master", "depth"}
    assert not any(path.suffix == ".mp4" for path in cache.root.rglob("*"))
    report = json.loads(
        (
            tmp_path / "gate-generated" / "scene_000000000000000000000099" / "motion.technical.json"
        ).read_text()
    )
    assert report["accepted"] is False
    assert report["semantic_identity_review"] == "not_performed"
    assert report["promotion_scope"] == "experimental-technical-only"


def test_motion_gate_thresholds_cover_loop_mechanics_and_projection() -> None:
    gate = MotionTechnicalGate()
    evidence = MotionTechnicalEvidence(
        sha256="a" * 64,
        duration_seconds=2.5,
        fps=18,
        endpoint_ssim=0.8,
        motion_stability=0.7,
        projection_legibility=0.1,
    )

    reasons = gate.rejection_reasons(evidence)

    assert len(reasons) == 5
    assert any("duration" in reason for reason in reasons)
    assert any("endpoint SSIM" in reason for reason in reasons)


def test_live_scene_adapter_emits_progressive_checksum_cached_story_packs(
    tmp_path: Path,
) -> None:
    cloud_prompts: list[str] = []

    class StubFiniteProvider:
        async def generate_fast(
            self,
            request: FastSceneRequest,
            *,
            output_dir: Path,
        ) -> FiniteSceneBundle:
            cloud_prompts.append(request.prompt)
            output_dir.mkdir(parents=True)
            master = _artifact(output_dir / "master.png", b"master", "image/png")
            depth = _artifact(output_dir / "depth.png", b"depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": request.scene_id,
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                        "remote_seconds": 5.0,
                        "inference_seconds": 4.5,
                        "provider_overhead_seconds": 0.5,
                        "packaging_seconds": 0.04,
                        "image_seconds": 4.0,
                        "depth_seconds": 0.5,
                        "warm_state": "cold",
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            path = output_dir / "scene.manifest.json"
            path.write_text(json.dumps(payload))
            return load_finite_scene_bundle(path)

        async def upgrade_motion(
            self,
            bundle: FiniteSceneBundle,
            request: MotionUpgradeRequest,
        ) -> FiniteSceneBundle:
            cloud_prompts.append(request.prompt)
            payload = json.loads(bundle.manifest_path.read_text())
            motion = _artifact(
                bundle.manifest_path.parent / "motion.mp4",
                b"motion",
                "video/mp4",
            )
            motion.update({"duration_ms": 3375, "frames": 81, "fps": 24})
            payload["stages"]["motion"] = {
                "model": MOTION_MODEL,
                "model_revision": MOTION_MODEL_REVISION,
                "gpu": "L4",
                "finite_call": True,
                "estimated_gpu_usd": 0.02,
                "inference_seconds": 18.0,
                "remote_seconds": 20.0,
                "provider_overhead_seconds": 2.0,
                "warm_state": "cold",
            }
            payload["artifacts"]["motion"] = motion
            bundle.manifest_path.write_text(json.dumps(payload))
            assert request.seed == 8
            return load_finite_scene_bundle(bundle.manifest_path)

    async def run():
        cache = AssetCache(tmp_path / "cache")
        await cache.initialize()
        provider = FiniteModalLiveSceneProvider(
            StubFiniteProvider(),  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "generated",
            enable_motion=True,
            motion_evaluator=lambda path: MotionTechnicalEvidence(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                duration_seconds=3.375,
                fps=24,
                endpoint_ssim=0.98,
                motion_stability=0.9,
                projection_legibility=0.7,
            ),
        )
        active_promotions = 0
        peak_promotions = 0
        original_promote = provider._promote_artifact  # noqa: SLF001

        async def observed_promote(**kwargs):
            nonlocal active_promotions, peak_promotions
            if kwargs["role"] in {"master", "depth"}:
                active_promotions += 1
                peak_promotions = max(peak_promotions, active_promotions)
                try:
                    await asyncio.sleep(0.01)
                    return await original_promote(**kwargs)
                finally:
                    active_promotions -= 1
            return await original_promote(**kwargs)

        provider._promote_artifact = observed_promote  # type: ignore[method-assign]  # noqa: SLF001
        request = LiveSceneCreateRequest(
            text="A winged library rises into the stars.",
            seed=7,
        )
        updates = [
            update
            async for update in provider.generate(
                request,
                job_id="scene_000000000000000000000001",
            )
        ]
        return cache, updates, peak_promotions

    cache, updates, peak_promotions = asyncio.run(run())

    assert [update.stage for update in updates] == [
        LiveSceneStage.DRAFT_READY,
        LiveSceneStage.MASTER_READY,
        LiveSceneStage.MOTION_READY,
    ]
    assert updates[-1].complete is True
    assert len(cloud_prompts) == 2
    assert all("A winged library rises into the stars." not in prompt for prompt in cloud_prompts)
    assert all("luminous watercolor paper theater" in prompt for prompt in cloud_prompts)
    assert [artifact.kind for artifact in updates[-1].artifacts] == [
        LiveSceneArtifactKind.MASTER,
        LiveSceneArtifactKind.DEPTH,
        LiveSceneArtifactKind.MOTION,
    ]
    assert len(updates[-1].story_pack.assets) == 3
    assert peak_promotions == 2
    assert updates[-1].story_pack.compiler_model == "deterministic-live-scene-planner-v3"
    assert updates[1].metrics is not None
    assert updates[1].metrics.provider_ms == 5_000
    assert updates[1].metrics.inference_ms == 4_500
    assert updates[1].metrics.overhead_ms == 500
    assert updates[1].metrics.packaging_ms == 40
    assert updates[1].metrics.warm_state is LiveSceneWarmState.COLD
    assert updates[1].metrics.cost_source is LiveSceneCostSource.PROVIDER_MANIFEST
    assert updates[1].metrics.planning_status.value == "deterministic"
    assert updates[1].metrics.planning_ms == 0
    assert [model.role for model in updates[1].metrics.models] == [
        "scene_plan",
        "master",
        "depth",
    ]
    assert updates[-1].metrics is not None
    assert updates[-1].metrics.provider_ms == 25_000
    assert updates[-1].metrics.inference_ms == 22_500
    assert updates[-1].metrics.estimated_gpu_usd == pytest.approx(0.03)
    assert [model.role for model in updates[-1].metrics.models] == [
        "scene_plan",
        "master",
        "depth",
        "motion",
    ]
    for artifact in updates[-1].artifacts:
        filename = artifact.uri.rsplit("/", 1)[-1]
        cached = cache.resolve(artifact.checksum_sha256, filename)
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == artifact.checksum_sha256


def test_live_scene_adapter_uses_model_plan_after_immediate_deterministic_draft(
    tmp_path: Path,
) -> None:
    class PlannerStub:
        def __init__(self) -> None:
            self.calls = 0

        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            assert kwargs["text"] == (
                "Quenlora whispers the amber-key refrain while paper birds cross the stars."
            )
            self.calls += 1
            return LiveScenePlanningResult(
                plan=_gemma_live_plan(),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=8_400,
                    input_tokens=224,
                    output_tokens=168,
                ),
                model_revision="sha256:gemma-fixture",
                wall_ms=8_450,
            )

    class CapturingProvider:
        def __init__(self) -> None:
            self.request: FastSceneRequest | None = None

        async def generate_fast(
            self,
            request: FastSceneRequest,
            *,
            output_dir: Path,
        ) -> FiniteSceneBundle:
            self.request = request
            output_dir.mkdir(parents=True)
            master = _artifact(output_dir / "master.png", b"planned-master", "image/png")
            depth = _artifact(output_dir / "depth.png", b"planned-depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": request.scene_id,
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                        "remote_seconds": 4.0,
                        "inference_seconds": 3.5,
                        "provider_overhead_seconds": 0.5,
                        "warm_state": "warm",
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            manifest = output_dir / "scene.manifest.json"
            manifest.write_text(json.dumps(payload))
            return load_finite_scene_bundle(manifest)

    async def run():
        cache = AssetCache(tmp_path / "model-plan-cache")
        await cache.initialize()
        planner = PlannerStub()
        finite = CapturingProvider()
        adapter = FiniteModalLiveSceneProvider(
            finite,  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "model-plan-output",
            planner=planner,
            fidelity_mode="deferred",
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(
                text=("Quenlora whispers the amber-key refrain while paper birds cross the stars."),
                seed=31,
            ),
            job_id="scene_000000000000000000000031",
        )
        draft = await anext(iterator)
        assert planner.calls == 0
        master = await anext(iterator)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return draft, master, planner, finite

    draft, master, planner, finite = asyncio.run(run())

    assert draft.stage is LiveSceneStage.DRAFT_READY
    assert draft.story_pack.compiler_model == "deterministic-live-scene-planner-v3"
    assert planner.calls == 1
    assert finite.request is not None
    assert _gemma_live_plan().art_direction in finite.request.prompt
    assert "luminous watercolor paper theater" in finite.request.prompt
    assert ".." not in finite.request.prompt
    assert "Quenlora" not in finite.request.prompt
    assert "amber-key refrain" not in finite.request.prompt
    assert finite.request.fidelity_label == ""
    assert finite.request.fidelity_object_label == ""
    assert finite.request.require_subject_object_overlap is False
    assert master.complete is True
    assert master.story_pack.compiler_model == "gemma3:1b"
    assert master.story_pack.pages[0].scene_summary == _gemma_live_plan().scene_summary
    assert [layer.layer_id for layer in master.story_pack.pages[0].layers] == [
        "scene-background",
        "scene-focus",
        "scene-accent",
    ]
    assert {asset.layer_id for asset in master.story_pack.assets} == {"scene-background"}
    scene_spec = master.story_pack.pages[0].scene_spec
    assert scene_spec is not None
    normalized_focus = scene_spec.composition[1]
    normalized_accent = scene_spec.composition[2]
    assert (normalized_focus.width, normalized_focus.height) == (0.6, 0.72)
    assert (normalized_focus.center_x, normalized_focus.center_y) == (0.5, 0.5)
    assert (normalized_accent.width, normalized_accent.height) == (0.3, 0.34)
    assert (normalized_accent.center_x, normalized_accent.center_y) == pytest.approx((0.81, 0.21))
    assert normalized_focus.depth != normalized_accent.depth
    assert "[0.5" not in master.story_pack.pages[0].layers[0].prompt
    assert master.story_pack.pages[0].source_text.startswith("Quenlora whispers")
    assert master.metrics is not None
    assert master.metrics.planning_status.value == "model"
    assert master.metrics.planning_ms == 8_450
    assert master.metrics.elapsed_ms == (
        master.metrics.planning_ms + master.metrics.provider_ms + master.metrics.cache_ms
    )
    assert [(model.role, model.model, model.revision) for model in master.metrics.models] == [
        ("scene_plan", "gemma3:1b", "sha256:gemma-fixture"),
        ("master", FAST_MODEL, FAST_MODEL_REVISION),
        (
            "depth",
            finite_modal_provider_module.DEPTH_MODEL,
            finite_modal_provider_module.DEPTH_MODEL_REVISION,
        ),
    ]


def test_live_scene_adapter_stops_before_cloud_render_when_model_planner_fails(
    tmp_path: Path,
) -> None:
    class FailingPlanner:
        async def plan(self, **kwargs):
            del kwargs
            raise LiveScenePlannerError("Jetson planner unavailable")

    class CapturingProvider:
        def __init__(self) -> None:
            self.request: FastSceneRequest | None = None

        async def generate_fast(
            self,
            request: FastSceneRequest,
            *,
            output_dir: Path,
        ) -> FiniteSceneBundle:
            self.request = request
            output_dir.mkdir(parents=True)
            master = _artifact(output_dir / "master.png", b"fallback-master", "image/png")
            depth = _artifact(output_dir / "depth.png", b"fallback-depth", "image/png")
            payload = {
                "schema_version": "1.0",
                "provider": "modal-finite",
                "scene_id": request.scene_id,
                "stages": {
                    "fast": {
                        "model": FAST_MODEL,
                        "model_revision": FAST_MODEL_REVISION,
                        "gpu": "L4",
                        "finite_call": True,
                        "estimated_gpu_usd": 0.01,
                        "remote_seconds": 4.0,
                        "inference_seconds": 3.5,
                        "provider_overhead_seconds": 0.5,
                    }
                },
                "artifacts": {"master": master, "depth": depth},
            }
            manifest = output_dir / "scene.manifest.json"
            manifest.write_text(json.dumps(payload))
            return load_finite_scene_bundle(manifest)

    async def run():
        cache = AssetCache(tmp_path / "fallback-cache")
        await cache.initialize()
        finite = CapturingProvider()
        adapter = FiniteModalLiveSceneProvider(
            finite,  # type: ignore[arg-type]
            cache=cache,
            output_root=tmp_path / "fallback-output",
            planner=FailingPlanner(),  # type: ignore[arg-type]
        )
        updates = []
        with pytest.raises(
            LiveSceneProviderUnavailableError,
            match="no cloud render was started",
        ):
            async for update in adapter.generate(
                LiveSceneCreateRequest(text="A whale carries a library over the ocean.", seed=9),
                job_id="scene_000000000000000000000009",
            ):
                updates.append(update)
        return updates, finite

    updates, finite = asyncio.run(run())
    assert [update.stage for update in updates] == [LiveSceneStage.DRAFT_READY]
    assert finite.request is None


def test_live_scene_adapter_stops_before_cloud_render_for_unsafe_model_plan(
    tmp_path: Path,
) -> None:
    class UnsafePlanner:
        async def plan(self, **kwargs) -> LiveScenePlanningResult:
            assert "Quenlora" in kwargs["text"]
            payload = _gemma_live_plan().model_dump()
            payload["focus"]["prompt"] = "Quenlora opens the secret amber gate"
            return LiveScenePlanningResult(
                plan=LiveScenePlan.model_validate(payload),
                metrics=ModelMetrics(
                    backend="ollama",
                    model="gemma3:1b",
                    total_ms=500,
                    input_tokens=100,
                    output_tokens=100,
                ),
                model_revision="ollama-manifest-sha256:unsafe-fixture",
                wall_ms=550,
            )

    async def run():
        request = LiveSceneCreateRequest(
            text="A child named Quenlora opens the secret amber gate.",
            seed=41,
        )
        draft = build_live_scene_story_pack(
            request,
            job_id="scene_000000000000000000000041",
            seed=41,
            assets=[],
            compiler_model="deterministic-live-scene-planner-v1",
        )
        adapter = FiniteModalLiveSceneProvider(
            object(),  # type: ignore[arg-type]
            cache=AssetCache(tmp_path / "privacy-fallback-cache"),
            planner=UnsafePlanner(),  # type: ignore[arg-type]
        )
        with pytest.raises(
            LiveSceneProviderUnavailableError,
            match="no cloud render was started",
        ):
            await adapter._resolve_plan(
                request,
                job_id="scene_000000000000000000000041",
                seed=41,
                draft=draft,
            )

    asyncio.run(run())
