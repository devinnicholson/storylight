import asyncio
import hashlib
import json
import struct
from pathlib import Path

import pytest

import bookforge.finite_modal_provider as finite_modal_provider_module
from bookforge.asset_cache import AssetCache
from bookforge.domain import ModelMetrics
from bookforge.finite_modal_provider import (
    FAST_MODEL,
    FAST_MODEL_REVISION,
    MOTION_MODEL,
    MOTION_MODEL_REVISION,
    WARM_FULL_SESSION_CEILING_USD,
    FastSceneRequest,
    FiniteModalBudgetError,
    FiniteModalLiveSceneProvider,
    FiniteModalProviderError,
    FiniteModalSceneProvider,
    FiniteSceneBundle,
    MotionTechnicalEvidence,
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
    LiveSceneWarmState,
    build_live_scene_story_pack,
)
from bookforge.live_scene_planner import (
    LiveSceneGraphPlan,
    LiveScenePlacedLayerPlan,
    LiveScenePlan,
    LiveScenePlannerError,
    LiveScenePlanningResult,
)
from bookforge.modal_budget import budget_envelope_from_plan
from bookforge.scene_facts import (
    SceneEventFact,
    SceneFactsV2,
    SceneObjectFact,
    SceneSettingFact,
    SceneSubjectFact,
    SceneTemporalOrderFact,
    SceneTransformationFact,
)
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
    ['[{"description":"missing"}]', '[{"cost":"NaN"}]', '[{"cost":"1.0","Cost":"2.0"}]'],
)
def test_modal_billing_parser_rejects_ambiguous_or_unsafe_reports(payload: str) -> None:
    with pytest.raises((TypeError, ValueError, json.JSONDecodeError)):
        finite_modal_provider_module._parse_modal_billing_total(payload)


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


def _scene_scope_plan(facts: SceneFactsV2) -> LiveSceneGraphPlan:
    return LiveSceneGraphPlan(
        scene_summary="A cave scene",
        art_direction="watercolor",
        camera_motion="locked",
        background_prompt="cave",
        focus_label="fox",
        focus=LiveScenePlacedLayerPlan(
            kind="character", prompt="fox", anchor=(0.4, 0.5, 0.3, 0.4), depth=5,
            motion="parallax",
        ),
        accent=LiveScenePlacedLayerPlan(
            kind="prop", prompt="box", anchor=(0.7, 0.5, 0.2, 0.2), depth=3,
            motion="parallax",
        ),
        scene_facts=facts,
    )


@pytest.mark.parametrize("actors", ["group", "colors"])
def test_scene_scope_validates_before_billing_and_renders_graph_subject_count(
    tmp_path: Path, actors
):
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"),
        subjects=(SceneSubjectFact(ref="fox", label="fox", count=3, actions=("stand",)),)
        if actors == "group" else (
            SceneSubjectFact(ref="silver", label="fox", color="silver", actions=("stands",)),
            SceneSubjectFact(ref="red", label="fox", color="red", actions=("stands",)),
        ),
        objects=(SceneObjectFact(ref="box", label="box"),),
    )
    source = (
        "In a cave, three foxes stand beside a box." if actors == "group" else
        "In a cave, a silver fox stands beside a box. A red fox stands beside the box."
    )
    planner_started, release_planner = asyncio.Event(), asyncio.Event()
    preparation_started = []
    original_authorize = warm.prepare_fast_authorization

    async def authorize(**kwargs):
        preparation_started.append("authorize")
        await original_authorize(**kwargs)

    warm.prepare_fast_authorization = authorize

    class Planner:
        planning_scope = "scene"

        async def has_cached_plan(self, **kwargs):
            raise AssertionError("scene scope must skip the generic preview route")

        async def plan(self, **kwargs):
            planner_started.set()
            await release_planner.wait()
            return LiveScenePlanningResult(
                plan=_scene_scope_plan(facts).model_copy(
                    update={"focus_label": "fox" if actors == "group" else "silver fox"}
                ),
                metrics=ModelMetrics(backend="fixture", model="fixture", total_ms=1),
                model_revision="fixture", wall_ms=1,
            )

    async def run():
        cache = AssetCache(tmp_path / "scene-cache")
        await cache.initialize()
        adapter = FiniteModalLiveSceneProvider(
            warm, cache=cache, planner=Planner(), output_root=tmp_path / "scene-output"
        )
        iterator = adapter.generate(
            LiveSceneCreateRequest(text=source), job_id="scene_scoped_count"
        )
        draft = await anext(iterator)
        task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(planner_started.wait(), timeout=1)
        assert preparation_started == [] and invoker.calls == []
        release_planner.set()
        master = await asyncio.wait_for(task, timeout=2)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return draft, master

    draft, master = asyncio.run(run())
    assert draft.story_pack.planning_scope == master.story_pack.planning_scope == "scene"
    assert preparation_started == ["authorize"]
    assert len(invoker.calls) == 1 and invoker.calls[0][1] == "generate"
    arguments = invoker.calls[0][2]
    assert arguments["expected_subject_count"] == (3 if actors == "group" else 2)
    assert arguments["fidelity_label"] == "fox"
    assert arguments["prompt"] == facts.to_renderer_prompt(
        source_text=source, visual_style="luminous watercolor paper theater"
    )
    assert "duplicate character" not in arguments["negative_prompt"]


@pytest.mark.parametrize(
    "unsupported", [
        "count", "mixed", "transformation", "order", "legacy", "long-style", "oversized-prompt"
    ]
)
def test_scene_scope_refuses_unsupported_plan_before_paid_preparation(
    tmp_path: Path, unsupported, monkeypatch
):
    invoker = StubWarmInvoker()
    warm = _warm_provider(tmp_path, invoker)
    subjects = (SceneSubjectFact(ref="fox", label="fox", count=5 if unsupported == "count" else 1),)
    if unsupported == "mixed":
        subjects += (SceneSubjectFact(ref="owl", label="owl"),)
    facts = SceneFactsV2(
        setting=SceneSettingFact(label="cave"), subjects=subjects,
        objects=(SceneObjectFact(ref="box", label="box"),),
        transformation=(
            SceneTransformationFact(source="box", result_label="birds")
            if unsupported == "transformation" else None
        ),
        events=(
            SceneEventFact(ref="open", source="fox", action="opens", object="box"),
            SceneEventFact(ref="close", source="fox", action="closes", object="box"),
        ) if unsupported == "order" else (),
        temporal_order=(SceneTemporalOrderFact(before="open", after="close"),)
        if unsupported == "order" else (),
    )

    async def unexpected_preparation(**kwargs):
        raise AssertionError("unsupported scene reached paid preparation")

    warm.prepare_fast_authorization = unexpected_preparation
    warm.prewarm = unexpected_preparation
    if unsupported == "oversized-prompt":
        monkeypatch.setattr(SceneFactsV2, "to_renderer_prompt", lambda self, **kwargs: "x" * 4_001)

    class Planner:
        planning_scope = "scene"

        async def plan(self, **kwargs):
            return LiveScenePlanningResult(
                plan=_gemma_live_plan() if unsupported == "legacy" else _scene_scope_plan(facts),
                metrics=ModelMetrics(backend="fixture", model="fixture", total_ms=1),
                model_revision="fixture", wall_ms=1,
            )

    async def run():
        adapter = FiniteModalLiveSceneProvider(
            warm, cache=AssetCache(tmp_path / "cache"), planner=Planner(),
            auto_prewarm_on_submit=True,
        )
        with pytest.raises(LiveSceneProviderUnavailableError, match="no cloud render"):
            async for _ in adapter.generate(
                LiveSceneCreateRequest(
                    text="In a cave, a fox opens a box.",
                    visual_style=(
                        "watercolor " * 12 if unsupported == "long-style" else "watercolor"
                    ),
                ),
                job_id="scene_scope_refusal",
            ):
                pass

    asyncio.run(run())
    assert invoker.calls == []


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
        async with asyncio.timeout(1):
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
