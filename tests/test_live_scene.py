import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter

import pytest
from pydantic import ValidationError

from bookforge.asset_cache import AssetCache
from bookforge.live_scene import (
    DeterministicFakeLiveSceneProvider,
    LiveSceneCapacityError,
    LiveSceneCreateRequest,
    LiveSceneJobRegistry,
    LiveSceneMetrics,
    LiveSceneModelProvenance,
    LiveSceneNotFoundError,
    LiveSceneRegistryClosedError,
    LiveSceneStage,
    LiveSceneUpdate,
    build_live_scene_provider,
    build_live_scene_story_pack,
)


def test_live_scene_request_is_strict_text_only() -> None:
    request = LiveSceneCreateRequest(
        text="  A fox opens a book beneath the stars.  ",
        visual_style=" watercolor theater ",
        seed=7,
        session_id="reader-1",
    )

    assert request.text == "A fox opens a book beneath the stars."
    assert request.visual_style == "watercolor theater"
    with pytest.raises(ValidationError, match="raw_audio"):
        LiveSceneCreateRequest.model_validate(
            {
                "text": "A valid scene description.",
                "visual_style": "paper theater",
                "raw_audio": "base64-is-not-accepted",
            }
        )
    with pytest.raises(ValidationError):
        LiveSceneCreateRequest(text="x")
    with pytest.raises(ValidationError):
        LiveSceneCreateRequest(text="valid text", seed=2**32)


def test_fake_provider_emits_valid_progressive_story_packs_and_cached_assets(
    tmp_path: Path,
) -> None:
    async def exercise():
        cache = AssetCache(tmp_path / "cache")
        await cache.initialize()
        registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(cache=cache),
            event_queue_size=8,
        )
        created = await registry.submit(
            LiveSceneCreateRequest(
                text="A silver fox discovers a glowing word in an old library.",
                visual_style="luminous cut-paper watercolor",
                seed=42,
            )
        )
        subscription = await registry.subscribe(created.job_id)
        snapshots = [created]
        async with subscription:
            while not snapshots[-1].terminal:
                snapshots.append(await subscription.receive())
        terminal = snapshots[-1]

        assert terminal.stage is LiveSceneStage.MOTION_READY
        assert terminal.complete is True
        assert terminal.progress == 1
        assert terminal.story_pack is not None
        assert terminal.story_pack.pages[0].scene_spec is not None
        assert {artifact.kind.value for artifact in terminal.artifacts} == {
            "master",
            "depth",
            "motion",
        }
        for artifact in terminal.artifacts:
            assert artifact.uri.startswith("/v1/assets/")
            filename = artifact.uri.rsplit("/", 1)[1]
            assert cache.resolve(artifact.checksum_sha256, filename).is_file()

        observed = [snapshot.stage for snapshot in snapshots]
        assert observed[0] is LiveSceneStage.QUEUED
        assert observed[-1] is LiveSceneStage.MOTION_READY
        assert LiveSceneStage.DRAFT_READY in observed
        assert LiveSceneStage.MASTER_READY in observed
        revisions = [snapshot.revision for snapshot in snapshots]
        assert revisions == sorted(revisions)
        elapsed = [snapshot.metrics.elapsed_ms for snapshot in snapshots]
        assert elapsed == sorted(elapsed)
        for snapshot in snapshots:
            assert snapshot.stage in snapshot.metrics.milestones_ms
        assert terminal.metrics.provider_ms == 30
        assert terminal.metrics.inference_ms == 22
        assert terminal.metrics.cache_ms == 3
        assert terminal.metrics.overhead_ms == 5
        assert terminal.metrics.warm_state.value == "unknown"
        assert terminal.metrics.gpu is None
        assert terminal.metrics.estimated_gpu_usd == 0
        assert terminal.metrics.cost_source.value == "fixture"
        assert [model.role for model in terminal.metrics.models] == [
            "scene_plan",
            "master",
            "depth",
            "motion",
        ]
        assert revisions[1:] == list(range(1, terminal.revision + 1))
        await registry.close()

    asyncio.run(exercise())


def test_live_scene_metrics_are_strict_and_require_immutable_model_roles() -> None:
    model = LiveSceneModelProvenance(
        role="master",
        model="example/model",
        revision="sha-123",
    )
    metrics = LiveSceneMetrics(
        elapsed_ms=12,
        provider_ms=8,
        inference_ms=6,
        cache_ms=1,
        overhead_ms=2,
        warm_state="cold",
        gpu="NVIDIA L4",
        estimated_gpu_usd=0.01,
        cost_source="provider_manifest",
        models=[model],
        milestones_ms={LiveSceneStage.DRAFT_READY: 12},
    )

    assert metrics.models[0].revision == "sha-123"
    with pytest.raises(ValidationError):
        LiveSceneMetrics(provider_ms=-1)
    with pytest.raises(ValidationError, match="unavailable cost evidence"):
        LiveSceneMetrics(estimated_gpu_usd=0.01)
    with pytest.raises(ValidationError, match="model roles must be unique"):
        LiveSceneMetrics(models=[model, model])


def test_deterministic_draft_is_visibly_passage_derived_before_inference() -> None:
    space = build_live_scene_story_pack(
        LiveSceneCreateRequest(text="A rocket crossed the moon and a field of stars."),
        job_id="scene_000000000000000000000001",
        seed=11,
        assets=[],
        compiler_model="draft-fixture@v1",
    )
    ocean = build_live_scene_story_pack(
        LiveSceneCreateRequest(text="A whale swam beneath the ocean waves."),
        job_id="scene_000000000000000000000002",
        seed=22,
        assets=[],
        compiler_model="draft-fixture@v1",
    )

    space_page = space.pages[0]
    ocean_page = ocean.pages[0]
    assert len(space_page.layers) == 3
    assert len(ocean_page.layers) == 3
    assert space_page.scene_spec is not None
    assert ocean_page.scene_spec is not None
    assert {effect.kind for effect in space_page.scene_spec.ambience} == {"stars"}
    assert {effect.color for effect in ocean_page.scene_spec.ambience} == {
        "#83f3e4",
        "#b9fff4",
    }
    assert space_page.scene_spec.composition != ocean_page.scene_spec.composition
    assert space_page.layers[-1].prompt != ocean_page.layers[-1].prompt


def test_fake_provider_is_deterministic_for_the_same_request() -> None:
    async def collect(provider, request):
        return [
            update
            async for update in provider.generate(
                request,
                job_id="scene_0123456789abcdef01234567",
            )
        ]

    request = LiveSceneCreateRequest(
        text="The moonlit letters rise from the page.",
        visual_style="ink and paper",
    )
    first = asyncio.run(collect(DeterministicFakeLiveSceneProvider(), request))
    second = asyncio.run(collect(DeterministicFakeLiveSceneProvider(), request))

    assert [update.model_dump() for update in first] == [
        update.model_dump() for update in second
    ]


@pytest.mark.parametrize(
    ("field", "forged_value", "message"),
    [
        ("checksum_sha256", "0" * 64, "metadata differs"),
        ("uri", "/v1/assets/forged/master.png", "URI does not match"),
    ],
)
def test_live_scene_update_rejects_forged_artifact_asset_metadata(
    field: str,
    forged_value: str,
    message: str,
) -> None:
    async def master_update() -> LiveSceneUpdate:
        async for update in DeterministicFakeLiveSceneProvider().generate(
            LiveSceneCreateRequest(text="A fox reads a map."),
            job_id="scene_0123456789abcdef01234567",
        ):
            if update.stage is LiveSceneStage.MASTER_READY:
                return update
        raise AssertionError("fake provider did not emit master_ready")

    payload = asyncio.run(master_update()).model_dump(mode="json")
    payload["artifacts"][0][field] = forged_value

    with pytest.raises(ValidationError, match=message):
        LiveSceneUpdate.model_validate(payload)


class _MasterOnlyProvider:
    name = "fake"

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        async for update in DeterministicFakeLiveSceneProvider().generate(
            request,
            job_id=job_id,
        ):
            yield update
            if update.stage is LiveSceneStage.MASTER_READY:
                return


class _MotionFailureProvider:
    name = "fake"

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        async for update in DeterministicFakeLiveSceneProvider().generate(
            request,
            job_id=job_id,
        ):
            if update.stage is LiveSceneStage.MOTION_READY:
                raise RuntimeError("optional video worker was unavailable")
            yield update


@pytest.mark.parametrize(
    ("provider", "warning_code"),
    [
        (_MasterOnlyProvider(), None),
        (_MotionFailureProvider(), "motion_upgrade_failed"),
    ],
)
def test_master_scene_is_successful_without_optional_video(provider, warning_code) -> None:
    async def exercise():
        registry = LiveSceneJobRegistry(provider)
        created = await registry.submit(
            LiveSceneCreateRequest(text="A child follows a trail of golden letters.")
        )
        terminal = await registry.wait(created.job_id)
        await registry.close()
        return terminal

    terminal = asyncio.run(exercise())

    assert terminal.stage is LiveSceneStage.MASTER_READY
    assert terminal.complete is True
    assert terminal.story_pack is not None
    assert {artifact.kind.value for artifact in terminal.artifacts} == {"master", "depth"}
    assert terminal.error is None
    assert (terminal.warning.code if terminal.warning else None) == warning_code


class _FailBeforeMasterProvider:
    name = "broken"

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        del request, job_id
        raise RuntimeError("planning backend failed")
        yield  # pragma: no cover


def test_failure_before_master_is_terminal_and_explicit() -> None:
    async def exercise():
        registry = LiveSceneJobRegistry(_FailBeforeMasterProvider())
        created = await registry.submit(LiveSceneCreateRequest(text="A usable text prompt."))
        terminal = await registry.wait(created.job_id)
        await registry.close()
        return terminal

    terminal = asyncio.run(exercise())

    assert terminal.stage is LiveSceneStage.FAILED
    assert terminal.complete is True
    assert terminal.error is not None
    assert terminal.error.code == "generation_failed"
    assert terminal.story_pack is None


class _BlockingProvider:
    name = "blocking"

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        await self.release.wait()
        async for update in DeterministicFakeLiveSceneProvider().generate(
            request,
            job_id=job_id,
        ):
            yield update


def test_registry_enforces_active_capacity() -> None:
    async def exercise():
        provider = _BlockingProvider()
        registry = LiveSceneJobRegistry(provider, max_active_jobs=1, max_retained_jobs=2)
        first = await registry.submit(LiveSceneCreateRequest(text="First scene prompt."))
        with pytest.raises(LiveSceneCapacityError):
            await registry.submit(LiveSceneCreateRequest(text="Second scene prompt."))
        provider.release.set()
        assert (await registry.wait(first.job_id)).terminal
        await registry.close()

    asyncio.run(exercise())


def test_registry_evicts_oldest_completed_job_within_retention_bound() -> None:
    async def exercise():
        registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(),
            max_active_jobs=1,
            max_retained_jobs=2,
        )
        ids = []
        for number in range(3):
            created = await registry.submit(
                LiveSceneCreateRequest(text=f"Scene number {number} is valid.")
            )
            ids.append(created.job_id)
            await registry.wait(created.job_id)
        with pytest.raises(LiveSceneNotFoundError):
            await registry.get(ids[0])
        assert (await registry.get(ids[1])).terminal
        assert (await registry.get(ids[2])).terminal
        await registry.close()

    asyncio.run(exercise())


def test_abandoned_terminal_subscription_cannot_pin_retained_capacity() -> None:
    async def exercise() -> None:
        registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(),
            max_active_jobs=1,
            max_retained_jobs=1,
        )
        first = await registry.submit(
            LiveSceneCreateRequest(text="First retained job.", session_id="first-session")
        )
        await registry.wait(first.job_id)
        abandoned = await registry.subscribe(first.job_id)

        second = await registry.submit(
            LiveSceneCreateRequest(text="Second retained job.", session_id="second-session")
        )
        with pytest.raises(LiveSceneNotFoundError):
            await registry.get(first.job_id)
        with pytest.raises(LiveSceneNotFoundError):
            await registry.get_session("first-session")
        assert (await abandoned.receive()).terminal
        with pytest.raises(LiveSceneRegistryClosedError, match="subscription is closed"):
            await abandoned.receive()
        await registry.wait(second.job_id)
        await registry.close()

    asyncio.run(exercise())


def test_provider_factory_selects_fake_and_finite_modal_without_persistent_service(
    tmp_path: Path,
) -> None:
    cache = AssetCache(tmp_path / "cache")
    asyncio.run(cache.initialize())

    fake = build_live_scene_provider("auto", asset_backend="fake", cache=cache)
    modal = build_live_scene_provider(
        "auto",
        asset_backend="modal",
        cache=cache,
        output_root=tmp_path / "generated",
    )
    modal_warm = build_live_scene_provider(
        "modal_warm",
        asset_backend="disabled",
        cache=cache,
        output_root=tmp_path / "warm-generated",
    )
    explicit_fake = build_live_scene_provider(
        "fake",
        asset_backend="disabled",
        cache=cache,
    )

    assert fake.name == "fake"
    assert explicit_fake.name == "fake"
    assert modal.name == "modal-finite"
    assert modal_warm.name == "modal-finite"
    assert modal_warm.provider.__class__.__name__ == "WarmModalSceneProvider"  # type: ignore[attr-defined]
    assert modal.enable_motion is False  # type: ignore[attr-defined]
    assert modal.output_root == tmp_path / "generated"  # type: ignore[attr-defined]
    assert LiveSceneJobRegistry(modal, max_active_jobs=8).max_active_jobs == 1
    assert LiveSceneJobRegistry(modal_warm, max_active_jobs=8).max_active_jobs == 1
    assert LiveSceneJobRegistry(fake, max_active_jobs=8).max_active_jobs == 8


def test_session_rendezvous_is_monotonic_and_never_reverts_to_an_older_job() -> None:
    async def exercise() -> None:
        registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(stage_delay_seconds=0.01),
            max_active_jobs=2,
        )
        first = await registry.submit(
            LiveSceneCreateRequest(text="The first scene waits.", session_id="shared")
        )
        first_pointer = await registry.get_session("shared")
        second = await registry.submit(
            LiveSceneCreateRequest(text="The newer scene wins.", session_id="shared")
        )
        second_pointer = await registry.get_session("shared")

        assert first_pointer.job.job_id == first.job_id
        assert second_pointer.job.job_id == second.job_id
        assert second_pointer.session_revision > first_pointer.session_revision
        await registry.wait(second.job_id)
        assert (await registry.get(first.job_id)).error.code == "superseded"  # type: ignore[union-attr]

        final_pointer = await registry.get_session("shared")
        assert final_pointer.session_revision == second_pointer.session_revision
        assert final_pointer.job.job_id == second.job_id
        await registry.close()

    asyncio.run(exercise())


def test_new_same_session_job_supersedes_and_cancels_the_prior_billable_task() -> None:
    class BillableBlockingProvider:
        name = "fake"

        def __init__(self) -> None:
            self.started: asyncio.Queue[str] = asyncio.Queue()
            self.releases: dict[str, asyncio.Event] = {}
            self.active: set[str] = set()
            self.cancelled: set[str] = set()

        async def generate(
            self,
            request: LiveSceneCreateRequest,
            *,
            job_id: str,
        ) -> AsyncIterator[LiveSceneUpdate]:
            release = self.releases.setdefault(job_id, asyncio.Event())
            self.active.add(job_id)
            await self.started.put(job_id)
            try:
                await release.wait()
                async for update in DeterministicFakeLiveSceneProvider().generate(
                    request,
                    job_id=job_id,
                ):
                    yield update
            except asyncio.CancelledError:
                self.cancelled.add(job_id)
                raise
            finally:
                self.active.discard(job_id)

    async def exercise() -> None:
        provider = BillableBlockingProvider()
        registry = LiveSceneJobRegistry(
            provider,
            max_active_jobs=1,
            max_retained_jobs=2,
        )
        first = await registry.submit(
            LiveSceneCreateRequest(text="The old costly scene.", session_id="shared")
        )
        assert await provider.started.get() == first.job_id

        second = await registry.submit(
            LiveSceneCreateRequest(text="The replacement scene.", session_id="shared")
        )
        assert await provider.started.get() == second.job_id
        await asyncio.sleep(0)

        old = await registry.get(first.job_id)
        assert old.stage is LiveSceneStage.FAILED
        assert old.complete is True
        assert old.error is not None and old.error.code == "superseded"
        assert first.job_id in provider.cancelled
        assert provider.active == {second.job_id}
        assert (await registry.get_session("shared")).job.job_id == second.job_id

        provider.releases[second.job_id].set()
        assert (await registry.wait(second.job_id)).terminal
        await registry.close()

    asyncio.run(exercise())


def test_session_rendezvous_isolated_by_session_and_missing_for_unscoped_jobs() -> None:
    async def exercise() -> None:
        registry = LiveSceneJobRegistry(DeterministicFakeLiveSceneProvider())
        alpha = await registry.submit(
            LiveSceneCreateRequest(text="Alpha sees a moon.", session_id="alpha")
        )
        beta = await registry.submit(
            LiveSceneCreateRequest(text="Beta sees a star.", session_id="beta")
        )
        await registry.wait(alpha.job_id)
        await registry.wait(beta.job_id)

        assert (await registry.get_session("alpha")).job.job_id == alpha.job_id
        assert (await registry.get_session("beta")).job.job_id == beta.job_id
        await registry.submit(LiveSceneCreateRequest(text="No browser session here."))
        with pytest.raises(LiveSceneNotFoundError):
            await registry.get_session("missing")
        await registry.close()

    asyncio.run(exercise())


def test_session_revision_epoch_changes_across_registry_restart() -> None:
    async def exercise() -> None:
        first_registry = LiveSceneJobRegistry(DeterministicFakeLiveSceneProvider())
        first = await first_registry.submit(
            LiveSceneCreateRequest(text="Before restart.", session_id="kiosk")
        )
        first_pointer = await first_registry.get_session("kiosk")
        await first_registry.close()

        restarted_registry = LiveSceneJobRegistry(DeterministicFakeLiveSceneProvider())
        restarted = await restarted_registry.submit(
            LiveSceneCreateRequest(text="After restart.", session_id="kiosk")
        )
        restarted_pointer = await restarted_registry.get_session("kiosk")

        assert first_pointer.session_revision == 1
        assert restarted_pointer.session_revision == 1
        assert first_pointer.server_instance_id != restarted_pointer.server_instance_id
        assert first_pointer.job.job_id == first.job_id
        assert restarted_pointer.job.job_id == restarted.job_id
        await restarted_registry.close()

    asyncio.run(exercise())


def test_session_stream_subscribes_before_first_job_and_follows_replacement() -> None:
    async def exercise() -> None:
        registry = LiveSceneJobRegistry(
            DeterministicFakeLiveSceneProvider(stage_delay_seconds=0.01)
        )
        subscription = await registry.subscribe_session("cross-device")
        async with subscription:
            epoch = await asyncio.wait_for(subscription.receive(), timeout=0.1)
            assert epoch.server_instance_id == registry.server_instance_id
            assert epoch.session_revision == 0
            assert epoch.job is None

            started = perf_counter()
            first = await registry.submit(
                LiveSceneCreateRequest(
                    text="The first streamed scene.",
                    session_id="cross-device",
                )
            )
            queued = await asyncio.wait_for(subscription.receive(), timeout=0.1)
            assert perf_counter() - started < 0.1
            assert queued.session_revision == 1
            assert queued.job is not None and queued.job.job_id == first.job_id
            assert queued.job.stage is LiveSceneStage.QUEUED
            draft = None
            for _ in range(4):
                candidate = await asyncio.wait_for(subscription.receive(), timeout=0.1)
                if (
                    candidate.job is not None
                    and candidate.job.job_id == first.job_id
                    and candidate.job.stage is LiveSceneStage.DRAFT_READY
                ):
                    draft = candidate
                    break
            assert draft is not None
            assert perf_counter() - started < 0.1

            second = await registry.submit(
                LiveSceneCreateRequest(
                    text="The replacement streamed scene.",
                    session_id="cross-device",
                )
            )
            replacement = None
            for _ in range(8):
                candidate = await asyncio.wait_for(subscription.receive(), timeout=0.1)
                if candidate.job is not None and candidate.job.job_id == second.job_id:
                    replacement = candidate
                    break
            assert replacement is not None
            assert replacement.session_revision == 2
            assert replacement.job is not None
            assert replacement.job.stage in {LiveSceneStage.QUEUED, LiveSceneStage.PLANNING}
            await registry.wait(second.job_id)
        await registry.close()

    asyncio.run(exercise())
