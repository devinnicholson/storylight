"""Bounded, provider-neutral control plane for progressive live-scene generation.

The control plane deliberately accepts text only. Providers may run locally or in a
remote GPU environment, but callers never submit microphone recordings or camera
frames through this API.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from bookforge.asset_cache import AssetCache
from bookforge.domain import (
    AmbientEffect,
    AmbientMotion,
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetState,
    CameraMotion,
    GeneratedPagePlan,
    LayerComposition,
    SceneCanvas,
    SceneSpecV2,
    StoryPack,
    VisualLayer,
)
from bookforge.event_hub import SessionId

if TYPE_CHECKING:
    from bookforge.model_client import StructuredModelClient

LiveSceneJobId = Annotated[
    str,
    StringConstraints(pattern=r"^scene_[a-f0-9]{24}$"),
]
LiveSceneServerInstanceId = Annotated[
    str,
    StringConstraints(pattern=r"^server_[a-f0-9]{32}$"),
]
LiveScenePrewarmId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$"),
]
SceneText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=4_000),
]
VisualStyle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
Checksum = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LiveSceneStage(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    DRAFT_READY = "draft_ready"
    MASTER_READY = "master_ready"
    MOTION_READY = "motion_ready"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {self.MOTION_READY, self.FAILED}


class LiveSceneArtifactKind(StrEnum):
    DRAFT = "draft"
    MASTER = "master"
    DEPTH = "depth"
    MOTION = "motion"


class LiveSceneWarmState(StrEnum):
    COLD = "cold"
    WARM = "warm"
    UNKNOWN = "unknown"


class LiveSceneCostSource(StrEnum):
    FIXTURE = "fixture"
    PROVIDER_MANIFEST = "provider_manifest"
    UNAVAILABLE = "unavailable"


class LiveScenePlanningStatus(StrEnum):
    PENDING = "pending"
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    FALLBACK = "fallback"


class LiveSceneModelProvenance(FrozenStrictModel):
    role: Annotated[
        str,
        StringConstraints(pattern=r"^[a-z][a-z0-9_-]{1,63}$"),
    ]
    model: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    revision: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]


class LiveSceneMetrics(FrozenStrictModel):
    """Backend-owned cumulative timing, hardware, model, and cost evidence."""

    elapsed_ms: Annotated[float, Field(ge=0)] = 0
    provider_ms: Annotated[float, Field(ge=0)] = 0
    inference_ms: Annotated[float, Field(ge=0)] = 0
    cache_ms: Annotated[float, Field(ge=0)] = 0
    overhead_ms: Annotated[float, Field(ge=0)] = 0
    packaging_ms: Annotated[float, Field(ge=0)] = 0
    planning_ms: Annotated[float, Field(ge=0)] = 0
    preparation_ms: Annotated[float, Field(ge=0)] = 0
    planning_status: LiveScenePlanningStatus = LiveScenePlanningStatus.PENDING
    planning_cache_hit: bool = False
    scene_cache_hit: bool = False
    warm_state: LiveSceneWarmState = LiveSceneWarmState.UNKNOWN
    gpu: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
        ]
        | None
    ) = None
    estimated_gpu_usd: Annotated[float, Field(ge=0)] = 0
    cost_source: LiveSceneCostSource = LiveSceneCostSource.UNAVAILABLE
    models: list[LiveSceneModelProvenance] = Field(default_factory=list, max_length=8)
    milestones_ms: dict[LiveSceneStage, Annotated[float, Field(ge=0)]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_coherent_evidence(self) -> LiveSceneMetrics:
        roles = [model.role for model in self.models]
        if len(roles) != len(set(roles)):
            raise ValueError("metric model roles must be unique")
        if (
            self.planning_status is not LiveScenePlanningStatus.PENDING
            and "scene_plan" not in roles
        ):
            raise ValueError("completed planning metrics require scene-plan provenance")
        if self.planning_status is LiveScenePlanningStatus.PENDING and self.planning_ms != 0:
            raise ValueError("pending planning metrics cannot report planning time")
        if self.cost_source is LiveSceneCostSource.UNAVAILABLE and self.estimated_gpu_usd != 0:
            raise ValueError("unavailable cost evidence cannot report a GPU estimate")
        if self.scene_cache_hit and (
            self.provider_ms != 0 or self.inference_ms != 0 or self.estimated_gpu_usd != 0
        ):
            raise ValueError("restored scenes cannot report new provider work or GPU cost")
        return self


class LiveSceneCreateRequest(FrozenStrictModel):
    """A privacy-bounded request: story text and visual direction, never raw media."""

    text: SceneText
    visual_style: VisualStyle = "luminous watercolor paper theater"
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None
    session_id: SessionId | None = None


class LiveScenePrewarmRequest(FrozenStrictModel):
    prewarm_id: LiveScenePrewarmId
    include_motion: bool = False
    scaledown_window_seconds: Annotated[int, Field(ge=90, le=900)] = 90


class LiveScenePrewarmResponse(FrozenStrictModel):
    prewarm_id: LiveScenePrewarmId
    reservation_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    include_motion: bool
    fast_remote_seconds: Annotated[float, Field(ge=0)]
    motion_remote_seconds: Annotated[float, Field(ge=0)]
    full_session_ceiling_usd: Annotated[float, Field(gt=0)]
    fast_model_load_seconds: Annotated[float, Field(ge=0)]
    motion_model_load_seconds: Annotated[float, Field(ge=0)]
    expires_in_seconds: Annotated[float, Field(ge=0)]
    scaledown_window_seconds: Annotated[int, Field(ge=90, le=900)] = 90
    fast_inference_warmup_seconds: Annotated[float, Field(ge=0)] = 0


class LiveScenePlannerPrepareRequest(FrozenStrictModel):
    """Prime only the private local planner; this request never reaches a renderer."""

    text: SceneText
    visual_style: VisualStyle = "luminous watercolor paper theater"
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0


class LiveScenePlannerPrepareResponse(FrozenStrictModel):
    ready: bool = True
    planning_ms: Annotated[float, Field(ge=0)]
    cache_hit: bool
    model: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    revision: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class LiveSceneWarmProviderStatus(FrozenStrictModel):
    ready: bool
    detail: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]
    state: Annotated[str, StringConstraints(pattern=r"^(idle|prewarmed)$")]
    prewarm_id: LiveScenePrewarmId | None = None
    include_motion: bool = False
    expires_in_seconds: Annotated[float, Field(ge=0)] = 0
    scaledown_window_seconds: Annotated[int, Field(ge=90, le=900)] = 90

    @model_validator(mode="after")
    def require_prewarm_identity(self) -> LiveSceneWarmProviderStatus:
        if self.state == "prewarmed" and self.prewarm_id is None:
            raise ValueError("prewarmed status requires prewarm_id")
        if self.state == "idle" and self.prewarm_id is not None:
            raise ValueError("idle warm provider cannot include prewarm_id")
        return self


class LiveSceneArtifact(FrozenStrictModel):
    artifact_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
    ]
    kind: LiveSceneArtifactKind
    uri: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048)]
    checksum_sha256: Checksum
    media_type: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=100),
    ]
    provider: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    ]
    model: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    width: Annotated[int, Field(ge=1, le=8_192)] | None = None
    height: Annotated[int, Field(ge=1, le=8_192)] | None = None
    duration_ms: Annotated[int, Field(ge=0, le=300_000)] = 0

    @model_validator(mode="after")
    def require_paired_dimensions(self) -> LiveSceneArtifact:
        if (self.width is None) != (self.height is None):
            raise ValueError("artifact width and height must be provided together")
        if self.kind is LiveSceneArtifactKind.MOTION and self.duration_ms == 0:
            raise ValueError("motion artifacts require a positive duration_ms")
        if self.kind is not LiveSceneArtifactKind.MOTION and self.duration_ms != 0:
            raise ValueError("only motion artifacts accept duration_ms")
        return self


class LiveSceneError(FrozenStrictModel):
    code: Annotated[
        str,
        StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$"),
    ]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    retryable: bool = False


_HEAVY_ARTIFACTS = {
    LiveSceneArtifactKind.MASTER,
    LiveSceneArtifactKind.DEPTH,
    LiveSceneArtifactKind.MOTION,
}


class LiveSceneJob(FrozenStrictModel):
    job_id: LiveSceneJobId
    stage: LiveSceneStage
    revision: Annotated[int, Field(ge=1)]
    progress: Annotated[float, Field(ge=0, le=1)]
    complete: bool = False
    provider: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    ]
    request: LiveSceneCreateRequest
    artifacts: list[LiveSceneArtifact] = Field(default_factory=list)
    story_pack: StoryPack | None = None
    error: LiveSceneError | None = None
    warning: LiveSceneError | None = None
    metrics: LiveSceneMetrics = Field(default_factory=LiveSceneMetrics)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_stage_payload(self) -> LiveSceneJob:
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("job timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("live-scene artifact IDs must be unique")

        if self.story_pack is not None:
            assets = {asset.asset_id: asset for asset in self.story_pack.assets}
            heavy_ids = {
                artifact.artifact_id
                for artifact in self.artifacts
                if artifact.kind is not LiveSceneArtifactKind.DRAFT
            }
            if heavy_ids != set(assets):
                raise ValueError("live-scene artifacts must exactly cover Story Pack assets")
            expected_asset_types = {
                LiveSceneArtifactKind.MASTER: (AssetKind.IMAGE, AssetRole.MASTER),
                LiveSceneArtifactKind.DEPTH: (AssetKind.DEPTH_MAP, AssetRole.DEPTH),
                LiveSceneArtifactKind.MOTION: (AssetKind.VIDEO_LOOP, AssetRole.MOTION),
            }
            for artifact in self.artifacts:
                if artifact.kind is LiveSceneArtifactKind.DRAFT:
                    continue
                asset = assets[artifact.artifact_id]
                expected_kind, expected_role = expected_asset_types[artifact.kind]
                if asset.kind is not expected_kind or asset.role is not expected_role:
                    raise ValueError(
                        f"artifact {artifact.artifact_id!r} kind/role does not match its asset"
                    )
                if artifact.uri != asset.local_uri:
                    raise ValueError(
                        f"artifact {artifact.artifact_id!r} URI does not match its asset"
                    )
                parity = {
                    "checksum": artifact.checksum_sha256 == asset.checksum_sha256,
                    "seed": artifact.seed == asset.seed,
                    "width": artifact.width == asset.width,
                    "height": artifact.height == asset.height,
                    "duration": artifact.duration_ms == asset.duration_ms,
                }
                mismatches = [name for name, matches in parity.items() if not matches]
                if mismatches:
                    raise ValueError(
                        f"artifact {artifact.artifact_id!r} metadata differs from its asset: "
                        f"{', '.join(mismatches)}"
                    )
                accepted_provenance = {
                    artifact.provider,
                    f"{artifact.provider}:{artifact.model}",
                }
                if self.provider != artifact.provider or asset.provider not in accepted_provenance:
                    raise ValueError(
                        f"artifact {artifact.artifact_id!r} provider provenance is inconsistent"
                    )
            scene_plan = next(
                (model for model in self.metrics.models if model.role == "scene_plan"),
                None,
            )
            if scene_plan is not None and scene_plan.model != self.story_pack.compiler_model:
                raise ValueError("Story Pack compiler_model must match scene-plan provenance")

        if self.stage is LiveSceneStage.FAILED:
            if self.error is None:
                raise ValueError("failed jobs require an error")
            if not self.complete:
                raise ValueError("failed jobs must be complete")
            if self.warning is not None:
                raise ValueError("failed jobs cannot include a warning")
            return self
        if self.error is not None:
            raise ValueError("only failed jobs may include an error")

        if self.stage in {LiveSceneStage.QUEUED, LiveSceneStage.PLANNING}:
            if self.story_pack is not None or self.artifacts or self.complete:
                raise ValueError("queued and planning jobs cannot include generated output")
            if self.warning is not None:
                raise ValueError("incomplete jobs cannot include a warning")
            return self

        if self.story_pack is None:
            raise ValueError(f"{self.stage.value} jobs require a Story Pack snapshot")

        kinds = {artifact.kind for artifact in self.artifacts}
        if self.stage is LiveSceneStage.DRAFT_READY and kinds & _HEAVY_ARTIFACTS:
            raise ValueError("draft-ready jobs cannot include heavy generated assets")
        if self.stage is LiveSceneStage.DRAFT_READY and self.complete:
            raise ValueError("draft-ready jobs cannot be complete")
        if self.stage is LiveSceneStage.MASTER_READY:
            if not {LiveSceneArtifactKind.MASTER, LiveSceneArtifactKind.DEPTH} <= kinds:
                raise ValueError("master-ready jobs require master and depth artifacts")
            if LiveSceneArtifactKind.MOTION in kinds:
                raise ValueError("master-ready jobs cannot include motion artifacts")
            if self.complete and self.progress != 1:
                raise ValueError("completed master-ready jobs require progress=1")
        if self.stage is LiveSceneStage.MOTION_READY:
            required = {
                LiveSceneArtifactKind.MASTER,
                LiveSceneArtifactKind.DEPTH,
                LiveSceneArtifactKind.MOTION,
            }
            if not required <= kinds:
                raise ValueError("motion-ready jobs require master, depth, and motion artifacts")
            if self.progress != 1:
                raise ValueError("motion-ready jobs require progress=1")
            if not self.complete:
                raise ValueError("motion-ready jobs must be complete")
        if self.warning is not None and not (
            self.stage is LiveSceneStage.MASTER_READY and self.complete
        ):
            raise ValueError("warnings are only valid on completed master-ready jobs")

        return self

    @property
    def terminal(self) -> bool:
        return self.complete


class LiveSceneSessionStatus(FrozenStrictModel):
    """Authoritative rendezvous pointer for browsers sharing one API server."""

    session_id: SessionId
    server_instance_id: LiveSceneServerInstanceId
    session_revision: Annotated[int, Field(ge=1)]
    job: LiveSceneJob

    @model_validator(mode="after")
    def require_matching_session(self) -> LiveSceneSessionStatus:
        if self.job.request.session_id != self.session_id:
            raise ValueError("rendezvous job does not belong to this session")
        return self


class LiveSceneSessionEvent(FrozenStrictModel):
    """Long-lived session stream event, including the pre-job server epoch."""

    session_id: SessionId
    server_instance_id: LiveSceneServerInstanceId
    session_revision: Annotated[int, Field(ge=0)]
    job: LiveSceneJob | None = None

    @model_validator(mode="after")
    def require_coherent_pointer(self) -> LiveSceneSessionEvent:
        if self.job is None and self.session_revision != 0:
            raise ValueError("an empty session event must use revision zero")
        if self.job is not None:
            if self.session_revision < 1:
                raise ValueError("a session job event requires a positive session revision")
            if self.job.request.session_id != self.session_id:
                raise ValueError("session event job does not belong to this session")
        return self


class LiveSceneUpdate(FrozenStrictModel):
    stage: LiveSceneStage
    progress: Annotated[float, Field(gt=0, le=1)]
    complete: bool = False
    artifacts: list[LiveSceneArtifact] = Field(default_factory=list)
    story_pack: StoryPack
    metrics: LiveSceneMetrics | None = None

    @model_validator(mode="after")
    def require_generation_stage(self) -> LiveSceneUpdate:
        if self.stage not in {
            LiveSceneStage.DRAFT_READY,
            LiveSceneStage.MASTER_READY,
            LiveSceneStage.MOTION_READY,
        }:
            raise ValueError("providers may only emit generated ready stages")
        # Reuse the externally visible contract as the canonical payload validator.
        update_provider = self.artifacts[0].provider if self.artifacts else "contract-validator"
        LiveSceneJob(
            job_id="scene_000000000000000000000000",
            stage=self.stage,
            revision=1,
            progress=self.progress,
            complete=self.complete,
            provider=update_provider,
            request=LiveSceneCreateRequest(text="validation text"),
            artifacts=self.artifacts,
            story_pack=self.story_pack,
            metrics=self.metrics or LiveSceneMetrics(),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        return self


class LiveSceneProvider(Protocol):
    @property
    def name(self) -> str: ...

    def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]: ...


class LiveSceneRegistryError(RuntimeError):
    pass


class LiveSceneCapacityError(LiveSceneRegistryError):
    pass


class LiveSceneNotFoundError(LiveSceneRegistryError, LookupError):
    pass


class LiveSceneRegistryClosedError(LiveSceneRegistryError):
    pass


class LiveSceneProviderProtocolError(LiveSceneRegistryError):
    pass


class LiveSceneProviderUnavailableError(RuntimeError):
    pass


_CLOSED = object()
_NEXT_STAGE = {
    LiveSceneStage.QUEUED: LiveSceneStage.PLANNING,
    LiveSceneStage.PLANNING: LiveSceneStage.DRAFT_READY,
    LiveSceneStage.DRAFT_READY: LiveSceneStage.MASTER_READY,
    LiveSceneStage.MASTER_READY: LiveSceneStage.MOTION_READY,
}


@dataclass(eq=False, slots=True)
class _JobRecord:
    snapshot: LiveSceneJob
    started_monotonic: float
    subscribers: set[LiveSceneSubscription] = field(default_factory=set)


@dataclass(eq=False, slots=True)
class LiveSceneSubscription:
    _registry: LiveSceneJobRegistry
    job_id: str
    _queue: asyncio.Queue[LiveSceneJob | object]
    _closed: bool = False

    async def receive(self) -> LiveSceneJob:
        if self._closed and self._queue.empty():
            raise LiveSceneRegistryClosedError("Live-scene subscription is closed")
        item = await self._queue.get()
        if item is _CLOSED:
            raise LiveSceneRegistryClosedError("Live-scene job registry is closed")
        assert isinstance(item, LiveSceneJob)
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._registry.unsubscribe(self)

    async def __aenter__(self) -> LiveSceneSubscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


@dataclass(eq=False, slots=True)
class LiveSceneSessionSubscription:
    _registry: LiveSceneJobRegistry
    session_id: str
    _queue: asyncio.Queue[LiveSceneSessionEvent | object]
    _closed: bool = False

    async def receive(self) -> LiveSceneSessionEvent:
        if self._closed and self._queue.empty():
            raise LiveSceneRegistryClosedError("Live-scene session subscription is closed")
        item = await self._queue.get()
        if item is _CLOSED:
            raise LiveSceneRegistryClosedError("Live-scene job registry is closed")
        assert isinstance(item, LiveSceneSessionEvent)
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._registry.unsubscribe_session(self)

    async def __aenter__(self) -> LiveSceneSessionSubscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class LiveSceneJobRegistry:
    """Process-local bounded registry and fan-out for progressive generation jobs."""

    def __init__(
        self,
        provider: LiveSceneProvider,
        *,
        max_active_jobs: int = 2,
        max_retained_jobs: int = 64,
        event_queue_size: int = 8,
        completed_pack_sink: Callable[[StoryPack], Awaitable[object]] | None = None,
        completed_pack_source: (
            Callable[[LiveSceneCreateRequest], Awaitable[StoryPack | None]] | None
        ) = None,
        completed_pack_validator: (Callable[[StoryPack], Awaitable[StoryPack]] | None) = None,
    ) -> None:
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        effective_max_active_jobs = 1 if provider.name == "modal-finite" else max_active_jobs
        if max_retained_jobs < effective_max_active_jobs:
            raise ValueError("max_retained_jobs must be at least the effective max_active_jobs")
        if event_queue_size < 1:
            raise ValueError("event_queue_size must be at least 1")
        self.provider = provider
        self.server_instance_id = f"server_{uuid4().hex}"
        # Paid finite jobs are serialized even if a broader local concurrency was configured.
        self.max_active_jobs = effective_max_active_jobs
        self.max_retained_jobs = max_retained_jobs
        self.event_queue_size = event_queue_size
        self.completed_pack_sink = completed_pack_sink
        self.completed_pack_source = completed_pack_source
        self.completed_pack_validator = completed_pack_validator
        self._jobs: OrderedDict[str, _JobRecord] = OrderedDict()
        self._session_jobs: dict[str, tuple[int, str]] = {}
        self._session_subscribers: dict[str, set[LiveSceneSessionSubscription]] = {}
        self._session_revision_sequence = 0
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def submit(self, request: LiveSceneCreateRequest) -> LiveSceneJob:
        async with self._lock:
            if self._closed:
                raise LiveSceneRegistryClosedError("Live-scene job registry is closed")
            if request.session_id is not None:
                pointer = self._session_jobs.get(request.session_id)
                if pointer is not None:
                    previous = self._jobs.get(pointer[1])
                    if previous is not None and not previous.snapshot.terminal:
                        current = previous.snapshot
                        superseded = LiveSceneJob.model_validate(
                            current.model_copy(
                                update={
                                    "stage": LiveSceneStage.FAILED,
                                    "revision": current.revision + 1,
                                    "complete": True,
                                    "error": LiveSceneError(
                                        code="superseded",
                                        message=(
                                            "A newer generation job replaced this session job"
                                        ),
                                        retryable=False,
                                    ),
                                    "metrics": self._metrics_locked(
                                        previous,
                                        stage=LiveSceneStage.FAILED,
                                    ),
                                    "updated_at": datetime.now(UTC),
                                }
                            ).model_dump()
                        )
                        self._publish_locked(previous, superseded)
                        previous_task = self._tasks.get(current.job_id)
                        if previous_task is not None:
                            previous_task.cancel()
            active_count = sum(not record.snapshot.terminal for record in self._jobs.values())
            if active_count >= self.max_active_jobs:
                raise LiveSceneCapacityError(
                    f"Live-scene capacity is full ({self.max_active_jobs} active jobs)"
                )
            self._evict_completed_locked()
            if len(self._jobs) >= self.max_retained_jobs:
                raise LiveSceneCapacityError("Live-scene retained-job capacity is full")

            job_id = f"scene_{uuid4().hex[:24]}"
            now = datetime.now(UTC)
            snapshot = LiveSceneJob(
                job_id=job_id,
                stage=LiveSceneStage.QUEUED,
                revision=1,
                progress=0,
                provider=self.provider.name,
                request=request,
                metrics=LiveSceneMetrics(milestones_ms={LiveSceneStage.QUEUED: 0}),
                created_at=now,
                updated_at=now,
            )
            record = _JobRecord(snapshot=snapshot, started_monotonic=perf_counter())
            self._jobs[job_id] = record
            if request.session_id is not None:
                self._session_revision_sequence += 1
                self._session_jobs[request.session_id] = (
                    self._session_revision_sequence,
                    job_id,
                )
                self._publish_session_locked(request.session_id)
            task = asyncio.create_task(self._run(job_id), name=f"live-scene:{job_id}")
            self._tasks[job_id] = task
            task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
            return snapshot

    async def get(self, job_id: str) -> LiveSceneJob:
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise LiveSceneNotFoundError(f"Live-scene job {job_id!r} was not found")
            return record.snapshot

    async def get_session(self, session_id: str) -> LiveSceneSessionStatus:
        async with self._lock:
            pointer = self._session_jobs.get(session_id)
            if pointer is None:
                raise LiveSceneNotFoundError(
                    f"Live-scene session {session_id!r} has no generation job"
                )
            session_revision, job_id = pointer
            record = self._jobs.get(job_id)
            if record is None:
                # A stale pointer must never fall back to an older job.
                self._session_jobs.pop(session_id, None)
                raise LiveSceneNotFoundError(
                    f"Live-scene session {session_id!r} has no retained generation job"
                )
            return LiveSceneSessionStatus(
                session_id=session_id,
                server_instance_id=self.server_instance_id,
                session_revision=session_revision,
                job=record.snapshot,
            )

    async def subscribe(
        self,
        job_id: str,
        *,
        after_revision: int = 0,
    ) -> LiveSceneSubscription:
        if after_revision < 0:
            raise ValueError("after_revision cannot be negative")
        queue: asyncio.Queue[LiveSceneJob | object] = asyncio.Queue(self.event_queue_size)
        subscription = LiveSceneSubscription(self, job_id, queue)
        async with self._lock:
            if self._closed:
                raise LiveSceneRegistryClosedError("Live-scene job registry is closed")
            record = self._jobs.get(job_id)
            if record is None:
                raise LiveSceneNotFoundError(f"Live-scene job {job_id!r} was not found")
            record.subscribers.add(subscription)
            if (
                record.snapshot.revision > after_revision
                or record.snapshot.terminal
                or after_revision > record.snapshot.revision
            ):
                self._put_snapshot(queue, record.snapshot)
        return subscription

    async def subscribe_session(self, session_id: str) -> LiveSceneSessionSubscription:
        """Subscribe before a job exists and follow replacements for one browser session."""

        queue: asyncio.Queue[LiveSceneSessionEvent | object] = asyncio.Queue(self.event_queue_size)
        subscription = LiveSceneSessionSubscription(self, session_id, queue)
        async with self._lock:
            if self._closed:
                raise LiveSceneRegistryClosedError("Live-scene job registry is closed")
            self._session_subscribers.setdefault(session_id, set()).add(subscription)
            self._put_session_event(queue, self._session_event_locked(session_id))
        return subscription

    async def unsubscribe(self, subscription: LiveSceneSubscription) -> None:
        async with self._lock:
            record = self._jobs.get(subscription.job_id)
            if record is not None:
                record.subscribers.discard(subscription)

    async def unsubscribe_session(
        self,
        subscription: LiveSceneSessionSubscription,
    ) -> None:
        async with self._lock:
            subscriptions = self._session_subscribers.get(subscription.session_id)
            if subscriptions is None:
                return
            subscriptions.discard(subscription)
            if not subscriptions:
                self._session_subscribers.pop(subscription.session_id, None)

    async def wait(self, job_id: str) -> LiveSceneJob:
        """Wait for a terminal snapshot; useful for tests and local orchestration."""

        subscription = await self.subscribe(job_id)
        async with subscription:
            while True:
                snapshot = await subscription.receive()
                if snapshot.terminal:
                    return snapshot

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks.values())
            subscriptions = [
                subscription
                for record in self._jobs.values()
                for subscription in record.subscribers
            ]
            session_subscriptions = [
                subscription
                for group in self._session_subscribers.values()
                for subscription in group
            ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            for subscription in subscriptions:
                subscription._closed = True
                self._put_closed(subscription._queue)
            for subscription in session_subscriptions:
                subscription._closed = True
                self._put_closed(subscription._queue)
            for record in self._jobs.values():
                record.subscribers.clear()
            self._tasks.clear()
            self._session_subscribers.clear()

    async def _run(self, job_id: str) -> None:
        try:
            await self._transition(job_id, stage=LiveSceneStage.PLANNING, progress=0.1)
            request = (await self.get(job_id)).request
            if await self._restore_completed_scene(job_id, request):
                return
            emitted = False
            async for update in self.provider.generate(request, job_id=job_id):
                emitted = True
                if update.complete and update.story_pack is not None:
                    await self._persist_completed_pack(update.story_pack)
                snapshot = await self._transition(
                    job_id,
                    stage=update.stage,
                    progress=update.progress,
                    complete=update.complete,
                    artifacts=update.artifacts,
                    story_pack=update.story_pack,
                    metrics=update.metrics,
                )
                if snapshot.terminal:
                    return
            current = await self.get(job_id)
            if emitted and current.stage is LiveSceneStage.MASTER_READY:
                assert current.story_pack is not None
                await self._persist_completed_pack(current.story_pack)
                await self._complete_master(job_id)
                return
            if not emitted or not current.terminal:
                raise LiveSceneProviderProtocolError(
                    "Provider ended before emitting a usable master scene"
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # Providers are an explicit failure boundary.
            current = await self.get(job_id)
            if current.stage is LiveSceneStage.MASTER_READY:
                assert current.story_pack is not None
                await self._persist_completed_pack(current.story_pack)
                await self._complete_master(
                    job_id,
                    warning=LiveSceneError(
                        code="motion_upgrade_failed",
                        message=_error_message(error),
                        retryable=True,
                    ),
                )
                return
            code = (
                "provider_unavailable"
                if isinstance(error, LiveSceneProviderUnavailableError)
                else "generation_failed"
            )
            await self._fail(
                job_id,
                LiveSceneError(
                    code=code,
                    message=_error_message(error),
                    retryable=isinstance(error, LiveSceneProviderUnavailableError),
                ),
            )

    async def _restore_completed_scene(
        self,
        job_id: str,
        request: LiveSceneCreateRequest,
    ) -> bool:
        """Publish a checksum-verified exact scene without invoking the provider."""

        if self.completed_pack_source is None or self.completed_pack_validator is None:
            return False
        started = perf_counter()
        try:
            pack = await self.completed_pack_source(request)
            if pack is None:
                return False
            pack = await self.completed_pack_validator(pack)
            artifacts = _cached_live_scene_artifacts(pack, provider=self.provider.name)
        except Exception:
            # A stale/corrupt/incompatible cache is only a missed optimization.
            return False

        try:
            cache_ms = max(0.0, (perf_counter() - started) * 1000)
            master_artifacts = [
                artifact
                for artifact in artifacts
                if artifact.kind
                in {
                    LiveSceneArtifactKind.MASTER,
                    LiveSceneArtifactKind.DEPTH,
                }
            ]
            master_asset_ids = {artifact.artifact_id for artifact in master_artifacts}
            master_pack = pack.model_copy(
                update={
                    "assets": [
                        asset for asset in pack.assets if asset.asset_id in master_asset_ids
                    ]
                }
            )
            draft_pack = master_pack.model_copy(update={"assets": []})
            draft_metrics = _cached_live_scene_metrics(
                master_pack,
                artifacts=master_artifacts,
                cache_ms=cache_ms,
                include_heavy_models=False,
            )
            motion_artifact = next(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.kind is LiveSceneArtifactKind.MOTION
                ),
                None,
            )
            master_metrics = _cached_live_scene_metrics(
                master_pack,
                artifacts=master_artifacts,
                cache_ms=cache_ms,
                include_heavy_models=True,
            )
            motion_metrics = (
                _cached_live_scene_metrics(
                    pack,
                    artifacts=artifacts,
                    cache_ms=cache_ms,
                    include_heavy_models=True,
                )
                if motion_artifact is not None
                else None
            )
            # Validate every stage before publishing the draft. An invalid cache
            # must remain a clean miss so the provider can still run normally.
            LiveSceneUpdate(
                stage=LiveSceneStage.DRAFT_READY,
                progress=0.55,
                story_pack=draft_pack,
                metrics=draft_metrics,
            )
            LiveSceneUpdate(
                stage=LiveSceneStage.MASTER_READY,
                progress=1 if motion_artifact is None else 0.82,
                complete=motion_artifact is None,
                artifacts=master_artifacts,
                story_pack=master_pack,
                metrics=master_metrics,
            )
            if motion_artifact is not None:
                LiveSceneUpdate(
                    stage=LiveSceneStage.MOTION_READY,
                    progress=1,
                    complete=True,
                    artifacts=artifacts,
                    story_pack=pack,
                    metrics=motion_metrics,
                )
        except Exception:
            return False

        await self._transition(
            job_id,
            stage=LiveSceneStage.DRAFT_READY,
            progress=0.55,
            story_pack=draft_pack,
            metrics=draft_metrics,
        )
        master_snapshot = await self._transition(
            job_id,
            stage=LiveSceneStage.MASTER_READY,
            progress=1 if motion_artifact is None else 0.82,
            complete=motion_artifact is None,
            artifacts=master_artifacts,
            story_pack=master_pack,
            metrics=master_metrics,
        )
        if master_snapshot.terminal:
            return True
        assert motion_artifact is not None
        assert motion_metrics is not None
        await self._transition(
            job_id,
            stage=LiveSceneStage.MOTION_READY,
            progress=1,
            complete=True,
            artifacts=artifacts,
            story_pack=pack,
            metrics=motion_metrics,
        )
        return True

    async def _persist_completed_pack(self, pack: StoryPack) -> None:
        if self.completed_pack_sink is None:
            return
        try:
            await self.completed_pack_sink(pack)
        except Exception:
            # Generation already succeeded and its assets remain usable. Local
            # persistence is a recovery optimization, not a reason to discard it.
            return

    async def _transition(
        self,
        job_id: str,
        *,
        stage: LiveSceneStage,
        progress: float,
        complete: bool = False,
        artifacts: list[LiveSceneArtifact] | None = None,
        story_pack: StoryPack | None = None,
        metrics: LiveSceneMetrics | None = None,
    ) -> LiveSceneJob:
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise LiveSceneNotFoundError(f"Live-scene job {job_id!r} was not found")
            current = record.snapshot
            expected = _NEXT_STAGE.get(current.stage)
            if stage is not expected:
                raise LiveSceneProviderProtocolError(
                    f"Invalid live-scene transition {current.stage.value} -> {stage.value}"
                )
            if progress < current.progress:
                raise LiveSceneProviderProtocolError("Live-scene progress cannot move backwards")
            self._validate_metrics_progression(current.metrics, metrics)
            now = datetime.now(UTC)
            snapshot = current.model_copy(
                update={
                    "stage": stage,
                    "revision": current.revision + 1,
                    "progress": progress,
                    "complete": complete,
                    "artifacts": artifacts or [],
                    "story_pack": story_pack,
                    "metrics": self._metrics_locked(
                        record,
                        stage=stage,
                        provider_metrics=metrics,
                    ),
                    "updated_at": now,
                }
            )
            # model_copy does not revalidate in Pydantic v2.
            snapshot = LiveSceneJob.model_validate(snapshot.model_dump())
            self._publish_locked(record, snapshot)
            return snapshot

    async def _complete_master(
        self,
        job_id: str,
        *,
        warning: LiveSceneError | None = None,
    ) -> LiveSceneJob:
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise LiveSceneNotFoundError(f"Live-scene job {job_id!r} was not found")
            current = record.snapshot
            if current.stage is not LiveSceneStage.MASTER_READY:
                raise LiveSceneProviderProtocolError(
                    "Only a master-ready scene can complete without a motion artifact"
                )
            if current.complete:
                return current
            snapshot = LiveSceneJob.model_validate(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "complete": True,
                        "progress": 1,
                        "warning": warning,
                        "metrics": self._metrics_locked(
                            record,
                            stage=LiveSceneStage.MASTER_READY,
                        ),
                        "updated_at": datetime.now(UTC),
                    }
                ).model_dump()
            )
            self._publish_locked(record, snapshot)
            return snapshot

    async def _fail(self, job_id: str, error: LiveSceneError) -> LiveSceneJob:
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.snapshot.terminal:
                if record is None:
                    raise LiveSceneNotFoundError(f"Live-scene job {job_id!r} was not found")
                return record.snapshot
            current = record.snapshot
            snapshot = LiveSceneJob.model_validate(
                current.model_copy(
                    update={
                        "stage": LiveSceneStage.FAILED,
                        "revision": current.revision + 1,
                        "complete": True,
                        "error": error,
                        "metrics": self._metrics_locked(
                            record,
                            stage=LiveSceneStage.FAILED,
                        ),
                        "updated_at": datetime.now(UTC),
                    }
                ).model_dump()
            )
            self._publish_locked(record, snapshot)
            return snapshot

    def _metrics_locked(
        self,
        record: _JobRecord,
        *,
        stage: LiveSceneStage,
        provider_metrics: LiveSceneMetrics | None = None,
    ) -> LiveSceneMetrics:
        current = record.snapshot.metrics
        source = provider_metrics or current
        elapsed_ms = max(
            current.elapsed_ms,
            round((perf_counter() - record.started_monotonic) * 1000, 3),
        )
        milestones = dict(current.milestones_ms)
        milestones.update(source.milestones_ms)
        milestones[stage] = elapsed_ms
        return LiveSceneMetrics.model_validate(
            source.model_copy(
                update={
                    "elapsed_ms": elapsed_ms,
                    "milestones_ms": milestones,
                }
            ).model_dump()
        )

    @staticmethod
    def _validate_metrics_progression(
        current: LiveSceneMetrics,
        update: LiveSceneMetrics | None,
    ) -> None:
        if update is None:
            return
        cumulative_fields = (
            "provider_ms",
            "inference_ms",
            "cache_ms",
            "overhead_ms",
            "packaging_ms",
            "planning_ms",
            "preparation_ms",
            "estimated_gpu_usd",
        )
        for field_name in cumulative_fields:
            if getattr(update, field_name) < getattr(current, field_name):
                raise LiveSceneProviderProtocolError(
                    f"Live-scene metric {field_name} cannot move backwards"
                )
        if (
            current.planning_status is not LiveScenePlanningStatus.PENDING
            and update.planning_status is not current.planning_status
        ):
            raise LiveSceneProviderProtocolError(
                "Live-scene planning status cannot change after planning completes"
            )
        if (
            current.planning_status is not LiveScenePlanningStatus.PENDING
            and update.planning_cache_hit is not current.planning_cache_hit
        ):
            raise LiveSceneProviderProtocolError(
                "Live-scene planning cache evidence cannot change after planning completes"
            )
        if current.scene_cache_hit and not update.scene_cache_hit:
            raise LiveSceneProviderProtocolError(
                "Live-scene completed-scene cache evidence cannot be removed"
            )
        previous_models = {model.role: model for model in current.models}
        for model in update.models:
            previous = previous_models.get(model.role)
            if previous is not None and previous != model:
                raise LiveSceneProviderProtocolError(
                    f"Live-scene model revision changed for role {model.role!r}"
                )

    def _publish_locked(self, record: _JobRecord, snapshot: LiveSceneJob) -> None:
        record.snapshot = snapshot
        for subscription in tuple(record.subscribers):
            self._put_snapshot(subscription._queue, snapshot)
        session_id = snapshot.request.session_id
        if session_id is not None:
            pointer = self._session_jobs.get(session_id)
            if pointer is not None and pointer[1] == snapshot.job_id:
                self._publish_session_locked(session_id)

    def _session_event_locked(self, session_id: str) -> LiveSceneSessionEvent:
        pointer = self._session_jobs.get(session_id)
        if pointer is None:
            return LiveSceneSessionEvent(
                session_id=session_id,
                server_instance_id=self.server_instance_id,
                session_revision=0,
            )
        session_revision, job_id = pointer
        record = self._jobs.get(job_id)
        if record is None:
            return LiveSceneSessionEvent(
                session_id=session_id,
                server_instance_id=self.server_instance_id,
                session_revision=0,
            )
        return LiveSceneSessionEvent(
            session_id=session_id,
            server_instance_id=self.server_instance_id,
            session_revision=session_revision,
            job=record.snapshot,
        )

    def _publish_session_locked(self, session_id: str) -> None:
        event = self._session_event_locked(session_id)
        for subscription in tuple(self._session_subscribers.get(session_id, ())):
            self._put_session_event(subscription._queue, event)

    def _evict_completed_locked(self) -> None:
        while len(self._jobs) >= self.max_retained_jobs:
            completed_id = next(
                (job_id for job_id, record in self._jobs.items() if record.snapshot.terminal),
                None,
            )
            if completed_id is None:
                return
            completed = self._jobs.pop(completed_id)
            for subscription in completed.subscribers:
                subscription._closed = True
                if subscription._queue.empty():
                    subscription._queue.put_nowait(_CLOSED)
            completed.subscribers.clear()
            session_id = completed.snapshot.request.session_id
            if session_id is not None:
                pointer = self._session_jobs.get(session_id)
                if pointer is not None and pointer[1] == completed_id:
                    self._session_jobs.pop(session_id, None)
                    self._publish_session_locked(session_id)

    @staticmethod
    def _put_snapshot(
        queue: asyncio.Queue[LiveSceneJob | object],
        snapshot: LiveSceneJob,
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(snapshot)

    @staticmethod
    def _put_closed(queue: asyncio.Queue[LiveSceneJob | object]) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(_CLOSED)

    @staticmethod
    def _put_session_event(
        queue: asyncio.Queue[LiveSceneSessionEvent | object],
        event: LiveSceneSessionEvent,
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(event)


def live_scene_request_seed(request: LiveSceneCreateRequest) -> int:
    """Resolve the stable renderer seed shared by generation and exact reuse."""

    if request.seed is not None:
        return request.seed
    payload = f"{request.text}\0{request.visual_style}\0{request.session_id or ''}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _cached_live_scene_artifacts(
    pack: StoryPack,
    *,
    provider: str,
) -> list[LiveSceneArtifact]:
    role_settings = {
        AssetRole.MASTER: (LiveSceneArtifactKind.MASTER, AssetKind.IMAGE),
        AssetRole.DEPTH: (LiveSceneArtifactKind.DEPTH, AssetKind.DEPTH_MAP),
        AssetRole.MOTION: (LiveSceneArtifactKind.MOTION, AssetKind.VIDEO_LOOP),
    }
    artifacts: list[LiveSceneArtifact] = []
    observed_roles: set[AssetRole] = set()
    for asset in pack.assets:
        settings = role_settings.get(asset.role)
        if settings is None or asset.kind is not settings[1]:
            raise ValueError("stored live scene contains unsupported asset roles")
        if asset.role in observed_roles:
            raise ValueError("stored live scene contains duplicate asset roles")
        observed_roles.add(asset.role)
        if asset.provider != provider and not asset.provider.startswith(f"{provider}:"):
            raise ValueError("stored live scene came from a different provider")
        model = (
            asset.provider.removeprefix(f"{provider}:")
            if asset.provider.startswith(f"{provider}:")
            else "stored-artifact"
        )
        suffix = Path(asset.local_uri).suffix.casefold()
        media_type = {
            ".avif": "image/avif",
            ".jpeg": "image/jpeg",
            ".jpg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
        }.get(suffix)
        if media_type is None:
            raise ValueError("stored live scene asset has an unsupported media type")
        artifacts.append(
            LiveSceneArtifact(
                artifact_id=asset.asset_id,
                kind=settings[0],
                uri=asset.local_uri,
                checksum_sha256=asset.checksum_sha256,
                media_type=media_type,
                provider=provider,
                model=model,
                seed=asset.seed,
                width=asset.width,
                height=asset.height,
                duration_ms=asset.duration_ms,
            )
        )
    if not {AssetRole.MASTER, AssetRole.DEPTH} <= observed_roles:
        raise ValueError("stored live scene is missing master or depth output")
    return artifacts


def _cached_live_scene_metrics(
    pack: StoryPack,
    *,
    artifacts: list[LiveSceneArtifact],
    cache_ms: float,
    include_heavy_models: bool,
) -> LiveSceneMetrics:
    compiler = pack.compiler_model
    if "fallback" in compiler.casefold():
        planning_status = LiveScenePlanningStatus.FALLBACK
    elif "deterministic" in compiler.casefold() or "fixture" in compiler.casefold():
        planning_status = LiveScenePlanningStatus.DETERMINISTIC
    else:
        planning_status = LiveScenePlanningStatus.MODEL
    models = [
        LiveSceneModelProvenance(
            role="scene_plan",
            model=compiler,
            revision="stored-story-pack-v1",
        )
    ]
    if include_heavy_models:
        for artifact in artifacts:
            model, separator, revision = artifact.model.rpartition("@")
            models.append(
                LiveSceneModelProvenance(
                    role=artifact.kind.value,
                    model=model if separator else artifact.model,
                    revision=revision if separator else artifact.checksum_sha256[:16],
                )
            )
    return LiveSceneMetrics(
        elapsed_ms=cache_ms,
        cache_ms=cache_ms,
        planning_status=planning_status,
        planning_cache_hit=False,
        scene_cache_hit=True,
        models=models,
    )


class DisabledLiveSceneProvider:
    def __init__(self, reason: str = "Live-scene generation provider is disabled") -> None:
        self.reason = reason

    @property
    def name(self) -> str:
        return "disabled"

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        del request, job_id
        raise LiveSceneProviderUnavailableError(self.reason)
        yield  # pragma: no cover - keeps this method an async iterator


class DeterministicFakeLiveSceneProvider:
    """Fast deterministic fixture provider for API/UI development and tests."""

    def __init__(
        self,
        *,
        cache: AssetCache | None = None,
        stage_delay_seconds: float = 0,
        enable_motion: bool = True,
    ) -> None:
        if stage_delay_seconds < 0:
            raise ValueError("stage_delay_seconds cannot be negative")
        self.cache = cache
        self.stage_delay_seconds = stage_delay_seconds
        self.enable_motion = enable_motion

    @property
    def name(self) -> str:
        return "fake"

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        seed = request.seed if request.seed is not None else _stable_seed(request)
        draft = build_live_scene_story_pack(
            request,
            job_id=job_id,
            seed=seed,
            assets=[],
            compiler_model="deterministic-live-scene-fixture-v1",
        )
        await asyncio.sleep(self.stage_delay_seconds)
        yield LiveSceneUpdate(
            stage=LiveSceneStage.DRAFT_READY,
            progress=0.35,
            story_pack=draft,
            metrics=_fake_live_scene_metrics(LiveSceneStage.DRAFT_READY),
        )

        master = await _fake_asset(
            job_id,
            request,
            seed,
            LiveSceneArtifactKind.MASTER,
            cache=self.cache,
        )
        depth = await _fake_asset(
            job_id,
            request,
            seed,
            LiveSceneArtifactKind.DEPTH,
            cache=self.cache,
        )
        master_artifacts = [master[1], depth[1]]
        master_pack = build_live_scene_story_pack(
            request,
            job_id=job_id,
            seed=seed,
            assets=[master[0], depth[0]],
            compiler_model="deterministic-live-scene-fixture-v1",
        )
        await asyncio.sleep(self.stage_delay_seconds)
        yield LiveSceneUpdate(
            stage=LiveSceneStage.MASTER_READY,
            progress=0.7,
            story_pack=master_pack,
            artifacts=master_artifacts,
            metrics=_fake_live_scene_metrics(LiveSceneStage.MASTER_READY),
        )

        if not self.enable_motion:
            return

        motion = await _fake_asset(
            job_id,
            request,
            seed,
            LiveSceneArtifactKind.MOTION,
            cache=self.cache,
        )
        motion_pack = build_live_scene_story_pack(
            request,
            job_id=job_id,
            seed=seed,
            assets=[master[0], depth[0], motion[0]],
            compiler_model="deterministic-live-scene-fixture-v1",
        )
        await asyncio.sleep(self.stage_delay_seconds)
        yield LiveSceneUpdate(
            stage=LiveSceneStage.MOTION_READY,
            progress=1,
            complete=True,
            story_pack=motion_pack,
            artifacts=[*master_artifacts, motion[1]],
            metrics=_fake_live_scene_metrics(LiveSceneStage.MOTION_READY),
        )


def build_live_scene_provider(
    live_scene_backend: str,
    *,
    asset_backend: str,
    cache: AssetCache | None = None,
    output_root: Path = Path("artifacts/live-scenes/generated"),
    enable_motion: bool = False,
    modal_session_gpu_cap_usd: float = 1.0,
    planner_mode: str = "deterministic",
    model_client: StructuredModelClient | None = None,
    planner_timeout_seconds: float = 12.0,
    planner_model_revision: str = "configured-local-model",
    planner_compact_wire: bool = False,
    planner_cache_entries: int = 32,
    master_width: int = 896,
    master_height: int = 512,
    master_steps: int = 2,
    master_guidance_scale: float = 4.5,
    auto_prewarm_on_submit: bool = False,
) -> LiveSceneProvider:
    planner = None
    if planner_mode == "model":
        if model_client is None:
            raise ValueError("model live-scene planning requires a structured model client")
        from bookforge.live_scene_planner import StructuredLiveScenePlanner

        planner = StructuredLiveScenePlanner(
            model_client,
            timeout_seconds=planner_timeout_seconds,
            model_revision=planner_model_revision,
            compact_wire=planner_compact_wire,
            cache_entries=planner_cache_entries,
        )
    elif planner_mode != "deterministic":
        raise ValueError(f"unknown live-scene planner mode {planner_mode!r}")

    selected = asset_backend if live_scene_backend == "auto" else live_scene_backend
    if selected == "fake":
        # A small delay makes each progressive stage observable in local UI smoke tests.
        return DeterministicFakeLiveSceneProvider(
            cache=cache,
            stage_delay_seconds=0.05,
            enable_motion=enable_motion,
        )
    if selected == "modal":
        if cache is None:
            raise ValueError("The finite Modal live-scene provider requires an AssetCache")
        # Lazy import avoids a cycle: the finite adapter implements this module's protocol.
        from bookforge.finite_modal_provider import (
            FiniteModalLiveSceneProvider,
            FiniteModalSceneProvider,
        )

        return FiniteModalLiveSceneProvider(
            FiniteModalSceneProvider(session_gpu_cap_usd=modal_session_gpu_cap_usd),
            cache=cache,
            output_root=output_root,
            enable_motion=enable_motion,
            planner=planner,
            master_width=master_width,
            master_height=master_height,
            master_steps=master_steps,
            master_guidance_scale=master_guidance_scale,
            auto_prewarm_on_submit=auto_prewarm_on_submit,
        )
    if selected == "modal_warm":
        if cache is None:
            raise ValueError("The warm Modal live-scene provider requires an AssetCache")
        from bookforge.finite_modal_provider import (
            FiniteModalLiveSceneProvider,
            WarmModalSceneProvider,
        )

        return FiniteModalLiveSceneProvider(
            WarmModalSceneProvider(session_gpu_cap_usd=modal_session_gpu_cap_usd),
            cache=cache,
            output_root=output_root,
            enable_motion=enable_motion,
            planner=planner,
            master_width=master_width,
            master_height=master_height,
            master_steps=master_steps,
            master_guidance_scale=master_guidance_scale,
            auto_prewarm_on_submit=auto_prewarm_on_submit,
        )
    return DisabledLiveSceneProvider(
        f"No live-scene provider is configured for backend {selected!r}"
    )


def _stable_seed(request: LiveSceneCreateRequest) -> int:
    digest = hashlib.sha256(
        f"{request.text}\0{request.visual_style}\0{request.session_id or ''}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _fake_live_scene_metrics(stage: LiveSceneStage) -> LiveSceneMetrics:
    evidence = {
        LiveSceneStage.DRAFT_READY: (3.0, 2.0, 0.0, 1.0, ("scene_plan",)),
        LiveSceneStage.MASTER_READY: (
            15.0,
            10.0,
            2.0,
            3.0,
            ("scene_plan", "master", "depth"),
        ),
        LiveSceneStage.MOTION_READY: (
            30.0,
            22.0,
            3.0,
            5.0,
            ("scene_plan", "master", "depth", "motion"),
        ),
    }
    provider_ms, inference_ms, cache_ms, overhead_ms, roles = evidence[stage]
    return LiveSceneMetrics(
        provider_ms=provider_ms,
        inference_ms=inference_ms,
        cache_ms=cache_ms,
        overhead_ms=overhead_ms,
        planning_ms=2,
        planning_status=LiveScenePlanningStatus.DETERMINISTIC,
        warm_state=LiveSceneWarmState.UNKNOWN,
        estimated_gpu_usd=0,
        cost_source=LiveSceneCostSource.FIXTURE,
        models=[
            LiveSceneModelProvenance(
                role=role,
                model="deterministic-live-scene-fixture-v1",
                revision="v1",
            )
            for role in roles
        ],
    )


def _error_message(error: Exception) -> str:
    return (str(error) or error.__class__.__name__)[:500]


def build_live_scene_story_pack(
    request: LiveSceneCreateRequest,
    *,
    job_id: str,
    seed: int,
    assets: list[AssetRecord],
    compiler_model: str,
    cloud_safe_prompts: bool = False,
) -> StoryPack:
    page_id = "page-01"
    background_id = "scene-background"
    focus_id = "scene-focus"
    accent_id = "scene-accent"
    theme = _passage_draft_theme(request.text, seed)
    focus_x = 0.36 + ((seed >> 4) & 0xFF) / 255 * 0.28
    focus_y = 0.52 + ((seed >> 12) & 0xFF) / 255 * 0.12
    accent_x = 0.76 if focus_x < 0.5 else 0.24
    accent_y = 0.32 + ((seed >> 20) & 0xFF) / 255 * 0.28
    if cloud_safe_prompts:
        prompt = (
            f"{request.visual_style}. {theme['prompt']}. "
            "One cohesive cinematic storybook moment with a clear central subject, "
            "projection-ready silhouettes, layered depth, and no readable text."
        )
        background_prompt = str(theme["prompt"])
        focus_prompt = "Clear central storytelling subject and action"
        accent_prompt = str(theme["accent"])
    else:
        prompt = (
            f"{request.visual_style}. {theme['prompt']}. "
            "A cinematic, projection-ready illustration of: "
            f"{request.text}"
        )
        background_prompt = f"Environment and atmosphere for {request.text}"
        focus_prompt = f"Clear central storytelling subject from {request.text}"
        accent_prompt = f"{theme['accent']} inspired by {request.text}"
    layers = [
        VisualLayer(
            layer_id=background_id,
            kind="background",
            prompt=background_prompt,
            z_index=0,
            motion="subtle parallax drift",
        ),
        VisualLayer(
            layer_id=focus_id,
            kind="character",
            prompt=focus_prompt,
            z_index=5,
            motion="gentle breathing and cloth movement",
        ),
        VisualLayer(
            layer_id=accent_id,
            kind=theme["layer_kind"],
            prompt=accent_prompt,
            z_index=8,
            motion=theme["motion"],
        ),
    ]
    page = GeneratedPagePlan(
        page_id=page_id,
        source_text=request.text,
        scene_summary=request.text[:500],
        scene_spec=SceneSpecV2(
            canvas=SceneCanvas(),
            master_prompt=prompt,
            negative_prompt="text, watermark, logo, duplicate subjects, distorted anatomy",
            camera=CameraMotion(
                kind=theme["camera"],
                travel_x=theme["travel_x"],
                travel_y=theme["travel_y"],
                duration_ms=8_000 + seed % 4_001,
            ),
            composition=[
                LayerComposition(
                    layer_id=background_id,
                    center_x=0.5,
                    center_y=0.5,
                    width=1,
                    height=1,
                    depth=12,
                    ambient_motion=AmbientMotion(kind="parallax", period_ms=8_000),
                ),
                LayerComposition(
                    layer_id=focus_id,
                    center_x=focus_x,
                    center_y=focus_y,
                    width=0.38 + ((seed >> 8) & 0x3F) / 1_000,
                    height=0.58 + ((seed >> 16) & 0x3F) / 1_000,
                    depth=4,
                    ambient_motion=AmbientMotion(
                        kind="breathe",
                        scale_delta=0.012,
                        period_ms=4_000,
                    ),
                ),
                LayerComposition(
                    layer_id=accent_id,
                    center_x=accent_x,
                    center_y=accent_y,
                    width=0.18 + ((seed >> 2) & 0x3F) / 1_000,
                    height=0.2 + ((seed >> 10) & 0x3F) / 1_000,
                    depth=2,
                    ambient_motion=AmbientMotion(
                        kind=theme["ambient_motion"],
                        amplitude_x=theme["amplitude_x"],
                        amplitude_y=theme["amplitude_y"],
                        scale_delta=theme["scale_delta"],
                        period_ms=3_200 + seed % 2_401,
                    ),
                ),
            ],
            ambience=theme["ambience"],
        ),
        layers=layers,
        triggers=[],
        literacy_support=[],
        comprehension=[],
    )
    title_words = request.text.rstrip(".!?").split()[:8]
    title = " ".join(title_words) or "Live Scene"
    story_prefix = request.session_id or "live-scene"
    return StoryPack(
        schema_version="2.0",
        story_id=f"{story_prefix}-{job_id[-12:]}",
        title=title,
        reading_level=2,
        visual_style=request.visual_style,
        compiler_model=compiler_model,
        pages=[page],
        assets=assets,
    )


def build_planned_live_scene_story_pack(
    request: LiveSceneCreateRequest,
    *,
    job_id: str,
    page: GeneratedPagePlan,
    assets: list[AssetRecord],
    compiler_model: str,
) -> StoryPack:
    """Wrap one validated model-authored page without recomputing its SceneSpec."""

    if page.page_id != "page-01":
        raise ValueError("live-scene model plans must use page-01")
    page = page.model_copy(update={"source_text": request.text})
    title_words = request.text.rstrip(".!?").split()[:8]
    title = " ".join(title_words) or "Live Scene"
    story_prefix = request.session_id or "live-scene"
    return StoryPack(
        schema_version="2.0",
        story_id=f"{story_prefix}-{job_id[-12:]}",
        title=title,
        reading_level=2,
        visual_style=request.visual_style,
        compiler_model=compiler_model,
        pages=[page],
        assets=assets,
    )


def _passage_draft_theme(text: str, seed: int) -> dict[str, object]:
    """Fast, deterministic passage cues for the pre-inference animated draft."""

    lowered = text.casefold()
    rules = (
        (
            "space",
            ("star", "moon", "planet", "space", "sky", "galaxy", "rocket"),
            {
                "prompt": "Deep indigo celestial world with luminous points of light",
                "accent": "a small orbiting constellation",
                "layer_kind": "effect",
                "motion": "slow orbital drift",
                "camera": "float",
                "travel_x": 0.012,
                "travel_y": -0.008,
                "ambient_motion": "float",
                "amplitude_x": 0.025,
                "amplitude_y": 0.018,
                "scale_delta": 0.008,
                "ambience": [
                    AmbientEffect(kind="stars", density=0.42, speed=0.18, color="#d8e7ff"),
                ],
            },
        ),
        (
            "ocean",
            ("ocean", "sea", "wave", "whale", "fish", "river", "boat", "water"),
            {
                "prompt": "Layered teal water and caustic aqua light",
                "accent": "a trail of luminous bubbles",
                "layer_kind": "effect",
                "motion": "bubbles rise and sway",
                "camera": "float",
                "travel_x": 0.008,
                "travel_y": 0.012,
                "ambient_motion": "drift",
                "amplitude_x": 0.018,
                "amplitude_y": -0.035,
                "scale_delta": 0.006,
                "ambience": [
                    AmbientEffect(kind="dust", density=0.34, speed=0.38, color="#83f3e4"),
                    AmbientEffect(kind="light_rays", density=0.1, speed=0.12, color="#b9fff4"),
                ],
            },
        ),
        (
            "forest",
            ("forest", "tree", "fox", "deer", "mushroom", "garden", "leaf", "wood"),
            {
                "prompt": "Moss green woodland with pools of warm gold",
                "accent": "a cluster of fireflies among leaves",
                "layer_kind": "prop",
                "motion": "fireflies wander around the prop",
                "camera": "slow_push",
                "travel_x": 0.006,
                "travel_y": -0.004,
                "ambient_motion": "float",
                "amplitude_x": 0.022,
                "amplitude_y": -0.018,
                "scale_delta": 0.01,
                "ambience": [
                    AmbientEffect(kind="fireflies", density=0.3, speed=0.32, color="#ffe58a"),
                    AmbientEffect(kind="fog", density=0.08, speed=0.1, color="#a8d6bd"),
                ],
            },
        ),
        (
            "storm",
            ("storm", "rain", "thunder", "lightning", "cloud", "wind", "snow"),
            {
                "prompt": "Slate clouds, silver mist, and a sharp seam of light",
                "accent": "a distant branching flash",
                "layer_kind": "effect",
                "motion": "light pulses through drifting mist",
                "camera": "pan_right",
                "travel_x": 0.025,
                "travel_y": 0.004,
                "ambient_motion": "pulse",
                "amplitude_x": 0.008,
                "amplitude_y": 0.006,
                "scale_delta": 0.025,
                "ambience": [
                    AmbientEffect(kind="fog", density=0.32, speed=0.55, color="#b9c8dc"),
                    AmbientEffect(kind="light_rays", density=0.12, speed=0.7, color="#e7f2ff"),
                ],
            },
        ),
        (
            "literacy",
            ("book", "letter", "word", "read", "library", "story", "page", "school"),
            {
                "prompt": "Warm paper-gold library light with ink-blue shadows",
                "accent": "a ribbon of glowing letter-like marks without readable text",
                "layer_kind": "effect",
                "motion": "marks gather into a gentle arc",
                "camera": "slow_push",
                "travel_x": 0.004,
                "travel_y": -0.006,
                "ambient_motion": "drift",
                "amplitude_x": 0.028,
                "amplitude_y": -0.012,
                "scale_delta": 0.008,
                "ambience": [
                    AmbientEffect(kind="dust", density=0.26, speed=0.22, color="#ffe2a1"),
                    AmbientEffect(kind="light_rays", density=0.08, speed=0.12, color="#fff0c7"),
                ],
            },
        ),
    )
    for name, keywords, theme in rules:
        if any(keyword in lowered for keyword in keywords):
            return {"name": name, **theme}

    fallback_colors = ("#82e6bd", "#f4d27d", "#8fb8ff", "#e79ad8")
    accent = fallback_colors[seed % len(fallback_colors)]
    return {
        "name": "imaginative",
        "prompt": "Layered twilight paper shapes with a passage-derived color rhythm",
        "accent": "a symbolic glowing keepsake from the scene",
        "layer_kind": "prop",
        "motion": "the keepsake floats in a small arc",
        "camera": ("slow_push", "pan_left", "pan_right", "float")[seed % 4],
        "travel_x": (-0.012, 0.008, 0.014)[seed % 3],
        "travel_y": (-0.008, 0.006)[seed % 2],
        "ambient_motion": "float",
        "amplitude_x": 0.02,
        "amplitude_y": -0.016,
        "scale_delta": 0.012,
        "ambience": [AmbientEffect(kind="dust", density=0.2, speed=0.26, color=accent)],
    }


async def _fake_asset(
    job_id: str,
    request: LiveSceneCreateRequest,
    base_seed: int,
    kind: LiveSceneArtifactKind,
    *,
    cache: AssetCache | None,
) -> tuple[AssetRecord, LiveSceneArtifact]:
    settings = {
        LiveSceneArtifactKind.MASTER: (
            AssetKind.IMAGE,
            AssetRole.MASTER,
            "image/png",
            ".png",
            0,
            0,
            "moon-gate-hero-v1.png",
            1_672,
            941,
        ),
        LiveSceneArtifactKind.DEPTH: (
            AssetKind.DEPTH_MAP,
            AssetRole.DEPTH,
            "image/png",
            ".png",
            0,
            0,
            "moon-gate-hero-v1.png",
            1_672,
            941,
        ),
        LiveSceneArtifactKind.MOTION: (
            AssetKind.VIDEO_LOOP,
            AssetRole.MOTION,
            "video/mp4",
            ".mp4",
            4_000,
            0,
            "silver-fox-loop-v1.mp4",
            768,
            512,
        ),
    }
    (
        asset_kind,
        role,
        media_type,
        suffix,
        duration_ms,
        offset,
        fixture_name,
        width,
        height,
    ) = settings[kind]
    seed = (base_seed + offset) % (2**32)
    artifact_id = f"{job_id}-{kind.value}"
    fixture_path = Path(__file__).parent / "static" / "assets" / fixture_name
    content = await asyncio.to_thread(fixture_path.read_bytes)
    if cache is None:
        digest = hashlib.sha256(content).hexdigest()
        uri = f"/workbench-assets/assets/{fixture_name}"
    else:
        digest, uri = await cache.store_generated(
            asset_id=artifact_id,
            kind=asset_kind,
            content=content,
            suffix=suffix,
        )
    prompt = f"{request.visual_style}: {request.text}"
    asset = AssetRecord(
        asset_id=artifact_id,
        page_id="page-01",
        layer_id="scene-background",
        kind=asset_kind,
        role=role,
        provider="fake",
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        duration_ms=duration_ms,
        checksum_sha256=digest,
        local_uri=uri,
        state=AssetState.READY,
        generation_ms=1,
    )
    artifact = LiveSceneArtifact(
        artifact_id=artifact_id,
        kind=kind,
        uri=uri,
        checksum_sha256=digest,
        media_type=media_type,
        provider="fake",
        model="deterministic-live-scene-fixture-v1",
        seed=seed,
        width=width,
        height=height,
        duration_ms=duration_ms,
    )
    return asset, artifact
