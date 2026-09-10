"""Privacy-bounded speculative scene orchestration.

The edge creates these contracts after local story understanding.  The contract has
no field for a passage, transcript, audio, image of the reader, or camera frame.  A
remote orchestrator may render and visually critique the resulting synthetic scene,
but it cannot reconstruct the live reading session from this schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import Field, StringConstraints, model_validator

from storylight.domain import FrozenStrictModel
from storylight.nemotron_critic import NemotronCriticDecision, NemotronCriticEvidence

SessionToken = Annotated[
    str,
    StringConstraints(pattern=r"^anticipate_[a-f0-9]{24}$"),
]
BranchId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_-]{1,47}$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
ArtifactRef = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
BoundedPhrase = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]


class AnticipationSource(StrEnum):
    EXACT_LOOKAHEAD = "exact_lookahead"
    PREDICTED_BRANCH = "predicted_branch"


class CandidateState(StrEnum):
    QUEUED = "queued"
    RENDERING = "rendering"
    CRITIQUING = "critiquing"
    REPAIRING = "repairing"
    READY = "ready"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in {
            self.READY,
            self.COMMITTED,
            self.CANCELLED,
            self.REJECTED,
            self.FAILED,
            self.EXPIRED,
        }


class PrivacyAttestation(FrozenStrictModel):
    boundary: Literal["sanitized_scene_spec_v1"] = "sanitized_scene_spec_v1"
    raw_text_removed: Literal[True] = True
    audio_removed: Literal[True] = True
    camera_data_removed: Literal[True] = True
    reader_identity_removed: Literal[True] = True
    edge_gate_revision: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    ]


class AnticipatorySceneSpec(FrozenStrictModel):
    """A renderable story beat with no raw reader or source-media field."""

    branch_id: BranchId
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    source: AnticipationSource
    visual_brief: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=1_200),
    ]
    visual_style: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    ] = "luminous watercolor paper theater"
    expected_subjects: list[BoundedPhrase] = Field(default_factory=list, max_length=8)
    forbidden_content: list[BoundedPhrase] = Field(
        default_factory=lambda: [
            "readable text",
            "duplicate principal subject",
            "interface chrome",
        ],
        max_length=8,
    )
    negative_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_200),
    ] = "readable text, watermark, logo, duplicate principal subject, interface chrome"
    continuity_sha256: Sha256
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    width: Annotated[int, Field(ge=512, le=1_536, multiple_of=32)] = 1024
    height: Annotated[int, Field(ge=512, le=1_536, multiple_of=32)] = 576
    not_after: datetime
    max_render_cost_usd: Annotated[float, Field(gt=0, le=0.25)] = 0.02
    privacy: PrivacyAttestation

    @model_validator(mode="after")
    def require_safe_shape(self) -> AnticipatorySceneSpec:
        if self.not_after.tzinfo is None:
            raise ValueError("not_after must be timezone-aware")
        if not math.isfinite(self.max_render_cost_usd):
            raise ValueError("max_render_cost_usd must be finite")
        if len(set(self.expected_subjects)) != len(self.expected_subjects):
            raise ValueError("expected subjects must be unique")
        if len(set(self.forbidden_content)) != len(self.forbidden_content):
            raise ValueError("forbidden content entries must be unique")
        return self

    @property
    def cache_key(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"branch_id", "sequence", "source", "not_after", "privacy"},
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class AnticipatoryBatchRequest(FrozenStrictModel):
    session_token: SessionToken
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    candidates: list[AnticipatorySceneSpec] = Field(min_length=1, max_length=2)
    session_cost_ceiling_usd: Annotated[float, Field(gt=0, le=1)] = 0.10

    @model_validator(mode="after")
    def require_coherent_candidates(self) -> AnticipatoryBatchRequest:
        if not math.isfinite(self.session_cost_ceiling_usd):
            raise ValueError("session_cost_ceiling_usd must be finite")
        if any(candidate.sequence != self.sequence for candidate in self.candidates):
            raise ValueError("every candidate must match the batch sequence")
        branch_ids = [candidate.branch_id for candidate in self.candidates]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("candidate branch IDs must be unique")
        reserved = sum(candidate.max_render_cost_usd * 2 for candidate in self.candidates)
        if reserved > self.session_cost_ceiling_usd + 1e-9:
            raise ValueError(
                "candidate render and one-repair reservations exceed the session cost ceiling"
            )
        exact_count = sum(
            candidate.source is AnticipationSource.EXACT_LOOKAHEAD for candidate in self.candidates
        )
        if exact_count and len(self.candidates) != 1:
            raise ValueError(
                "exact lookahead accepts one known candidate, not speculative siblings"
            )
        return self


class RenderedScene(FrozenStrictModel):
    master_ref: ArtifactRef
    depth_ref: ArtifactRef
    master_sha256: Sha256
    depth_sha256: Sha256
    provider: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    ]
    model: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    model_revision: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    render_latency_ms: Annotated[float, Field(ge=0)]
    estimated_gpu_usd: Annotated[float, Field(ge=0, le=0.25)]


class CandidateRecord(FrozenStrictModel):
    spec: AnticipatorySceneSpec
    state: CandidateState
    selected: bool = False
    cache_hit: bool = False
    render_attempts: Annotated[int, Field(ge=0, le=2)] = 0
    repair_attempts: Annotated[int, Field(ge=0, le=1)] = 0
    rendered: RenderedScene | None = None
    critic_history: list[NemotronCriticEvidence] = Field(default_factory=list, max_length=2)
    render_cost_usd: Annotated[float, Field(ge=0, le=0.50)] = 0
    ready_latency_ms: Annotated[float, Field(ge=0)] | None = None
    error: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        ]
        | None
    ) = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def require_coherent_state(self) -> CandidateRecord:
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("candidate updated_at cannot precede created_at")
        if self.repair_attempts > self.render_attempts:
            raise ValueError("repair attempts cannot exceed render attempts")
        if self.cache_hit and (self.render_attempts or self.render_cost_usd):
            raise ValueError("cache hits cannot report new render work")
        if self.state in {CandidateState.READY, CandidateState.COMMITTED}:
            if self.rendered is None or not self.critic_history:
                raise ValueError("promotable candidates require a render and critic evidence")
            if self.critic_history[-1].verdict.decision is not NemotronCriticDecision.ACCEPT:
                raise ValueError("promotable candidates require an accepting critic verdict")
            if self.ready_latency_ms is None:
                raise ValueError("promotable candidates require ready latency")
        if self.state is CandidateState.COMMITTED and not self.selected:
            raise ValueError("committed candidates must be selected")
        if self.state in {CandidateState.FAILED, CandidateState.REJECTED} and not self.error:
            raise ValueError("failed and rejected candidates require an error")
        if self.error and self.state not in {
            CandidateState.FAILED,
            CandidateState.REJECTED,
            CandidateState.EXPIRED,
        }:
            raise ValueError("only failed, rejected, or expired candidates may include errors")
        return self


class AnticipationMetrics(FrozenStrictModel):
    candidates: Annotated[int, Field(ge=0)]
    ready: Annotated[int, Field(ge=0)]
    committed: Annotated[int, Field(ge=0)]
    cache_hits: Annotated[int, Field(ge=0)]
    cancelled: Annotated[int, Field(ge=0)]
    rejected: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    repair_attempts: Annotated[int, Field(ge=0)]
    render_cost_usd: Annotated[float, Field(ge=0)]
    wasted_render_cost_usd: Annotated[float, Field(ge=0)]
    mean_ready_ms: Annotated[float, Field(ge=0)] | None = None


class AnticipationStatus(FrozenStrictModel):
    session_token: SessionToken
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    candidates: list[CandidateRecord] = Field(max_length=2)
    metrics: AnticipationMetrics


class CommitRequest(FrozenStrictModel):
    session_token: SessionToken
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    branch_id: BranchId


class AnticipatoryRenderer(Protocol):
    async def render(
        self,
        spec: AnticipatorySceneSpec,
        *,
        attempt: Literal[1, 2],
        repair_guidance: str | None = None,
    ) -> RenderedScene: ...


class AnticipatoryCritic(Protocol):
    async def evaluate(
        self,
        spec: AnticipatorySceneSpec,
        scene: RenderedScene,
    ) -> NemotronCriticEvidence: ...


class AnticipationError(RuntimeError):
    pass


class AnticipationConflictError(AnticipationError):
    pass


class AnticipationNotFoundError(AnticipationError, LookupError):
    pass


class AnticipationCapacityError(AnticipationError):
    pass


@dataclass(frozen=True, slots=True)
class _CachedResult:
    rendered: RenderedScene
    critic_history: tuple[NemotronCriticEvidence, ...]
    expires_at: datetime


class AnticipatorySceneOrchestrator:
    """In-memory bounded coordinator; losing it only loses optional lookahead work."""

    def __init__(
        self,
        *,
        renderer: AnticipatoryRenderer,
        critic: AnticipatoryCritic,
        max_concurrent_renders: int = 2,
        max_candidates: int = 64,
        cache_entries: int = 32,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_concurrent_renders < 1:
            raise ValueError("max_concurrent_renders must be at least one")
        if max_candidates < 2:
            raise ValueError("max_candidates must be at least two")
        if cache_entries < 0:
            raise ValueError("cache_entries cannot be negative")
        self.renderer = renderer
        self.critic = critic
        self.max_candidates = max_candidates
        self.cache_entries = cache_entries
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic
        self._semaphore = asyncio.Semaphore(max_concurrent_renders)
        self._records: OrderedDict[tuple[str, int, str], CandidateRecord] = OrderedDict()
        self._started: dict[tuple[str, int, str], float] = {}
        self._tasks: dict[tuple[str, int, str], asyncio.Task[None]] = {}
        self._terminal_events: dict[tuple[str, int, str], asyncio.Event] = {}
        self._cache: OrderedDict[str, _CachedResult] = OrderedDict()
        self._lock = asyncio.Lock()
        self._closed = False

    async def submit(self, batch: AnticipatoryBatchRequest) -> AnticipationStatus:
        async with self._lock:
            if self._closed:
                raise AnticipationError("anticipatory orchestrator is closed")
            self._expire_locked()
            now = self._now()
            if any(candidate.not_after <= now for candidate in batch.candidates):
                raise AnticipationConflictError("cannot submit an already-expired candidate")
            keys = [
                (batch.session_token, batch.sequence, candidate.branch_id)
                for candidate in batch.candidates
            ]
            existing = [self._records.get(key) for key in keys]
            if any(record is not None for record in existing):
                if not all(record is not None for record in existing):
                    raise AnticipationConflictError(
                        "batch partially conflicts with existing branches"
                    )
                assert all(record is not None for record in existing)
                if any(
                    record.spec != candidate
                    for record, candidate in zip(existing, batch.candidates, strict=True)
                ):
                    raise AnticipationConflictError(
                        "branch ID already exists with different content"
                    )
                return self._status_locked(batch.session_token, batch.sequence)
            self._evict_locked(required=len(batch.candidates))
            if len(self._records) + len(batch.candidates) > self.max_candidates:
                raise AnticipationCapacityError("anticipatory candidate capacity is full")

            for key, spec in zip(keys, batch.candidates, strict=True):
                started = self._monotonic()
                self._started[key] = started
                event = asyncio.Event()
                self._terminal_events[key] = event
                cached = self._cache.get(spec.cache_key)
                if cached is not None and cached.expires_at < spec.not_after:
                    self._cache.pop(spec.cache_key)
                    cached = None
                cache_validator = getattr(self.renderer, "cached_result_available", None)
                if cached is not None and callable(cache_validator):
                    try:
                        cache_available = await cache_validator(cached.rendered)
                    except Exception:
                        cache_available = False
                    if not cache_available:
                        self._cache.pop(spec.cache_key)
                        cached = None
                if cached is not None:
                    self._cache.move_to_end(spec.cache_key)
                    record = CandidateRecord(
                        spec=spec,
                        state=CandidateState.READY,
                        cache_hit=True,
                        rendered=cached.rendered,
                        critic_history=list(cached.critic_history),
                        ready_latency_ms=0,
                        created_at=now,
                        updated_at=now,
                    )
                    event.set()
                else:
                    record = CandidateRecord(
                        spec=spec,
                        state=CandidateState.QUEUED,
                        created_at=now,
                        updated_at=now,
                    )
                self._records[key] = record
                if cached is None:
                    task = asyncio.create_task(
                        self._run_candidate(key),
                        name=f"storylight-anticipate:{batch.sequence}:{spec.branch_id}",
                    )
                    self._tasks[key] = task
                    task.add_done_callback(
                        lambda _task, task_key=key: self._tasks.pop(task_key, None)
                    )
            return self._status_locked(batch.session_token, batch.sequence)

    async def status(self, session_token: str, sequence: int) -> AnticipationStatus:
        async with self._lock:
            self._expire_locked()
            return self._status_locked(session_token, sequence)

    async def commit(self, request: CommitRequest) -> AnticipationStatus:
        async with self._lock:
            self._expire_locked()
            selected_key = (request.session_token, request.sequence, request.branch_id)
            selected = self._records.get(selected_key)
            if selected is None:
                raise AnticipationNotFoundError("selected anticipation branch was not found")
            if selected.state in {
                CandidateState.CANCELLED,
                CandidateState.REJECTED,
                CandidateState.FAILED,
                CandidateState.EXPIRED,
            }:
                raise AnticipationConflictError(
                    f"cannot commit a candidate in state {selected.state.value}"
                )
            for key, record in tuple(self._records.items()):
                if key[:2] != selected_key[:2] or key == selected_key:
                    continue
                if record.state not in {CandidateState.COMMITTED, CandidateState.CANCELLED}:
                    self._cancel_locked(key)
            new_state = (
                CandidateState.COMMITTED
                if selected.state is CandidateState.READY
                else selected.state
            )
            self._records[selected_key] = selected.model_copy(
                update={"selected": True, "state": new_state, "updated_at": self._now()}
            )
            return self._status_locked(request.session_token, request.sequence)

    async def cancel(self, session_token: str, sequence: int) -> AnticipationStatus:
        async with self._lock:
            matching = [key for key in self._records if key[:2] == (session_token, sequence)]
            if not matching:
                raise AnticipationNotFoundError("anticipation sequence was not found")
            for key in matching:
                record = self._records[key]
                if record.state is not CandidateState.COMMITTED:
                    self._cancel_locked(key)
            return self._status_locked(session_token, sequence)

    async def wait_terminal(
        self,
        session_token: str,
        sequence: int,
        branch_id: str,
        *,
        timeout_seconds: float = 30,
    ) -> CandidateRecord:
        key = (session_token, sequence, branch_id)
        async with self._lock:
            event = self._terminal_events.get(key)
            if event is None:
                raise AnticipationNotFoundError("anticipation branch was not found")
        await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
        async with self._lock:
            record = self._records.get(key)
            if record is None:
                raise AnticipationNotFoundError("anticipation branch was evicted while waiting")
            return record

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks.values())
            for key in tuple(self._tasks):
                self._cancel_locked(key)
        await asyncio.gather(*tasks, return_exceptions=True)
        close_renderer = getattr(self.renderer, "aclose", None)
        if callable(close_renderer):
            await close_renderer()
        close_critic = getattr(self.critic, "aclose", None)
        if callable(close_critic):
            await close_critic()

    async def _run_candidate(self, key: tuple[str, int, str]) -> None:
        try:
            async with self._semaphore:
                spec = await self._begin_render(key)
                first = await self.renderer.render(spec, attempt=1)
                self._validate_render_cost(spec, first)
                if not await self._record_render(key, first, repair=False):
                    return
                first_evidence = await self.critic.evaluate(spec, first)
                if first_evidence.verdict.decision is NemotronCriticDecision.ACCEPT:
                    await self._promote(key, first, [first_evidence], repair_attempts=0)
                    return

                correction = first_evidence.verdict.correction_visual_brief
                if correction is None:
                    await self._reject(key, "critic requested repair without a correction")
                    return
                if not await self._begin_repair(key, first_evidence):
                    return
                repaired = await self.renderer.render(spec, attempt=2, repair_guidance=correction)
                self._validate_render_cost(spec, repaired)
                if not await self._record_render(key, repaired, repair=True):
                    return
                # A correction guides rendering; it must never replace the acceptance contract.
                repaired_evidence = await self.critic.evaluate(spec, repaired)
                history = [first_evidence, repaired_evidence]
                if repaired_evidence.verdict.decision is NemotronCriticDecision.ACCEPT:
                    await self._promote(key, repaired, history, repair_attempts=1)
                    return
                await self._reject(
                    key,
                    "candidate failed the visual critic after its single allowed repair",
                    critic_history=history,
                )
        except Exception as error:
            await self._fail(key, _bounded_error(error))

    async def _begin_render(self, key: tuple[str, int, str]) -> AnticipatorySceneSpec:
        async with self._lock:
            record = self._records[key]
            if record.spec.not_after <= self._now():
                self._expire_record_locked(key)
                raise asyncio.CancelledError
            if record.state is CandidateState.CANCELLED:
                raise asyncio.CancelledError
            self._records[key] = record.model_copy(
                update={"state": CandidateState.RENDERING, "updated_at": self._now()}
            )
            return record.spec

    async def _record_render(
        self,
        key: tuple[str, int, str],
        scene: RenderedScene,
        *,
        repair: bool,
    ) -> bool:
        async with self._lock:
            record = self._records[key]
            if record.state in {CandidateState.CANCELLED, CandidateState.EXPIRED}:
                return False
            if record.spec.not_after <= self._now():
                self._expire_record_locked(key)
                return False
            attempts = record.render_attempts + 1
            self._records[key] = record.model_copy(
                update={
                    "state": CandidateState.CRITIQUING,
                    "rendered": scene,
                    "render_attempts": attempts,
                    "repair_attempts": 1 if repair else record.repair_attempts,
                    "render_cost_usd": record.render_cost_usd + scene.estimated_gpu_usd,
                    "updated_at": self._now(),
                }
            )
            return True

    async def _begin_repair(
        self,
        key: tuple[str, int, str],
        evidence: NemotronCriticEvidence,
    ) -> bool:
        async with self._lock:
            record = self._records[key]
            if record.state in {CandidateState.CANCELLED, CandidateState.EXPIRED}:
                return False
            if record.spec.not_after <= self._now():
                self._expire_record_locked(key)
                return False
            self._records[key] = record.model_copy(
                update={
                    "state": CandidateState.REPAIRING,
                    "critic_history": [evidence],
                    "updated_at": self._now(),
                }
            )
            return True

    async def _promote(
        self,
        key: tuple[str, int, str],
        scene: RenderedScene,
        critic_history: list[NemotronCriticEvidence],
        *,
        repair_attempts: int,
    ) -> None:
        async with self._lock:
            record = self._records[key]
            if record.state in {CandidateState.CANCELLED, CandidateState.EXPIRED}:
                return
            selected = record.selected
            state = CandidateState.COMMITTED if selected else CandidateState.READY
            ready_ms = max(0.0, (self._monotonic() - self._started[key]) * 1_000)
            promoted = record.model_copy(
                update={
                    "state": state,
                    "rendered": scene,
                    "critic_history": critic_history,
                    "repair_attempts": repair_attempts,
                    "ready_latency_ms": ready_ms,
                    "updated_at": self._now(),
                }
            )
            self._records[key] = CandidateRecord.model_validate(promoted.model_dump())
            if self.cache_entries:
                self._cache[record.spec.cache_key] = _CachedResult(
                    rendered=scene,
                    critic_history=tuple(critic_history),
                    expires_at=record.spec.not_after,
                )
                self._cache.move_to_end(record.spec.cache_key)
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)
            self._terminal_events[key].set()

    async def _reject(
        self,
        key: tuple[str, int, str],
        message: str,
        *,
        critic_history: list[NemotronCriticEvidence] | None = None,
    ) -> None:
        async with self._lock:
            record = self._records[key]
            if record.state in {CandidateState.CANCELLED, CandidateState.EXPIRED}:
                return
            history = critic_history if critic_history is not None else record.critic_history
            self._records[key] = record.model_copy(
                update={
                    "state": CandidateState.REJECTED,
                    "critic_history": history,
                    "error": message,
                    "updated_at": self._now(),
                }
            )
            self._terminal_events[key].set()

    async def _fail(self, key: tuple[str, int, str], message: str) -> None:
        async with self._lock:
            record = self._records.get(key)
            if record is None or record.state in {
                CandidateState.CANCELLED,
                CandidateState.EXPIRED,
                CandidateState.COMMITTED,
            }:
                return
            self._records[key] = record.model_copy(
                update={
                    "state": CandidateState.FAILED,
                    "error": message,
                    "updated_at": self._now(),
                }
            )
            self._terminal_events[key].set()

    def _cancel_locked(self, key: tuple[str, int, str]) -> None:
        record = self._records[key]
        if record.state in {
            CandidateState.COMMITTED,
            CandidateState.CANCELLED,
            CandidateState.REJECTED,
            CandidateState.FAILED,
            CandidateState.EXPIRED,
        }:
            return
        self._records[key] = record.model_copy(
            update={"state": CandidateState.CANCELLED, "updated_at": self._now()}
        )
        task = self._tasks.get(key)
        if task is not None and not task.done():
            task.cancel()
        self._terminal_events[key].set()

    def _expire_locked(self) -> None:
        now = self._now()
        for key, record in tuple(self._records.items()):
            if (
                record.state not in {CandidateState.COMMITTED, CandidateState.EXPIRED}
                and record.spec.not_after <= now
            ):
                self._expire_record_locked(key)

    def _expire_record_locked(self, key: tuple[str, int, str]) -> None:
        record = self._records[key]
        self._records[key] = record.model_copy(
            update={
                "state": CandidateState.EXPIRED,
                "error": "candidate expired before it could be committed",
                "updated_at": self._now(),
            }
        )
        task = self._tasks.get(key)
        if task is not None and not task.done():
            task.cancel()
        self._terminal_events[key].set()

    def _evict_locked(self, *, required: int) -> None:
        while len(self._records) + required > self.max_candidates:
            evictable = next(
                (
                    key
                    for key, record in self._records.items()
                    if record.state.terminal and record.state is not CandidateState.COMMITTED
                ),
                None,
            )
            if evictable is None:
                return
            self._records.pop(evictable)
            self._started.pop(evictable, None)
            self._terminal_events.pop(evictable, None)

    def _status_locked(self, session_token: str, sequence: int) -> AnticipationStatus:
        records = [
            record for key, record in self._records.items() if key[:2] == (session_token, sequence)
        ]
        if not records:
            raise AnticipationNotFoundError("anticipation sequence was not found")
        records.sort(key=lambda record: record.spec.branch_id)
        ready_values = [
            record.ready_latency_ms for record in records if record.ready_latency_ms is not None
        ]
        metrics = AnticipationMetrics(
            candidates=len(records),
            ready=sum(record.state is CandidateState.READY for record in records),
            committed=sum(record.state is CandidateState.COMMITTED for record in records),
            cache_hits=sum(record.cache_hit for record in records),
            cancelled=sum(record.state is CandidateState.CANCELLED for record in records),
            rejected=sum(record.state is CandidateState.REJECTED for record in records),
            failed=sum(record.state is CandidateState.FAILED for record in records),
            repair_attempts=sum(record.repair_attempts for record in records),
            render_cost_usd=sum(record.render_cost_usd for record in records),
            wasted_render_cost_usd=sum(
                record.render_cost_usd
                for record in records
                if record.state
                in {
                    CandidateState.CANCELLED,
                    CandidateState.REJECTED,
                    CandidateState.FAILED,
                    CandidateState.EXPIRED,
                }
            ),
            mean_ready_ms=(sum(ready_values) / len(ready_values) if ready_values else None),
        )
        return AnticipationStatus(
            session_token=session_token,
            sequence=sequence,
            candidates=records,
            metrics=metrics,
        )

    @staticmethod
    def _validate_render_cost(spec: AnticipatorySceneSpec, scene: RenderedScene) -> None:
        if scene.estimated_gpu_usd > spec.max_render_cost_usd + 1e-9:
            raise AnticipationError("renderer exceeded the candidate cost ceiling")


def _bounded_error(error: Exception) -> str:
    text = " ".join(str(error).split()) or error.__class__.__name__
    return text[:500]
