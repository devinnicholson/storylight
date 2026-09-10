"""Bounded, local-only next-page preparation and offline projector activation."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from PIL import Image
from pydantic import Field, StringConstraints

from storylight.anticipatory import (
    AnticipationStatus,
    CandidateRecord,
    CandidateState,
    CommitRequest,
)
from storylight.anticipatory_edge import (
    AnticipatoryEdgeCoordinator,
    AnticipatoryEdgeError,
    LocalAnticipationCandidate,
    LocalAnticipationPrepareRequest,
    LocalAnticipationPrepareResponse,
)
from storylight.asset_cache import AssetCache
from storylight.domain import (
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetState,
    FrozenStrictModel,
    GeneratedPagePlan,
    StoryPack,
)
from storylight.event_hub import SessionId
from storylight.live_scene import (
    LiveSceneCreateRequest,
    LiveSceneJobRegistry,
    LiveSceneServerInstanceId,
    LiveSceneSessionStatus,
    live_scene_request_seed,
)

PreparedProjectionId = Annotated[str, StringConstraints(pattern=r"^prepared_[a-f0-9]{24}$")]


class PrepareProjectionRequest(LiveSceneCreateRequest):
    session_id: SessionId


class ActivateProjectionRequest(FrozenStrictModel):
    prepared_id: PreparedProjectionId
    expected_server_instance_id: LiveSceneServerInstanceId
    expected_session_revision: Annotated[int, Field(ge=0)]


class PrewarmProjectionRequest(FrozenStrictModel):
    authorization: Literal["I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU"]


class PreparedProjectionStatus(FrozenStrictModel):
    prepared_id: PreparedProjectionId
    state: Literal["preparing", "approved", "staged", "failed", "cancelled", "expired"]
    detail: str
    expires_at: datetime
    planning_ms: float = 0
    render_ms: float = 0
    critic_ms: float = 0
    cache_hit: bool = False
    assets_verified: bool = False


@dataclass
class _PreparedPage:
    request: PrepareProjectionRequest
    response: LocalAnticipationPrepareResponse
    page: GeneratedPagePlan
    candidate: CandidateRecord
    expires_at: datetime
    pack: StoryPack | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class AnticipatoryPlayback:
    """Keep at most eight exact next pages; never publish during preparation."""

    def __init__(
        self,
        *,
        edge: AnticipatoryEdgeCoordinator,
        cache: AssetCache,
        registry: LiveSceneJobRegistry,
    ) -> None:
        self.edge = edge
        self.cache = cache
        self.registry = registry
        self._pages: dict[str, _PreparedPage] = {}
        self._lock = asyncio.Lock()
        self._preparing = 0

    async def prepare(self, request: PrepareProjectionRequest) -> PreparedProjectionStatus:
        async with self._lock:
            now = datetime.now(UTC)
            self._pages = {key: page for key, page in self._pages.items() if page.expires_at > now}
            if len(self._pages) + self._preparing >= 8:
                raise AnticipatoryEdgeError("Prepared-page capacity is full; discard a page first")
            self._preparing += 1
        try:
            await self.edge.client.ensure_critic_ready()
            response, pages = await self.edge.prepare_with_pages(
                LocalAnticipationPrepareRequest(
                    sequence=1,
                    visual_style=request.visual_style,
                    expires_in_seconds=900,
                    max_render_cost_usd=0.012,
                    session_cost_ceiling_usd=0.024,
                    candidates=[
                        LocalAnticipationCandidate(
                            branch_id="known_next",
                            text=request.text,
                            source="exact_lookahead",
                            seed=live_scene_request_seed(request),
                        )
                    ],
                )
            )
            candidate = response.anticipation.candidates[0]
            prepared_id = f"prepared_{uuid4().hex[:24]}"
            page = _PreparedPage(
                request=request,
                response=response,
                page=pages[0],
                candidate=candidate,
                expires_at=candidate.spec.not_after,
            )
            self._pages[prepared_id] = page
            return self._snapshot(prepared_id, page)
        finally:
            self._preparing -= 1

    def _get(self, prepared_id: str) -> _PreparedPage:
        page = self._pages.get(prepared_id)
        if page is None:
            raise AnticipatoryEdgeError("Prepared page is unavailable; prepare it again")
        if page.expires_at <= datetime.now(UTC):
            raise AnticipatoryEdgeError("Prepared page expired; prepare it again")
        return page

    async def status(
        self, prepared_id: str, *, wait_seconds: float = 0
    ) -> PreparedProjectionStatus:
        page = self._get(prepared_id)
        async with page.lock:
            self._get(prepared_id)
            if page.pack is None:
                await self._refresh(page, wait_seconds=wait_seconds)
            return self._snapshot(prepared_id, page)

    async def _refresh(self, page: _PreparedPage, *, wait_seconds: float = 0) -> None:
        status = await self.edge.status(
            page.response.session_token,
            page.response.sequence,
            wait_seconds=wait_seconds,
            branch_id=page.candidate.spec.branch_id,
        )
        self._accept_status(page, status)

    @staticmethod
    def _accept_status(page: _PreparedPage, status: AnticipationStatus) -> None:
        if (
            status.session_token != page.response.session_token
            or status.sequence != page.response.sequence
            or len(status.candidates) != 1
            or status.candidates[0].spec != page.response.anticipation.candidates[0].spec
        ):
            raise AnticipatoryEdgeError("Remote scene identity changed; refusing playback")
        page.candidate = status.candidates[0]

    async def stage(self, prepared_id: str) -> PreparedProjectionStatus:
        page = self._get(prepared_id)
        async with page.lock:
            self._get(prepared_id)
            if page.pack is not None:
                return self._snapshot(prepared_id, page)
            await self._refresh(page)
            candidate = page.candidate
            if candidate.state not in {CandidateState.READY, CandidateState.COMMITTED}:
                raise AnticipatoryEdgeError("Nemotron has not approved this scene")
            scene = candidate.rendered
            assert scene is not None  # CandidateRecord validates promotable states.
            master, depth = await self.edge.client.fetch_scene(scene)
            shapes = await asyncio.to_thread(_image_shapes, master, depth)
            if any(shape[:2] != (candidate.spec.width, candidate.spec.height) for shape in shapes):
                raise AnticipatoryEdgeError("Prepared images do not match the scene dimensions")
            records = []
            for role, kind, content, shape, digest in (
                (AssetRole.MASTER, AssetKind.IMAGE, master, shapes[0], scene.master_sha256),
                (AssetRole.DEPTH, AssetKind.DEPTH_MAP, depth, shapes[1], scene.depth_sha256),
            ):
                checksum, uri = await self.cache.store_generated(
                    asset_id=f"{prepared_id}-{role.value}",
                    kind=kind,
                    content=content,
                    suffix=shape[2],
                )
                if checksum != digest:
                    raise AnticipatoryEdgeError("Prepared cache checksum differs from the scene")
                records.append(
                    AssetRecord(
                        asset_id=f"{prepared_id}-{role.value}",
                        page_id=page.page.page_id,
                        layer_id=page.page.layers[0].layer_id,
                        kind=kind,
                        role=role,
                        provider=f"{scene.provider}:{scene.model}@{scene.model_revision}",
                        prompt=candidate.spec.visual_brief,
                        seed=candidate.spec.seed,
                        width=shape[0],
                        height=shape[1],
                        checksum_sha256=checksum,
                        local_uri=uri,
                        state=AssetState.READY,
                    )
                )
            pack = StoryPack(
                schema_version="2.0",
                story_id=prepared_id,
                title="Prepared next page",
                reading_level=2,
                visual_style=page.request.visual_style,
                compiler_model=page.response.planning[0].model,
                pages=[page.page],
                assets=records,
            )
            committed = await self.edge.commit(
                CommitRequest(
                    session_token=page.response.session_token,
                    sequence=page.response.sequence,
                    branch_id=candidate.spec.branch_id,
                )
            )
            self._get(prepared_id)
            self._accept_status(page, committed)
            if (
                page.candidate.state is not CandidateState.COMMITTED
                or page.candidate.rendered != scene
            ):
                raise AnticipatoryEdgeError("Cloud did not commit the verified scene")
            # This selects the single known-next branch, not the physical projector.
            # Every later activation is local and works if the cloud disconnects.
            page.pack = pack
            return self._snapshot(prepared_id, page)

    async def activate(self, request: ActivateProjectionRequest) -> LiveSceneSessionStatus:
        page = self._get(request.prepared_id)
        async with page.lock:
            self._get(request.prepared_id)
            if page.pack is None:
                raise AnticipatoryEdgeError(
                    "Download and verify the approved scene before showing it"
                )
            assert page.candidate.rendered is not None
            return await self.registry.activate_prepared_pack(
                activation_id=request.prepared_id,
                request=page.request,
                pack=page.pack,
                provider=page.candidate.rendered.provider,
                expected_server_instance_id=request.expected_server_instance_id,
                expected_session_revision=request.expected_session_revision,
            )

    async def discard(self, prepared_id: str) -> None:
        page = self._pages.get(prepared_id)
        if page is None:
            return
        async with page.lock:
            if page.pack is None and page.expires_at > datetime.now(UTC):
                await self.edge.cancel(page.response.session_token, page.response.sequence)
            self._pages.pop(prepared_id, None)

    @staticmethod
    def _snapshot(prepared_id: str, page: _PreparedPage) -> PreparedProjectionStatus:
        candidate = page.candidate
        state = "preparing"
        detail = "Preparing in the background; the current projection is unchanged."
        if page.pack is not None:
            state, detail = "staged", "Verified on this device. Ready to show without a cloud call."
        elif candidate.state in {CandidateState.READY, CandidateState.COMMITTED}:
            state, detail = "approved", "Nemotron approved the scene; local download is next."
        elif candidate.state in {CandidateState.REJECTED, CandidateState.FAILED}:
            state, detail = "failed", candidate.error or "Scene preparation failed."
        elif candidate.state in {CandidateState.CANCELLED, CandidateState.EXPIRED}:
            state, detail = candidate.state.value, "Scene preparation is no longer available."
        return PreparedProjectionStatus(
            prepared_id=prepared_id,
            state=state,
            detail=detail,
            expires_at=page.expires_at,
            planning_ms=page.response.planning[0].planning_ms,
            render_ms=0
            if candidate.cache_hit or candidate.rendered is None
            else candidate.rendered.render_latency_ms,
            critic_ms=0
            if candidate.cache_hit
            else sum(item.latency_ms for item in candidate.critic_history),
            cache_hit=candidate.cache_hit,
            assets_verified=page.pack is not None,
        )


def _image_shapes(*contents: bytes) -> list[tuple[int, int, str]]:
    result = []
    for content in contents:
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG"} or image.width * image.height > 1536 * 1536:
                    raise ValueError("unsupported scene image")
                shape = (image.width, image.height, ".jpg" if image.format == "JPEG" else ".png")
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                result.append(shape)
        except (OSError, ValueError) as error:
            raise AnticipatoryEdgeError("Prepared scene image is invalid") from error
    return result
