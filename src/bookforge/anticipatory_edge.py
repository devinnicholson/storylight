"""Jetson-side bridge from a privacy-gated local plan to GKE anticipation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, StringConstraints, model_validator

from bookforge.anticipatory import (
    AnticipationSource,
    AnticipationStatus,
    AnticipatoryBatchRequest,
    AnticipatorySceneSpec,
    BranchId,
    CandidateState,
    CommitRequest,
    PrivacyAttestation,
    RenderedScene,
)
from bookforge.domain import FrozenStrictModel
from bookforge.live_scene import SceneText, VisualStyle
from bookforge.live_scene_planner import LiveScenePlan, LiveScenePlanningResult

MAX_ASSET_BYTES = 8 * 1024 * 1024
RUNTIME_PREWARM_TIMEOUT_SECONDS = 240
EdgeTokenSource = Callable[[], Awaitable[str]]


class AnticipatoryEdgeError(RuntimeError):
    pass


class LocalScenePlanner(Protocol):
    async def plan(
        self,
        *,
        text: str,
        visual_style: str,
        seed: int,
    ) -> LiveScenePlanningResult: ...


class LocalAnticipationCandidate(FrozenStrictModel):
    branch_id: BranchId
    text: SceneText
    source: AnticipationSource
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]


class LocalAnticipationPrepareRequest(FrozenStrictModel):
    session_token: (
        Annotated[
            str,
            StringConstraints(pattern=r"^anticipate_[a-f0-9]{24}$"),
        ]
        | None
    ) = None
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    candidates: list[LocalAnticipationCandidate] = Field(min_length=1, max_length=2)
    visual_style: VisualStyle = "luminous watercolor paper theater"
    expires_in_seconds: Annotated[int, Field(ge=30, le=900)] = 300
    max_render_cost_usd: Annotated[float, Field(gt=0, le=0.25)] = 0.02
    session_cost_ceiling_usd: Annotated[float, Field(gt=0, le=1)] = 0.10

    @model_validator(mode="after")
    def require_bounded_branching(self) -> LocalAnticipationPrepareRequest:
        exact = sum(
            candidate.source is AnticipationSource.EXACT_LOOKAHEAD for candidate in self.candidates
        )
        if exact and len(self.candidates) != 1:
            raise ValueError("exact lookahead accepts one known candidate")
        if len({candidate.branch_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate branch IDs must be unique")
        reserved = self.max_render_cost_usd * 2 * len(self.candidates)
        if reserved > self.session_cost_ceiling_usd + 1e-9:
            raise ValueError("render and repair reservations exceed the session cost ceiling")
        return self


class LocalPlanningEvidence(FrozenStrictModel):
    branch_id: BranchId
    planning_ms: Annotated[float, Field(ge=0)]
    cache_hit: bool
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    model_revision: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class LocalAnticipationPrepareResponse(FrozenStrictModel):
    session_token: Annotated[
        str,
        StringConstraints(pattern=r"^anticipate_[a-f0-9]{24}$"),
    ]
    sequence: Annotated[int, Field(ge=0, le=2**31 - 1)]
    privacy_boundary: Literal["sanitized_scene_spec_v1"] = "sanitized_scene_spec_v1"
    planning: list[LocalPlanningEvidence] = Field(min_length=1, max_length=2)
    anticipation: AnticipationStatus


class AnticipatoryEdgeCoordinator:
    """Plan with private edge text, then send only the sanitized visual contract."""

    def __init__(
        self,
        *,
        planner: LocalScenePlanner,
        client: AnticipatoryEdgeClient,
        edge_gate_revision: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not edge_gate_revision.strip() or len(edge_gate_revision) > 120:
            raise ValueError("edge gate revision must contain 1-120 characters")
        self.planner = planner
        self.client = client
        self.edge_gate_revision = edge_gate_revision.strip()
        self._now = now or (lambda: datetime.now(UTC))

    async def prepare(
        self,
        request: LocalAnticipationPrepareRequest,
    ) -> LocalAnticipationPrepareResponse:
        session_token = request.session_token or new_anticipation_session_token()
        expires_at = self._now() + timedelta(seconds=request.expires_in_seconds)
        specs: list[AnticipatorySceneSpec] = []
        evidence: list[LocalPlanningEvidence] = []
        for candidate in request.candidates:
            result = await self.planner.plan(
                text=candidate.text,
                visual_style=request.visual_style,
                seed=candidate.seed,
            )
            specs.append(
                scene_spec_from_local_plan(
                    result.plan,
                    source_text=candidate.text,
                    branch_id=candidate.branch_id,
                    sequence=request.sequence,
                    source=candidate.source,
                    visual_style=request.visual_style,
                    seed=candidate.seed,
                    not_after=expires_at,
                    edge_gate_revision=self.edge_gate_revision,
                    max_render_cost_usd=request.max_render_cost_usd,
                )
            )
            evidence.append(
                LocalPlanningEvidence(
                    branch_id=candidate.branch_id,
                    planning_ms=result.wall_ms,
                    cache_hit=result.cache_hit,
                    model=result.metrics.model,
                    model_revision=result.model_revision,
                )
            )
        anticipation = await self.client.submit(
            AnticipatoryBatchRequest(
                session_token=session_token,
                sequence=request.sequence,
                candidates=specs,
                session_cost_ceiling_usd=request.session_cost_ceiling_usd,
            )
        )
        return LocalAnticipationPrepareResponse(
            session_token=session_token,
            sequence=request.sequence,
            planning=evidence,
            anticipation=anticipation,
        )

    async def status(self, session_token: str, sequence: int) -> AnticipationStatus:
        return await self.client.status(session_token, sequence)

    async def commit(self, request: CommitRequest) -> AnticipationStatus:
        return await self.client.commit(request)

    async def cancel(self, session_token: str, sequence: int) -> AnticipationStatus:
        return await self.client.cancel(session_token, sequence)

    async def asset(
        self,
        *,
        session_token: str,
        sequence: int,
        branch_id: str,
        kind: Literal["master", "depth"],
    ) -> tuple[bytes, str, str]:
        status = await self.client.status(session_token, sequence)
        candidate = next(
            (item for item in status.candidates if item.spec.branch_id == branch_id),
            None,
        )
        if candidate is None:
            raise AnticipatoryEdgeError("anticipatory branch was not found")
        if candidate.state not in {CandidateState.READY, CandidateState.COMMITTED}:
            raise AnticipatoryEdgeError("anticipatory branch is not ready")
        scene = candidate.rendered
        if scene is None:
            raise AnticipatoryEdgeError("anticipatory branch has no rendered scene")
        ref = scene.master_ref if kind == "master" else scene.depth_ref
        digest = scene.master_sha256 if kind == "master" else scene.depth_sha256
        content = await self.client.fetch_asset(ref, expected_sha256=digest)
        media_type = "image/png" if content.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
        return content, media_type, digest

    async def aclose(self) -> None:
        await self.client.aclose()


def new_anticipation_session_token() -> str:
    """Return an unlinkable per-reading token; do not derive it from reader data."""

    return f"anticipate_{secrets.token_hex(12)}"


def scene_spec_from_local_plan(
    plan: LiveScenePlan,
    *,
    source_text: str,
    branch_id: str,
    sequence: int,
    source: AnticipationSource,
    visual_style: str,
    seed: int,
    not_after: datetime,
    edge_gate_revision: str,
    max_render_cost_usd: float = 0.02,
) -> AnticipatorySceneSpec:
    """Run the existing privacy gate, then discard the source passage."""

    page = plan.to_page(
        source_text=source_text,
        visual_style=visual_style,
        seed=seed,
        page_id=f"anticipation-{sequence}",
    )
    if page.scene_spec is None:
        raise AnticipatoryEdgeError("local scene plan did not produce a visual specification")
    sanitized_plan = plan.model_dump(mode="json")
    continuity_sha256 = hashlib.sha256(
        json.dumps(sanitized_plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_subjects = [plan.focus.prompt, plan.accent.prompt]
    return AnticipatorySceneSpec(
        branch_id=branch_id,
        sequence=sequence,
        source=source,
        visual_brief=page.scene_spec.master_prompt,
        visual_style=visual_style,
        expected_subjects=expected_subjects,
        forbidden_content=[
            "readable text",
            "duplicate principal subject",
            "interface chrome",
        ],
        negative_prompt=page.scene_spec.negative_prompt,
        continuity_sha256=continuity_sha256,
        seed=seed,
        not_after=not_after,
        max_render_cost_usd=max_render_cost_usd,
        privacy=PrivacyAttestation(edge_gate_revision=edge_gate_revision),
    )


class AnticipatoryEdgeClient:
    """Strict client that never accepts caller-controlled remote asset URLs."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30,
        token_source: EdgeTokenSource | None = None,
        allow_loopback_http: bool = False,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed = urlsplit(normalized_url)
        loopback_http = (
            allow_loopback_http
            and parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        )
        if parsed.scheme != "https" and not loopback_http:
            raise ValueError("anticipatory service URL must use HTTPS")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 120:
            raise ValueError("anticipatory service timeout must be 1-120 seconds")
        self.base_url = normalized_url
        self.timeout_seconds = timeout_seconds
        self.token_source = token_source
        self.client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def submit(self, batch: AnticipatoryBatchRequest) -> AnticipationStatus:
        payload = await self._json_request(
            "POST",
            "/v1/anticipations",
            json_body=batch.model_dump(mode="json"),
        )
        return AnticipationStatus.model_validate(payload)

    async def prewarm_runtime(self) -> dict[str, str | bool]:
        payload = await self._json_request(
            "POST",
            "/v1/runtime:prewarm",
            json_body={
                "authorization": "I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU",
            },
            timeout_seconds=RUNTIME_PREWARM_TIMEOUT_SECONDS,
        )
        if payload.get("ready") is not True:
            detail = str(payload.get("detail") or "runtime prewarm failed")
            raise AnticipatoryEdgeError(_bounded(RuntimeError(detail)))
        return payload

    async def status(self, session_token: str, sequence: int) -> AnticipationStatus:
        payload = await self._json_request(
            "GET",
            f"/v1/anticipations/{session_token}/{sequence}",
        )
        return AnticipationStatus.model_validate(payload)

    async def commit(self, request: CommitRequest) -> AnticipationStatus:
        payload = await self._json_request(
            "POST",
            "/v1/anticipations:commit",
            json_body=request.model_dump(mode="json"),
        )
        return AnticipationStatus.model_validate(payload)

    async def cancel(self, session_token: str, sequence: int) -> AnticipationStatus:
        payload = await self._json_request(
            "DELETE",
            f"/v1/anticipations/{session_token}/{sequence}",
        )
        return AnticipationStatus.model_validate(payload)

    async def fetch_scene(self, scene: RenderedScene) -> tuple[bytes, bytes]:
        master, depth = await asyncio.gather(
            self.fetch_asset(scene.master_ref, expected_sha256=scene.master_sha256),
            self.fetch_asset(scene.depth_ref, expected_sha256=scene.depth_sha256),
        )
        return master, depth

    async def _json_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        client = await self._get_client()
        try:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=await self._headers(),
                json=json_body,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            raise AnticipatoryEdgeError(
                f"anticipatory request failed: {_bounded(error)}"
            ) from error
        if not isinstance(payload, dict):
            raise AnticipatoryEdgeError("anticipatory response must be an object")
        return payload

    async def fetch_asset(self, ref: str, *, expected_sha256: str) -> bytes:
        if not ref.startswith("asset_") or len(ref) != 70:
            raise AnticipatoryEdgeError("anticipatory scene contains an invalid asset reference")
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.base_url}/v1/assets/{ref}",
                headers=await self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AnticipatoryEdgeError(
                f"anticipatory asset fetch failed: {_bounded(error)}"
            ) from error
        content = response.content
        if not content or len(content) > MAX_ASSET_BYTES:
            raise AnticipatoryEdgeError("anticipatory asset has an invalid byte length")
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected_sha256 or response.headers.get("x-content-sha256") != digest:
            raise AnticipatoryEdgeError("anticipatory asset checksum does not match its contract")
        if not content.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")):
            raise AnticipatoryEdgeError("anticipatory asset is not a supported image")
        return content

    async def _headers(self) -> dict[str, str]:
        if self.token_source is None:
            return {}
        token = await self.token_source()
        if not token:
            raise AnticipatoryEdgeError("anticipatory token source returned no token")
        return {"Authorization": f"Bearer {token}"}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = self.client_factory(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    follow_redirects=False,
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=4,
                        max_keepalive_connections=4,
                        keepalive_expiry=300,
                    ),
                )
            return self._client

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()


def _bounded(error: Exception) -> str:
    return (" ".join(str(error).split()) or error.__class__.__name__)[:300]
