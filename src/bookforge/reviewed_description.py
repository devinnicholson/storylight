"""Local reviewed-description planning, with an optional isolated syntax service."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import httpx
from pydantic import Field, StringConstraints

from bookforge.bounded_description import REVISION as BOUNDED_REVISION
from bookforge.bounded_description import plan_bounded_description
from bookforge.domain import FrozenStrictModel, ModelMetrics
from bookforge.live_scene_planner import (
    LiveSceneGraphWirePlan,
    LiveScenePlannerError,
    LiveScenePlanningResult,
    LiveSceneWireFocus,
    LiveSceneWireMagic,
    validate_live_scene_plan_privacy,
)
from bookforge.scene_facts import SceneFactsV2

REVISION = "reviewed-language-v1"
PARSER_REVISION = "dependency-scene-draft-v1"
AUDITED_REVISION = "reviewed-language-bounded-v1"
AUDIT_REVISION = "dependency-nominal-audit-v1"
CURRENT_REVISIONS = frozenset((BOUNDED_REVISION, REVISION, AUDITED_REVISION))


def is_reviewed_compiler(model: str) -> bool:
    return model.startswith(("bounded-description-", "reviewed-language-"))


class LocalSceneOmission(FrozenStrictModel):
    ref: Annotated[int, Field(strict=True, ge=0, le=500)]
    local_text: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    reason: Literal["proper_name_policy"]
    omission_requires_review: Literal[True]


class _SyntaxIssue(FrozenStrictModel):
    kind: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    token: Annotated[int, Field(strict=True, ge=0, le=500)] | None = None


class _ParserResponse(FrozenStrictModel):
    revision: Literal["dependency-scene-draft-v1"]
    status: Literal["draft_ready", "omission_review", "needs_review"]
    facts: SceneFactsV2 | None
    reason: Annotated[str, StringConstraints(max_length=100)] | None
    render_admitted: Literal[False]
    requires_fact_review: Literal[True]
    local_omissions: Annotated[tuple[LocalSceneOmission, ...], Field(max_length=50)]
    syntax_issues: Annotated[tuple[_SyntaxIssue, ...], Field(max_length=500)]
    renderer_prompt_preview: Annotated[str, StringConstraints(max_length=8_000)] | None


class _AuditIssue(FrozenStrictModel):
    label_index: Annotated[int, Field(strict=True, ge=0, le=9)]
    reason: Literal["not_nominal_span"]


class _NominalAudit(FrozenStrictModel):
    revision: Literal["dependency-nominal-audit-v1"]
    accepted: Annotated[bool, Field(strict=True)]
    issues: Annotated[tuple[_AuditIssue, ...], Field(max_length=10)]


@dataclass(frozen=True)
class ReviewedDescription:
    result: LiveScenePlanningResult
    requires_fact_review: bool = False
    local_omissions: tuple[LocalSceneOmission, ...] = ()
    visual_fact_digest: str | None = None

    def is_confirmed(self, confirm: bool, digest: str | None) -> bool:
        return not self.requires_fact_review or (
            confirm is True and digest is not None and digest == self.visual_fact_digest
        )


async def _parser_request(payload: dict, parser_socket: Path) -> bytes:
    if not 1 <= len(payload["text"]) <= 500 or not 1 <= len(payload["visual_style"]) <= 120:
        raise ValueError("input bounds")
    transport = httpx.AsyncHTTPTransport(uds=str(parser_socket), retries=0)
    async with asyncio.timeout(10), httpx.AsyncClient(
        transport=transport, trust_env=False, follow_redirects=False, timeout=10,
    ) as client, client.stream(
        "POST", "http://scene-parser/v1/scene-facts", json=payload,
    ) as response:
        if response.status_code != 200:
            raise ValueError("parser unavailable")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > 65_536:
                raise ValueError("response bounds")
    return bytes(body)


async def review_description(
    text: str, visual_style: str, seed: int, *, parser_socket: Path | None = None,
) -> ReviewedDescription:
    started = perf_counter()
    try:
        bounded = plan_bounded_description(text, visual_style, seed)
    except LiveScenePlannerError:
        if parser_socket is None:
            raise
        bounded = None
    if parser_socket is None:
        return ReviewedDescription(bounded)
    try:
        payload = {"text": text, "visual_style": visual_style}
        if bounded is not None:
            facts = bounded.plan.scene_facts
            labels = list(dict.fromkeys([
                *(item.label for item in (*facts.subjects, *facts.objects)),
                *([facts.setting.label] if facts.setting.label != "unspecified" else []),
                *(item.value for item in facts.negatives if item.kind == "additional_object"),
            ]))
            if not 1 <= len(labels) <= 10 or any(not 1 <= len(label) <= 64 for label in labels):
                raise ValueError("nominal bounds")
            audit = _NominalAudit.model_validate_json(await _parser_request(
                {**payload, "nominal_labels": labels}, parser_socket,
            ))
            if not audit.accepted or audit.issues:
                raise ValueError("unverified nominal label")
            elapsed = (perf_counter() - started) * 1_000
            return ReviewedDescription(bounded.model_copy(update={
                "metrics": ModelMetrics(
                    backend="local-syntax", model=AUDITED_REVISION, total_ms=elapsed,
                ),
                "model_revision": AUDIT_REVISION, "wall_ms": elapsed,
            }))
        parsed = _ParserResponse.model_validate_json(await _parser_request(payload, parser_socket))
        facts = parsed.facts
        if (parsed.status == "needs_review" or facts is None or parsed.syntax_issues
                or parsed.reason is not None or not facts.subjects
                or (parsed.status == "omission_review") != bool(parsed.local_omissions)):
            raise ValueError("unresolved draft")
        facts.validate_source_grounding(source_text=text)
        expected = facts.to_renderer_prompt(source_text=text, visual_style=visual_style)
        if expected != parsed.renderer_prompt_preview:
            raise ValueError("renderer differs from facts")
        for omission in parsed.local_omissions:
            if (omission.local_text.casefold() not in text.casefold()
                    or omission.local_text.casefold() in expected.casefold()):
                raise ValueError("invalid local omission")
        first = facts.subjects[0]
        if not first.actions:
            raise ValueError("missing subject action")
        plan = LiveSceneGraphWirePlan(
            background_prompt="unspecified",
            focus=LiveSceneWireFocus(
                kind="character", subject=first.label, action=first.actions[0].split()[0],
            ),
            magic=LiveSceneWireMagic(kind="effect", prompt="none"), scene_facts=facts,
        ).to_live_scene_plan(context_text=text)
        validate_live_scene_plan_privacy(plan, source_text=text)
        page = plan.to_page(source_text=text, visual_style=visual_style, seed=seed)
        if page.scene_spec.master_prompt != expected:
            raise ValueError("graph render contract changed")
    except (ValueError, KeyError, httpx.HTTPError, TimeoutError, LiveScenePlannerError) as error:
        raise LiveScenePlannerError(
            "Local description could not be verified; review the wording"
        ) from error
    elapsed = (perf_counter() - started) * 1_000
    digest = hashlib.sha256(json.dumps({
        "text": text, "visual_style": visual_style, "revision": PARSER_REVISION,
        "facts": facts.model_dump(mode="json"),
        "local_omissions": [item.model_dump(mode="json") for item in parsed.local_omissions],
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return ReviewedDescription(
        LiveScenePlanningResult(
            plan=plan,
            metrics=ModelMetrics(backend="local-syntax", model=REVISION, total_ms=elapsed),
            model_revision=PARSER_REVISION, wall_ms=elapsed,
        ),
        requires_fact_review=True, local_omissions=parsed.local_omissions,
        visual_fact_digest=digest,
    )
