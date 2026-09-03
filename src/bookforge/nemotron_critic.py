from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, StringConstraints, model_validator

from bookforge.domain import FrozenStrictModel

MAX_CRITIC_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CRITIC_CONTRACT_UTF8_BYTES = 1_400
DEFAULT_NEMOTRON_VL_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
NEMOTRON_CRITIC_MAX_OUTPUT_TOKENS = 128
NEMOTRON_CRITIC_SYSTEM_PROMPT = (
    "Judge the illustration only against the visual brief. Return JSON "
    "using f=fidelity score, c=composition score, p=projection-legibility score, "
    "i=identity-consistent, t=unintended-text, d=decision, r=reason, and optional "
    "x=correction. Scores are 0..1; d is accept, refine, or reject. Keep r under 12 words. "
    "Include x under 28 words only for refine or reject. Never request "
    "passage, reader, audio, or camera data."
)
NEMOTRON_CRITIC_WIRE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "f": {"type": "number"},
        "c": {"type": "number"},
        "p": {"type": "number"},
        "i": {"type": "boolean"},
        "t": {"type": "boolean"},
        "d": {"type": "string"},
        "r": {"type": "string"},
        "x": {"type": "string"},
    },
    "required": ["f", "c", "p", "i", "t", "d", "r"],
    "additionalProperties": False,
}

BearerTokenSource = Callable[[], Awaitable[str]]


class NemotronCriticError(RuntimeError):
    pass


class NemotronCriticUnavailableError(NemotronCriticError):
    pass


class NemotronCriticDecision(StrEnum):
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"


CriticPhrase = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]


class NemotronCriticRequest(FrozenStrictModel):
    """Privacy-minimized critic input; there is deliberately no source-text field."""

    visual_brief: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=1_200),
    ]
    expected_subjects: list[CriticPhrase] = Field(default_factory=list, max_length=8)
    forbidden_content: list[CriticPhrase] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def bound_total_text_context(self) -> NemotronCriticRequest:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > MAX_CRITIC_CONTRACT_UTF8_BYTES:
            raise ValueError("critic visual contract exceeds the 1400-byte NIM context budget")
        return self


class NemotronCriticVerdict(FrozenStrictModel):
    fidelity_score: Annotated[float, Field(ge=0, le=1)]
    composition_score: Annotated[float, Field(ge=0, le=1)]
    projection_legibility_score: Annotated[float, Field(ge=0, le=1)]
    identity_consistent: bool
    unintended_text: bool
    decision: NemotronCriticDecision
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
    ]
    correction_visual_brief: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=700)]
        | None
    ) = None

    @model_validator(mode="after")
    def require_actionable_correction(self) -> NemotronCriticVerdict:
        if (
            self.decision in {NemotronCriticDecision.REFINE, NemotronCriticDecision.REJECT}
            and self.correction_visual_brief is None
        ):
            raise ValueError("refine and reject decisions require a correction visual brief")
        return self


class NemotronCriticEvidence(FrozenStrictModel):
    verdict: NemotronCriticVerdict
    model: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    ]
    latency_ms: Annotated[float, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0


class NemotronVisionCritic:
    """OpenAI-compatible Nemotron VL client kept off the first-image critical path."""

    system_prompt = NEMOTRON_CRITIC_SYSTEM_PROMPT
    wire_schema = NEMOTRON_CRITIC_WIRE_SCHEMA

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_NEMOTRON_VL_MODEL,
        timeout_seconds: float = 90,
        api_key: str = "",
        token_source: BearerTokenSource | None = None,
        allow_loopback_http: bool = False,
        allow_cluster_http: bool = False,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)
        loopback_http = (
            allow_loopback_http
            and parsed_url.scheme == "http"
            and parsed_url.hostname in {"127.0.0.1", "::1", "localhost"}
        )
        cluster_http = (
            allow_cluster_http
            and parsed_url.scheme == "http"
            and parsed_url.hostname is not None
            and parsed_url.hostname.endswith(".svc.cluster.local")
        )
        if parsed_url.scheme != "https" and not loopback_http and not cluster_http:
            raise ValueError("Nemotron critic URL must use HTTPS")
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Nemotron critic model is required")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 300:
            raise ValueError("Nemotron critic timeout must be 1-300 seconds")
        if api_key and token_source is not None:
            raise ValueError("configure either an API key or a token source, not both")
        self.base_url = normalized_url
        self.model = normalized_model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.token_source = token_source
        self.client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def probe(self) -> tuple[bool, str]:
        try:
            headers = await self._headers()
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/v1/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            return False, f"Nemotron critic is unreachable: {error}"
        models = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if models and self.model not in models:
            return False, f"Nemotron endpoint does not expose {self.model}"
        return True, "Nemotron multimodal critic is reachable"

    async def evaluate(
        self,
        request: NemotronCriticRequest,
        *,
        image_bytes: bytes,
        media_type: Literal["image/jpeg", "image/png"],
        timeout_seconds: float | None = None,
    ) -> NemotronCriticEvidence:
        _validate_image(image_bytes, media_type)
        request_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(request_timeout) or not 1 <= request_timeout <= 300:
            raise ValueError("Nemotron request timeout must be 1-300 seconds")
        headers = await self._headers()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": NEMOTRON_CRITIC_MAX_OUTPUT_TOKENS,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                                },
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        request.model_dump(mode="json"),
                                        separators=(",", ":"),
                                        ensure_ascii=False,
                                    ),
                                },
                            ],
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "nemotron_critic_verdict",
                            "strict": True,
                            # NIM 1.3.1 falls back from xgrammar to the much slower
                            # outlines backend for enums, ranges, patterns, and nullable
                            # unions. Pydantic still enforces those constraints after the
                            # compact wire object is generated.
                            "schema": self.wire_schema,
                        },
                    },
                },
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            verdict = _verdict_from_wire(content)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, ValueError) as error:
            raise NemotronCriticUnavailableError(
                f"Nemotron critic request failed: {error}"
            ) from error
        usage = payload.get("usage", {})
        return NemotronCriticEvidence(
            verdict=verdict,
            model=str(payload.get("model") or self.model),
            latency_ms=(time.perf_counter() - started) * 1_000,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

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
                        max_connections=2,
                        max_keepalive_connections=2,
                        # NIM 1.3.1 closes idle connections after five seconds.
                        keepalive_expiry=4,
                    ),
                )
            return self._client

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()

    async def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.token_source is not None:
            token = await self.token_source()
            if not token:
                raise NemotronCriticUnavailableError("Nemotron token source returned no token")
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _validate_image(image_bytes: bytes, media_type: str) -> None:
    if not image_bytes:
        raise ValueError("critic image cannot be empty")
    if len(image_bytes) > MAX_CRITIC_IMAGE_BYTES:
        raise ValueError("critic image exceeds the 8 MiB limit")
    signatures = {
        "image/jpeg": (b"\xff\xd8\xff",),
        "image/png": (b"\x89PNG\r\n\x1a\n",),
    }
    if not any(image_bytes.startswith(signature) for signature in signatures[media_type]):
        raise ValueError(f"critic image bytes do not match {media_type}")


def _verdict_from_wire(content: str) -> NemotronCriticVerdict:
    wire = json.loads(content)
    if not isinstance(wire, dict):
        raise ValueError("Nemotron critic verdict must be an object")
    decision = wire.get("d")
    return NemotronCriticVerdict.model_validate(
        {
            "fidelity_score": wire.get("f"),
            "composition_score": wire.get("c"),
            "projection_legibility_score": wire.get("p"),
            "identity_consistent": wire.get("i"),
            "unintended_text": wire.get("t"),
            "decision": decision,
            "reason": wire.get("r"),
            # Some constrained decoders materialize an optional string as an
            # empty/"None" sentinel. An accepted scene has no correction by
            # definition; refine/reject still fail closed on a missing brief.
            "correction_visual_brief": None if decision == "accept" else wire.get("x"),
        }
    )
