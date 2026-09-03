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
DEFAULT_NEMOTRON_VL_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

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

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_NEMOTRON_VL_MODEL,
        timeout_seconds: float = 90,
        api_key: str = "",
        token_source: BearerTokenSource | None = None,
        allow_loopback_http: bool = False,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)
        loopback_http = (
            allow_loopback_http
            and parsed_url.scheme == "http"
            and parsed_url.hostname in {"127.0.0.1", "::1", "localhost"}
        )
        if parsed_url.scheme != "https" and not loopback_http:
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
    ) -> NemotronCriticEvidence:
        _validate_image(image_bytes, media_type)
        headers = await self._headers()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        schema = NemotronCriticVerdict.model_json_schema()
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict visual fidelity critic for projected "
                                "storybook illustrations. Inspect the synthetic image against "
                                "only the supplied visual brief. Never infer or request the "
                                "original passage, reader identity, audio, or camera data. "
                                "Return only schema-valid JSON."
                            ),
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
                            "schema": schema,
                        },
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            verdict = NemotronCriticVerdict.model_validate_json(content)
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
