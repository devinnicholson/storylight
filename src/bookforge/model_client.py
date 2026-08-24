import json
import time
from abc import ABC, abstractmethod
from typing import TypeVar

import httpx
from pydantic import BaseModel

from bookforge.config import Settings
from bookforge.domain import (
    AmbientEffect,
    AmbientMotion,
    CameraMotion,
    ComprehensionPrompt,
    GeneratedPagePlan,
    GeneratedStoryPlan,
    InterventionDecision,
    LayerComposition,
    LiteracySupport,
    ModelMetrics,
    SceneSpecV2,
    StoryTrigger,
    SupportAction,
    VisualLayer,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelUnavailableError(RuntimeError):
    pass


class StructuredModelClient(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        raise NotImplementedError

    @abstractmethod
    async def probe(self) -> tuple[bool, str]:
        raise NotImplementedError


class OllamaClient(StructuredModelClient):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.model_base_url.rstrip("/"),
            timeout=settings.model_timeout_seconds,
        )
        self._gpu_offload_verified = False

    async def _verify_required_gpu_offload(self) -> None:
        if not self.settings.model_require_gpu or self._gpu_offload_verified:
            return
        try:
            response = await self.client.get("/api/ps")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelUnavailableError(
                f"Ollama GPU-offload verification failed: {error}"
            ) from error
        configured = next(
            (
                model
                for model in response.json().get("models", [])
                if model.get("name") == self.settings.model_name
                or model.get("model") == self.settings.model_name
            ),
            None,
        )
        if configured is None:
            raise ModelUnavailableError(
                "Ollama GPU-offload verification failed: configured model is not loaded"
            )
        if int(configured.get("size_vram") or 0) <= 0:
            raise ModelUnavailableError(
                "Ollama GPU offload is required, but the configured model is running on CPU"
            )
        self._gpu_offload_verified = True

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        started = time.perf_counter()
        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": self.settings.model_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "format": output_type.model_json_schema(),
                    "keep_alive": self.settings.model_keep_alive,
                    "options": {
                        "temperature": 0,
                        "num_ctx": self.settings.model_context_tokens,
                        "num_predict": self.settings.model_max_output_tokens,
                    },
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelUnavailableError(f"Ollama request failed: {error}") from error

        await self._verify_required_gpu_offload()

        payload = response.json()
        content = payload.get("message", {}).get("content", "")
        try:
            parsed = output_type.model_validate_json(content)
        except ValueError as error:
            message = f"Model returned invalid structured output: {error}"
            raise ModelUnavailableError(message) from error

        return parsed, ModelMetrics(
            backend="ollama",
            model=payload.get("model", self.settings.model_name),
            total_ms=(time.perf_counter() - started) * 1000,
            load_ms=payload.get("load_duration", 0) / 1_000_000,
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
        )

    async def probe(self) -> tuple[bool, str]:
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as error:
            return False, f"Ollama is unreachable: {error}"
        names = {model.get("name") for model in response.json().get("models", [])}
        if self.settings.model_name not in names:
            return False, f"Ollama is running; pull {self.settings.model_name}"
        if self.settings.model_require_gpu:
            try:
                running = await self.client.get("/api/ps")
                running.raise_for_status()
            except httpx.HTTPError as error:
                return False, f"Ollama GPU-offload verification failed: {error}"
            configured = next(
                (
                    model
                    for model in running.json().get("models", [])
                    if model.get("name") == self.settings.model_name
                    or model.get("model") == self.settings.model_name
                ),
                None,
            )
            if configured is None:
                return True, "Ollama model is installed; GPU offload will be verified after warmup"
            if int(configured.get("size_vram") or 0) <= 0:
                return False, "Ollama model is loaded without required GPU offload"
            self._gpu_offload_verified = True
            return True, "Ollama is running with required GPU offload"
        return True, "Ollama is running and the configured model is installed"


class OpenAICompatibleClient(StructuredModelClient):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.model_api_key:
            headers["Authorization"] = f"Bearer {settings.model_api_key}"
        self.client = httpx.AsyncClient(
            base_url=settings.model_base_url.rstrip("/"),
            timeout=settings.model_timeout_seconds,
            headers=headers,
        )

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        started = time.perf_counter()
        schema = output_type.model_json_schema()
        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.settings.model_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": self.settings.model_max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": output_type.__name__, "schema": schema},
                    },
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelUnavailableError(f"OpenAI-compatible request failed: {error}") from error

        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = output_type.model_validate_json(content)
        usage = payload.get("usage", {})
        return parsed, ModelMetrics(
            backend="openai",
            model=payload.get("model", self.settings.model_name),
            total_ms=(time.perf_counter() - started) * 1000,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def probe(self) -> tuple[bool, str]:
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as error:
            return False, f"Model endpoint is unreachable: {error}"
        return True, "OpenAI-compatible model endpoint is reachable"


class FakeModelClient(StructuredModelClient):
    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        del system
        if output_type is InterventionDecision:
            output: BaseModel = InterventionDecision(
                action=SupportAction.HIGHLIGHT_GRAPHEME,
                target="th",
                display_text="Try the highlighted sound.",
                rationale_code="initial_pause",
                confidence=0.91,
            )
        elif output_type is GeneratedStoryPlan:
            data = json.loads(prompt.split("Book input:\n", 1)[1])
            pages = []
            for page in data["pages"]:
                first_word = page["text"].split()[0].strip('.,!?"').lower()
                pages.append(
                    GeneratedPagePlan(
                        page_id=page["page_id"],
                        scene_summary="A test scene generated without a model runtime.",
                        scene_spec=SceneSpecV2(
                            master_prompt=(
                                "Cinematic luminous paper theater landscape, wide 16:9 composition"
                            ),
                            camera=CameraMotion(kind="slow_push"),
                            composition=[
                                LayerComposition(
                                    layer_id="background",
                                    center_x=0.5,
                                    center_y=0.5,
                                    width=1,
                                    height=1,
                                    depth=0.15,
                                    ambient_motion=AmbientMotion(
                                        kind="parallax",
                                        amplitude_x=0.01,
                                        period_ms=12_000,
                                    ),
                                )
                            ],
                            ambience=[AmbientEffect(kind="dust")],
                        ),
                        layers=[
                            VisualLayer(
                                layer_id="background",
                                kind="background",
                                prompt="luminous paper theater landscape, wide composition",
                                z_index=0,
                                motion="slow 2D parallax",
                            )
                        ],
                        triggers=[
                            StoryTrigger(
                                trigger_id="reveal-background",
                                word=first_word,
                                action="reveal",
                                target_layer_id="background",
                                duration_ms=250,
                            )
                        ],
                        literacy_support=[
                            LiteracySupport(
                                word=first_word,
                                grapheme=first_word[0],
                                hint_ladder=[f"Highlight {first_word[0]}"],
                            )
                        ],
                        comprehension=[
                            ComprehensionPrompt(
                                question="What happened on this page?",
                                expected_concepts=["story event"],
                            )
                        ],
                    )
                )
            output = GeneratedStoryPlan(pages=pages)
        else:
            raise TypeError(f"Fake client has no fixture for {output_type.__name__}")
        return output, ModelMetrics(backend="fake", model="fake", total_ms=1)

    async def probe(self) -> tuple[bool, str]:
        return True, "Fake model is ready"


def build_model_client(settings: Settings) -> StructuredModelClient:
    if settings.model_backend == "ollama":
        return OllamaClient(settings)
    if settings.model_backend == "openai":
        return OpenAICompatibleClient(settings)
    return FakeModelClient()
