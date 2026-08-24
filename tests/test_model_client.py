import asyncio
import json

import httpx
from pydantic import BaseModel

from bookforge.config import Settings
from bookforge.model_client import OllamaClient, OpenAICompatibleClient


class _Output(BaseModel):
    value: str


def test_ollama_client_honors_bounded_configured_context_window() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gemma3:1b-it-q4_K_M",
                "message": {"content": '{"value":"ready"}'},
                "prompt_eval_count": 20,
                "eval_count": 5,
            },
        )

    async def run():
        settings = Settings(
            _env_file=None,
            model_backend="ollama",
            model_name="gemma3:1b-it-q4_K_M",
            model_context_tokens=4_096,
            model_max_output_tokens=320,
        )
        client = OllamaClient(settings)
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.generate(
                system="Return JSON.",
                prompt="Plan a scene.",
                output_type=_Output,
            )
        finally:
            await client.client.aclose()

    output, metrics = asyncio.run(run())

    assert output.value == "ready"
    assert metrics.model == "gemma3:1b-it-q4_K_M"
    assert observed["options"] == {
        "temperature": 0,
        "num_ctx": 4_096,
        "num_predict": 320,
    }


def test_openai_compatible_client_honors_bounded_output_tokens() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "local-gemma",
                "choices": [{"message": {"content": '{"value":"ready"}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
        )

    async def run():
        settings = Settings(
            _env_file=None,
            model_backend="openai",
            model_name="local-gemma",
            model_base_url="http://127.0.0.1:8088",
            model_max_output_tokens=256,
        )
        client = OpenAICompatibleClient(settings)
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            base_url=settings.model_base_url,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.generate(
                system="Return JSON.",
                prompt="Plan a scene.",
                output_type=_Output,
            )
        finally:
            await client.client.aclose()

    output, metrics = asyncio.run(run())

    assert output.value == "ready"
    assert metrics.model == "local-gemma"
    assert observed["max_tokens"] == 256
