import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from bookforge.nemotron_critic import (
    DEFAULT_NEMOTRON_VL_MODEL,
    NemotronCriticDecision,
    NemotronCriticRequest,
    NemotronCriticUnavailableError,
    NemotronCriticVerdict,
    NemotronVisionCritic,
)


def _request() -> NemotronCriticRequest:
    return NemotronCriticRequest(
        visual_brief=(
            "A small reader silhouette holds a glowing book beneath indigo paper trees; "
            "one flock of gold origami birds rises toward the moon."
        ),
        expected_subjects=["one reader", "glowing book", "origami bird flock"],
        forbidden_content=["readable text", "duplicate reader", "interface"],
    )


def _verdict() -> dict[str, object]:
    return {
        "fidelity_score": 0.94,
        "composition_score": 0.9,
        "projection_legibility_score": 0.88,
        "identity_consistent": True,
        "unintended_text": False,
        "decision": "accept",
        "reason": "The generated plate contains the expected subjects with clear silhouettes.",
        "correction_visual_brief": None,
    }


def test_evaluate_sends_only_bounded_visual_contract_and_generated_image() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_NEMOTRON_VL_MODEL,
                "choices": [{"message": {"content": json.dumps(_verdict())}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 24},
            },
        )

    async def token_source() -> str:
        return "signed-gcp-token"

    critic = NemotronVisionCritic(
        base_url="https://nemotron-private.example.run.app",
        token_source=token_source,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    evidence = asyncio.run(
        critic.evaluate(
            _request(),
            image_bytes=b"\xff\xd8\xffsynthetic-jpeg",
            media_type="image/jpeg",
        )
    )

    assert evidence.verdict.decision is NemotronCriticDecision.ACCEPT
    assert evidence.model == DEFAULT_NEMOTRON_VL_MODEL
    assert evidence.input_tokens == 50
    assert len(observed) == 1
    assert observed[0].headers["authorization"] == "Bearer signed-gcp-token"
    payload = json.loads(observed[0].content)
    user_content = payload["messages"][1]["content"]
    assert json.loads(user_content[0]["text"]) == _request().model_dump(mode="json")
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    serialized = observed[0].content.decode()
    assert "source_text" not in serialized
    assert "audio" not in user_content[0]["text"]
    assert "camera frame" not in user_content[0]["text"]


def test_request_rejects_raw_passage_field_at_schema_boundary() -> None:
    with pytest.raises(ValidationError, match="source_text"):
        NemotronCriticRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "source_text": "Mira whispered the private sentence.",
            }
        )


def test_refinement_requires_actionable_visual_correction() -> None:
    with pytest.raises(ValidationError, match="correction visual brief"):
        NemotronCriticVerdict.model_validate(
            {
                **_verdict(),
                "decision": "refine",
                "correction_visual_brief": None,
            }
        )


def test_invalid_media_fails_before_network() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    critic = NemotronVisionCritic(
        base_url="https://nemotron-private.example.run.app",
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        asyncio.run(
            critic.evaluate(
                _request(),
                image_bytes=b"not-an-image",
                media_type="image/png",
            )
        )
    assert called is False


def test_invalid_model_output_fails_closed() -> None:
    critic = NemotronVisionCritic(
        base_url="https://nemotron-private.example.run.app",
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "{}"}}]},
                )
            ),
            **kwargs,
        ),
    )
    with pytest.raises(NemotronCriticUnavailableError, match="critic request failed"):
        asyncio.run(
            critic.evaluate(
                _request(),
                image_bytes=b"\x89PNG\r\n\x1a\nsynthetic",
                media_type="image/png",
            )
        )


def test_critic_requires_https_and_single_auth_mechanism() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        NemotronVisionCritic(base_url="http://127.0.0.1:9000")

    async def token_source() -> str:
        return "token"

    with pytest.raises(ValueError, match="either an API key"):
        NemotronVisionCritic(
            base_url="https://nemotron.example",
            api_key="key",
            token_source=token_source,
        )
