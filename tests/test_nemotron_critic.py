import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from bookforge.nemotron_critic import (
    DEFAULT_NEMOTRON_VL_MODEL,
    NEMOTRON_CRITIC_MAX_OUTPUT_TOKENS,
    NEMOTRON_CRITIC_WIRE_SCHEMA,
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


def _wire_verdict() -> dict[str, object]:
    verdict = _verdict()
    return {
        "f": verdict["fidelity_score"],
        "c": verdict["composition_score"],
        "p": verdict["projection_legibility_score"],
        "i": verdict["identity_consistent"],
        "t": verdict["unintended_text"],
        "d": verdict["decision"],
        "r": verdict["reason"],
    }


def test_evaluate_sends_only_bounded_visual_contract_and_generated_image() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_NEMOTRON_VL_MODEL,
                "choices": [{"message": {"content": json.dumps(_wire_verdict())}}],
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
    assert payload["max_tokens"] == NEMOTRON_CRITIC_MAX_OUTPUT_TOKENS
    assert payload["response_format"]["json_schema"]["schema"] == (NEMOTRON_CRITIC_WIRE_SCHEMA)
    serialized_schema = json.dumps(NEMOTRON_CRITIC_WIRE_SCHEMA)
    assert '"enum"' not in serialized_schema
    assert '"minimum"' not in serialized_schema
    assert '"anyOf"' not in serialized_schema
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert json.loads(user_content[1]["text"]) == _request().model_dump(mode="json")
    serialized = observed[0].content.decode()
    assert "source_text" not in serialized
    assert "audio" not in user_content[1]["text"]
    assert "camera frame" not in user_content[1]["text"]


def test_request_rejects_raw_passage_field_at_schema_boundary() -> None:
    with pytest.raises(ValidationError, match="source_text"):
        NemotronCriticRequest.model_validate(
            {
                **_request().model_dump(mode="json"),
                "source_text": "Mira whispered the private sentence.",
            }
        )


def test_request_rejects_an_aggregate_contract_that_can_overrun_nim_context() -> None:
    with pytest.raises(ValidationError, match="1400-byte NIM context budget"):
        NemotronCriticRequest(
            visual_brief="A" * 1_200,
            expected_subjects=["B" * 300],
            forbidden_content=["readable text"],
        )


@pytest.mark.parametrize(
    ("decision", "correction", "wire_correction"),
    [
        (
            "refine",
            "Show only one fox and make the moon gate clearly visible.",
            "Show only one fox and make the moon gate clearly visible.",
        )
    ],
)
def test_compact_wire_protocol_preserves_actionable_refinement(
    decision: str,
    correction: str | None,
    wire_correction: str,
) -> None:
    wire = {**_wire_verdict(), "d": decision, "x": wire_correction}

    critic = NemotronVisionCritic(
        base_url="http://127.0.0.1:8000",
        allow_loopback_http=True,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.dumps(wire)}}]},
                )
            ),
            **kwargs,
        ),
    )
    evidence = asyncio.run(
        critic.evaluate(
            _request(),
            image_bytes=b"\xff\xd8\xffsynthetic-jpeg",
            media_type="image/jpeg",
        )
    )
    assert evidence.verdict.decision.value == decision
    assert evidence.verdict.correction_visual_brief == correction


def test_refinement_requires_actionable_visual_correction() -> None:
    with pytest.raises(ValidationError, match="correction visual brief"):
        NemotronCriticVerdict.model_validate(
            {
                **_verdict(),
                "decision": "refine",
                "correction_visual_brief": None,
            }
        )


@pytest.mark.parametrize("changes", [{"identity_consistent": False}, {"unintended_text": True}])
def test_accept_cannot_contradict_hard_quality_failures(changes):
    with pytest.raises(ValidationError, match="accept contradicts"):
        NemotronCriticVerdict.model_validate({**_verdict(), **changes})


def test_truncated_response_cannot_pass_even_if_partial_json_is_parseable():
    async def scenario():
        critic = NemotronVisionCritic(
            base_url="https://example.org",
            client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                    "choices": [{"finish_reason": "length", "message": {
                        "content": json.dumps(_wire_verdict())
                    }}]
                })), **kwargs,
            ),
        )
        try:
            with pytest.raises(NemotronCriticUnavailableError, match="did not finish"):
                await critic.evaluate(
                    _request(), image_bytes=b"\xff\xd8\xffsynthetic", media_type="image/jpeg"
                )
        finally:
            await critic.aclose()

    asyncio.run(scenario())


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
