import asyncio
import hashlib

import httpx
import pytest

from bookforge import nemotron_critic_benchmark as benchmark
from bookforge.nemotron_critic import (
    NemotronCriticEvidence,
    NemotronCriticUnavailableError,
    NemotronCriticVerdict,
)


def test_regression_suite_contains_positive_and_action_count_position_negatives():
    cases = benchmark.regression_cases()
    assert sum(accept for _, accept, _ in cases) == 2
    assert len(cases) == 6
    assert "carrying_negative" in {name for name, _, _ in cases}
    assert "fox" not in benchmark.OBSERVATION_PROMPT
    assert next(iter(benchmark.OBSERVATION_SCHEMA["properties"])) == "r"


def test_rejects_unreviewed_fixture_before_any_network_call():
    with pytest.raises(ValueError, match="inspected generated fox fixture"):
        asyncio.run(
            benchmark.run_benchmark(image_bytes=b"private image", base_url="https://example.org")
        )


def test_paired_order_errors_and_client_cleanup(monkeypatch):
    image = b"synthetic fixture"
    monkeypatch.setattr(benchmark, "FOX_IMAGE_SHA256", hashlib.sha256(image).hexdigest())
    monkeypatch.setattr(benchmark, "_nemotron_review_copy", lambda data: data)
    calls = []
    closed = []

    class FakeCritic:
        def __init__(self, **kwargs):
            self.system_prompt = ""

        async def evaluate(self, request, **kwargs):
            name = next(
                key for key, value in benchmark.PROMPTS.items() if value == self.system_prompt
            )
            calls.append(name)
            if "Two silver foxes" in request.visual_brief:
                raise NemotronCriticUnavailableError("one failed request; no retry")
            return NemotronCriticEvidence(
                verdict=NemotronCriticVerdict(
                    fidelity_score=0.9,
                    composition_score=0.9,
                    projection_legibility_score=0.9,
                    identity_consistent=True,
                    unintended_text=False,
                    decision="accept",
                    reason="Fixture verdict",
                ),
                model="fixture",
                latency_ms=100,
                input_tokens=30,
                output_tokens=20,
            )

        async def aclose(self):
            closed.append(self)

    report = asyncio.run(
        benchmark.run_benchmark(
            image_bytes=image,
            base_url="http://127.0.0.1:8000",
            repeats=1,
            critic_factory=FakeCritic,
        )
    )
    assert len(closed) == 2
    assert len(calls) == 14  # Two warmups plus six paired contracts, including failures.
    assert calls[2:6] == ["baseline", "observation_first", "observation_first", "baseline"]
    assert report["summary"]["baseline"]["errors"] == 1
    assert report["summary"]["baseline"]["false_accepts"] == 3
    assert report["summary"]["baseline"]["correct"] == 2
    assert report["summary"]["baseline"]["median_ms"] == 100


@pytest.mark.parametrize("repeats", [0, 4])
def test_call_limit(repeats):
    with pytest.raises(ValueError, match="repeats"):
        asyncio.run(
            benchmark.run_benchmark(
                image_bytes=b"", base_url="https://example.org", repeats=repeats
            )
        )


def test_transport_failure_stops_remaining_calls_and_preserves_evidence(monkeypatch):
    image = b"synthetic fixture"
    monkeypatch.setattr(benchmark, "FOX_IMAGE_SHA256", hashlib.sha256(image).hexdigest())
    monkeypatch.setattr(benchmark, "_nemotron_review_copy", lambda data: data)
    calls = []
    closed = []

    class FailingCritic:
        def __init__(self, **kwargs):
            pass

        async def evaluate(self, request, **kwargs):
            calls.append(request)
            if len(calls) > 2:
                raise NemotronCriticUnavailableError("transport down") from httpx.ConnectError(
                    "down"
                )
            return NemotronCriticEvidence(
                verdict=NemotronCriticVerdict(
                    fidelity_score=0.9,
                    composition_score=0.9,
                    projection_legibility_score=0.9,
                    identity_consistent=True,
                    unintended_text=False,
                    decision="accept",
                    reason="Fixture verdict",
                ),
                model="fixture",
                latency_ms=1,
            )

        async def aclose(self):
            closed.append(self)

    report = asyncio.run(
        benchmark.run_benchmark(
            image_bytes=image,
            base_url="https://example.org",
            critic_factory=FailingCritic,
        )
    )
    assert len(calls) == 3
    assert len(closed) == 2
    assert report["transport_interrupted"] is True
    assert len(report["results"]) == 1
    assert report["results"][0]["error"] == "transport down"
