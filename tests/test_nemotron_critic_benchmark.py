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


def test_rejects_unreviewed_fixture_before_any_network_call():
    with pytest.raises(ValueError, match="inspected generated fox fixture"):
        asyncio.run(
            benchmark.run_benchmark(image_bytes=b"private image", base_url="https://example.org")
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
