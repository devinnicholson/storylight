import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from storylight.live_scene import (
    LiveSceneCapacityError,
    LiveSceneConflictError,
    LiveSceneCreateRequest,
    LiveSceneJobRegistry,
    LiveSceneStage,
)


def guarded(registry, *, revision=0, session="voice", submission=None):
    return LiveSceneCreateRequest(
        text="A fox beside a stream.", session_id=session,
        submission_id=submission or str(uuid4()),
        expected_server_instance_id=registry.server_instance_id,
        expected_session_revision=revision,
    )


class BlockingProvider:
    name = "fake"

    def __init__(self):
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.cancelled = []
        self.calls = 0

    async def generate(self, request, *, job_id):
        self.calls += 1
        await self.started.put(job_id)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(job_id)
            raise
        raise RuntimeError("Ambiguous upstream completion")
        yield  # Make this the provider's async iterator contract.


def test_guarded_request_requires_complete_session_fence():
    registry = LiveSceneJobRegistry(BlockingProvider())
    payload = guarded(registry).model_dump()
    for field in ["session_id", "submission_id", "expected_server_instance_id",
                  "expected_session_revision"]:
        with pytest.raises(ValidationError, match="Guarded submission"):
            LiveSceneCreateRequest.model_validate({**payload, field: None})


def test_concurrent_guarded_submissions_recover_without_cancel_or_second_paid_call():
    async def exercise():
        provider = BlockingProvider()
        registry = LiveSceneJobRegistry(provider)
        request = guarded(registry)
        first, duplicate = await asyncio.gather(registry.submit(request), registry.submit(request))
        assert first.job_id == duplicate.job_id
        assert await provider.started.get() == first.job_id
        with pytest.raises(LiveSceneConflictError, match="session changed"):
            await registry.submit(guarded(registry))
        with pytest.raises(LiveSceneConflictError, match="another request"):
            await registry.submit(request.model_copy(update={"text": "A different fox."}))
        other_server = LiveSceneJobRegistry(BlockingProvider())
        with pytest.raises(LiveSceneConflictError, match="another server"):
            await other_server.submit(request)
        assert provider.calls == 1 and provider.cancelled == []
        assert (await registry.get_session("voice")).job.job_id == first.job_id

        # A new ID with the current pointer is an intentional replacement.
        revision = (await registry.get_session("voice")).session_revision
        replacement_request = guarded(registry, revision=revision)
        replacement = await registry.submit(replacement_request)
        assert await provider.started.get() == replacement.job_id
        assert provider.cancelled == [first.job_id]
        provider.release.set()
        failed = await registry.wait(replacement.job_id)
        assert failed.stage is LiveSceneStage.FAILED
        assert (await registry.submit(replacement_request)) == failed
        assert provider.calls == 2  # Ambiguous terminal recovery does not rerender.
        await registry.close()
        await other_server.close()

    asyncio.run(exercise())


def test_evicted_claims_never_readmit_and_full_claim_store_still_allows_replay():
    async def exercise():
        provider = BlockingProvider()
        provider.release.set()
        registry = LiveSceneJobRegistry(provider, max_active_jobs=1, max_retained_jobs=1)
        first_request = guarded(registry)
        first = await registry.submit(first_request)
        await registry.wait(first.job_id)
        # Same random ID in a different session is a separate private claim.
        request = guarded(registry, session="other", submission=first_request.submission_id)
        job = await registry.submit(request)
        await registry.wait(job.job_id)
        with pytest.raises(LiveSceneConflictError, match="no longer retained"):
            await registry.submit(first_request)  # Original revision=0 despite pointer eviction.
        while len(registry._submission_jobs) < registry.max_submission_claims:
            revision = (await registry.get_session("other")).session_revision
            request = guarded(registry, session="other", revision=revision)
            job = await registry.submit(request)
            await registry.wait(job.job_id)
        revision = (await registry.get_session("other")).session_revision
        with pytest.raises(LiveSceneCapacityError, match="Guarded submission capacity"):
            await registry.submit(guarded(registry, session="other", revision=revision))
        assert (await registry.submit(request)).job_id == job.job_id
        assert provider.calls == registry.max_submission_claims
        await registry.close()

    asyncio.run(exercise())
