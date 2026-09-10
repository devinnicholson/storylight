"""Nonbillable readiness reuse, ownership and routing boundaries; no cloud calls."""

import asyncio
import base64
import time
from types import SimpleNamespace

import httpx
import pytest

from storylight.finite_modal_provider import FastSceneRequest
from storylight.provider_router import ProviderRoute, ResilientFastSceneProvider
from storylight.vertex_scene_provider import (
    VertexGeminiImageSceneProvider,
    VertexSceneAmbiguousError,
    _projection_depth_png,
)


class ForbiddenFallback:
    async def probe(self):
        raise AssertionError("preconnect touched another provider")


def router_for(handler):
    async def token():
        return "fixture-access-token"

    vertex = VertexGeminiImageSceneProvider(
        project_id="fixture-project", token_source=token,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs,
        ),
    )
    router = ResilientFastSceneProvider([
        ProviderRoute("vertex", vertex, healthy_probe_ttl_seconds=300),
        ProviderRoute("forbidden-fallback", ForbiddenFallback()),
    ])
    return router, vertex


def test_recording_and_submission_share_head_despite_cancelled_recording(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        methods = []

        async def handler(request):
            methods.append(request.method)
            assert request.headers["authorization"] == "Bearer fixture-access-token"
            if request.method == "HEAD":
                assert request.content == b""
                entered.set()
                await release.wait()
                return httpx.Response(405)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{
                "inlineData": {"mimeType": "image/png", "data": base64.b64encode(
                    _projection_depth_png(32, 18),
                ).decode()},
            }]}}]})

        router, vertex = router_for(handler)
        early = asyncio.create_task(router.prepare_connection())
        await entered.wait()
        submitted = asyncio.create_task(router.generate_fast(
            FastSceneRequest(scene_id="fixture-scene", prompt="A red balloon."),
            output_dir=tmp_path / "scene",
        ))
        await asyncio.sleep(0)
        early.cancel()
        with pytest.raises(asyncio.CancelledError):
            await early
        assert not vertex._probe_task.done()
        release.set()
        await submitted
        assert methods == ["HEAD", "POST"]
        ready, remaining = await router.prepare_connection()
        assert ready and 0 < remaining <= 300
        assert methods == ["HEAD", "POST"]
        await router.aclose()

    asyncio.run(exercise())


def test_cache_reads_do_not_extend_expiry_and_failures_are_not_cached(monkeypatch):
    import storylight.provider_router as module

    now, responses, methods = [0.0], [404, 401, 404], []
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: now[0], perf_counter=time.perf_counter,
    ))

    def handler(request):
        methods.append(request.method)
        return httpx.Response(responses.pop(0))

    async def exercise():
        router, vertex = router_for(handler)
        assert await router.prepare_connection() == (True, 300)
        now[0] = 299
        assert await router.prepare_connection() == (True, 1)
        now[0] = 301
        assert await router.prepare_connection() == (False, 0)
        assert vertex._cached_token == ""
        assert "vertex" not in router._unavailable_until
        assert await router.prepare_connection() == (True, 300)
        assert methods == ["HEAD"] * 3
        await router.aclose()

    asyncio.run(exercise())


def test_submission_joins_head_until_router_cache_publication():
    async def exercise():
        committing, release, submitted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        methods = []

        def handler(request):
            methods.append(request.method)
            return httpx.Response(404)

        router, vertex = router_for(handler)
        mark = router._mark_healthy

        async def held_mark(*args, **kwargs):
            committing.set()
            await release.wait()
            return await mark(*args, **kwargs)

        router._mark_healthy = held_mark
        early = asyncio.create_task(router.prepare_connection())
        await committing.wait()
        assert vertex._probe_task.done() and "vertex" not in router._healthy_until

        async def submission():
            submitted.set()
            return await router._probe_route(router.routes[0], attempts=[])

        pending = asyncio.create_task(submission())
        await submitted.wait()
        await asyncio.sleep(0)
        assert not pending.done() and methods == ["HEAD"]
        early.cancel()
        with pytest.raises(asyncio.CancelledError):
            await early
        assert not router._connection_task.done()
        release.set()
        ready, _, probe_ms = await pending
        assert ready and probe_ms > 0 and methods == ["HEAD"]
        assert "vertex" in router._healthy_until
        assert (await router._probe_route(router.routes[0], attempts=[]))[2] == 0
        await router.aclose()

    asyncio.run(exercise())


def test_submission_owner_join_obeys_probe_timeout_without_cancelling_owner():
    async def exercise():
        committing, release = asyncio.Event(), asyncio.Event()
        methods = []

        def handler(request):
            methods.append(request.method)
            return httpx.Response(404)

        router, _ = router_for(handler)
        router.probe_timeout_seconds = 0.01
        mark = router._mark_healthy

        async def held_mark(*args, **kwargs):
            committing.set()
            await release.wait()
            return await mark(*args, **kwargs)

        router._mark_healthy = held_mark
        early = asyncio.create_task(router.prepare_connection())
        await committing.wait()
        attempts = []
        result = await asyncio.wait_for(
            router._probe_route(router.routes[0], attempts=attempts), timeout=0.5,
        )
        assert result[0] is False and "exceeded" in result[1]
        assert attempts[0]["outcome"] == "probe_unavailable"
        assert not router._connection_task.done() and methods == ["HEAD"]
        release.set()
        # The timed-out submission opened its normal circuit; the late HEAD
        # must not restore a readiness claim from the previous epoch.
        assert await early == (False, 0)
        assert "vertex" not in router._healthy_until
        await router.aclose()

    asyncio.run(exercise())


def test_old_head_cannot_clear_newer_circuit_or_warm_a_fallback():
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        methods = []

        async def handler(request):
            methods.append(request.method)
            entered.set()
            await release.wait()
            return httpx.Response(404)

        router, vertex = router_for(handler)
        pending = asyncio.create_task(router.prepare_connection())
        await entered.wait()
        await router._mark_unavailable("vertex", "explicit rejection", cooldown_seconds=30)
        deadline = router._unavailable_until["vertex"]
        release.set()
        assert await pending == (False, 0)
        assert await router.prepare_connection() == (False, 0)
        assert router._unavailable_until["vertex"] == deadline
        assert "vertex" not in router._healthy_until
        reversed_router = ResilientFastSceneProvider(tuple(reversed(router.routes)))
        with pytest.raises(ValueError):
            await reversed_router.prepare_connection()
        assert methods == ["HEAD"]
        await vertex.aclose()

    asyncio.run(exercise())


def test_close_joins_owned_head_and_blocks_new_connections():
    async def exercise():
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def handler(_request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        _, vertex = router_for(handler)
        waiter = asyncio.create_task(vertex.probe())
        await entered.wait()
        await vertex.aclose()
        assert stopped.is_set() and vertex._probe_task.done() and vertex._client is None
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert (await vertex.probe())[0] is False

    asyncio.run(exercise())


def test_invalidation_before_readiness_response_is_not_reported_ready():
    async def exercise():
        router, _ = router_for(lambda _request: httpx.Response(404))
        mark = router._mark_healthy

        async def invalidate_after_mark(*args, **kwargs):
            result = await mark(*args, **kwargs)
            await router._invalidate_readiness("vertex")
            return result

        router._mark_healthy = invalidate_after_mark
        assert await router.prepare_connection() == (False, 0)
        await router.aclose()
        assert await router.prepare_connection() == (False, 0)

    asyncio.run(exercise())


def test_ambiguous_generation_invalidates_readiness_without_retry(tmp_path):
    methods = []

    def handler(request):
        methods.append(request.method)
        if request.method == "POST":
            raise httpx.ConnectError("fixture connection failure")
        return httpx.Response(404)

    async def exercise():
        router, vertex = router_for(handler)
        assert (await router.prepare_connection())[0]
        with pytest.raises(VertexSceneAmbiguousError):
            await router.generate_fast(
                FastSceneRequest(scene_id="fixture-error", prompt="A red balloon."),
                output_dir=tmp_path / "failed",
            )
        assert methods == ["HEAD", "POST"]
        assert "vertex" not in router._healthy_until
        assert vertex._reserved_usd == vertex.estimated_image_usd
        await router.aclose()

    asyncio.run(exercise())


def test_api_is_empty_local_only_and_does_not_accept_generic_warmup(monkeypatch):
    from storylight.api import app

    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(404)

    async def exercise():
        router, vertex = router_for(handler)
        registry = SimpleNamespace(provider=SimpleNamespace(provider=router))
        monkeypatch.setattr(app.state, "live_scenes", registry, raising=False)
        endpoint = "/v1/live-scene-provider/preconnect"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1234)),
            base_url="http://localhost",
        ) as client:
            response = await client.post(endpoint, json={})
            assert response.status_code == 200
            assert set(response.json()) == {
                "supported", "ready", "expires_in_seconds", "inference_started",
            }
            assert response.json()["inference_started"] is False
            assert (await client.post(endpoint, json={"text": "private input"})).status_code == 422
            registry.provider.provider = vertex
            assert (await client.post(endpoint, json={})).status_code == 409
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("203.0.113.1", 1234)),
            base_url="http://localhost",
        ) as client:
            assert (await client.post(endpoint, json={})).status_code == 403
        assert methods == ["HEAD"]
        await vertex.aclose()

    asyncio.run(exercise())
