from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import klein_latency_http as boundary  # noqa: E402
from klein_latency_protocol import digest, unpack_response  # noqa: E402

REQUEST = {"request_id": "a" * 32, "operation": "render", "prompt": "synthetic scene", "seed": 7}


def start_request(app, raw=None, *, path="/invoke", method="POST", disconnect_after_response=False):
    inbox = asyncio.Queue()
    inbox.put_nowait(
        {"type": "http.request", "body": raw if raw is not None else json.dumps(REQUEST).encode()}
    )
    sent = []

    async def send(message):
        sent.append(message)
        if disconnect_after_response and message["type"] == "http.response.body":
            inbox.put_nowait({"type": "http.disconnect"})

    task = asyncio.create_task(
        app({"type": "http", "path": path, "method": method}, inbox.get, send)
    )
    return task, inbox, sent


def status(sent):
    return next(row["status"] for row in sent if row["type"] == "http.response.start")


def test_only_frozen_bounded_requests_run_once_without_leaking_errors():
    async def exercise():
        claims, calls, stops = set(), [], []

        def claim(identity):
            if identity in claims:
                return False
            claims.add(identity)
            return True

        def handler(request):
            calls.append(request)
            return {"metrics": {"count": 1}, "master": b"image", "depth": b"depth"}

        app = boundary.build_app(
            handler,
            claim,
            lambda: stops.append(True),
            {REQUEST["request_id"]: digest(REQUEST)},
            float("inf"),
        )
        task, _, sent = start_request(app, path="/ready", method="GET")
        await task
        assert status(sent) == 200 and not claims and not calls
        for raw, expected in (
            (b"x" * (boundary.MAX_REQUEST_BYTES + 1), 400),
            (b'{"request_id":"first","request_id":"second"}', 400),
            (json.dumps({**REQUEST, "prompt": "private mismatched prompt"}).encode(), 403),
        ):
            task, _, sent = start_request(app, raw)
            await task
            assert status(sent) == expected and sent[-1]["body"] == b'{"ok":false}'
            assert not claims and not calls and not stops
        task, _, sent = start_request(app, disconnect_after_response=True)
        await task
        assert status(sent) == 200 and not stops
        assert unpack_response(sent[-1]["body"])["master"] == b"image"
        task, _, sent = start_request(app)
        await task
        assert status(sent) == 409 and len(calls) == 1 and len(claims) == 1 and not stops
        expired = boundary.build_app(handler, claim, lambda: stops.append(True), {}, 0)
        task, _, sent = start_request(expired)
        await task
        assert status(sent) == 410 and len(calls) == 1 and not stops

    asyncio.run(exercise())


def test_blocking_work_terminates_on_disconnect_cancel_or_deadline_and_never_queues(monkeypatch):
    monkeypatch.setattr(boundary, "REQUEST_TIMEOUT_SECONDS", 0.1)

    async def scenario(failure):
        entered, release = asyncio.Event(), threading.Event()
        calls, stops, claims = [], [], set()
        loop = asyncio.get_running_loop()

        def handler(request):
            calls.append(request)
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            return {"master": b"done"}

        def terminate():
            stops.append(True)
            release.set()

        def claim(identity):
            claims.add(identity)
            return True

        app = boundary.build_app(
            handler, claim, terminate, {REQUEST["request_id"]: digest(REQUEST)}, float("inf")
        )
        task, inbox, sent = start_request(app)
        await asyncio.wait_for(entered.wait(), 1)
        parallel, _, rejected = start_request(app)
        await parallel
        assert status(rejected) == 429 and len(calls) == 1
        if failure == "disconnect":
            inbox.put_nowait({"type": "http.disconnect"})
        elif failure == "cancel":
            task.cancel()
        try:
            if failure == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await asyncio.wait_for(task, 1)
        finally:
            release.set()
        assert stops == [True] and claims == {REQUEST["request_id"]} and len(calls) == 1
        if failure == "deadline":
            assert status(sent) == 504
        else:
            assert not sent
        retry, _, rejected = start_request(app)
        await retry
        assert status(rejected) == 503 and len(calls) == 1

    async def exercise():
        for failure in ("disconnect", "cancel", "deadline"):
            await scenario(failure)

    asyncio.run(exercise())


def test_claim_failures_or_timeouts_cannot_later_dispatch_a_gpu_thread(monkeypatch):
    async def exercise():
        calls, stops = [], []

        def failed_claim(identity):
            raise RuntimeError("private claim transport details")

        app = boundary.build_app(
            calls.append,
            failed_claim,
            lambda: stops.append(True),
            {REQUEST["request_id"]: digest(REQUEST)},
            float("inf"),
        )
        task, _, sent = start_request(app)
        await task
        assert status(sent) == 503 and sent[-1]["body"] == b'{"ok":false}'
        assert not calls and not stops
        monkeypatch.setattr(boundary, "REQUEST_TIMEOUT_SECONDS", 0.05)
        release, finished = threading.Event(), threading.Event()

        def slow_claim(identity):
            assert release.wait(2)
            finished.set()
            return True

        app = boundary.build_app(
            calls.append,
            slow_claim,
            lambda: stops.append(True),
            {REQUEST["request_id"]: digest(REQUEST)},
            float("inf"),
        )
        task, _, sent = start_request(app)
        try:
            await asyncio.wait_for(task, 1)
            assert status(sent) == 504 and not calls and not stops
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        assert not calls and not stops

    asyncio.run(exercise())


def test_oversize_completed_response_is_never_streamed_and_closes_the_worker():
    async def exercise():
        stops, claims = [], []
        app = boundary.build_app(
            lambda request: {"master": b"x" * boundary.MAX_RESPONSE_BYTES},
            lambda identity: claims.append(identity) is None,
            lambda: stops.append(True),
            {REQUEST["request_id"]: digest(REQUEST)},
            float("inf"),
        )
        task, _, sent = start_request(app)
        await task
        assert status(sent) == 500 and sent[-1]["body"] == b'{"ok":false}'
        assert stops == [True] and claims == [REQUEST["request_id"]]

    asyncio.run(exercise())
