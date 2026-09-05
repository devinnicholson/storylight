"""Bounded ASGI boundary behind Modal proxy authentication for the latency experiment."""

from __future__ import annotations

import asyncio
import time

from klein_latency_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    decode_json,
    digest,
    pack_response,
    validate_request,
)

REQUEST_TIMEOUT_SECONDS = 180


class _Disconnected(Exception):
    pass


async def _respond(send, status, body=b'{"ok":false}', content_type=b"application/json"):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _body(receive, deadline):
    body = bytearray()
    while True:
        message = await asyncio.wait_for(
            receive(), max(0, deadline - asyncio.get_running_loop().time())
        )
        if message["type"] == "http.disconnect":
            raise _Disconnected
        if message["type"] != "http.request" or not isinstance(message.get("body", b""), bytes):
            raise ValueError("invalid ASGI body")
        chunk = message.get("body", b"")
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds bound")
        body.extend(chunk)
        if not message.get("more_body", False):
            return bytes(body)


async def _disconnected(receive):
    while True:
        try:
            message = await receive()
        except Exception:
            return
        if message["type"] == "http.disconnect":
            return
        await asyncio.sleep(0)


async def _until(task, disconnect, deadline, *, prefer_completed=False):
    done, _ = await asyncio.wait(
        (task, disconnect),
        timeout=max(0, deadline - asyncio.get_running_loop().time()),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if prefer_completed and task in done:
        return task.result()
    if disconnect in done:
        raise _Disconnected
    if task not in done:
        raise TimeoutError
    return task.result()


def build_app(handler, claim, terminate, allowed_requests, expires_at):
    """Serve only frozen requests; a claim can outlive cancellation but cannot start a render."""
    allowed = dict(allowed_requests)
    lock = asyncio.Lock()
    stopped = False

    def stop():
        nonlocal stopped
        stopped = True
        terminate()

    def invoke(request):
        result = pack_response(handler(request))
        if len(result) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds bound")
        return result

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT_SECONDS
        if time.time() > expires_at:
            await _respond(send, 410)
            return
        if stopped:
            await _respond(send, 503)
            return
        if scope["path"] == "/ready" and scope["method"] == "GET":
            await _respond(send, 200, b'{"ready":true}')
            return
        if scope["path"] != "/invoke" or scope["method"] != "POST":
            await _respond(send, 404)
            return
        if lock.locked():
            await _respond(send, 429)
            return
        try:
            request = decode_json(await _body(receive, deadline))
            validate_request(request)
            if allowed.get(request["request_id"]) != digest(request):
                await _respond(send, 403)
                return
        except _Disconnected:
            return
        except TimeoutError:
            await _respond(send, 408)
            return
        except (ValueError, TypeError, KeyError, RecursionError):
            await _respond(send, 400)
            return
        if lock.locked():
            await _respond(send, 429)
            return
        async with lock:
            disconnect = asyncio.create_task(_disconnected(receive))
            tasks = [disconnect]
            dispatched = response_started = False
            try:
                if time.time() > expires_at:
                    await _respond(send, 410)
                    return
                pending = asyncio.create_task(asyncio.to_thread(claim, request["request_id"]))
                tasks.append(pending)
                claimed = await _until(pending, disconnect, deadline)
                if claimed is not True:
                    await _respond(send, 409)
                    return
                if time.time() > expires_at:
                    await _respond(send, 410)
                    return
                dispatched = True
                pending = asyncio.create_task(asyncio.to_thread(invoke, request))
                tasks.append(pending)
                payload = await _until(pending, disconnect, deadline)
                response_started = True
                pending = asyncio.create_task(
                    _respond(send, 200, payload, b"application/octet-stream")
                )
                tasks.append(pending)
                await _until(pending, disconnect, deadline, prefer_completed=True)
            except asyncio.CancelledError:
                if dispatched:
                    stop()
                raise
            except _Disconnected:
                if dispatched:
                    stop()
            except TimeoutError:
                if dispatched:
                    stop()
                if not response_started:
                    await _respond(send, 504)
            except Exception:
                if dispatched:
                    stop()
                if not response_started:
                    await _respond(send, 500 if dispatched else 503)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    return app
