"""Offline session lifecycle; the worker owns its expiry watcher and GPU callbacks.

prepare_bucket must finish or join all GPU work before returning or propagating
cancellation. Cancelling an asyncio task does not stop a to_thread CUDA call.
Each successful callback must attest an actual render in its requested bucket.
compile() alone does not establish readiness. Cleanup is not cloud teardown.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import suppress


class SessionUnavailable(RuntimeError):
    pass


def _identity(value):
    if type(value) is not dict:
        raise ValueError("identity must be a dictionary")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SessionState:
    BUCKETS = (128, 256)

    def __init__(self, instance_id, expected_identity, lifetime_seconds,
                 prepare_bucket, cleanup, *, clock=time.monotonic):
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance identity required")
        if (type(lifetime_seconds) not in (int, float)
                or not math.isfinite(lifetime_seconds) or lifetime_seconds <= 0):
            raise ValueError("finite positive lifetime required")
        self.instance_id = instance_id
        self._identity = _identity(expected_identity)
        self._clock = clock
        self.started_at = clock()
        self.expires_at = self.started_at + lifetime_seconds
        if not math.isfinite(self.started_at) or not math.isfinite(self.expires_at):
            raise ValueError("invalid session clock")
        self.state = "NEW"
        self.failure_type = None
        self._prepare_bucket = prepare_bucket
        self._cleanup = cleanup
        self._prepare_task = self._cleanup_task = self._close_task = None

    async def _cleanup_once(self):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        try:
            await asyncio.shield(self._cleanup_task)
        except Exception as error:
            self.failure_type = type(error).__name__
            raise SessionUnavailable("session cleanup failed") from None

    def _check_live(self):
        if self._clock() >= self.expires_at:
            self.state = "EXPIRED"
        if self.state != "PREPARING":
            raise SessionUnavailable("session preparation ended")

    async def _prepare(self):
        try:
            for bucket in self.BUCKETS:
                self._check_live()
                receipt = await self._prepare_bucket(bucket)
                self._check_live()
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                if (type(receipt) is not dict
                        or set(receipt) != {"instance_id", "identity", "sequence_bucket"}
                        or receipt["instance_id"] != self.instance_id
                        or type(receipt["sequence_bucket"]) is not int
                        or receipt["sequence_bucket"] != bucket
                        or _identity(receipt["identity"]) != self._identity):
                    raise ValueError("preparation identity mismatch")
            self.state = "READY"
        except BaseException as error:
            if self.state == "PREPARING":
                self.state = "FAILED"
            self.failure_type = type(error).__name__
            await self._cleanup_once()
            if isinstance(error, asyncio.CancelledError):
                raise
            if not isinstance(error, Exception):
                raise
            raise SessionUnavailable("session preparation failed") from None

    async def prepare(self):
        await self.expire()
        if self.state == "NEW":
            self.state = "PREPARING"
            self._prepare_task = asyncio.create_task(self._prepare())
        if self.state == "READY":
            return
        if self.state != "PREPARING":
            raise SessionUnavailable("session is unavailable")
        try:
            await asyncio.shield(self._prepare_task)
        except asyncio.CancelledError:
            # One owner cancels/joins the task, even if several waiters disconnect.
            if self.state == "PREPARING":
                self.failure_type = "CancelledError"
            await self._terminate("FAILED")
            raise

    async def require_ready(self, instance_id, identity, bucket):
        await self.expire()
        if (self.state != "READY" or instance_id != self.instance_id
                or _identity(identity) != self._identity
                or type(bucket) is not int or bucket not in self.BUCKETS):
            raise SessionUnavailable("session is not ready for this request")

    async def _close(self):
        task = self._prepare_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, SessionUnavailable):
                await task
        await self._cleanup_once()

    async def _terminate(self, state):
        if self._close_task is None:
            self.state = state
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def close(self):
        await self._terminate("CLOSED")

    async def expire(self):
        """Called by the worker's deadline watcher as well as admission methods."""
        if self._clock() >= self.expires_at:
            await self._terminate("EXPIRED")
            return True
        return False
