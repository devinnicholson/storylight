"""Finite prepared synthetic session; cloud deletion remains supervisor-owned."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sys
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import runtime_worker
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from session_state import SessionState, SessionUnavailable

ROOT = Path(__file__).resolve().parent
# The pinned FlashPack helper imports this public worker-module constant.
PACK_PROOF_SHA256 = runtime_worker.PACK_PROOF_SHA256


def log_render_failure(worker, error):
    def safe_name(value):
        return value if re.fullmatch(r"[A-Za-z0-9_.<>-]{1,96}", value) else "unknown"

    frames = []
    frame = error.__traceback__
    for _ in range(128):
        if frame is None:
            break
        code = frame.tb_frame.f_code
        frames.append(
            {
                "file": safe_name(Path(code.co_filename).name),
                "function": safe_name(code.co_name),
                "line": frame.tb_lineno,
            }
        )
        frame = frame.tb_next
    stage = getattr(worker, "stage", "unknown")
    if stage not in {"idle", "load", "cache_restore", "compile", "render", "verify", "complete"}:
        stage = "unknown"
    print(
        runtime_worker.encoded(
            {
                "event": "session.render_failed",
                "stage": stage,
                "error_type": safe_name(type(error).__name__),
                "instance_id": worker.instance_id,
                "request_ordinal": worker.requests,
                "frames": frames[-8:],
            }
        ).decode(),
        file=sys.stderr,
        flush=True,
    )


def read_session(path):
    raw = runtime_worker.bounded_file(path, 4096)
    value = runtime_worker.decode(raw)
    require = runtime_worker.require
    require(
        set(value) == {"schema_version", "lifetime_seconds", "max_measured_requests", "sources"}
    )
    for key, expected in (
        ("schema_version", 1),
        ("lifetime_seconds", 300),
        ("max_measured_requests", 10),
    ):
        require(type(value[key]) is int and value[key] == expected)
    require(set(value["sources"]) == {"runtime_worker", "session_state"})
    for name, digest in value["sources"].items():
        require(hashlib.sha256((ROOT / f"{name}.py").read_bytes()).hexdigest() == digest)
    require(raw == runtime_worker.encoded(value))
    return value, hashlib.sha256(raw).hexdigest()


def prompt_bucket(runtime, prompt):
    from klein_scene_runtime import sequence_bucket

    tokenizer = runtime.pipe.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return sequence_bucket(len(tokenizer(text)["input_ids"]))


async def joined_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(task)
        with suppress(asyncio.CancelledError, Exception):
            task.result()
        raise


class SessionWorker:
    def __init__(
        self,
        manifest_path,
        session_path,
        *,
        factory=runtime_worker.load_runtime,
        worker_type=runtime_worker.Worker,
        manifest_reader=runtime_worker.read_manifest,
        session_reader=read_session,
        bucket_for=prompt_bucket,
        clock=time.monotonic,
        wall_clock=time.time,
    ):
        self.worker = worker_type(manifest_path, factory)
        self.manifest_path, self.session_path = manifest_path, session_path
        self.manifest_reader, self.session_reader = manifest_reader, session_reader
        self.bucket_for, self.clock, self.wall_clock = bucket_for, clock, wall_clock
        self.manifest = self.config = self.state = self.watcher = None
        self.manifest_sha = self.session_sha = None
        self.started_wall = None
        self.warmups = []
        self.closed = False
        self.cold_cleanup_task = None

    def validate(self):
        manifest, digest = self.manifest_reader(self.manifest_path)
        config, config_digest = self.session_reader(self.session_path)
        runtime_worker.require(manifest["experiment_id"] == os.environ.get("K_SERVICE"))
        runtime_worker.require(self.manifest_sha in (None, digest))
        runtime_worker.require(self.session_sha in (None, config_digest))
        self.manifest, self.config = manifest, config
        self.manifest_sha, self.session_sha = digest, config_digest
        self.worker.manifest_sha256 = digest

    async def cleanup(self):
        async with self.worker.lock:
            self.worker.failed = True
            self.worker.runtime = None

    async def watch_expiry(self):
        await asyncio.sleep(max(0, self.state.expires_at - self.clock()))
        await self.state.expire()

    async def render(self, case):
        self.worker.requests += 1
        try:
            result = await joined_thread(self.worker.render, self.manifest, case)
            runtime_worker.require(result["instance_id"] == self.worker.instance_id)
            runtime_worker.require(result["identity"] == self.manifest["expected_identity"])
            runtime_worker.require(result["request_ordinal"] == self.worker.requests)
            runtime_worker.require(result["case_id"] == case["case_id"])
            return result | {"manifest_sha256": self.manifest_sha}
        except BaseException as error:
            self.worker.failed = True
            log_render_failure(self.worker, error)
            raise

    async def prepare_bucket(self, bucket):
        async with self.worker.lock:
            result = await self.render(self.manifest["cases"][len(self.warmups)])
            runtime_worker.require(result["metrics"]["sequence_bucket"] == bucket)
            self.warmups.append(result)
            return {
                "instance_id": result["instance_id"],
                "identity": result["identity"],
                "sequence_bucket": bucket,
            }

    async def prepare(self):
        if self.closed:
            raise SessionUnavailable("session closed")
        self.validate()
        if self.state is None:
            runtime_worker.require(self.manifest["expires_at"] > self.wall_clock() + 300)
            self.started_wall = self.wall_clock()
            self.state = SessionState(
                self.worker.instance_id,
                self.manifest["expected_identity"],
                self.config["lifetime_seconds"],
                self.prepare_bucket,
                self.cleanup,
                clock=self.clock,
            )
            self.watcher = asyncio.create_task(self.watch_expiry())
        await self.state.prepare()
        return {"status": await self.status(), "warmups": self.warmups}

    async def status(self):
        if self.manifest is None:
            self.validate()
        if self.state is not None:
            await self.state.expire()
        state = self.state.state if self.state else ("CLOSED" if self.closed else "NEW")
        return {
            "schema_version": 1,
            "state": state,
            **self.worker.identity(),
            "identity": self.manifest["expected_identity"],
            "manifest_sha256": self.manifest_sha,
            "session_sha256": self.session_sha,
            "buckets": [128, 256] if state == "READY" else [],
            "started_at": self.started_wall,
            "expires_at": self.started_wall + 300 if self.started_wall is not None else None,
            "manifest_expires_at": self.manifest["expires_at"],
            "requests": self.worker.requests,
            "measured_requests": max(0, self.worker.requests - 2),
        }

    async def generate(self, payload):
        runtime_worker.require(
            type(payload) is dict and set(payload) == {"case_id", "seed", "instance_id", "ordinal"}
        )
        runtime_worker.require(type(payload["ordinal"]) is int and type(payload["seed"]) is int)
        self.validate()
        if self.state is None or self.closed or self.worker.failed:
            raise SessionUnavailable("session unavailable")
        await self.state.require_ready(
            payload["instance_id"], self.manifest["expected_identity"], 128
        )
        if self.worker.lock.locked():
            raise SessionUnavailable("session busy")
        rendered = False
        try:
            async with self.worker.lock:
                if self.state.state != "READY" or self.clock() >= self.state.expires_at:
                    raise SessionUnavailable("session expired")
                ordinal = self.worker.requests + 1
                runtime_worker.require(3 <= ordinal <= 12 and payload["ordinal"] == ordinal)
                case = (self.manifest["cases"] + self.manifest["cases"][:2])[ordinal - 3]
                runtime_worker.require(
                    (payload["case_id"], payload["seed"]) == (case["case_id"], case["seed"])
                )
                bucket = self.bucket_for(self.worker.runtime, case["prompt"])
                runtime_worker.require(type(bucket) is int and bucket in (128, 256))
                rendered = True
                result = await self.render(case)
                runtime_worker.require(result["metrics"]["sequence_bucket"] == bucket)
                runtime_worker.require(result["bucket_was_warm"] is True)
            await self.state.require_ready(payload["instance_id"], result["identity"], bucket)
            return result
        except BaseException:
            if rendered:
                self.worker.failed = True
            if self.worker.failed:
                await self.state.close()
            raise

    async def close(self):
        self.closed = True
        if self.state is not None:
            await self.state.close()
        else:
            if self.cold_cleanup_task is None:
                self.cold_cleanup_task = asyncio.create_task(self.cleanup())
            await asyncio.shield(self.cold_cleanup_task)
        if self.watcher is not None and self.watcher is not asyncio.current_task():
            self.watcher.cancel()
            with suppress(asyncio.CancelledError):
                await self.watcher
        return await self.status()

    async def evidence(self):
        if self.state is None or self.closed or self.worker.failed or self.worker.lock.locked():
            raise SessionUnavailable("session evidence unavailable")
        await self.state.require_ready(
            self.worker.instance_id, self.manifest["expected_identity"], 128
        )
        runtime_worker.require(self.worker.requests == 12)
        runtime_worker.require(len(self.worker.receipts) == len(self.worker.noise_receipts) == 12)
        runtime_worker.require(
            [row["ordinal"] for row in self.worker.receipts] == list(range(1, 13))
            and [row["ordinal"] for row in self.worker.noise_receipts] == list(range(1, 13))
        )
        return {
            "schema_version": 1,
            **self.worker.identity(),
            "identity": self.manifest["expected_identity"],
            "manifest_sha256": self.manifest_sha,
            "session_sha256": self.session_sha,
            "receipts": self.worker.receipts,
            "noise_receipts": self.worker.noise_receipts,
        }


def create_app(manifest_path=ROOT / "manifest.json", session_path=ROOT / "session.json", **kwargs):
    session = SessionWorker(manifest_path, session_path, **kwargs)

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            await session.close()

    application = FastAPI(lifespan=lifespan)
    application.state.session = session

    async def body(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            runtime_worker.require(len(raw) <= 1024)
        return runtime_worker.decode(raw)

    async def respond(function, *args):
        try:
            return await function(*args)
        except SessionUnavailable:
            return JSONResponse({"error": "session_unavailable"}, status_code=409)
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "request_rejected"}, status_code=422)
        except Exception:
            return JSONResponse({"error": "session_failed"}, status_code=500)

    @application.get("/status")
    async def status():
        return await respond(session.status)

    @application.post("/prepare")
    async def prepare(request: Request):
        async def call():
            runtime_worker.require(await body(request) == {})
            return await session.prepare()

        return await respond(call)

    @application.post("/generate")
    async def generate(request: Request):
        async def call():
            return await session.generate(await body(request))

        return await respond(call)

    @application.post("/close")
    async def close(request: Request):
        async def call():
            runtime_worker.require(await body(request) == {})
            return await session.close()

        return await respond(call)

    @application.get("/evidence")
    async def evidence():
        return await respond(session.evidence)

    return application


app = create_app()
