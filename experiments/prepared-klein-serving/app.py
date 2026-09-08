"""Prospective IAM-private serving wrapper; deployment and deletion remain external."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import math
import re
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "session_worker": "4301d7525a67c8009b13d6670ba46f8fc8706ee26f51f0019e4a6cbb39a14c96",
    "runtime_worker": "b2e6551ac73d696b7b46d877c4423af9172b110674feaf9cfb943afbd3a47d5c",
    "session_state": "7ed0c641d4e63c514e88ac3502dee2cdff1da6bfbc61fc606800cac10264a7e6",
    "noise": "88f993d8bb6e4ae16da8c8918d7f8c2cd5e70b6378d6daefd76def18603381b4",
}


def verify_sources(directory=ROOT):
    for name, digest in SOURCES.items():
        path = directory / (name + ".py")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 65536
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("serving source changed")


def load(name):
    path = ROOT / (name + ".py")
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError("serving module collision")
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_sources()
runtime_worker = load("runtime_worker")
load("session_state")
session_worker = load("session_worker")
# The unchanged packed loader imports this constant from the serving app module.
PACK_PROOF_SHA256 = runtime_worker.PACK_PROOF_SHA256
SessionUnavailable = session_worker.SessionUnavailable
require = runtime_worker.require

# These four patterns match the edge privacy policy; original-source checks remain there.
PROMPT_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)"),
    re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
    re.compile(
        r"\b(?:account|credential|password|passcode|secret|social security|ssn)\b", re.IGNORECASE
    ),
)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value))


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
    return value


class ServingWorker:
    def __init__(self, session):
        self.session = session
        self.session_id = None
        self.preparation = None
        self.claims = {}

    async def lease(self):
        status = await self.session.status()
        if status["state"] != "READY" or self.session.worker.failed or self.session.closed:
            raise SessionUnavailable("session unavailable")
        return {
            "schema_version": 1,
            "state": "READY",
            "session_id": self.session_id,
            **{
                key: status[key]
                for key in (
                    "instance_id",
                    "service",
                    "revision",
                    "identity",
                    "started_at",
                    "expires_at",
                )
            },
            "warmups": [
                {
                    key: result["metrics"][key]
                    for key in ("sequence_bucket", "seed", "master_sha256", "depth_sha256")
                }
                for result in self.session.warmups
            ],
        }

    async def prewarm(self, payload):
        require(type(payload) is dict and set(payload) == {"session_id"})
        identifier(payload["session_id"])
        verify_sources()
        if self.session_id not in (None, payload["session_id"]):
            raise SessionUnavailable("session already claimed")
        self.session_id = payload["session_id"]
        prepared = await self.session.prepare()
        if self.preparation is None:
            self.preparation = {
                "lease": await self.lease(),
                "model_load_seconds": finite(prepared["warmups"][0]["load_seconds"]),
                "warmup_seconds": sum(
                    finite(row["metrics"]["total_seconds"]) for row in prepared["warmups"]
                ),
            }
        return copy.deepcopy(self.preparation)

    async def generate(self, payload):
        require(
            type(payload) is dict
            and set(payload)
            == {"session_id", "instance_id", "request_id", "scene_id", "prompt", "seed"}
        )
        for key in ("session_id", "instance_id", "request_id"):
            identifier(payload[key])
        require(
            isinstance(payload["scene_id"], str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", payload["scene_id"])
        )
        prompt = payload["prompt"]
        require(
            isinstance(prompt, str)
            and 1 <= len(prompt.strip()) <= 4000
            and len(prompt) <= 4000
            and not any(ord(char) < 32 and char not in "\n\t" for char in prompt)
        )
        require(not any(pattern.search(prompt) for pattern in PROMPT_PATTERNS))
        require(type(payload["seed"]) is int and 0 <= payload["seed"] < 2**32)
        verify_sources()
        self.session.validate()
        if payload["session_id"] != self.session_id or self.preparation is None:
            raise SessionUnavailable("session not prepared")
        session, worker = self.session, self.session.worker
        await session.state.require_ready(
            payload["instance_id"], session.manifest["expected_identity"], 128
        )
        if worker.failed or session.closed:
            raise SessionUnavailable("session unavailable")
        fingerprint = hashlib.sha256(runtime_worker.encoded(payload)).hexdigest()
        claim = self.claims.get(payload["request_id"])
        if claim is not None:
            if claim[0] != fingerprint or claim[1] is None:
                raise SessionUnavailable("request already claimed")
            return copy.deepcopy(claim[1])
        if len(self.claims) >= 10 or worker.requests >= 12 or worker.lock.locked():
            raise SessionUnavailable("session capacity unavailable")
        rendered = False
        try:
            async with worker.lock:
                if session.state.state != "READY" or session.clock() >= session.state.expires_at:
                    raise SessionUnavailable("session expired")
                bucket = session.bucket_for(worker.runtime, prompt)
                require(type(bucket) is int and bucket in (128, 256))
                self.claims[payload["request_id"]] = (fingerprint, None)
                rendered = True
                result = await session.render(
                    {"case_id": payload["request_id"], "prompt": prompt, "seed": payload["seed"]}
                )
                require(
                    result["metrics"]["sequence_bucket"] == bucket
                    and result["bucket_was_warm"] is True
                )
            await session.state.require_ready(payload["instance_id"], result["identity"], bucket)
            response = {
                "lease": await self.lease(),
                "request_id": payload["request_id"],
                "scene_id": payload["scene_id"],
                **{key: result[key] for key in ("master_b64", "depth_b64", "metrics")},
            }
            require(len(runtime_worker.encoded(response)) <= 12 * 1024**2)
            self.claims[payload["request_id"]] = (fingerprint, response)
            return copy.deepcopy(response)
        except BaseException:
            if rendered:
                worker.failed = True
                await session.state.close()
            raise


async def owned_request(request, operation):
    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    task = asyncio.create_task(operation())
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            raise SessionUnavailable("request disconnected")
        return await task
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        while not task.done():
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(task)
        with suppress(asyncio.CancelledError, Exception):
            task.result()
        with suppress(asyncio.CancelledError):
            await watcher


def create_app(manifest_path=ROOT / "manifest.json", session_path=ROOT / "session.json", **kwargs):
    serving = ServingWorker(session_worker.SessionWorker(manifest_path, session_path, **kwargs))

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            await serving.session.close()

    application = FastAPI(lifespan=lifespan)
    application.state.serving = serving

    async def respond(request, function):
        try:
            async with asyncio.timeout(180):
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    require(len(body) <= 20 * 1024)
                payload = runtime_worker.decode(body)
                return await owned_request(request, lambda: function(payload))
        except SessionUnavailable:
            return JSONResponse({"error": "session_unavailable"}, status_code=409)
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "request_rejected"}, status_code=422)
        except Exception:
            return JSONResponse({"error": "session_failed"}, status_code=500)

    @application.post("/v1/prewarm")
    async def prewarm(request: Request):
        return await respond(request, serving.prewarm)

    @application.post("/v1/generate")
    async def generate(request: Request):
        return await respond(request, serving.generate)

    return application


app = create_app()
