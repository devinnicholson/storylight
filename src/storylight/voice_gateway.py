"""Loopback voice demo: local workbench/ASR, existing Jetson scene jobs and assets."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from storylight.controller_gateway import _route_allowed

_STATIC = Path(__file__).resolve().parent / "static"
_AUDIO_LIMIT = 20 * 1024 * 1024
_REQUEST_LIMIT = 64 * 1024
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_PRIVATE_HEADERS = {
    "host",
    "cookie",
    "authorization",
    "content-length",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "origin",
    "referer",
}


def _loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return host is not None and ip_address(host).is_loopback
    except ValueError:
        return False


def _backend(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or not _loopback(parsed.hostname)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or parsed.port is None
    ):
        raise ValueError("voice gateway upstream must be an explicit HTTP loopback port")
    return value.rstrip("/")


def create_voice_gateway(
    *,
    scene_url: str = "http://127.0.0.1:18768",
    asr_url: str = "http://127.0.0.1:18766",
    port: int = 18767,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    scene_url, asr_url = _backend(scene_url), _backend(asr_url)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("invalid voice gateway port")
    authorities = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=5, read=600, write=30, pool=5),
        ) as client:
            app.state.upstream = client
            yield

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "")
        peer = request.client.host if request.client else None
        # Check the actual socket peer; forwarded headers never establish local trust.
        origin = request.headers.get("origin")
        if (
            not _loopback(peer)
            or host not in authorities
            or (origin is not None and origin != f"http://{host}")
            or request.headers.get("sec-fetch-site") == "cross-site"
        ):
            return JSONResponse({"detail": "Voice gateway is loopback-only"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/workbench-assets", StaticFiles(directory=_STATIC), name="voice-assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/workbench?voice=1", status_code=303)

    @app.api_route("/workbench", methods=["GET", "HEAD"])
    async def workbench(request: Request):
        if request.query_params.get("demo") == "1":
            return RedirectResponse(asr_url + "/workbench?demo=1", status_code=303)
        return FileResponse(_STATIC / "workbench.html")

    @app.api_route("/projector", methods=["GET", "HEAD"])
    async def projector():
        return FileResponse(_STATIC / "projector.html")

    @app.api_route(
        "/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )
    async def proxy(path: str, request: Request):
        route = "/" + path
        local = (request.method, route) in {
            ("POST", "/v1/audio:transcribe"),
            ("GET", "/v1/runtime:status"),
        }
        canonical = not any(part in {".", "..", ""} for part in path.split("/"))
        if (
            not canonical
            or "\\" in path
            or "%" in path
            or not (
                local
                or _route_allowed(request.method, route)
                or (request.method == "POST" and route == "/v1/live-scene-provider/preconnect")
                or (request.method == "POST" and route == "/v1/projector-telemetry")
                or (request.method, route) in {
                    ("GET", "/v1/demo-cues"), ("POST", "/v1/demo-cues/activate"),
                }
            )
        ):
            raise HTTPException(status_code=403, detail="Voice gateway route is not exposed")
        if route in {"/v1/live-scene-provider/prewarm", "/v1/prepared-projections:prewarm"}:
            raise HTTPException(
                status_code=403, detail="GPU prewarm is not part of this voice demo"
            )
        limit = _AUDIO_LIMIT if route == "/v1/audio:transcribe" else _REQUEST_LIMIT
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid Content-Length") from error
            if length < 0 or length > limit:
                raise HTTPException(status_code=413, detail="Voice gateway request exceeds limit")
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > limit:
                        raise HTTPException(
                            status_code=413, detail="Voice gateway request exceeds limit"
                        )
                    body.extend(chunk)
        except TimeoutError as error:
            raise HTTPException(status_code=408, detail="Voice gateway upload timed out") from error
        query = "?" + request.url.query if request.url.query else ""
        connection_headers = {
            name.strip().lower() for name in request.headers.get("connection", "").split(",")
        }
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_HEADERS | _PRIVATE_HEADERS | connection_headers
        }
        client = request.app.state.upstream
        upstream_request = client.build_request(
            request.method,
            (asr_url if local else scene_url) + route + query,
            headers=headers,
            content=bytes(body),
        )
        # HTTPX's cookie jar also sees upstream Set-Cookie; never relay that state across APIs.
        upstream_request.headers.pop("cookie", None)
        deadline = asyncio.get_running_loop().time() + 600
        try:
            async with asyncio.timeout_at(deadline):
                upstream = await client.send(upstream_request, stream=True)
        except (httpx.HTTPError, TimeoutError) as error:
            raise HTTPException(
                status_code=502, detail="Local demo upstream is unavailable"
            ) from error

        async def close():
            with anyio.move_on_after(5, shield=True):
                await upstream.aclose()

        if 300 <= upstream.status_code < 400 and upstream.status_code != 304:
            await close()
            raise HTTPException(status_code=502, detail="Local demo upstream redirected")

        async def stream():
            try:
                async with asyncio.timeout_at(deadline):
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            finally:
                await close()

        connection_headers = {
            name.strip().lower() for name in upstream.headers.get("connection", "").split(",")
        }
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_HEADERS | {"set-cookie"} | connection_headers
        }
        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(close),
        )

    return app


def create_app_from_env() -> FastAPI:
    """Create the loopback voice gateway for uvicorn's --factory option."""
    return create_voice_gateway()
