"""Paired, allowlisted LAN gateway for the standalone Jetson controller.

The private Bookforge API keeps listening on Jetson loopback. This process is the
only LAN listener: it exchanges a high-entropy pairing token for an ephemeral
HttpOnly session, then proxies only the workbench and its bounded scene-control
routes back to loopback. Camera, microphone, diagnostics, compilation, and other
private APIs are deliberately absent from the allowlist.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from html import escape
from ipaddress import ip_address
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_COOKIE_NAME = "bookforge_controller"
_MAX_REQUEST_BYTES = 64 * 1024
_FORWARDED_HEADERS = {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip"}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ControllerGatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOOKFORGE_CONTROLLER_",
        extra="ignore",
    )

    backend_url: str = "http://127.0.0.1:8080"
    pairing_token: Annotated[str, Field(min_length=32, max_length=256)]
    session_ttl_seconds: Annotated[int, Field(ge=300, le=86_400)] = 43_200
    maximum_sessions: Annotated[int, Field(ge=1, le=32)] = 8
    cookie_secure: bool = False
    public_name: Annotated[str, Field(min_length=1, max_length=80)] = "Bookforge"

    @field_validator("backend_url")
    @classmethod
    def require_loopback_backend(cls, value: str) -> str:
        parsed = urlsplit(value.rstrip("/"))
        if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query:
            raise ValueError("controller backend must be a plain HTTP loopback URL")
        try:
            loopback = parsed.hostname is not None and ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if not loopback or parsed.path not in {"", "/"}:
            raise ValueError("controller backend must resolve explicitly to loopback")
        return value.rstrip("/")


class PairingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: Annotated[str, Field(min_length=1, max_length=256)]


@dataclass(frozen=True)
class _ControllerSession:
    session_id: str
    expires_at: float


def _route_allowed(method: str, path: str) -> bool:
    method = method.upper()
    if method in {"GET", "HEAD"}:
        return (
            path in {"/workbench", "/projector", "/readyz", "/v1/story-packs/latest"}
            or path.startswith("/workbench-assets/")
            or path.startswith("/v1/assets/")
            or path.startswith("/v1/live-scenes/")
            or path.startswith("/v1/live-scene-sessions/")
            or path == "/v1/live-scene-provider/warm-status"
        )
    if method == "POST":
        return path in {
            "/v1/live-scenes",
            "/v1/live-scene-provider/prewarm",
            "/v1/live-scene-planner/prepare",
            "/v1/live-scene-planner/warmup",
        }
    return False


def _pairing_page(settings: ControllerGatewaySettings, *, nonce: str) -> str:
    title = escape(settings.public_name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pair with {title}</title>
  <style nonce="{nonce}">
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ min-height: 100vh; margin: 0; display: grid; place-items: center; background: #081018;
      color: #f5f0df; }}
    main {{ width: min(32rem, calc(100vw - 3rem)); padding: 2rem; border: 1px solid #2c4b56;
      border-radius: 1.25rem; background: #10202a; box-shadow: 0 1.5rem 5rem #0008; }}
    h1 {{ margin-top: 0; }} p {{ color: #bdd0d2; line-height: 1.5; }}
    input, button {{ box-sizing: border-box; width: 100%; border-radius: .75rem; padding: .9rem;
      font: inherit; }}
    input {{ border: 1px solid #496873; background: #07151d; color: white; }}
    button {{ margin-top: .8rem; border: 0; background: #f2b35d; color: #17120c;
      font-weight: 750; }}
    #status {{ min-height: 1.5rem; color: #ffb7a8; }}
  </style>
</head>
<body>
  <main>
    <p>Private edge controller</p>
    <h1>Pair with {title}</h1>
    <p>Enter the pairing token shown by the Jetson operator. It is exchanged locally for a
      temporary controller session and is never forwarded to the story API or cloud renderer.</p>
    <form id="pair-form">
      <label for="token">Pairing token</label>
      <input id="token" name="token" type="password" autocomplete="one-time-code" required>
      <button type="submit">Open controller</button>
    </form>
    <p id="status" role="status"></p>
  </main>
  <script nonce="{nonce}">
    const form = document.querySelector("#pair-form");
    const input = document.querySelector("#token");
    const status = document.querySelector("#status");
    const fragment = decodeURIComponent(location.hash.slice(1));
    history.replaceState(null, "", location.pathname);
    if (fragment) input.value = fragment;
    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      status.textContent = "Pairing…";
      const response = await fetch("/pair-session", {{
        method: "POST",
        headers: {{"content-type": "application/json"}},
        body: JSON.stringify({{token: input.value}}),
      }});
      if (!response.ok) {{
        status.textContent = "Pairing failed. Check the token and try again.";
        return;
      }}
      const result = await response.json();
      location.replace(result.redirect);
    }});
    if (fragment) form.requestSubmit();
  </script>
</body>
</html>"""


def create_controller_gateway(
    settings: ControllerGatewaySettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    sessions: dict[str, _ControllerSession] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.backend = httpx.AsyncClient(
            base_url=settings.backend_url,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(connect=5, read=None, write=10, pool=5),
            transport=transport,
        )
        yield
        await app.state.backend.aclose()
        sessions.clear()

    app = FastAPI(
        title="Bookforge paired controller gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def authenticated(request: Request) -> bool:
        now = time.monotonic()
        expired = [key for key, session in sessions.items() if session.expires_at <= now]
        for key in expired:
            sessions.pop(key, None)
        session_id = request.cookies.get(_COOKIE_NAME, "")
        session = sessions.get(session_id)
        return session is not None and session.expires_at > now

    async def bounded_body(request: Request) -> bytes:
        declared_length = request.headers.get("content-length")
        if declared_length:
            try:
                if int(declared_length) > _MAX_REQUEST_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Controller request exceeds the limit",
                    )
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid Content-Length") from error
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Controller request exceeds the limit",
                )
        return bytes(body)

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        try:
            response = await request.app.state.backend.get("/readyz")
            backend_ready = response.status_code == 200
        except httpx.HTTPError:
            backend_ready = False
        return JSONResponse(
            status_code=200 if backend_ready else 503,
            content={"ready": backend_ready, "paired_sessions": len(sessions)},
        )

    @app.get("/pair", response_class=HTMLResponse)
    async def pair() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        return HTMLResponse(
            _pairing_page(settings, nonce=nonce),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; connect-src 'self'; form-action 'self'; "
                    f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                    "base-uri 'none'; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/pair-session")
    async def pair_session(payload: PairingRequest) -> JSONResponse:
        if not secrets.compare_digest(payload.token, settings.pairing_token):
            await asyncio.sleep(0.2)
            raise HTTPException(status_code=403, detail="Pairing token was rejected")
        session_id = secrets.token_urlsafe(32)
        if len(sessions) >= settings.maximum_sessions:
            oldest = min(sessions.values(), key=lambda session: session.expires_at)
            sessions.pop(oldest.session_id, None)
        sessions[session_id] = _ControllerSession(
            session_id=session_id,
            expires_at=time.monotonic() + settings.session_ttl_seconds,
        )
        response = JSONResponse(
            {"ready": True, "redirect": "/workbench?session=bookforge-live"}
        )
        response.set_cookie(
            _COOKIE_NAME,
            session_id,
            max_age=settings.session_ttl_seconds,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="strict",
            path="/",
        )
        return response

    @app.api_route(
        "/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(path: str, request: Request):
        route = f"/{path}"
        if not authenticated(request):
            if request.method in {"GET", "HEAD"} and not route.startswith("/v1/"):
                return RedirectResponse("/pair", status_code=303)
            raise HTTPException(status_code=401, detail="Pairing is required")
        if not _route_allowed(request.method, route):
            raise HTTPException(status_code=403, detail="Controller route is not exposed")

        body = await bounded_body(request)
        query = f"?{request.url.query}" if request.url.query else ""
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.casefold()
            not in _HOP_BY_HOP_HEADERS
            | _FORWARDED_HEADERS
            | {"host", "cookie", "content-length"}
        }
        upstream_request = request.app.state.backend.build_request(
            request.method,
            f"{route}{query}",
            headers=headers,
            content=body,
        )
        try:
            upstream = await request.app.state.backend.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail="Private Bookforge API is unavailable",
            ) from error

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.casefold() not in _HOP_BY_HOP_HEADERS | {"set-cookie"}
        }
        response_headers.setdefault("X-Content-Type-Options", "nosniff")
        response_headers.setdefault("Referrer-Policy", "no-referrer")
        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory; configuration is loaded only when this service is started."""

    return create_controller_gateway(ControllerGatewaySettings())
