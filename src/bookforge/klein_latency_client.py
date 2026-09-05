"""Bounded, non-retrying transports for the opt-in Klein latency experiment."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import time
from urllib.parse import urlsplit

import httpx

from bookforge.finite_modal_provider import _jpeg_dimensions

TIMINGS = (
    "lock_seconds",
    "lookup_seconds",
    "readiness_seconds",
    "submission_seconds",
    "result_download_seconds",
    "validation_seconds",
    "storage_seconds",
    "total_artifact_ready_seconds",
)
METRIC_TIMES = (
    "tokenization_seconds",
    "pipeline_seconds",
    "image_seconds",
    "depth_seconds",
    "encoding_seconds",
    "total_seconds",
    "peak_allocated_gib",
    "peak_reserved_gib",
)


class LatencyError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_payload(payload: dict, request: dict, expected: dict, bucket: int | None) -> None:
    from deploy.klein_latency_protocol import MAX_RESPONSE_BYTES

    prewarm = request["operation"] == "prewarm"
    fields = {
        "request_id",
        "identity",
        "deployment_sha256",
        "location",
        "model_load_seconds",
        "startup_seconds",
        "cache_setup_seconds",
        "instrumentation_sha256",
        "server_seconds",
        "master",
        "depth",
    }
    fields |= {"renders", "warmup_seconds"} if prewarm else {"metrics", "warm_state"}
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload.get("request_id") != request["request_id"]
        or json.dumps(payload.get("identity"), sort_keys=True)
        != json.dumps(expected["expected_identity"], sort_keys=True)
        or payload.get("deployment_sha256") != expected["deployment_sha256"]
        or payload.get("instrumentation_sha256") != expected["instrumentation_sha256"]
    ):
        raise LatencyError("response_identity_mismatch")
    location = payload.get("location")
    if (
        not isinstance(location, dict)
        or set(location) != {"compute_region", "cloud", "container_sha256", "routing_region"}
        or location["routing_region"] != "us-east"
        or any(
            location[name] is not None
            and (
                not isinstance(location[name], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", location[name])
            )
            for name in ("compute_region", "cloud")
        )
        or not isinstance(location["container_sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", location["container_sha256"])
        or any(
            not finite(payload.get(name))
            for name in (
                "model_load_seconds",
                "startup_seconds",
                "cache_setup_seconds",
                "server_seconds",
            )
        )
    ):
        raise LatencyError("response_provenance_mismatch")
    reports = payload.get("renders") if prewarm else [payload.get("metrics")]
    if not isinstance(reports, list) or len(reports) != (2 if prewarm else 1):
        raise LatencyError("response_metrics_mismatch")
    for index, metrics in enumerate(reports):
        expected_bucket = (128, 256)[index] if prewarm else bucket
        if (
            not isinstance(metrics, dict)
            or set(metrics)
            != {
                *METRIC_TIMES,
                "seed",
                "token_count",
                "sequence_bucket",
                "bucket_was_warm",
                "instrumentation_sha256",
                "cuda_image_seconds",
                "master_sha256",
                "depth_sha256",
            }
            or metrics.get("seed") != request["seed"]
            or type(metrics.get("seed")) is not int
            or metrics.get("sequence_bucket") != expected_bucket
            or type(metrics.get("token_count")) is not int
            or not 0 < metrics["token_count"] <= 256
            or (128 if metrics["token_count"] <= 128 else 256) != expected_bucket
            or type(metrics.get("bucket_was_warm")) is not bool
            or metrics.get("instrumentation_sha256") != expected["instrumentation_sha256"]
            or any(not finite(metrics.get(name)) for name in METRIC_TIMES)
            or (
                metrics.get("cuda_image_seconds") is not None
                and not finite(metrics["cuda_image_seconds"])
            )
            or any(
                not isinstance(metrics.get(name), str)
                or not re.fullmatch(r"[a-f0-9]{64}", metrics[name])
                for name in ("master_sha256", "depth_sha256")
            )
        ):
            raise LatencyError("response_metrics_mismatch")
    if prewarm:
        if payload.get("instrumentation_sha256") != expected[
            "instrumentation_sha256"
        ] or not finite(payload.get("warmup_seconds")):
            raise LatencyError("warmup_metrics_mismatch")
    elif payload.get("warm_state") not in {"cold", "warm"}:
        raise LatencyError("response_warmth_mismatch")
    for role in ("master", "depth"):
        content = payload.get(role)
        if not isinstance(content, bytes) or len(content) > MAX_RESPONSE_BYTES:
            raise LatencyError("artifact_size_mismatch")
        if prewarm:
            if content:
                raise LatencyError("warmup_artifacts_unexpected")
        elif hashlib.sha256(content).hexdigest() != reports[0][
            f"{role}_sha256"
        ] or _jpeg_dimensions(content) != (1024, 576):
            raise LatencyError("artifact_verification_failed")


class LatencyClient:
    def __init__(self, expected: dict, *, modal_module=None, http_client=None, secrets=None):
        self.expected = expected
        self.modal = modal_module
        self.http = http_client
        self.owns_http = http_client is None
        self.secrets = secrets
        self.instance = None
        self.url = None
        self.lock = asyncio.Lock()
        self.pending_cleanup: set[asyncio.Task] = set()
        self.last_failure = None
        self.last_timings = None
        self.http_ready = False

    def headers(self) -> dict:
        values = os.environ if self.secrets is None else self.secrets
        key, secret = (
            values.get(f"BOOKFORGE_LATENCY_MODAL_{name}", "") for name in ("KEY", "SECRET")
        )
        if any(
            not isinstance(value, str)
            or not 1 <= len(value) <= 4096
            or any(not 33 <= ord(character) <= 126 for character in value)
            for value in (key, secret)
        ):
            raise LatencyError("proxy_credentials_missing")
        return {"Modal-Key": key, "Modal-Secret": secret}

    async def lookup(self, transport: str) -> None:
        if self.modal is None:
            self.modal = importlib.import_module("modal")
        if transport == "sdk" and self.instance is None:
            cls = self.modal.Cls.from_name("bookforge-klein-latency-sdk", "LatencyStudio")
            await asyncio.wait_for(cls.hydrate.aio(), 10)
            self.instance = cls()
        elif transport == "http" and self.url is None:
            server = self.modal.Server.from_name("bookforge-klein-latency-http", "LatencyServer")
            url = await asyncio.wait_for(server.get_url.aio(), 10)
            parsed = urlsplit(url or "")
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or not parsed.hostname.endswith((".modal.run", ".us-east.modal.direct"))
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
            ):
                raise LatencyError("server_url_invalid")
            self.url = url.rstrip("/")

    async def readiness(self, client) -> None:
        async with asyncio.timeout(120):
            while True:
                response = await client.get(self.url + "/ready", headers=self.headers())
                if response.status_code == 200:
                    self.http_ready = True
                    return
                if response.status_code != 503:
                    raise LatencyError("http_readiness_failed")
                await asyncio.sleep(0.5)

    async def cancel_call(self, call) -> None:
        await asyncio.wait_for(call.cancel.aio(terminate_containers=True), 10)

    def late_submission(self, task) -> None:
        if task.cancelled():
            self.last_failure = "sdk_submission_cleanup_required"
            return

        async def cleanup():
            try:
                call = task.result()
                await self.cancel_call(call)
            except BaseException:
                self.last_failure = "sdk_submission_cleanup_required"

        cleanup_task = asyncio.create_task(cleanup())
        self.pending_cleanup.add(cleanup_task)
        cleanup_task.add_done_callback(self.pending_cleanup.discard)

    async def sdk(self, request: dict, timings: dict):
        begun = time.perf_counter()
        task = asyncio.create_task(self.instance.invoke.spawn.aio(request=request))
        try:
            call = await asyncio.wait_for(asyncio.shield(task), 30)
        except BaseException:
            self.last_failure = "sdk_submission_unknown"
            task.add_done_callback(self.late_submission)
            self.pending_cleanup.add(task)
            task.add_done_callback(self.pending_cleanup.discard)
            raise
        finally:
            timings["submission_seconds"] = time.perf_counter() - begun
        begun = time.perf_counter()
        try:
            return await asyncio.wait_for(call.get.aio(), 310)
        except BaseException:
            self.last_failure = "sdk_result_failed"
            try:
                await asyncio.shield(self.cancel_call(call))
            except BaseException:
                self.last_failure = "sdk_result_cleanup_required"
            raise
        finally:
            timings["result_download_seconds"] = time.perf_counter() - begun

    async def http_invoke(self, request: dict, timings: dict):
        from deploy.klein_latency_protocol import MAX_RESPONSE_BYTES

        if self.http is None:
            self.http = httpx.AsyncClient(timeout=180, trust_env=False, follow_redirects=False)
        client = self.http
        if request["operation"] == "prewarm" and not self.http_ready:
            begun = time.perf_counter()
            try:
                await self.readiness(client)
            except BaseException:
                self.last_failure = "http_readiness_cleanup_required"
                raise
            finally:
                timings["readiness_seconds"] = time.perf_counter() - begun
        if not self.http_ready:
            raise LatencyError("http_prewarm_required")
        begun = time.perf_counter()
        submitted = False
        try:
            async with asyncio.timeout(180):
                async with client.stream(
                    "POST", self.url + "/invoke", json=request, headers=self.headers()
                ) as response:
                    submitted = True
                    timings["submission_seconds"] = time.perf_counter() - begun
                    if response.status_code != 200:
                        raise LatencyError("http_operation_failed")
                    download = time.perf_counter()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_RESPONSE_BYTES:
                            raise LatencyError("http_response_too_large")
                    timings["result_download_seconds"] = time.perf_counter() - download
                    return bytes(content)
        except BaseException:
            self.last_failure = "http_remote_completion_unknown"
            if not submitted:
                timings["submission_seconds"] = time.perf_counter() - begun
            raise

    async def invoke(self, transport: str, request: dict, expected_bucket: int | None):
        from deploy.klein_latency_protocol import unpack_response, validate_request

        validate_request(request)
        if transport not in {"sdk", "http"}:
            raise LatencyError("transport_invalid")
        begun = time.perf_counter()
        timings = dict.fromkeys(TIMINGS, 0.0)
        self.last_timings = timings
        self.last_failure = None
        async with self.lock:
            timings["lock_seconds"] = time.perf_counter() - begun
            stage = time.perf_counter()
            try:
                await self.lookup(transport)
                timings["lookup_seconds"] = time.perf_counter() - stage
                raw = await (
                    self.sdk(request, timings)
                    if transport == "sdk"
                    else self.http_invoke(request, timings)
                )
                stage = time.perf_counter()
                payload = unpack_response(raw)
                validate_payload(payload, request, self.expected, expected_bucket)
                timings["validation_seconds"] = time.perf_counter() - stage
                return payload, timings
            except asyncio.CancelledError:
                if self.last_failure is None:
                    self.last_failure = "operation_cancelled"
                raise
            except Exception:
                if self.last_failure is None:
                    self.last_failure = "operation_failed"
                raise LatencyError(self.last_failure) from None
            finally:
                timings["total_artifact_ready_seconds"] = time.perf_counter() - begun

    async def cleanup(self) -> bool:
        deadline = time.monotonic() + 10
        while self.pending_cleanup and time.monotonic() < deadline:
            await asyncio.wait(self.pending_cleanup, timeout=max(0, deadline - time.monotonic()))
            await asyncio.sleep(0)
        if self.pending_cleanup:
            self.last_failure = "sdk_submission_cleanup_required"
            remaining = list(self.pending_cleanup)
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
        if self.owns_http and self.http is not None:
            await self.http.aclose()
        return not self.pending_cleanup and self.last_failure not in {
            "sdk_submission_unknown",
            "sdk_submission_cleanup_required",
            "sdk_result_cleanup_required",
            "http_remote_completion_unknown",
            "http_readiness_cleanup_required",
            "operation_cancelled",
        }
