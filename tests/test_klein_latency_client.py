from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookforge.klein_latency_client import (
    METRIC_TIMES,
    TIMINGS,
    LatencyClient,
    LatencyError,
    validate_payload,
)  # noqa: E402
from deploy import klein_latency_protocol as protocol  # noqa: E402
from scripts import benchmark_klein_latency as benchmark  # noqa: E402
from scripts import freeze_klein_latency as freezer  # noqa: E402


def response(manifest, request, bucket):
    image = io.BytesIO()
    Image.new("RGB", (1024, 576), (80, 120, 160)).save(image, format="JPEG")
    content = image.getvalue()

    def metrics(selected, seed):
        return {
            **dict.fromkeys(METRIC_TIMES, 0.1),
            "seed": seed,
            "token_count": selected,
            "sequence_bucket": selected,
            "bucket_was_warm": True,
            "cuda_image_seconds": None,
            "master_sha256": hashlib.sha256(content).hexdigest(),
            "depth_sha256": hashlib.sha256(content).hexdigest(),
            "instrumentation_sha256": manifest["instrumentation_sha256"],
        }

    payload = {
        "request_id": request["request_id"],
        "identity": manifest["expected_identity"],
        "deployment_sha256": manifest["deployment_sha256"],
        "instrumentation_sha256": manifest["instrumentation_sha256"],
        "server_seconds": 1.0,
        "model_load_seconds": 1.0,
        "startup_seconds": 2.0,
        "cache_setup_seconds": 1.0,
        "location": {
            "compute_region": "us-east-1",
            "cloud": "aws",
            "routing_region": "us-east",
            "container_sha256": "e" * 64,
        },
        "master": content,
        "depth": content,
    }
    if request["operation"] == "prewarm":
        payload.update(
            master=b"", depth=b"", warmup_seconds=1.0, renders=[metrics(128, 0), metrics(256, 0)]
        )
    else:
        payload.update(metrics=metrics(bucket, request["seed"]), warm_state="warm")
    return payload


def test_sdk_and_http_share_verified_bytes_and_refuse_mutated_evidence():
    manifest = freezer.freeze("transport-test", int(time.time()) + 1200)
    operation = manifest["operations"][2]
    payload = response(manifest, operation["request"], 128)
    raw = protocol.pack_response(payload)
    posted, readiness = [], []

    async def scenario():
        async def get():
            return raw

        async def spawn(**kwargs):
            posted.append(kwargs["request"])
            return SimpleNamespace(get=SimpleNamespace(aio=get))

        sdk = LatencyClient(manifest)
        sdk.instance = SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))
        sdk.modal = object()
        first, timings = await sdk.invoke("sdk", operation["request"], 128)
        assert set(timings) == set(TIMINGS)

        def serve(request):
            if request.method == "GET":
                readiness.append(True)
                return httpx.Response(503 if len(readiness) == 1 else 200)
            value = json.loads(request.content)
            posted.append(value)
            assert request.headers["Modal-Key"] == "private-key"
            return httpx.Response(
                200,
                content=protocol.pack_response(
                    response(manifest, value, None if value["operation"] == "prewarm" else 128)
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
            client = LatencyClient(
                manifest,
                modal_module=object(),
                http_client=http,
                secrets={
                    "BOOKFORGE_LATENCY_MODAL_KEY": "private-key",
                    "BOOKFORGE_LATENCY_MODAL_SECRET": "private-secret",
                },
            )
            client.url = "https://isolated.modal.run"
            await client.invoke("http", manifest["operations"][1]["request"], None)
            second, _ = await client.invoke("http", operation["request"], 128)
            assert first["master"] == second["master"] and first["depth"] == second["depth"]
        assert len(readiness) == 2 and len(posted) == 3

    asyncio.run(scenario())
    for mutate in (
        lambda value: value.update(request_id="0" * 32),
        lambda value: value.update(master=b"private corrupt image"),
        lambda value: value["metrics"].update(seed=True),
        lambda value: value["metrics"].update(sequence_bucket=256),
        lambda value: value["metrics"].update(instrumentation_sha256="0" * 64),
        lambda value: value["metrics"].update(total_seconds=float("nan")),
        lambda value: value["metrics"].update(unexpected="private untrusted text"),
    ):
        changed = copy.deepcopy(payload)
        mutate(changed)
        with pytest.raises(LatencyError):
            validate_payload(changed, operation["request"], manifest, 128)


def test_cancel_during_submit_cleans_late_handle_and_http_failure_never_retries():
    manifest = freezer.freeze("cancel-test", int(time.time()) + 1200)
    operation = manifest["operations"][2]

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        cancelled, posts = [], []

        async def cancel(**kwargs):
            cancelled.append(kwargs)

        async def submit(**kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(cancel=SimpleNamespace(aio=cancel))

        client = LatencyClient(manifest, modal_module=object())
        client.instance = SimpleNamespace(invoke=SimpleNamespace(spawn=SimpleNamespace(aio=submit)))
        task = asyncio.create_task(client.invoke("sdk", operation["request"], 128))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.last_failure == "sdk_submission_unknown"
        release.set()
        await client.cleanup()
        assert cancelled == [{"terminate_containers": True}]

        def fail(request):
            posts.append(request.method)
            return httpx.Response(503, content=b"private server failure details")

        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            client = LatencyClient(
                manifest,
                modal_module=object(),
                http_client=http,
                secrets={
                    "BOOKFORGE_LATENCY_MODAL_KEY": "key",
                    "BOOKFORGE_LATENCY_MODAL_SECRET": "secret",
                },
            )
            client.url, client.http_ready = "https://isolated.modal.run", True
            with pytest.raises(LatencyError, match="^http_remote_completion_unknown$"):
                await client.invoke("http", operation["request"], 128)
            assert not await client.cleanup()
        assert posts == ["POST"]

    asyncio.run(scenario())


def test_frozen_schedule_claim_and_offline_comparison_preserve_failed_or_cold_cases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "operator-home")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    manifest = freezer.freeze("offline-test", int(time.time()) + 1200)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    authorization_path = private / "authorization.json"
    authorization = {
        "schema_version": 1,
        "manifest_sha256": benchmark.file_hash(manifest_path),
        "reservation_id": "private-canonical-reservation",
        "reserved_usd": 4.58,
        "maximum_operations": 14,
        "ledger_sha256": "a" * 64,
    }
    benchmark.write_exclusive(authorization_path, json.dumps(authorization).encode())
    assert benchmark.load_inputs(manifest_path, authorization_path) == (manifest, authorization)
    args = argparse.Namespace(
        manifest=manifest_path, authorization=authorization_path, output=tmp_path / "run"
    )
    header = benchmark.header(args, manifest, authorization)
    benchmark.claim(args, header)
    copied = private / "authorization-copy.json"
    benchmark.write_exclusive(copied, authorization_path.read_bytes())
    with pytest.raises(FileExistsError):
        benchmark.claim(
            argparse.Namespace(
                **{**vars(args), "authorization": copied, "output": tmp_path / "another-run"}
            ),
            header,
        )
    args.output.mkdir(mode=0o700)
    journal = args.output / "journal.jsonl"
    monkeypatch.delenv("BOOKFORGE_LATENCY_MODAL_KEY", raising=False)
    monkeypatch.delenv("BOOKFORGE_LATENCY_MODAL_SECRET", raising=False)
    with pytest.raises(LatencyError, match="proxy_credentials_missing"):
        asyncio.run(benchmark.execute(args, manifest, header))
    assert not journal.exists()
    events = [header]
    for ordinal, operation in enumerate(manifest["operations"]):
        payload = response(manifest, operation["request"], operation["expected_bucket"])
        path = args.output / f"operation-{ordinal:02}"
        path.mkdir()
        for role in ("master", "depth"):
            content = payload.pop(role)
            if content:
                (path / f"{role}.jpg").write_bytes(content)
        timings = dict.fromkeys(TIMINGS, 0.0)
        timings["total_artifact_ready_seconds"] = 10 if operation["transport"] == "sdk" else 6
        events.extend(
            [
                {"kind": "start", "ordinal": ordinal, "request_sha256": protocol.digest(operation)},
                {
                    "kind": "result",
                    "ordinal": ordinal,
                    "status": "ok",
                    "timings": timings,
                    "payload": payload,
                },
            ]
        )
    events.extend(
        [
            {"kind": "complete", "operations": 14},
            {"kind": "cleanup", "known_calls_cancelled": True, "external_app_stop_required": False},
        ]
    )

    def report(records):
        journal.write_text("\n".join(json.dumps(event) for event in records) + "\n")
        return benchmark.aggregate(args, manifest, header)

    result = report(events)
    assert result["decision"] == "pass" and result["http_p50_improvement_fraction"] == 0.4
    assert not result["external_app_shutdown_verified"]
    assert report(events[:-1])["decision"] == "reject"
    assert len(result["pairs"]) == 6 and all(
        pair["master_equal"] and pair["depth_equal"] for pair in result["pairs"]
    )
    assert "private-canonical-reservation" not in json.dumps(result)
    cold = copy.deepcopy(events)
    cold[6]["payload"]["metrics"]["bucket_was_warm"] = False
    assert report(cold)["decision"] == "reject"
    unknown = copy.deepcopy(events)
    unknown[6]["payload"]["location"]["cloud"] = None
    assert report(unknown)["decision"] == "reject"
    failed = events[:6] + [
        {
            "kind": "result",
            "ordinal": 2,
            "status": "failed",
            "code": "sdk_submission_unknown",
            "timings": {},
        },
        events[-1],
    ]
    result = report(failed)
    assert result["decision"] == "reject" and result["failures"] == 1 and result["started"] == 3
    changed = copy.deepcopy(manifest)
    changed["operations"][2]["request"]["seed"] += 1
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        benchmark.load_inputs(manifest_path, authorization_path)
