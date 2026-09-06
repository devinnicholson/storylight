"""The finite HTTP flow must preserve failures and distinguish bucket priming."""

import asyncio
import base64
import copy
import io
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_gcp_klein as bench  # noqa: E402


def setup_manifest():
    identity, _ = bench.cold.frozen_cases()
    identity = {**identity, "gpu": "NVIDIA RTX PRO 6000 Blackwell", "capability": [12, 0]}
    sources = {k: bench.sha(p.read_bytes()) for k, p in bench.SOURCE_PATHS.items()}
    manifest = bench.prepare_manifest("test-gcp", identity, sources)
    manifest.update(status="authorized", expires_at=int(time.time()) + 600)
    return manifest


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (1024, 576), "orange").save(stream, "JPEG")
    return stream.getvalue()


def response(manifest, index, raw):
    case = bench.schedule(manifest)[index]
    bucket = 256 if case["case_id"] == "retained-1" else 128
    metrics = dict.fromkeys(bench.TIMINGS, 0.01)
    metrics.update(
        seed=case["seed"],
        sequence_bucket=bucket,
        token_count=bucket - 1,
        master_sha256=bench.sha(raw),
        depth_sha256=bench.sha(raw),
    )
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "instance_id": "a" * 32,
        "manifest_sha256": bench.sha(bench.encoded(manifest)),
        "identity": manifest["expected_identity"],
        "metrics": metrics,
        "master_b64": base64.b64encode(raw).decode(),
        "depth_b64": base64.b64encode(raw).decode(),
        "cold": index == 0,
        "bucket_was_warm": index > 1,
        "load_seconds": 2,
        "compile_seconds": 1,
        "worker_seconds": 4 if index == 0 else 0.1,
        "request_ordinal": index + 1,
        "revision": "renderer-00001-a",
        "service": "renderer",
    }


def test_ten_requests_exact_schedule_and_priming_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest, raw = setup_manifest(), jpeg()
    assert len(manifest["cases"]) == 8
    assert bench.schedule(manifest)[-2:] == manifest["cases"][:2]
    assert all("watercolor" in c["prompt"].lower() for c in manifest["cases"])
    sent = []

    async def handler(request):
        index = len(sent)
        sent.append(request)
        assert request.method == "POST" and request.url.path == "/generate"
        assert json.loads(request.content) == {
            k: bench.schedule(manifest)[index][k] for k in ("case_id", "seed")
        }
        return httpx.Response(200, json=response(manifest, index, raw))

    async def token(audience):
        assert audience == "https://renderer.run.app"
        return "private-token"

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await bench.run(
                manifest,
                bench.encoded(manifest),
                tmp_path / "results",
                "https://renderer.run.app",
                "renderer",
                "renderer-00001-a",
                client=client,
                token_source=token,
            )
            with pytest.raises(FileExistsError):
                await bench.run(
                    manifest,
                    bench.encoded(manifest),
                    tmp_path / "different-output",
                    "https://renderer.run.app",
                    "renderer",
                    "renderer-00001-a",
                    client=client,
                    token_source=token,
                )

    asyncio.run(execute())
    assert len(sent) == 10
    summary = bench.aggregate(
        manifest, bench.encoded(manifest), tmp_path / "results", "renderer", "renderer-00001-a"
    )
    assert summary["decision"] == "ungraded" and summary["completed_cases"] == 10
    assert summary["later_bucket_client_seconds"]["count"] == 8
    assert [r["first_bucket_on_worker"] for r in summary["cases"]] == [True, True] + [False] * 8
    assert len(list((tmp_path / "results").glob("*.jpg"))) == 20
    assert all(r["human_quality"] == "ungraded" for r in summary["cases"])
    assert "private-token" not in (tmp_path / "results/journal.jsonl").read_text()
    (tmp_path / "results/9-master.jpg").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        bench.aggregate(
            manifest, bench.encoded(manifest), tmp_path / "results", "renderer", "renderer-00001-a"
        )


def test_failure_stops_without_retry_and_boundary_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest, raw = setup_manifest(), jpeg()
    valid = response(manifest, 0, raw)
    for field, bad in (
        ("revision", "other"),
        ("manifest_sha256", "d" * 64),
        ("case_id", "retained-1"),
        ("request_ordinal", True),
        ("master_b64", "invalid!"),
        ("cold", False),
    ):
        changed = {**valid, field: bad}
        with pytest.raises((ValueError, TypeError)):
            bench.validate_payload(
                changed,
                manifest["cases"][0],
                manifest,
                bench.sha(bench.encoded(manifest)),
                "renderer",
                "renderer-00001-a",
            )
    for mutation in (
        lambda m: m["cases"][0].update(seed=90401.0),
        lambda m: m["cases"][0].update(prompt="unapproved input"),
        lambda m: m.update(expires_at=0),
    ):
        changed = copy.deepcopy(manifest)
        mutation(changed)
        with pytest.raises(ValueError):
            bench.validate_manifest(changed, active=True)
    sent = []

    async def handler(request):
        sent.append(request)
        return httpx.Response(503, text="secret server details")

    async def token(_):
        return "private-token"

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await bench.run(
                manifest,
                bench.encoded(manifest),
                tmp_path / "failed",
                "https://renderer.run.app",
                "renderer",
                "renderer-00001-a",
                client=client,
                token_source=token,
            )

    asyncio.run(execute())
    assert len(sent) == 1 and not list((tmp_path / "failed").glob("*.jpg"))
    summary = bench.aggregate(
        manifest, bench.encoded(manifest), tmp_path / "failed", "renderer", "renderer-00001-a"
    )
    assert summary["decision"] == "reject" and summary["completed_cases"] == 0
    assert "secret" not in (tmp_path / "failed/journal.jsonl").read_text()
    for endpoint in (
        "https://renderer.run.app.evil.invalid",
        "http://renderer.run.app",
        "https://renderer.run.app/redirect",
    ):
        with pytest.raises(ValueError):
            bench.validate_endpoint(endpoint, "renderer", "renderer-00001-a")
