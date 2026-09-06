"""Compiler-cache export must match completed render evidence before saving bytes."""

import asyncio
import copy
import importlib.util
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load(ROOT / "experiments/renderer-native-cache/export_cache.py", "cache_export")
fixtures = load(ROOT / "tests/test_gcp_klein_benchmark.py", "cache_benchmark_fixtures")
bench = export.bench


def completed(tmp_path):
    manifest, jpeg = fixtures.setup_manifest(), fixtures.jpeg()
    raw = bench.encoded(manifest)
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    renders = tmp_path / "renders"
    renders.mkdir()
    rows = [bench.context(manifest, bench.sha(raw), "renderer", "renderer-00001-a")]
    for n, case in enumerate(bench.schedule(manifest)):
        payload = fixtures.response(manifest, n, jpeg)
        for role in ("master", "depth"):
            (renders / f"{n}-{role}.jpg").write_bytes(jpeg)
            del payload[role + "_b64"]
        rows.extend(
            [
                {"kind": "start", "ordinal": n, "case_sha256": bench.protocol.digest(case)},
                {
                    "kind": "result",
                    "ordinal": n,
                    "status": "ok",
                    "payload": payload,
                    "client_artifact_ready_seconds": 5.0,
                },
            ]
        )
    rows.append({"kind": "complete"})
    (renders / "journal.jsonl").write_bytes(b"\n".join(map(bench.encoded, rows)) + b"\n")
    summary = bench.aggregate(manifest, raw, renders, "renderer", "renderer-00001-a")
    (renders / "summary.json").write_bytes(bench.encoded(summary))
    return path, renders, bench.sha(raw)


def test_export_binds_receipts_identity_and_bounded_envelope(tmp_path, monkeypatch):
    path, renders, digest = completed(tmp_path)
    original = bench.validate_manifest
    seen_sources = []

    def validate(value, *, worker_source, **kwargs):
        seen_sources.append(worker_source)
        assert worker_source == bench.SOURCE_PATHS["worker"]
        original(value, worker_source=worker_source, **kwargs)

    monkeypatch.setattr(bench, "validate_manifest", validate)
    _, expected, _ = export.evidence(
        path, renders, "renderer", "renderer-00001-a", bench.SOURCE_PATHS["worker"]
    )
    artifact = b"opaque compiler artifact"
    header = expected | {"sha256": bench.sha(artifact), "size_bytes": len(artifact)}

    def wire(value, body=artifact):
        raw = bench.encoded(value)
        return export.MAGIC + len(raw).to_bytes(4, "big") + raw + body

    sent = []
    payload = wire(header)

    async def handler(request):
        sent.append(request)
        assert request.method == "GET" and request.url.path == "/compiler-cache"
        assert dict(request.url.params) == {"instance_id": "a" * 32, "manifest_sha256": digest}
        assert request.headers["authorization"] == "Bearer private-token"
        return httpx.Response(
            200, content=payload, headers={"content-type": "application/octet-stream"}
        )

    async def token(_):
        return "private-token"

    async def run(output):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await export.export_cache(
                path,
                renders,
                output,
                "https://renderer.run.app",
                "renderer",
                "renderer-00001-a",
                worker_source=bench.SOURCE_PATHS["worker"],
                proof_manifest_sha256=digest,
                client=client,
                token_source=token,
            )

    output = tmp_path / "valid"
    assert asyncio.run(run(output))["status"] == "exported"
    assert (output / "artifacts.bin").read_bytes() == artifact
    assert (output / "metadata.json").read_bytes() == bench.encoded(header)
    assert all(b"private-token" not in p.read_bytes() for p in output.iterdir())
    with pytest.raises(FileExistsError):
        asyncio.run(run(output))
    assert len(sent) == 1 and seen_sources
    wrong = copy.deepcopy(header)
    wrong["receipts_sha256"] = "f" * 64
    variants = [
        wire(wrong),
        wire(header)[:-1],
        wire(header) + b"extra",
        export.MAGIC + (export.MAX_HEADER + 1).to_bytes(4, "big"),
    ]
    for n, invalid_payload in enumerate(variants):
        payload = invalid_payload
        with pytest.raises(ValueError):
            asyncio.run(run(tmp_path / f"invalid-{n}"))
        assert not (tmp_path / f"invalid-{n}" / "metadata.json").exists()
    assert len(sent) == 5
    (renders / "0-master.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        asyncio.run(run(tmp_path / "bad-evidence"))
    assert len(sent) == 5


def test_export_deadline_retains_attempt_and_does_not_retry(tmp_path, monkeypatch):
    path, renders, digest = completed(tmp_path)
    output = tmp_path / "timeout"
    calls, cancelled = [], []
    monkeypatch.setattr(export, "DEADLINE_SECONDS", 0.01)

    async def handler(request):
        calls.append(request)
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(True)

    async def token(_):
        return "private-token"

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await export.export_cache(
                path,
                renders,
                output,
                "https://renderer.run.app",
                "renderer",
                "renderer-00001-a",
                worker_source=bench.SOURCE_PATHS["worker"],
                proof_manifest_sha256=digest,
                client=client,
                token_source=token,
            )

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    assert len(calls) == len(cancelled) == 1
    assert [p.name for p in output.iterdir()] == ["attempt.json"]
    with pytest.raises(FileExistsError):
        asyncio.run(run())
    assert len(calls) == 1
