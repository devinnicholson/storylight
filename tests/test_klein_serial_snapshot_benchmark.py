from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_warmed_snapshot_benchmark import inputs as warmed_inputs
from test_klein_warmed_snapshot_benchmark import response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_serial_snapshot as benchmark  # noqa: E402


def inputs(tmp_path, monkeypatch):
    manifest, content = warmed_inputs(tmp_path, monkeypatch)
    manifest.update(
        experiment_id=benchmark.EXPERIMENT,
        client_sha256=benchmark.legacy.file_hash(Path(benchmark.__file__)),
        deployment_sha256=benchmark.legacy.file_hash(
            benchmark.ROOT / "deploy/modal_klein_serial_snapshot.py"
        ),
    )
    return manifest, content


def test_serial_proof_and_shared_byte_latency_gates(tmp_path, monkeypatch):
    manifest, content = inputs(tmp_path, monkeypatch)
    path = tmp_path / "manifest.json"
    path.write_bytes(benchmark.preparation.encoded(manifest))
    assert benchmark.read_manifest(path) == manifest
    assert (
        benchmark.main(
            ["--preflight", "--manifest", str(path), "--output", str(tmp_path / "preflight")]
        )
        == 0
    )
    assert (
        benchmark.main(["--run", "--manifest", str(path), "--output", str(tmp_path / "forbidden")])
        == 1
    )
    assert not (tmp_path / "forbidden").exists()
    changed = copy.deepcopy(manifest)
    changed["max_captures"] = 2
    path.write_bytes(benchmark.preparation.encoded(changed))
    with pytest.raises(ValueError):
        benchmark.read_manifest(path)
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    path.write_bytes(benchmark.preparation.encoded(manifest))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    auth_path = private / "authorization.json"
    auth = {
        "schema_version": 1,
        "manifest_sha256": benchmark.legacy.file_hash(path),
        "reservation_id": "synthetic-serial-snapshot",
        "reserved_usd": 1.05,
        "maximum_operations": 3,
        "ledger_sha256": "f" * 64,
    }
    benchmark.legacy.write_exclusive(auth_path, benchmark.preparation.encoded(auth))
    assert (
        benchmark.read_authorization(auth_path, benchmark.legacy.file_hash(auth_path), path) == auth
    )
    with pytest.raises(ValueError):
        benchmark.read_authorization(auth_path, "0" * 64, path)
    args = argparse.Namespace(
        manifest=path,
        authorization=auth_path,
        output=tmp_path / "results",
        deadline_unix=time.time() + 200,
    )
    header = benchmark.header(args, manifest, auth)
    assert header["harness_sha256"] == benchmark.legacy.file_hash(Path(benchmark.__file__))
    assert header["support_sha256"]["scripts/benchmark_klein_warmed_snapshot.py"] == (
        benchmark.legacy.file_hash(Path(benchmark.warm.__file__))
    )
    assert benchmark.execute is benchmark.warm.execute
    assert benchmark.validate_payload is benchmark.warm.validate_payload
    assert benchmark.qualification is benchmark.warm.qualification
    assert (
        benchmark.main(
            [
                "--run",
                "--manifest",
                str(path),
                "--output",
                str(tmp_path / "overlong"),
                "--authorization",
                str(auth_path),
                "--authorization-sha256",
                benchmark.legacy.file_hash(auth_path),
                "--deadline-unix",
                str(time.time() + 181),
            ]
        )
        == 1
    )
    assert not (tmp_path / "overlong").exists()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    benchmark.legacy.claim(args, header)
    with pytest.raises(FileExistsError):
        benchmark.legacy.claim(args, header)
    args.output.mkdir()

    class OfflineClient:
        last_failure = last_timings = None
        calls = []

        async def cycle(self, operation):
            self.calls.append(operation)
            return response(manifest, operation, content), dict.fromkeys(
                benchmark.cold.TIMINGS, 0.0
            )

        async def cleanup(self):
            return True

    client = OfflineClient()
    asyncio.run(benchmark.execute(args, manifest, header, client))
    assert len(client.calls) == 3 and len(list(args.output.glob("operation-*/*.jpg"))) == 24
    journal = args.output / "journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    for row in entries:
        if row["kind"] == "result":
            row["timings"]["total_artifact_ready_seconds"] = 8.0
            assert row["payload"]["location"]["compute_region"] == "us-central1"

    def write(rows):
        journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    write(entries)
    result = benchmark.aggregate(args, manifest, header)
    assert result["correctness_qualified"] and result["latency_pass"]
    assert result["kind"] == "klein-serial-snapshot-summary"
    assert result["compiler_configuration_audit_pending"]
    assert result["decision"] == "pending_platform_audit" and not result["comparison_is_causal"]
    slow_bucket = copy.deepcopy(entries)
    slow_bucket[4]["payload"]["samples"][1]["metrics"]["total_seconds"] = 2.500001
    write(slow_bucket)
    result = benchmark.aggregate(args, manifest, header)
    assert result["correctness_qualified"] and not result["first_bucket_latency_pass"]
    assert result["restore_cycle_latency_pass"] and not result["latency_pass"]
    slow_delivery = copy.deepcopy(entries)
    slow_delivery[4]["timings"]["total_artifact_ready_seconds"] = 24.0
    write(slow_delivery)
    result = benchmark.aggregate(args, manifest, header)
    assert result["correctness_qualified"] and result["first_bucket_latency_pass"]
    assert not result["restore_cycle_latency_pass"] and not result["latency_pass"]
    write(entries[:-1])
    assert benchmark.aggregate(args, manifest, header)["decision"] == "reject"
    write(entries)
    benchmark.cold.artifact_path(args.output, 0, 0, "master").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)


def test_capture_provenance_expiry_and_cancelled_submission_refuse_without_retries(
    tmp_path, monkeypatch
):
    manifest, content = inputs(tmp_path, monkeypatch)
    operation = manifest["operations"][0]
    valid = response(manifest, operation, content)
    for field in ("warmup_hash", "placement", "capture_timing"):
        value = copy.deepcopy(valid)
        if field == "warmup_hash":
            value["snapshot"]["warmups"][0]["master_sha256"] = "a" * 64
        elif field == "placement":
            value["location"]["compute_region"] = "europe-west1"
        else:
            value["snapshot"]["initialization"]["model_load_seconds"] += 1
        with pytest.raises(ValueError):
            benchmark.validate_payload(value, operation, manifest)
    recaptured = [response(manifest, op, content) for op in manifest["operations"][:2]]
    recaptured[1]["snapshot"]["capture_id"] = "e" * 32
    assert benchmark.qualification(recaptured, manifest)["capture_limit_exceeded"]

    async def cancellation():
        entered, release = asyncio.Event(), asyncio.Event()
        lookups, calls, cancelled = [], [], []

        async def hydrate_remote():
            return None

        async def cancel(*, terminate_containers):
            cancelled.append(terminate_containers)

        async def spawn(request_id):
            calls.append(request_id)
            entered.set()
            await release.wait()
            return SimpleNamespace(cancel=SimpleNamespace(aio=cancel))

        class Remote:
            hydrate = SimpleNamespace(aio=hydrate_remote)

            def __call__(self):
                return SimpleNamespace(cycle=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))

        def lookup(app, cls):
            lookups.append((app, cls))
            return Remote()

        client = benchmark.SerialClient(
            manifest,
            deadline=time.time() + 200,
            modal_module=SimpleNamespace(Cls=SimpleNamespace(from_name=lookup)),
        )
        task = asyncio.create_task(client.cycle(operation))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert not await client.cleanup()
        assert calls == [operation["request_id"]] and cancelled == [True]
        assert lookups == [("bookforge-klein-serial-snapshot", "SerialSnapshot")]

    asyncio.run(cancellation())
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    output = tmp_path / "expired"
    output.mkdir()
    args = argparse.Namespace(output=output, deadline_unix=time.time() - 1)

    class NoCalls:
        async def cycle(self, operation):
            raise AssertionError("expired probe cannot dispatch")

        async def cleanup(self):
            return True

    header = {"kind": "header", "synthetic": True}
    asyncio.run(benchmark.execute(args, manifest, header, NoCalls()))
    result = benchmark.aggregate(args, manifest, header)
    assert result["decision"] == "inconclusive" and result["started"] == 0
