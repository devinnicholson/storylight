from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_klein_cold_start import payload as cold_payload
from test_klein_cold_start import synthetic_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_import_snapshot as benchmark  # noqa: E402


def inputs(tmp_path, monkeypatch):
    manifest, content = synthetic_inputs(tmp_path, monkeypatch)
    manifest.update(
        experiment_id=benchmark.EXPERIMENT,
        max_operations=5,
        max_captures=3,
        client_sha256=benchmark.legacy.file_hash(Path(benchmark.__file__)),
        deployment_sha256=benchmark.legacy.file_hash(
            benchmark.ROOT / "deploy/modal_klein_import_snapshot.py"
        ),
        operations=[
            {"ordinal": i, "request_id": f"{i:032x}", "variant": "snapshot"} for i in range(5)
        ],
    )
    return manifest, content


def response(manifest, operation, content, *, capture=1, container=None):
    value = cold_payload(manifest, operation, content)
    if container is not None:
        value["location"]["container_sha256"] = container
    value["snapshot"] = {
        "capture_id": f"{capture:032x}",
        "activation_id": operation["request_id"],
        "capture_container_sha256": hashlib.sha256(f"capture-{capture}".encode()).hexdigest(),
        "imports_seconds": 0.01,
        "versions": {
            key: manifest["expected_identity"][key]
            for key in ("torch", "diffusers", "transformers", "triton")
        },
        "cuda_available": True,
    }
    return value


def test_private_proof_and_observed_restores_stop_before_five_calls(tmp_path, monkeypatch, capsys):
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
    changed["cases"][0]["prompt"] = "private unsupported prompt"
    path.write_bytes(benchmark.preparation.encoded(changed))
    with pytest.raises(ValueError):
        benchmark.read_manifest(path)
    assert "private unsupported prompt" not in capsys.readouterr().out
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    path.write_bytes(benchmark.preparation.encoded(manifest))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    auth_path = private / "authorization.json"
    auth = {
        "schema_version": 1,
        "manifest_sha256": benchmark.legacy.file_hash(path),
        "reservation_id": "synthetic-snapshot",
        "reserved_usd": 1.67,
        "maximum_operations": 5,
        "ledger_sha256": "f" * 64,
    }
    benchmark.legacy.write_exclusive(auth_path, benchmark.preparation.encoded(auth))
    proof = benchmark.legacy.file_hash(auth_path)
    assert benchmark.read_authorization(auth_path, proof, path) == auth
    with pytest.raises(ValueError):
        benchmark.read_authorization(auth_path, "0" * 64, path)
    args = argparse.Namespace(
        manifest=path,
        authorization=auth_path,
        output=tmp_path / "results",
        deadline_unix=time.time() + 300,
    )
    header = benchmark.header(args, manifest, auth)
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
            # A permitted new hardware capture; only the following two reuses prove restores.
            value = response(
                manifest, operation, content, capture=1 if operation["ordinal"] == 0 else 2
            )
            return value, dict.fromkeys(benchmark.cold.TIMINGS, 0.0)

        async def cleanup(self):
            return True

    client = OfflineClient()
    asyncio.run(benchmark.execute(args, manifest, header, client))
    assert len(client.calls) == 4 and len(list(args.output.glob("operation-*/*.jpg"))) == 32
    result = benchmark.aggregate(args, manifest, header)
    assert result["decision"] == "qualified_for_comparison"
    assert result["observed_restores"] == 2 and result["recapture_count"] == 1
    assert not result["provider_restore_logs_reviewed"] and not result["comparison_is_causal"]
    journal = args.output / "journal.jsonl"
    assert all(case["prompt"] not in journal.read_text() for case in manifest["cases"])
    original = journal.read_bytes()
    events = [json.loads(line) for line in original.splitlines()]
    journal.write_text("\n".join(json.dumps(row) for row in events[:-1]) + "\n")
    assert benchmark.aggregate(args, manifest, header)["decision"] == "reject"
    # Reusing an activation must not produce two restore proofs.
    events[6]["payload"]["snapshot"]["activation_id"] = events[4]["payload"]["snapshot"][
        "activation_id"
    ]
    journal.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)
    journal.write_bytes(original)
    artifact = benchmark.cold.artifact_path(args.output, 0, 0, "master")
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)


def test_capture_limits_deadline_and_ambiguous_submission_stay_bounded(tmp_path, monkeypatch):
    manifest, content = inputs(tmp_path, monkeypatch)
    all_new = [
        response(manifest, operation, content, capture=i + 1)
        for i, operation in enumerate(manifest["operations"][:4])
    ]
    state = benchmark.qualification(all_new, manifest)
    assert state["observed_restores"] == 0 and state["capture_limit_exceeded"]
    same_container = "e" * 64
    reused = [
        response(manifest, operation, content, container=same_container)
        for operation in manifest["operations"]
    ]
    assert benchmark.qualification(reused, manifest)["observed_restores"] == 0
    assert benchmark.qualification(reused, manifest)["container_reused"]
    changed = copy.deepcopy(reused)
    changed[1]["snapshot"]["imports_seconds"] += 1
    with pytest.raises(ValueError):
        benchmark.qualification(changed, manifest)
    invalid = copy.deepcopy(reused[0])
    invalid["snapshot"]["cuda_available"] = False
    with pytest.raises(ValueError):
        benchmark.validate_payload(invalid, manifest["operations"][0], manifest)

    async def cancellation():
        entered, released = asyncio.Event(), asyncio.Event()
        calls, cancelled = [], []

        async def hydrate_remote():
            return None

        async def cancel(*, terminate_containers):
            cancelled.append(terminate_containers)

        async def spawn(request_id):
            calls.append(request_id)
            entered.set()
            await released.wait()
            return SimpleNamespace(cancel=SimpleNamespace(aio=cancel))

        class Remote:
            hydrate = SimpleNamespace(aio=hydrate_remote)

            def __call__(self):
                return SimpleNamespace(cycle=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))

        modal = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *args: Remote()))
        client = benchmark.SnapshotClient(manifest, deadline=time.time() + 300, modal_module=modal)
        task = asyncio.create_task(client.cycle(manifest["operations"][0]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        released.set()
        assert not await client.cleanup()
        assert len(calls) == 1 and cancelled == [True]

    asyncio.run(cancellation())
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    output = tmp_path / "expired"
    output.mkdir()
    args = argparse.Namespace(output=output, deadline_unix=time.time() - 1)

    class NoCalls:
        async def cycle(self, operation):
            raise AssertionError("expired probe must not dispatch")

        async def cleanup(self):
            return True

    header = {"kind": "header", "synthetic": True}
    asyncio.run(benchmark.execute(args, manifest, header, NoCalls()))
    result = benchmark.aggregate(args, manifest, header)
    assert result["decision"] == "inconclusive" and result["started"] == 0
    assert result["stop_reason"] == "deadline"
