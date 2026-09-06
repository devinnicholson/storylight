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

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_cold_start as benchmark  # noqa: E402


def synthetic_inputs(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 576), (80, 120, 160)).save(buffer, "JPEG")
    content = buffer.getvalue()
    identity, cases = benchmark.frozen_cases()
    for case in cases:
        case.update(
            master_sha256=hashlib.sha256(content).hexdigest(),
            depth_sha256=hashlib.sha256(content).hexdigest(),
        )
    monkeypatch.setattr(benchmark, "frozen_cases", lambda: (identity, cases))
    value = {
        "schema_version": 1,
        "experiment_id": benchmark.EXPERIMENT,
        "status": "draft",
        "expires_at": None,
        "image_id": benchmark.IMAGE_ID,
        "cache_id": benchmark.CACHE_ID,
        "runtime_sha256": benchmark.legacy.file_hash(
            benchmark.ROOT / "deploy/klein_scene_runtime.py"
        ),
        "deployment_sha256": benchmark.legacy.file_hash(
            benchmark.ROOT / "deploy/modal_klein_cold_start.py"
        ),
        "client_sha256": benchmark.legacy.file_hash(Path(benchmark.__file__)),
        "expected_identity": identity,
        "cases": cases,
        "operations": [
            {"request_id": f"{index:032x}", "variant": variant, "pair_index": pair}
            for index, (variant, pair) in enumerate(benchmark.SCHEDULE)
        ],
    }
    return value, content


def payload(manifest, operation, content):
    samples = []
    for index in range(4):
        case = manifest["cases"][index % 2]
        samples.append(
            {
                "repeat": index // 2,
                "case_index": index % 2,
                "metrics": {
                    **dict.fromkeys(benchmark.METRICS, 0.01),
                    "seed": case["seed"],
                    "token_count": case["expected_bucket"],
                    "sequence_bucket": case["expected_bucket"],
                    "master_sha256": hashlib.sha256(content).hexdigest(),
                    "depth_sha256": hashlib.sha256(content).hexdigest(),
                },
                "master": content,
                "depth": content,
            }
        )
    return {
        "request_id": operation["request_id"],
        "variant": operation["variant"],
        "identity": manifest["expected_identity"],
        "location": {
            "cloud": "CLOUD_PROVIDER_AWS",
            "compute_region": "us-east-1",
            "container_sha256": hashlib.sha256(operation["request_id"].encode()).hexdigest(),
        },
        "stages": dict.fromkeys(benchmark.STAGES, 0.01),
        "samples": samples,
    }


def test_fixed_private_authorization_and_complete_offline_evidence(tmp_path, monkeypatch, capsys):
    manifest, content = synthetic_inputs(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(benchmark.preparation.encoded(manifest))
    assert benchmark.read_manifest(manifest_path) == manifest
    args = ["--manifest", str(manifest_path), "--output", str(tmp_path / "preflight")]
    assert benchmark.main(args) == 0
    assert benchmark.main([*args[:-1], str(tmp_path / "forbidden"), "--execute"]) == 1
    assert not (tmp_path / "forbidden").exists()
    for field in ("case", "schedule", "float", "duplicate"):
        changed = copy.deepcopy(manifest)
        if field == "case":
            changed["cases"][0]["prompt"] = "private unauthorized source"
        elif field == "schedule":
            changed["operations"].reverse()
        elif field == "float":
            changed["operations"][0]["pair_index"] = 0.0
        else:
            changed["operations"][1]["request_id"] = changed["operations"][0]["request_id"]
        manifest_path.write_bytes(benchmark.preparation.encoded(changed))
        with pytest.raises(ValueError):
            benchmark.read_manifest(manifest_path)
    assert "private unauthorized source" not in capsys.readouterr().out
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    manifest_path.write_bytes(benchmark.preparation.encoded(manifest))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    authorization_path = private / "authorization.json"
    authorization = {
        "schema_version": 1,
        "manifest_sha256": benchmark.legacy.file_hash(manifest_path),
        "reservation_id": "synthetic-cold",
        "reserved_usd": 2.84,
        "maximum_operations": 6,
        "ledger_sha256": "e" * 64,
    }
    benchmark.legacy.write_exclusive(
        authorization_path, benchmark.preparation.encoded(authorization)
    )
    proof = benchmark.legacy.file_hash(authorization_path)
    assert benchmark.read_authorization(authorization_path, proof, manifest_path) == authorization
    with pytest.raises(ValueError):
        benchmark.read_authorization(authorization_path, "f" * 64, manifest_path)
    output = tmp_path / "results"
    args = argparse.Namespace(
        manifest=manifest_path, authorization=authorization_path, output=output
    )
    header = benchmark.header(args, manifest, authorization)
    # Copying the authorization/output location cannot bypass the home-scoped attempt.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    benchmark.legacy.claim(args, header)
    with pytest.raises(FileExistsError):
        benchmark.legacy.claim(
            argparse.Namespace(authorization=authorization_path, output=tmp_path / "another"),
            header,
        )
    output.mkdir()

    class OfflineClient:
        last_failure = last_timings = None
        calls = []

        async def cycle(self, operation):
            self.calls.append(operation["request_id"])
            return payload(manifest, operation, content), dict.fromkeys(benchmark.TIMINGS, 0.0)

        async def cleanup(self):
            return True

    client = OfflineClient()
    asyncio.run(benchmark.execute(args, manifest, header, client))
    assert len(client.calls) == 6 and len(list(output.glob("operation-*/*.jpg"))) == 48
    journal = output / "journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    for row in entries:
        if row["kind"] == "result":
            row["timings"]["total_artifact_ready_seconds"] = (
                100 if row["payload"]["variant"] == "baseline" else 60
            )

    def write(rows):
        journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    write(entries)
    result = benchmark.aggregate(args, manifest, header)
    assert result["decision"] == "pass" and result["six_distinct_containers"]
    assert (
        result["all_reference_hashes_match"] and result["candidate_p50_improvement_fraction"] == 0.4
    )
    assert all(case["prompt"] not in journal.read_text() for case in manifest["cases"])
    reordered = copy.deepcopy(entries)
    reordered[1:5] = reordered[3:5] + reordered[1:3]
    write(reordered)
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)
    write(entries[:-1])
    assert benchmark.aggregate(args, manifest, header)["decision"] == "reject"
    same_container = copy.deepcopy(entries)
    same_container[4]["payload"]["location"] = same_container[2]["payload"]["location"]
    write(same_container)
    assert not benchmark.aggregate(args, manifest, header)["six_distinct_containers"]
    write(entries)
    mismatched = copy.deepcopy(manifest)
    mismatched["cases"][0]["master_sha256"] = "a" * 64
    assert benchmark.aggregate(args, mismatched, header)["decision"] == "reject"
    artifact = benchmark.artifact_path(output, 0, 0, "master")
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)


def test_cancelled_submission_is_not_retried_and_failure_stays_value_free(tmp_path, monkeypatch):
    manifest, content = synthetic_inputs(tmp_path, monkeypatch)
    operation = manifest["operations"][0]

    async def cancellation():
        entered, released = asyncio.Event(), asyncio.Event()
        calls, cancellations = [], []

        async def hydrate():
            return None

        async def cancel(*, terminate_containers):
            cancellations.append(terminate_containers)

        async def spawn(request_id):
            calls.append(request_id)
            entered.set()
            await released.wait()
            return SimpleNamespace(cancel=SimpleNamespace(aio=cancel))

        function = SimpleNamespace(
            hydrate=SimpleNamespace(aio=hydrate), spawn=SimpleNamespace(aio=spawn)
        )
        modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *args: function))
        client = benchmark.ColdClient(manifest, modal_module=modal)
        task = asyncio.create_task(client.cycle(operation))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        released.set()
        assert (
            not await client.cleanup()
        )  # Ambiguous submission still requires supervised app stop.
        assert calls == [operation["request_id"]] and cancellations == [True]

    asyncio.run(cancellation())
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    output = tmp_path / "failed"
    output.mkdir()
    args = argparse.Namespace(output=output)
    header = {"kind": "header", "synthetic": True}

    class FailedClient:
        last_failure = last_timings = None
        calls = []

        async def cycle(self, operation):
            self.calls.append(operation)
            raise RuntimeError("private provider exception payload")

        async def cleanup(self):
            return True

    client = FailedClient()
    with pytest.raises(RuntimeError):
        asyncio.run(benchmark.execute(args, manifest, header, client))
    assert len(client.calls) == 1
    result = benchmark.aggregate(args, manifest, header)
    assert result["decision"] == "reject" and result["failures"] == 1 and not result["complete"]
    assert "private provider exception payload" not in (output / "journal.jsonl").read_text()
    broken = payload(manifest, operation, content)
    broken["samples"][1]["metrics"]["seed"] += 1
    with pytest.raises(ValueError):
        benchmark.validate_payload(broken, operation, manifest)
