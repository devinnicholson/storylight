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
from scripts import benchmark_simple_scenes as benchmark  # noqa: E402


def inputs(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment.py"
    deployment.write_text("# offline declaration fixture\n")
    monkeypatch.setattr(benchmark, "PINS", {**benchmark.PINS, "deployment_sha256": deployment})

    class Tokenizer:
        chat_template = "synthetic-template"
        backend_tokenizer = SimpleNamespace(to_str=lambda: "synthetic-backend")

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text):
            return {"input_ids": text.split()}

    monkeypatch.setattr(benchmark, "load_tokenizer", Tokenizer)
    manifest = benchmark.prepare_manifest()
    path = tmp_path / "manifest.json"
    path.write_bytes(benchmark.preparation.encoded(manifest))
    return manifest, path


def response(manifest, operation, reference_sha):
    image = io.BytesIO()
    Image.new("RGB", (1024, 576), (operation["ordinal"] * 17, 60, 120)).save(image, format="JPEG")
    content = image.getvalue()
    control = next(c for c in benchmark.fixture()["controls"] if c["id"] == operation["control_id"])
    tokens = len(
        control["before_prompt" if operation["variant"] == "base" else "after_prompt"].split()
    )
    return {
        "request_id": operation["request_id"],
        "identity": manifest["expected_identity"],
        "location": {
            "cloud": "CLOUD_PROVIDER_AWS",
            "compute_region": "us-east-1",
            "container_sha256": "a" * 64,
        },
        "metrics": {
            **dict.fromkeys(benchmark.cold.METRICS, 0.1),
            "seed": operation["seed"],
            "token_count": tokens,
            "sequence_bucket": 128,
            "master_sha256": hashlib.sha256(content).hexdigest(),
            "depth_sha256": hashlib.sha256(content).hexdigest(),
            "reference_runtime_sha256": manifest["reference_runtime_sha256"],
            "reference_sha256": reference_sha,
            "reference_conditioned": reference_sha is not None,
            "reference_width": 1024 if reference_sha else None,
            "reference_height": 576 if reference_sha else None,
        },
        "master": content,
        "depth": content,
    }


def test_fixed_twelve_calls_bind_references_and_complete_client_timing(tmp_path, monkeypatch):
    manifest, path = inputs(tmp_path, monkeypatch)
    assert benchmark.read_manifest(path) == manifest
    token_proof = benchmark.token_preflight()
    assert len(token_proof["token_counts"]) == 8 and not token_proof["reference_shape_qualified"]
    ops = manifest["operations"]
    assert len(ops) == 12 and len({op["request_id"] for op in ops}) == 12
    assert [op["variant"] for op in ops] == [
        "base",
        "text_next",
        "reference_next",
        "base",
        "reference_next",
        "text_next",
    ] * 2
    for index in range(0, 12, 3):
        assert ops[index + 1]["prompt_sha256"] == ops[index + 2]["prompt_sha256"]
        assert ops[index]["seed"] == ops[index + 1]["seed"] == ops[index + 2]["seed"]
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
    modified = copy.deepcopy(manifest)
    modified["operations"][1]["seed"] = float(ops[1]["seed"])
    path.write_bytes(benchmark.preparation.encoded(modified))
    with pytest.raises(ValueError):
        benchmark.read_manifest(path)
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    path.write_bytes(benchmark.preparation.encoded(manifest))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    auth_path = private / "authorization.json"
    authorization = {
        "schema_version": 1,
        "manifest_sha256": benchmark.legacy.file_hash(path),
        "reservation_id": "synthetic-simple",
        "reserved_usd": 1.75,
        "maximum_operations": 12,
        "ledger_sha256": "b" * 64,
    }
    benchmark.legacy.write_exclusive(auth_path, benchmark.preparation.encoded(authorization))
    assert (
        benchmark.read_authorization(auth_path, benchmark.legacy.file_hash(auth_path), path)
        == authorization
    )
    with pytest.raises(ValueError):
        benchmark.read_authorization(auth_path, "0" * 64, path)
    args = argparse.Namespace(
        manifest=path,
        authorization=auth_path,
        output=tmp_path / "results",
        deadline_unix=time.time() + 500,
    )
    header = benchmark.header(args, manifest, authorization)

    class OversizeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "synthetic"

        def __call__(self, text):
            return {"input_ids": [0] * 257}

    with monkeypatch.context() as patch:
        patch.setattr(benchmark, "load_tokenizer", OversizeTokenizer)
        assert (
            benchmark.main(
                [
                    "--run",
                    "--manifest",
                    str(path),
                    "--output",
                    str(tmp_path / "oversize"),
                    "--authorization",
                    str(auth_path),
                    "--authorization-sha256",
                    benchmark.legacy.file_hash(auth_path),
                    "--deadline-unix",
                    str(time.time() + 500),
                ]
            )
            == 1
        )
        assert not (tmp_path / "oversize").exists()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    benchmark.legacy.claim(args, header)
    with pytest.raises(FileExistsError):
        benchmark.legacy.claim(args, header)
    args.output.mkdir()

    class OfflineClient:
        last_failure = last_timings = None
        calls = []

        async def render(self, operation, reference, reference_sha):
            self.calls.append(operation["ordinal"])
            if operation["variant"] == "reference_next":
                base = response(manifest, ops[operation["ordinal"] // 3 * 3], None)
                assert reference == base["master"]
                assert reference_sha == base["metrics"]["master_sha256"]
            else:
                assert reference is None and reference_sha is None
            return response(manifest, operation, reference_sha), dict.fromkeys(
                benchmark.cold.TIMINGS, 0.0
            )

        async def cleanup(self):
            return True

    client = OfflineClient()
    asyncio.run(benchmark.execute(args, manifest, header, client))
    assert client.calls == list(range(12))
    assert len(list(args.output.glob("operation-*/*.jpg"))) == 24
    journal = args.output / "journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "result":
            row["timings"]["total_artifact_ready_seconds"] = 50 if row["ordinal"] in (0, 2) else 2.0

    def write(entries):
        journal.write_text("\n".join(json.dumps(row) for row in entries) + "\n")

    write(rows)
    result = benchmark.aggregate(args, manifest, header)
    assert result["complete"] and result["latency_pass"] and result["artifacts_verified"] == 24
    assert result["decision"] == "ungraded" and not result["correctness_qualified"]
    assert result["variants"]["reference_next"]["eligible_ordinals"] == [4, 8, 10]
    assert result["variants"]["text_next"]["eligible_ordinals"] == [1, 5, 7, 11]
    slower = copy.deepcopy(rows)
    slower[12]["timings"]["total_artifact_ready_seconds"] = 2.50001
    write(slower)
    assert not benchmark.aggregate(args, manifest, header)["latency_pass"]
    write(rows[:-1])
    assert benchmark.aggregate(args, manifest, header)["decision"] == "reject"
    swapped = copy.deepcopy(rows)
    swapped[5]["reference_sha256"] = "f" * 64
    write(swapped)
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)
    write(rows)
    benchmark.artifact(args.output, 0, "master").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        benchmark.aggregate(args, manifest, header)


def test_failed_or_cancelled_request_stops_and_keeps_unknown_submission_claim(
    tmp_path, monkeypatch
):
    manifest, path = inputs(tmp_path, monkeypatch)
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    operation = manifest["operations"][2]
    valid = response(manifest, operation, "c" * 64)
    for field in ("reference_sha256", "reference_width", "seed"):
        invalid = copy.deepcopy(valid)
        invalid["metrics"][field] = "d" * 64 if field == "reference_sha256" else -1
        with pytest.raises(ValueError):
            benchmark.validate_payload(invalid, operation, manifest, "c" * 64)
    args = argparse.Namespace(output=tmp_path / "failure", deadline_unix=time.time() + 500)
    args.output.mkdir()

    class Failure:
        last_failure = "sdk_result_failed"
        last_timings = dict.fromkeys(benchmark.cold.TIMINGS, 0.0)
        calls = 0

        async def render(self, *args):
            self.calls += 1
            raise RuntimeError("synthetic private failure details")

        async def cleanup(self):
            return True

    client = Failure()
    with pytest.raises(RuntimeError):
        asyncio.run(benchmark.execute(args, manifest, {"kind": "header"}, client))
    assert client.calls == 1
    assert "private failure" not in (args.output / "journal.jsonl").read_text()
    summary = benchmark.aggregate(args, manifest, {"kind": "header"})
    assert (
        summary["failures"] == 1 and summary["decision"] == "reject" and not summary["latency_pass"]
    )

    async def cancellation():
        entered, release = asyncio.Event(), asyncio.Event()
        calls, cancelled, lookups = [], [], []

        async def hydrate_remote():
            return None

        async def cancel(*, terminate_containers):
            cancelled.append(terminate_containers)

        async def spawn(request_id, **kwargs):
            calls.append(request_id)
            entered.set()
            await release.wait()
            return SimpleNamespace(cancel=SimpleNamespace(aio=cancel))

        class Remote:
            hydrate = SimpleNamespace(aio=hydrate_remote)

            def __call__(self):
                return SimpleNamespace(render=SimpleNamespace(spawn=SimpleNamespace(aio=spawn)))

        def lookup(app, cls):
            lookups.append((app, cls))
            return Remote()

        client = benchmark.SimpleClient(
            manifest,
            deadline=time.time() + 500,
            modal_module=SimpleNamespace(Cls=SimpleNamespace(from_name=lookup)),
        )
        task = asyncio.create_task(client.render(manifest["operations"][0], None, None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert not await client.cleanup()
        assert cancelled == [True] and calls == [manifest["operations"][0]["request_id"]]
        assert lookups == [("bookforge-klein-simple-scenes", "SimpleSceneRenderer")]

    asyncio.run(cancellation())
