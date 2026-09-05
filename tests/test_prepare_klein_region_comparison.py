from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import stat
import sys
import time
from pathlib import Path

import pytest
from test_klein_latency_client import response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_region as runner  # noqa: E402
from scripts import prepare_klein_region_comparison as preparation  # noqa: E402


@pytest.fixture
def code_pins(tmp_path, monkeypatch):
    deployment, client = tmp_path / "regional.py", tmp_path / "regional-client.py"
    deployment.write_text("# Separate region-controlled deployment.\n")
    client.write_text("# Separate region-controlled client.\n")
    monkeypatch.setattr(preparation, "REGION_DEPLOYMENT", deployment)
    monkeypatch.setattr(preparation, "REGION_CLIENT", client)
    return deployment, client


def test_draft_preserves_requests_and_activation_requires_exact_private_proof(code_pins, tmp_path):
    original = preparation.SOURCE.read_bytes()
    source = json.loads(original)
    output = tmp_path / "draft.json"
    args = ["--experiment-id", "klein-region-b", "--output", str(output)]
    assert preparation.main(args) == 0
    draft = json.loads(output.read_text())
    assert draft["status"] == "draft" and draft["expires_at"] is None
    assert draft["placement"] == {
        "cloud": "aws",
        "compute_region": "us-east",
        "routing_region": "us-east",
        "expected_cloud": "CLOUD_PROVIDER_AWS",
        "expected_compute_region": "us-east-1",
    }
    assert draft["deployments"] == {
        "sdk": {"app": "bookforge-klein-region-sdk", "class": "RegionStudio"},
        "http": {"app": "bookforge-klein-region-http", "class": "RegionServer"},
    }
    assert len(draft["operations"]) == 14
    assert draft["baked_image_id"] == "im-WtXer8GjRPdgMqWAAUSMwJ"
    assert preparation.COST_CEILING_USD == 14 * 0.39 + 0.50
    for candidate, original_operation in zip(
        copy.deepcopy(draft["operations"]), source["operations"], strict=True
    ):
        request_id = candidate["request"]["request_id"]
        assert request_id != original_operation["request"]["request_id"]
        key = f"klein-region-b:{candidate['transport']}:{candidate['pair_id'] or 'warmup'}"
        assert request_id == hashlib.sha256(key.encode()).hexdigest()[:32]
        candidate["request"]["request_id"] = original_operation["request"]["request_id"]
        assert candidate == original_operation
    assert draft["expected_identity"] == source["expected_identity"]
    assert draft["original_batch_sha256"] == source["original_batch_sha256"]
    assert all(draft[field] == source[field] for field in preparation.SUPPORT)
    assert (
        draft["region_deployment_sha256"] == hashlib.sha256(code_pins[0].read_bytes()).hexdigest()
    )
    assert draft["region_client_sha256"] == hashlib.sha256(code_pins[1].read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert preparation.SOURCE.read_bytes() == original
    assert preparation.main(args) == 1
    for field in ("seed", "schema_version", "baked_image_id"):
        changed = copy.deepcopy(draft)
        if field == "seed":
            changed["operations"][2]["request"]["seed"] = float(
                changed["operations"][2]["request"]["seed"]
            )
        elif field == "schema_version":
            changed["schema_version"] = 2.0
        else:
            changed["baked_image_id"] = "im-another-image"
        invalid = tmp_path / f"invalid-{field}.json"
        invalid.write_bytes(preparation.encoded(changed))
        with pytest.raises(ValueError):
            preparation.validate_manifest(invalid)
    preflight = tmp_path / "preflight"
    assert runner.main(["--manifest", str(output), "--output", str(preflight)]) == 0
    assert json.loads((preflight / "preflight.json").read_text())["generation_calls"] == 0
    assert (
        runner.main(
            ["--manifest", str(output), "--output", str(tmp_path / "forbidden"), "--execute"]
        )
        == 1
    )
    assert not (tmp_path / "forbidden").exists()

    expiry = int(time.time()) + 1200
    active = {**draft, "status": "authorized", "expires_at": expiry}
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    authorization_path = private / "authorization.json"
    authorization = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(preparation.encoded(active)).hexdigest(),
        "reservation_id": "private-local-test-reservation",
        "reserved_usd": 5.96,
        "maximum_operations": 14,
        "ledger_sha256": "a" * 64,
    }
    runner.legacy.write_exclusive(authorization_path, preparation.encoded(authorization))
    proof = runner.legacy.file_hash(authorization_path)
    activated = tmp_path / "active.json"
    activate_args = [
        "--experiment-id",
        "klein-region-b",
        "--output",
        str(activated),
        "--activate-until",
        str(expiry),
    ]
    assert preparation.main(activate_args) == 1 and not activated.exists()
    assert (
        preparation.main(
            [
                *activate_args,
                "--authorization",
                str(authorization_path),
                "--authorization-sha256",
                "f" * 64,
            ]
        )
        == 1
    )
    assert (
        preparation.main(
            [
                *activate_args,
                "--authorization",
                str(authorization_path),
                "--authorization-sha256",
                proof,
            ]
        )
        == 0
    )
    assert preparation.validate_manifest(activated) == active

    class OfflineClient:
        last_failure = last_timings = None
        calls = []

        def headers(self):
            return {}

        async def invoke(self, transport, request, expected_bucket):
            self.calls.append((transport, request))
            payload = response(
                {**active, "deployment_sha256": active["region_deployment_sha256"]},
                request,
                expected_bucket,
            )
            payload["location"].update(cloud="CLOUD_PROVIDER_AWS", compute_region="us-east-1")
            return payload, dict.fromkeys(runner.legacy.TIMINGS, 0.0)

        async def cleanup(self):
            return True

    cli = argparse.Namespace(
        manifest=activated, authorization=authorization_path, output=tmp_path / "offline-results"
    )
    cli.output.mkdir()
    header = runner.header(cli, active, authorization)
    client = OfflineClient()
    asyncio.run(runner.execute(cli, active, header, client))
    result = runner.aggregate(cli, active, header)
    assert len(client.calls) == 14 and result["complete"] and result["placement_verified"]
    journal = cli.output / "journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    entries[2]["payload"]["location"]["compute_region"] = "us-west-2"
    journal.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    result = runner.aggregate(cli, active, header)
    assert not result["placement_verified"] and result["decision"] == "reject"


def test_tampered_source_or_reused_identity_cannot_produce_draft(
    code_pins, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "draft.json"
    source = json.loads(preparation.SOURCE.read_text())
    args = ["--experiment-id", source["experiment_id"], "--output", str(output)]
    assert preparation.main(args) == 1 and not output.exists()
    args[1] = "new-region-comparison"
    tampered = tmp_path / "source.json"
    source["operations"][2]["request"]["prompt"] = "private unauthorized prompt"
    tampered.write_text(json.dumps(source))
    monkeypatch.setattr(preparation, "SOURCE", tampered)
    assert preparation.main(args) == 1 and not output.exists()
    assert "private unauthorized prompt" not in capsys.readouterr().out
