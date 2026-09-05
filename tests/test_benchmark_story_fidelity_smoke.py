from __future__ import annotations

import json
import stat
from pathlib import Path

import httpx
import pytest

from scripts import benchmark_story_fidelity_smoke as smoke


def test_frozen_story_smoke_is_source_free_one_request_and_reproducible(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "engine_sha256": "a" * 64,
                "environment_sha256": "b" * 64,
                "snapshot_sha256": "c" * 64,
                "deployment_sha256": "d" * 64,
                "code_revision": "e" * 40,
            }
        )
    )
    manifest = Path(__file__).resolve().parents[1] / "examples/lantern-bridge-fidelity-story.json"
    evidence, output, contracts = (
        tmp_path / name for name in ("run.jsonl", "summary.json", "private.jsonl")
    )
    args = [
        "--manifest",
        str(manifest),
        "--model",
        "resident",
        "--provenance",
        str(provenance),
        "--evidence",
        str(evidence),
        "--output",
        str(output),
        "--contracts",
        str(contracts),
    ]
    calls = []
    raw = "SETTING: cave\nACTOR: orange foxes\nACTION: carry blue lantern\nMAGIC: ribbon"

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 2:
            raise httpx.ReadTimeout("private transport detail", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 29},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        smoke.httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(respond)}),
    )
    assert smoke.main([*args, "--aggregate-only"]) == 1
    assert not evidence.exists() and not contracts.exists()
    assert smoke.main(args) == 0
    summary = json.loads(output.read_text())
    assert len(calls) == 7
    assert all(call["max_tokens"] == 64 and call["temperature"] == 0 for call in calls)
    assert summary["request_failures"] == 1
    assert summary["decision"] == "stop_before_development_gate"
    assert summary["visual_fidelity_assessed"] is False
    assert summary["temporal_playback_assessed"] is False
    assert summary["semantic_accuracy_assessed"] is False
    assert summary["results"][6]["compiler_proof"] is True
    journal = evidence.read_text()
    assert raw not in journal
    assert "private transport detail" not in journal
    assert all(source not in journal for source, _ in smoke.load_manifest(manifest)[1])
    assert stat.S_IMODE(contracts.stat().st_mode) == 0o600
    retained = [json.loads(line) for line in contracts.read_text().splitlines()]
    assert any("candidate_master_prompt" in row for row in retained)
    assert all("source" not in row and "raw" not in row for row in retained)
    assert smoke.main([*args, "--aggregate-only"]) == 0
    assert json.loads(output.read_text()) == summary
    assert len(calls) == 7
    # A restart without private-contract export does not retry the failed request.
    assert smoke.main(args[:-2]) == 0
    assert len(calls) == 7


def test_smoke_rejects_changed_manifest_and_never_retries_interrupted_case(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="frozen fixture"):
        smoke.load_manifest(manifest)
    evidence = tmp_path / "run.jsonl"
    header = {"kind": "header", "schema_version": 1}
    assert smoke.load_evidence(evidence, header) == (set(), [])
    smoke.benchmark.append_event(evidence, {"kind": "start", "index": 0})
    started, rows = smoke.load_evidence(evidence, header)
    assert started == {0} and rows == []
    assert smoke.aggregate(header, started, rows)["interrupted_cases"] == 1
    with pytest.raises(ValueError, match="incompatible"):
        smoke.load_evidence(evidence, {**header, "schema_version": 2})
    smoke.benchmark.append_event(evidence, {"kind": "start", "index": 0})
    with pytest.raises(ValueError, match="invalid smoke start"):
        smoke.load_evidence(evidence, header)


def test_strict_raw_refusal_still_measures_tolerant_accepted_fallback():
    raw = "SETTING: cave ACTOR: foxes ACTION: carry lantern MAGIC: ribbon"
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            },
        )

    with httpx.Client(
        base_url="http://localhost", transport=httpx.MockTransport(respond)
    ) as client:
        result, contracts = smoke.run_case(
            client,
            smoke.PASSIVE_CONTROL,
            6,
            model="resident",
            style="watercolor",
            seed=90407,
        )
    assert len(calls) == 1
    assert result.raw_schema_valid is False
    assert result.accepted_valid and result.candidate_valid and result.fallback
    assert result.compiler_proof is False
    assert result.accepted_sha256 == result.candidate_sha256
    assert contracts["accepted_master_prompt"] == contracts["candidate_master_prompt"]
