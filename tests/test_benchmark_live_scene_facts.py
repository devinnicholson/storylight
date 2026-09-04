# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_schema import DatasetSplit
from bookforge.tensorrt_slot_client import _slot_messages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_live_scene_facts as benchmark


@pytest.fixture(scope="module")
def records():
    return tuple(generate_split(DatasetSplit.DEVELOPMENT))


def provenance():
    return benchmark.Provenance(
        engine_sha256="a" * 64,
        environment_sha256="b" * 64,
        snapshot_sha256="c" * 64,
        deployment_sha256="d" * 64,
        code_revision="e" * 40,
    )


def header(records, limit=512):
    return benchmark.context(
        records,
        provenance(),
        model="resident",
        max_output_tokens=64,
        limit=limit,
        endpoint="http://127.0.0.1:11435",
        timeout=30,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://cloud.invalid",
        "http://127.0.0.1.evil.invalid",
        "http://secret@localhost",
        "http://localhost/?secret=x",
        "http://localhost/v1",
        "http://localhost/#secret",
        "file://localhost",
    ],
)
def test_endpoint_rejects_nonlocal_and_payload_components(endpoint):
    with pytest.raises(ValueError, match="plain loopback"):
        benchmark.validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:11435", "http://[::1]:11435", "https://localhost:18435/"]
)
def test_loopback_origin(endpoint):
    assert benchmark.validate_endpoint(endpoint) == endpoint.rstrip("/")


def test_provenance_rejects_arbitrary_shell_payload_and_nonfinite_metrics():
    with pytest.raises(ValidationError):
        benchmark.Provenance.model_validate({**provenance().model_dump(), "hostname": "secret"})
    with pytest.raises(ValidationError):
        benchmark.Provenance.model_validate(
            {**provenance().model_dump(), "engine_sha256": "secret"}
        )


def test_exact_accepted_request_and_usage_unavailable(records):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "private"}, "finish_reason": "length"}]}
        )

    with httpx.Client(
        base_url="http://localhost", transport=httpx.MockTransport(respond)
    ) as client:
        text, elapsed, tokens, complete = benchmark.infer(
            client, records[0], "slots", "resident", 64
        )
    assert text == "private"
    assert elapsed >= 0
    assert tokens is None
    assert complete is False
    assert captured == [
        {
            "model": "resident",
            "messages": _slot_messages(records[0].passage),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 64,
            "stream": False,
        }
    ]


def test_score_excludes_private_model_output_and_expectation_values(records):
    private = "SETTING: cave\nACTOR: reader@example.invalid\nACTION: opens drum\nMAGIC: stars"
    value = benchmark.score(records[0], private, surface="raw", elapsed_ms=1, output_tokens=None)
    rendered = value.model_dump_json()
    assert "reader" not in rendered
    assert "cave" not in rendered
    assert value.privacy_pass is False
    assert value.output_sha256 == benchmark.digest(private)


def test_incomplete_generation_never_passes_schema(records):
    value = benchmark.score(
        records[0],
        records[0].target.as_wire(),
        surface="raw",
        elapsed_ms=1,
        output_tokens=64,
        generation_complete=False,
    )
    assert not value.schema_valid
    assert not value.exact_pass


def test_restart_skips_started_and_rejects_changed_context(tmp_path, records):
    path = tmp_path / "evidence.jsonl"
    expected = header(records)
    assert benchmark.load_evidence(path, expected) == (set(), [])
    benchmark.append_event(path, {"kind": "start", "index": 0})
    assert benchmark.load_evidence(path, expected) == ({0}, [])
    with pytest.raises(ValueError, match="incompatible"):
        benchmark.load_evidence(path, {**expected, "max_output_tokens": 65})
    report = benchmark.aggregate(expected, {0}, [], records)
    assert report["complete"] is False
    assert report["interrupted_cases"] == 1


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "start", "index": 0, "private": "secret"},
        {"kind": "start", "index": True},
        {"kind": "start", "index": 512},
        {"kind": "result", "index": 0, "private": "secret"},
    ],
)
def test_resume_rejects_private_or_malformed_evidence(tmp_path, records, event):
    path = tmp_path / "evidence.jsonl"
    expected = header(records)
    benchmark.load_evidence(path, expected)
    benchmark.append_event(path, event)
    with pytest.raises(ValueError, match="incompatible"):
        benchmark.load_evidence(path, expected)


def test_duplicate_or_orphan_result_rejected(tmp_path, records):
    path = tmp_path / "evidence.jsonl"
    expected = header(records)
    benchmark.load_evidence(path, expected)
    row = benchmark.CaseEvidence(
        index=0, status="request_failed", refusal="none", fallback=False, surfaces={}
    )
    benchmark.append_event(path, row.model_dump())
    with pytest.raises(ValueError):
        benchmark.load_evidence(path, expected)


def test_aggregation_reproduces_without_raw_text(tmp_path, records):
    expected = header(records, limit=1)
    path = tmp_path / "evidence.jsonl"
    benchmark.load_evidence(path, expected)
    benchmark.append_event(path, {"kind": "start", "index": 0})
    value = benchmark.score(
        records[0], {}, surface="postprocessed", elapsed_ms=3, output_tokens=None
    )
    row = benchmark.CaseEvidence(
        index=0,
        status="ok",
        refusal="adapter_refused",
        fallback=True,
        surfaces={name: value for name in benchmark.SURFACES},
    )
    benchmark.append_event(path, row.model_dump())
    started, loaded = benchmark.load_evidence(path, expected)
    report = benchmark.aggregate(expected, started, loaded, records)
    assert report == benchmark.aggregate(expected, {0}, [row], records)
    assert report["complete"] is False
    assert report["peak_memory_bytes"] is None
    assert report["thermal_max_millicelsius"] is None
    assert report["thermal_min_millicelsius"] is None
    assert report["surfaces"]["graph_candidate"]["output_tokens_unavailable"] == 1
    assert records[0].passage not in path.read_text()
    assert report["hidden_evaluated"] is False


def test_latency_nearest_rank_and_empty():
    assert benchmark.distribution(list(range(1, 101))) == {
        "count": 100,
        "p50": 50.5,
        "p95": 95,
        "max": 100,
    }
    assert benchmark.distribution([]) == {"count": 0, "p50": None, "p95": None, "max": None}


def test_requests_have_no_automatic_retry_and_do_not_leak_failures(monkeypatch, records):
    monkeypatch.setitem(
        sys.modules,
        "bookforge.live_scene_facts",
        SimpleNamespace(adapt_live_scene_facts=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "bookforge.tensorrt_slot_client.parse_tensor_graph_slots", lambda _: {}, raising=False
    )
    calls = []

    def fail(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-secret", request=request)

    with httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(fail)) as client:
        result = benchmark.run_case(client, records[0], 0, model="resident", max_output_tokens=64)
    assert len(calls) == 1
    assert result.status == "request_failed"
    assert "private-secret" not in result.model_dump_json()


def test_cli_rejects_hidden_split():
    with pytest.raises(SystemExit):
        benchmark.main(["--split", "hidden"])


def test_memory_samples_linux_system_and_process_separately(monkeypatch):
    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 100 kB\nMemAvailable: 40 kB\n"
        if str(path) == "/proc/123/status":
            return "Name: private\nVmRSS: 7 kB\n"
        raise OSError

    monkeypatch.setattr(Path, "read_text", read)
    with benchmark.MemorySampler(123) as sampler:
        sampler._sample()
    assert sampler.system_peak == 60 * 1024
    assert sampler.process_peak == 7 * 1024
    assert sampler.samples >= 3


def test_missing_memory_is_unavailable_not_zero(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(Path, "read_text", missing)
    with benchmark.MemorySampler(None) as sampler:
        pass
    assert sampler.system_peak is None
    assert sampler.process_peak is None
    assert sampler.thermal_min is None
    assert sampler.thermal_max is None
    assert sampler.thermal_samples == 0


def test_thermal_samples_all_zones_excluding_invalid_readings(monkeypatch):
    readings = {
        "cold": "42000",
        "hot": "61000",
        "bad": "private payload",
        "impossible": "999999",
        "missing": None,
    }
    monkeypatch.setattr(Path, "glob", lambda *a: iter(Path(name) for name in readings))

    def read(path):
        value = readings[str(path)]
        if value is None:
            raise OSError
        return value

    monkeypatch.setattr(Path, "read_text", read)
    sampler = benchmark.MemorySampler(None)
    sampler._sample_thermal()
    readings["hot"] = "65000"
    readings["cold"] = "40000"
    sampler._sample_thermal()
    assert sampler.thermal_min == 40000
    assert sampler.thermal_max == 65000
    assert sampler.thermal_samples == 2


def test_thermal_schema_rejects_impossible_values():
    with pytest.raises(ValidationError):
        benchmark.CaseEvidence(
            index=0,
            status="request_failed",
            refusal="none",
            fallback=False,
            surfaces={},
            thermal_max_millicelsius=999999,
        )


def test_aggregate_thermal_extrema_across_cases(records):
    rows = [
        benchmark.CaseEvidence(
            index=index,
            status="request_failed",
            refusal="none",
            fallback=False,
            surfaces={},
            thermal_min_millicelsius=low,
            thermal_max_millicelsius=high,
            thermal_samples=10,
        )
        for index, low, high in [(0, 41000, 59000), (1, 43000, 61000)]
    ]
    report = benchmark.aggregate(header(records), {0, 1}, rows, records)
    assert report["thermal_min_millicelsius"] == 41000
    assert report["thermal_max_millicelsius"] == 61000
    assert report["thermal_samples"] == 20


def test_fallback_sums_measured_planning_latency_and_tokens(monkeypatch, records):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "bookforge.live_scene_facts",
        SimpleNamespace(
            adapt_live_scene_facts=lambda *a, **k: SimpleNamespace(
                facts=None, refusal=SimpleNamespace(value="unsupported_syntax")
            )
        ),
    )
    monkeypatch.setattr(
        "bookforge.tensorrt_slot_client.parse_tensor_graph_slots", lambda _: {}, raising=False
    )
    monkeypatch.setattr(benchmark, "safe_slots", lambda *a: "safe contract")

    def inference(*args):
        calls.append(args[2])
        return records[0].target.as_wire(), 100.0, 10, True

    monkeypatch.setattr(benchmark, "infer", inference)
    result = benchmark.run_case(None, records[0], 0, model="resident", max_output_tokens=64)
    assert calls == ["slots", "hybrid"]
    assert result.fallback is True
    assert result.adapter_refusal_code == "unsupported_syntax"
    assert result.surfaces["final_renderer"].latency_ms >= 200
    assert result.surfaces["final_renderer"].output_tokens == 20
    assert result.surfaces["final_renderer"].output_sha256 == benchmark.digest("safe contract")
