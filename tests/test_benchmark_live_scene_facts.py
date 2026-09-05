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


@pytest.mark.parametrize("late_failure", [False, True])
def test_accepted_first_one_request_reuses_exact_fallback(monkeypatch, records, late_failure):
    requests = []
    constructed = []
    raw = records[0].target.as_wire()

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 31},
            },
        )

    def helper(output, *, source_text):
        constructed.append(output)
        return SimpleNamespace(
            scene_facts={} if late_failure else None, to_live_scene_plan=lambda **kwargs: object()
        )

    def render(*args):
        if late_failure:
            raise ValueError("private compilation detail")
        return "accepted safe contract"

    monkeypatch.setattr(
        "bookforge.tensorrt_slot_client.tensor_accepted_graph_wire_plan", helper, raising=False
    )
    monkeypatch.setattr(benchmark, "safe_slots", lambda *args: "accepted safe contract")
    monkeypatch.setattr(benchmark, "validate_live_scene_plan_privacy", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "renderer_contract", render)
    with httpx.Client(
        base_url="http://localhost", transport=httpx.MockTransport(respond)
    ) as client:
        result = benchmark.run_case(
            client, records[0], 0, model="resident", max_output_tokens=64, mode="accepted_first"
        )
    assert requests == [benchmark.request_payload(records[0].passage, "slots", "resident", 64)]
    assert constructed == [raw]
    assert set(result.surfaces) == set(benchmark.surfaces_for("accepted_first"))
    assert "hybrid_raw" not in result.surfaces
    assert result.fallback is True
    accepted = result.surfaces["accepted_renderer"]
    final = result.surfaces["final_renderer"]
    assert accepted.output_sha256 == final.output_sha256
    assert accepted.output_tokens == final.output_tokens == 31
    assert result.learned_inference_ms == result.surfaces["accepted_raw"].latency_ms
    assert result.graph_construction_ms is not None
    assert result.refusal == ("renderer_refused" if late_failure else "adapter_refused")
    assert "private compilation detail" not in result.model_dump_json()


def test_casewise_atom_comparison_detects_swaps_despite_equal_recall(records):
    value = benchmark.score(records[0], {}, surface="postprocessed", elapsed_ms=1, output_tokens=7)
    accepted = value.model_copy(
        update={"required_atoms": 2, "passed_atoms": 1, "required_atom_passes": [True, False]}
    )
    final = accepted.model_copy(update={"required_atom_passes": [False, True]})
    row = benchmark.CaseEvidence(
        index=0,
        status="ok",
        refusal="none",
        fallback=False,
        surfaces={
            "accepted_raw": accepted,
            "accepted_renderer": accepted,
            "graph_candidate": final,
            "final_renderer": final,
        },
    )
    report = benchmark.aggregate({**header(records), "mode": "accepted_first"}, {0}, [row], records)
    comparison = report["final_vs_accepted"]["overall"]
    assert comparison["atoms_lost"] == comparison["atoms_gained"] == 1
    assert comparison["cases_with_atom_loss"] == comparison["cases_with_atom_gain"] == 1
    assert comparison["exact_regressions"] == 0
    surface = report["surfaces"]["final_renderer"]
    assert surface["required_atoms"] == 2
    assert surface["passed_atoms"] == 1
    assert surface["semantic_atom_recall"] == 0.5
    assert surface["categories"][records[0].categories[0]]["semantic_atom_recall"] == 0.5


@pytest.mark.parametrize("stage", ["baseline", "helper", "renderer"])
def test_accepted_first_privacy_refusal_retains_raw_without_retry(monkeypatch, records, stage):
    calls = []
    raw = records[0].target.as_wire()

    def infer(*args):
        calls.append(args[2])
        return raw, 100.0, 31, True

    def refuse():
        raise benchmark.LiveScenePlannerPrivacyError("private refusal value")

    def baseline(*args):
        if stage == "baseline":
            refuse()
        return "safe"

    def helper(*args, **kwargs):
        if stage == "helper":
            refuse()
        return SimpleNamespace(scene_facts=None, to_live_scene_plan=lambda **k: object())

    def renderer(*args):
        if stage in {"baseline", "renderer"}:
            refuse()
        return "safe"

    monkeypatch.setattr(benchmark, "infer", infer)
    monkeypatch.setattr(benchmark, "safe_slots", baseline)
    monkeypatch.setattr("bookforge.tensorrt_slot_client.tensor_accepted_graph_wire_plan", helper)
    monkeypatch.setattr(benchmark, "validate_live_scene_plan_privacy", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "renderer_contract", renderer)
    result = benchmark.run_case(
        None, records[0], 0, model="resident", max_output_tokens=64, mode="accepted_first"
    )
    assert calls == ["slots"]
    assert result.status == "ok"
    assert result.surfaces["accepted_raw"].output_sha256 == benchmark.digest(raw)
    assert result.surfaces["accepted_raw"].exact_pass is True
    assert (
        result.surfaces["accepted_renderer"].output_sha256
        == result.surfaces["final_renderer"].output_sha256
    )
    assert result.surfaces["accepted_renderer"].schema_valid is (stage != "baseline")
    assert "private refusal value" not in result.model_dump_json()


def test_graph_renderer_scoring_requires_exact_prompt_and_hashes_only_prompt(records):
    from bookforge.tensorrt_slot_client import tensor_accepted_graph_wire_plan

    source = "In a cave, a fox holds a lantern. A ribbon appears."
    wire = "SETTING: cave\nACTOR: fox\nACTION: holds lantern\nMAGIC: ribbon"
    plan = tensor_accepted_graph_wire_plan(wire, source_text=source).to_live_scene_plan(
        context_text=source
    )
    envelope = benchmark.renderer_contract(plan, source)
    assert isinstance(envelope, dict)
    record = records[0].model_copy(update={"passage": source})
    verified = benchmark.score(record, envelope, surface="renderer", elapsed_ms=1, output_tokens=7)
    assert verified.schema_valid
    assert verified.output_sha256 == benchmark.digest(envelope["master_prompt"])
    tampered = {**envelope, "master_prompt": "a dragon replaces the fox"}
    rejected = benchmark.score(record, tampered, surface="renderer", elapsed_ms=1, output_tokens=7)
    assert rejected.schema_valid is False
    assert rejected.exact_pass is False
    assert source not in verified.model_dump_json()


@pytest.mark.parametrize(
    "failure", [None, "incomplete", "no_graphs", "category_loss", "privacy", "p95"]
)
def test_accepted_first_gate_requires_complete_safe_gain_with_category_and_latency_limits(
    records, failure
):
    records = tuple(
        record.model_copy(update={"categories": ("rare" if index == 0 else "common",)})
        for index, record in enumerate(records)
    )
    expected = benchmark.context(
        records,
        provenance(),
        model="resident",
        max_output_tokens=64,
        limit=512,
        endpoint="http://localhost",
        timeout=30,
        mode="accepted_first",
    )
    assert expected["evaluator_revision"] == benchmark.FIDELITY_EVALUATOR_REVISION
    assert expected["gate"] == {
        "revision": "product-fidelity-v1",
        "expected_cases": 512,
        "median_limit_ms": 1500,
        "p95_ratio_limit": 1.10,
    }
    accepted = benchmark.Score(
        schema_valid=True,
        exact_pass=False,
        privacy_pass=True,
        required_atoms=2,
        passed_atoms=1,
        forbidden_count=0,
        unsupported_count=0,
        output_sha256="a" * 64,
        latency_ms=1000,
        output_tokens=31,
        required_atom_passes=[True, False],
    )
    improved = accepted.model_copy(
        update={
            "exact_pass": True,
            "passed_atoms": 2,
            "required_atom_passes": [True, True],
            "latency_ms": 1100,
        }
    )
    rows = []
    for index in range(511 if failure == "incomplete" else 512):
        final = improved
        if failure == "category_loss" and index == 0:
            final = final.model_copy(
                update={
                    "exact_pass": False,
                    "passed_atoms": 0,
                    "required_atom_passes": [False, False],
                }
            )
        elif failure == "privacy" and index == 0:
            final = final.model_copy(update={"privacy_pass": False, "exact_pass": False})
        elif failure == "p95" and index >= 480:
            final = final.model_copy(update={"latency_ms": 1101})
        rows.append(
            benchmark.CaseEvidence(
                index=index,
                status="ok",
                refusal="none",
                fallback=failure == "no_graphs",
                surfaces={
                    "accepted_raw": accepted,
                    "accepted_renderer": accepted,
                    "graph_candidate": improved,
                    "final_renderer": final,
                },
            )
        )
    report = benchmark.aggregate(expected, set(range(512)), rows, records)
    gate = report["gate"]
    assert gate["decision"] == ("advance_to_visual_comparison" if failure is None else "reject")
    assert gate["graph_coverage_denominator"] == 512
    assert gate["appliance_promotion_permitted"] is False
    if failure == "category_loss":
        assert gate["checks"]["strict_required_fact_gain"]
        assert gate["category_regressions"] == ["rare"]
