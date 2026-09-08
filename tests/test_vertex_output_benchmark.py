import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

source = Path(__file__).parents[1] / "scripts/benchmark_vertex_output.py"
spec = importlib.util.spec_from_file_location("vertex_output_benchmark", source)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_pairs_change_only_modality_and_balance_order():
    rows = benchmark.schedule()
    assert len(rows) == 24
    for case in range(6):
        case_rows = [r for r in rows if r["case"] == case]
        assert [r["arm"] for r in case_rows] == (
            ["image", "text_image", "text_image", "image"] if case % 2
            else ["text_image", "image", "image", "text_image"]
        )
        for repeat in range(2):
            pair = [copy.deepcopy(r["payload"]) for r in case_rows if r["repeat"] == repeat]
            assert len(pair) == 2
            for payload in pair:
                payload["generationConfig"].pop("responseModalities")
            assert pair[0] == pair[1]

    masked = benchmark.schedule(field_mask=True)
    assert sum(r["arm"] == "field_mask" for r in masked) == 12
    for case in range(6):
        for repeat in range(2):
            pair = [r["payload"] for r in masked if (r["case"], r["repeat"]) == (case, repeat)]
            assert pair[0] == pair[1]
    assert "thoughtSignature" not in benchmark.FIELD_MASK
    assert "finishReason" in benchmark.FIELD_MASK
    assert "safetyRatings" in benchmark.FIELD_MASK


def test_ambiguous_dispatch_stops_and_existing_run_cannot_replay(tmp_path, monkeypatch):
    calls = []

    async def token():
        return "test-token"

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("ambiguous", request=request)

    factory = httpx.AsyncClient
    monkeypatch.setattr(benchmark, "google_access_token", token)
    monkeypatch.setattr(benchmark.httpx, "AsyncClient", lambda **kwargs: factory(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    out = tmp_path / "experiment"
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(benchmark.run(out))
    assert len(calls) == 1
    journal = [json.loads(line) for line in (out / "journal.jsonl").read_text().splitlines()]
    assert [r["event"] for r in journal] == ["dispatch", "failed"]
    assert "test-token" not in (out / "journal.jsonl").read_text()
    with pytest.raises(FileExistsError):
        asyncio.run(benchmark.run(out))
    assert len(calls) == 1


@pytest.mark.parametrize("reason,thought", [("MAX_TOKENS", False), ("STOP", True)])
def test_partial_or_thought_image_is_not_a_completed_result(reason, thought):
    payload = {"candidates": [{"finishReason": reason, "content": {"parts": [
        {"thought": thought, "inlineData": {"mimeType": "image/png", "data": "dGVzdA=="}},
    ]}}]}
    with pytest.raises(ValueError, match="completed non-thought"):
        benchmark.validate_output(payload)
    evidence, _ = benchmark.response_evidence(payload)
    inline = evidence["candidates"][0]["content"]["parts"][0]["inlineData"]
    assert "data" not in inline
    assert inline["encoded_sha256"] == benchmark.sha(b"dGVzdA==")


def test_filter_gate_requires_complete_equal_images_without_failures(tmp_path):
    rows = benchmark.schedule(field_mask=True)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "field_mask": benchmark.FIELD_MASK, "operations": rows,
    }))
    benchmark.Image.new("RGB", (2, 2), "red").save(tmp_path / "image.png")
    checksum = benchmark.sha((tmp_path / "image.png").read_bytes())
    records = []
    for row in rows:
        metadata = {k: v for k, v in row.items() if k != "payload"}
        filtered = row["arm"] == "field_mask"
        records.extend([
            {**metadata, "event": "dispatch"},
            {**metadata, "event": "complete", "image_file": "image.png",
             "image_sha256": checksum, "artifact_ready_ms": 90 if filtered else 100,
             "response_bytes": 30 if filtered else 100},
        ])

    def summarize(records):
        (tmp_path / "journal.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
        return benchmark.summarize(tmp_path)["promotion_qualified"]

    assert summarize(records)
    assert not summarize(records[:4])
    failed = copy.deepcopy(records)
    failed[-1]["event"] = "failed"
    assert not summarize(failed)
    benchmark.Image.new("RGB", (2, 2), "blue").save(tmp_path / "different.png")
    records[3].update(image_file="different.png", image_sha256=benchmark.sha(
        (tmp_path / "different.png").read_bytes()))
    assert not summarize(records)
