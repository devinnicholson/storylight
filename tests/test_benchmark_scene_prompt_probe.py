from __future__ import annotations

import json
import stat

import httpx
import pytest

from scripts import benchmark_scene_prompt_probe as probe


@pytest.fixture
def control_run(tmp_path, monkeypatch):
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
    slots = (
        ("quarry", "two badgers", "carry red bucket", "white arch"),
        ("garden", "hedgehog", "pushes cart", "one blue balloon"),
        ("courtyard", "badger", "lifts purple stone", "two white kites"),
        ("workshop", "mouse", "lifts red button", "two blue bells"),
        ("cave", "red badger", "holds blue cup", "ribbon"),
        ("cave", "badger", "holds lantern", "none"),
        ("cave", "badger", "holds lantern", "none"),
        ("cave", "badger", "holds lantern", "two blue birds"),
    )
    cases = probe.load_controls()["cases"]
    raws = {
        case["source"]: "\n".join(
            f"{key}: {value}"
            for key, value in zip(("SETTING", "ACTOR", "ACTION", "MAGIC"), values, strict=True)
        )
        for case, values in zip(cases, slots, strict=True)
    }
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        source = payload["messages"][-1]["content"]
        raw = next((raw for text, raw in raws.items() if text in source), raws[cases[0]["source"]])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 32},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        probe.httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(respond)}),
    )
    construct = probe.smoke.construct_case

    def stable_timing(*args, **kwargs):
        row, contracts = construct(*args, **kwargs)
        row.learned_inference_ms = 10.0
        row.candidate_construction_ms = 10.0
        return row, contracts

    monkeypatch.setattr(probe.smoke, "construct_case", stable_timing)
    args = [
        "--stage",
        "controls",
        "--model",
        "resident",
        "--provenance",
        str(provenance),
        "--evidence",
        str(tmp_path / "controls.jsonl"),
        "--output",
        str(tmp_path / "summary.json"),
    ]
    assert probe.main(args) == 0
    return args, calls, raws


def test_probe_exact_oracles_request_cap_and_resume(control_run, tmp_path):
    args, calls, raws = control_run
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["decision"] == "advance_to_story"
    assert summary["criteria_passed"] == {"accepted": 8, "candidate": 8}
    assert len(calls) == 16
    assert all(call["max_tokens"] == 64 and call["temperature"] == 0 for call in calls)
    journal = tmp_path / "controls.jsonl"
    assert not any(value in journal.read_text() for value in [*raws, *raws.values()])
    assert probe.main(args) == 0
    assert probe.main([*args, "--aggregate-only"]) == 0
    assert len(calls) == 16
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:-1]) + "\n")
    assert probe.main(args) == 0
    assert len(calls) == 16
    assert json.loads((tmp_path / "summary.json").read_text())["decision"] == "reject"


def test_probe_requires_complete_timed_baseline(control_run, tmp_path):
    args, calls, _ = control_run
    journal = tmp_path / "controls.jsonl"
    original = [json.loads(line) for line in journal.read_text().splitlines()]
    baseline = next(
        row for row in original if row.get("kind") == "result" and row["variant"] == "accepted"
    )
    for field, value, counter in (
        ("status", "request_failed", "request_failures"),
        ("generation_complete", False, "incomplete_generations"),
        ("learned_inference_ms", None, "missing_timings"),
    ):
        old = baseline[field]
        baseline[field] = value
        journal.write_text("".join(json.dumps(row) + "\n" for row in original))
        assert probe.main([*args, "--aggregate-only"]) == 0
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["decision"] == "reject" and summary[counter] == 1
        baseline[field] = old
    assert len(calls) == 16


def test_story_is_explicit_and_private_archive_is_bound(control_run, tmp_path):
    args, calls, _ = control_run
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    archive = private / "raw.jsonl"
    story_args = [
        *args[:1],
        "story",
        *args[2:6],
        "--evidence",
        str(tmp_path / "story.jsonl"),
        "--output",
        str(tmp_path / "story.json"),
        "--private-responses",
        str(archive),
    ]
    assert probe.main(story_args) == 1
    assert len(calls) == 16 and not archive.exists()
    story_args += ["--controls-evidence", str(tmp_path / "controls.jsonl")]
    changed = [*story_args]
    changed[3] = "different-model"
    assert probe.main(changed) == 1
    assert len(calls) == 16 and not archive.exists()
    assert probe.main(story_args) == 0
    assert len(calls) == 23
    entries = [json.loads(line) for line in archive.read_text().splitlines()]
    header = json.loads((tmp_path / "story.jsonl").read_text().splitlines()[0])
    assert entries[0] == {"kind": "private_header", "context": header}
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    story = probe.smoke.load_manifest(probe.STORY)[1]
    assert len(entries) == 8
    for index, entry in enumerate(entries[1:]):
        assert entry["index"] == index
        assert entry["source_sha256"] == probe.benchmark.digest(story[index][0])
        assert entry["raw_sha256"] == probe.benchmark.digest(entry["raw"])
        assert "source" not in entry
    assert probe.main(story_args) == 1
    assert len(calls) == 23
