from __future__ import annotations

import json
import stat

import httpx
import pytest

from scripts import benchmark_scene_prompt_probe as probe


@pytest.fixture
def control_run(tmp_path, monkeypatch, request):
    profile = getattr(request, "param", "scene-v1")
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
    if profile == "routed-v1":
        slots += (
            ("meadow", "three white mice", "carry purple drum", "silver ribbon"),
            ("harbor", "one brown rabbit", "carry green baskets", "blue arch"),
            ("workshop", "two red otters", "carry white box", "golden kite"),
            ("valley", "otter", "carries purple cup", "two blue birds"),
            ("courtyard", "red rabbit", "carries green box", "silver ribbon"),
            ("meadow", "two yellow goats", "carry blue bag", "silver arch"),
            ("harbor", "two green rabbits", "carry red drum", "white ribbon"),
            ("courtyard", "two black badgers", "carry white shell", "golden ribbon"),
        )
    cases = probe.load_controls(profile)["cases"]
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
    if profile == "routed-v1":
        args += ["--profile", profile]
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


@pytest.mark.parametrize("control_run", ["routed-v1"], indirect=True)
def test_routed_profile_preserves_original_controls_and_binds_all_32_requests(
    control_run, tmp_path
):
    args, calls, _ = control_run
    summary = json.loads((tmp_path / "summary.json").read_text())
    header = summary["header"]
    assert summary["decision"] == "advance_to_story"
    assert summary["criteria_passed"] == {"accepted": 16, "candidate": 16}
    assert header["schema_version"] == 2 and header["profile"] == "routed-v1"
    assert header["max_requests"] == 32 and len(calls) == 32
    assert probe.load_controls("routed-v1")["cases"][:8] == probe.load_controls()["cases"]
    assert header["request_schedule_sha256"] == probe.benchmark.digest(header["request_schedule"])
    assert [probe.benchmark.digest(call) for call in calls] == [
        row["request_sha256"] for row in header["request_schedule"]
    ]
    routed = [row for row in header["request_schedule"] if row["variant"] == "candidate"]
    assert [row["index"] for row in routed if row["prompt_route"] == "accepted_passive"] == [
        8,
        9,
        10,
    ]
    assert probe.main(args) == 0 and len(calls) == 32
    story_args = [*args]
    story_args[1] = "story"
    story_args[story_args.index("--evidence") + 1] = str(tmp_path / "routed-story.jsonl")
    story_args[story_args.index("--output") + 1] = str(tmp_path / "routed-story.json")
    story_args += ["--controls-evidence", str(tmp_path / "controls.jsonl")]
    wrong_profile = [*story_args]
    wrong_profile[wrong_profile.index("--profile") + 1] = "scene-v1"
    assert probe.main(wrong_profile) == 1 and len(calls) == 32
    assert probe.main(story_args) == 0 and len(calls) == 39
    story = probe.smoke.load_manifest(probe.STORY)[1]
    assert calls[32:] == [
        probe.request_payload(source, "candidate", "resident", "routed-v1") for source, _ in story
    ]
    for case in probe.load_controls("routed-v1")["cases"][8:]:
        routed_payload = probe.request_payload(case["source"], "candidate", "resident", "routed-v1")
        variant = "accepted" if case["expected_route"] == "accepted_passive" else "candidate"
        assert routed_payload == probe.request_payload(case["source"], variant, "resident")
    journal = tmp_path / "controls.jsonl"
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    reordered = [entries[0], *entries[3:5], *entries[1:3], *entries[5:]]
    journal.write_text("\n".join(json.dumps(row) for row in reordered) + "\n")
    assert probe.main([*args, "--aggregate-only"]) == 1
    assert len(calls) == 39
    result = next(row for row in entries if row.get("kind") == "result" and row["index"] == 8)
    result["request_sha256"] = "0" * 64
    journal.write_text("\n".join(json.dumps(row) for row in entries) + "\n")
    assert probe.main([*args, "--aggregate-only"]) == 1
    assert len(calls) == 39
