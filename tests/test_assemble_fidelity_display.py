from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import assemble_fidelity_display as assembly  # noqa: E402


@pytest.fixture
def captured_story(tmp_path, monkeypatch):
    probe, smoke = assembly.probe, assembly.smoke
    monkeypatch.setattr(probe.httpx, "Client", lambda **kwargs: pytest.fail("assembly used HTTP"))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    args = argparse.Namespace(
        model="resident",
        endpoint="http://127.0.0.1:11435",
        baseline_model=None,
        baseline_endpoint=None,
        **{
            name: tmp_path / f"{name}.json"
            for name in (
                "story_evidence",
                "provenance",
                "baseline_evidence",
                "baseline_summary",
                "baseline_provenance",
                "output",
            )
        },
        **{
            name: private / f"{name}.json"
            for name in (
                "story_private",
                "baseline_private",
                "private_pack",
            )
        },
    )
    provenance = dict(
        engine_sha256="a" * 64,
        environment_sha256="b" * 64,
        snapshot_sha256="c" * 64,
        deployment_sha256="d" * 64,
        code_revision="e" * 40,
    )
    args.provenance.write_text(json.dumps(provenance))
    args.baseline_provenance.write_text(
        json.dumps(
            {
                **provenance,
                "code_revision": assembly.BASELINE_REVISION,
            }
        )
    )
    context_args = argparse.Namespace(
        profile="routed-v1",
        model=args.model,
        endpoint=args.endpoint,
        memory_pid=None,
        timeout=30,
        planning_scope="focal",
        max_output_tokens=64,
    )
    candidate_header = probe.context(
        context_args, probe.benchmark.Provenance(**provenance), "story"
    )
    baseline_header = smoke.context(
        context_args,
        probe.benchmark.Provenance.model_validate_json(args.baseline_provenance.read_text()),
    )
    for path, header in (
        (args.story_evidence, candidate_header),
        (args.baseline_evidence, baseline_header),
    ):
        probe.benchmark.append_event(path, header)
    smoke.create_private_archive(args.story_private, candidate_header)
    smoke.create_private_archive(args.baseline_private, baseline_header)
    slots = (
        ("forest", "two foxes", "carry blue lantern", "golden ribbon"),
        ("forest", "silver fox", "holds blue lantern", "golden ribbon"),
        ("forest", "silver fox", "carries blue lantern", "golden ribbon"),
        ("forest", "silver fox", "holds blue lantern above bridge", "golden ribbon"),
        ("forest", "silver fox", "lifts white feather", "three golden birds"),
        ("forest", "silver fox", "opens wooden basket", "three golden birds"),
        ("cave", "two orange foxes", "carry blue lantern", "ribbon"),
    )
    manifest, cases = smoke.load_manifest(probe.STORY)
    baseline_rows = []
    for index, ((source, seed), values) in enumerate(zip(cases, slots, strict=True)):
        raw = "\n".join(
            f"{key}: {value}"
            for key, value in zip(("SETTING", "ACTOR", "ACTION", "MAGIC"), values, strict=True)
        )
        row = probe.ProbeResult(
            **probe.request_identity(source, index, "candidate", args.model, context_args.profile),
            status="ok",
            generation_complete=True,
            raw_sha256=assembly.digest(raw),
            learned_inference_ms=10,
        )
        row, _ = smoke.construct_case(
            row, raw, source, style=manifest["visual_style"], seed=seed, planning_scope="scene"
        )
        probe.check_result(row, raw, None, source)
        probe.benchmark.append_event(
            args.story_evidence, {"kind": "start", "index": index, "variant": "candidate"}
        )
        probe.benchmark.append_event(args.story_evidence, row.model_dump(mode="json"))
        baseline = smoke.Result.model_validate_json(
            row.model_dump_json(
                exclude={
                    "variant",
                    "checks",
                    "criteria_pass",
                    "prompt_route",
                    "source_sha256",
                    "request_sha256",
                }
            )
        )
        if index in {0, 1, 3}:
            baseline.accepted_valid = False
            baseline.accepted_sha256 = None
            baseline.accepted_privacy_pass = None
        baseline_rows.append(baseline)
        probe.benchmark.append_event(args.baseline_evidence, {"kind": "start", "index": index})
        probe.benchmark.append_event(args.baseline_evidence, baseline.model_dump(mode="json"))
        entry = dict(
            kind="response",
            index=index,
            source_sha256=assembly.digest(source),
            raw=raw,
            raw_sha256=assembly.digest(raw),
            generation_complete=True,
        )
        for path in (args.story_private, args.baseline_private):
            probe.benchmark.append_event(path, entry)
    args.baseline_summary.write_text(
        json.dumps(smoke.aggregate(baseline_header, set(range(7)), baseline_rows))
    )
    monkeypatch.setattr(
        assembly, "BASELINE_SUMMARY_SHA256", assembly.file_hash(args.baseline_summary)
    )
    return args, cases


def cli(args):
    return [
        part
        for name, value in vars(args).items()
        if value is not None
        for part in (f"--{name.replace('_', '-')}", str(value))
    ]


def test_frozen_assembly_has_eight_steps_and_only_three_original_baselines(captured_story):
    args, cases = captured_story
    assert assembly.main(cli(args)) == 0
    batch = json.loads(args.output.read_text())
    pack = json.loads(args.private_pack.read_text())
    assert len(pack["pages"]) == 8 and len(batch["requests"]) == 11
    assert stat.S_IMODE(args.private_pack.stat().st_mode) == 0o600
    accepted = [row for row in batch["requests"] if row["variant"] == "accepted"]
    assert [row["source_page_index"] for row in accepted] == [3, 5, 6]
    assert all(row["display_step_id"] is None and row["graph_sha256"] is None for row in accepted)
    assert [len(page["display_step_ids"]) for page in batch["pages"]] == [1, 1, 1, 1, 2, 2]
    assert [page["baseline_available"] for page in batch["pages"]] == [
        False,
        False,
        True,
        False,
        True,
        True,
    ]
    assert all(source not in args.output.read_text() for source, _ in cases)
    assert all(assembly.digest(row["prompt"]) == row["prompt_sha256"] for row in batch["requests"])
    before, after, first, then = batch["requests"][4:8]
    assert "feather" in before["prompt"] and "feather" not in after["prompt"]
    assert "lifts" not in after["prompt"]
    assert "opens" in first["prompt"] and "lifts" not in first["prompt"]
    assert "lifts" in then["prompt"] and "opens" not in then["prompt"]
    assert "fly" not in first["prompt"] and "fly" in then["prompt"]
    assert assembly.main(cli(args)) == 1


def test_assembly_refuses_changed_capture_or_proof_before_export(captured_story, capsys):
    args, _ = captured_story
    for target, mutate in (
        (args.story_private, lambda rows: rows[1].update(raw="private@example.invalid")),
        (args.story_private, lambda rows: rows[0]["context"].update(planning_scope="focal")),
        (args.story_evidence, lambda rows: rows[2].update(graph_sha256="f" * 64)),
        (args.baseline_private, lambda rows: rows[1].update(raw_sha256="f" * 64)),
        (args.baseline_summary, lambda value: value.update(context_sha256="f" * 64)),
    ):
        original = target.read_text()
        rows = (
            json.loads(original)
            if target == args.baseline_summary
            else [json.loads(line) for line in original.splitlines()]
        )
        mutate(rows)
        target.write_text(
            json.dumps(rows)
            if target == args.baseline_summary
            else "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        assert assembly.main(cli(args)) == 1
        assert not args.output.exists() and not args.private_pack.exists()
        target.write_text(original)
    args.story_private.chmod(0o644)
    assert assembly.main(cli(args)) == 1
    assert "private@example.invalid" not in capsys.readouterr().out
