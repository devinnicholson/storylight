# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import pytest

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_evaluation import evaluate_surface
from bookforge.fidelity_graph_targets import derive_fidelity_graph_target
from bookforge.fidelity_schema import DatasetSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_live_scene_facts_scoring as diagnostic


def test_fixed_training_targets_expose_unbound_actions_without_inventing_graph_facts():
    misses = Counter()
    for record in islice(generate_split(DatasetSplit.TRAIN), 128):
        facts = derive_fidelity_graph_target(record).facts
        assert facts is not None
        score = evaluate_surface(record, facts, surface="postprocessed")
        if score.exact_example_pass:
            continue
        assert [atom.slot for atom in score.expectation_results if not atom.passed] == ["ACTION"]
        category = record.categories[0]
        misses[category] += 1
        subject = facts.subjects[0]
        if category == "containment_relations":
            assert subject.actions == ("watches",)
            assert all(edge.source != subject.ref for edge in facts.relationships)
        else:
            assert category == "salience"
            assert not subject.actions
            assert record.target.action in {"stands in the foreground", "fills the foreground"}
            assert facts.salience[0].source == subject.ref
            assert facts.salience[0].layer.value == "foreground"
    assert misses == {"containment_relations": 8, "salience": 8}


def test_public_controls_and_aggregate_only_evidence(monkeypatch):
    requested = []

    def development_only(split):
        requested.append(split)
        assert split is DatasetSplit.DEVELOPMENT
        return generate_split(split)

    monkeypatch.setattr(diagnostic, "generate_split", development_only)
    report = diagnostic.build_report()
    assert requested == [DatasetSplit.DEVELOPMENT]
    assert report["total"] == 512
    assert report["controls"]["exact_target_raw"]["exact"] == 512
    assert report["controls"]["literal_target_renderer"]["exact"] == 512
    assert report["controls"]["typed_target"]["evaluated"] == 509
    typed = report["controls"]["typed_target"]
    # Bare watching does not identify its object. Foreground placement
    # establishes neither standing posture nor extent across that region.
    assert typed["exact"] == 460
    assert typed["missed_required_slots"] == {"ACTION": 49}
    assert typed["missed_required_categories"] == {"containment_relations": 21, "salience": 28}
    assert (
        report["controls"]["native_compiled_target_renderer"]["evaluated"]
        + sum(report["native_compile_refusals"].values())
        == 509
    )
    assert (
        report["controls"]["live_adapter_from_exact_target"]["evaluated"]
        + sum(report["adapter_refusals"].values())
        == 512
    )
    assert report["model_requests"] == report["network_calls"] == 0
    assert report["hidden_evaluated"] is False
    encoded = json.dumps(report)
    for record in generate_split(DatasetSplit.DEVELOPMENT):
        assert record.record_id not in encoded
        assert record.passage not in encoded
        assert record.target.as_wire() not in encoded


def test_cli_only_allows_public_control_mode():
    with pytest.raises(SystemExit):
        diagnostic.main(["--split", "hidden"])


def test_cli_serializes_stable_report(tmp_path, monkeypatch):
    report = {"schema_version": 1, "counts": {"exact": 512}}
    monkeypatch.setattr(diagnostic, "build_report", lambda: report)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert diagnostic.main(["--output", str(first)]) == 0
    assert diagnostic.main(["--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
