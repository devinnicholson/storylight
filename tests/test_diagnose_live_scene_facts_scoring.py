# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_schema import DatasetSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_live_scene_facts_scoring as diagnostic


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
    assert report["controls"]["typed_target"]["exact"] == 509
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
