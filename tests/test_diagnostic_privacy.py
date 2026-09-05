# ruff: noqa: E402
import json
import sys
from pathlib import Path

import pytest

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_schema import DatasetSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_live_scene_facts_scoring, diagnose_live_scene_training


@pytest.mark.parametrize(
    ("diagnostic", "split"),
    [
        (diagnose_live_scene_facts_scoring, DatasetSplit.DEVELOPMENT),
        (diagnose_live_scene_training, DatasetSplit.TRAIN),
    ],
)
def test_diagnostics_keep_source_private_and_request_only_their_public_split(
    monkeypatch, diagnostic, split
):
    record = next(generate_split(split))
    requested = []

    def one_public_record(selected):
        requested.append(selected)
        assert selected is split
        return iter((record,))

    monkeypatch.setattr(diagnostic, "generate_split", one_public_record)
    report = diagnostic.build_report()
    assert requested == [split]
    assert report["model_requests"] == 0
    assert report["hidden_evaluated"] is False
    encoded = json.dumps(report)
    assert record.record_id not in encoded
    assert record.passage not in encoded
    assert record.target.as_wire() not in encoded
