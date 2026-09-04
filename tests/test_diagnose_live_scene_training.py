# ruff: noqa: E402
import json
import sys
from itertools import islice
from pathlib import Path

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_schema import DatasetSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_live_scene_training as diagnostic


def test_fixed_training_controls_never_request_other_splits_or_retain_text(monkeypatch):
    requested = []

    def training_only(split):
        requested.append(split)
        assert split is DatasetSplit.TRAIN
        return generate_split(split)

    monkeypatch.setattr(diagnostic, "generate_split", training_only)
    report = diagnostic.build_report()
    assert requested == [DatasetSplit.TRAIN]
    assert report["totals"]["cases"] == len(report["cases"]) == 32
    assert report["model_requests"] == 0
    assert not report["development_evaluated"] and not report["hidden_evaluated"]
    assert not report["learned_accuracy_measured"]
    encoded = json.dumps(report)
    for record in islice(generate_split(DatasetSplit.TRAIN), 32):
        assert record.record_id not in encoded
        assert record.passage not in encoded
        assert record.target.as_wire() not in encoded
    assert (
        sum(
            report["totals"].get("integrated_" + stage, 0)
            for stage in ("graph", "fallback", "refused")
        )
        == 32
    )
