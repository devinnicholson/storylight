# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bookforge.fidelity_dataset import generate_split
from bookforge.fidelity_evaluation import FIDELITY_EVALUATOR_REVISION
from bookforge.fidelity_schema import DatasetSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_fidelity_graph_targets as benchmark


def test_build_report_never_requests_hidden_and_canonicalizes_split_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[DatasetSplit] = []

    def empty_public_split(split: DatasetSplit):  # type: ignore[no-untyped-def]
        assert split is not DatasetSplit.HIDDEN
        requested.append(split)
        return iter(())

    monkeypatch.setattr(benchmark, "generate_split", empty_public_split)

    report = benchmark.build_report(
        (DatasetSplit.DEVELOPMENT, DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT),
        token_budget=64,
    )

    assert requested == [DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT]
    assert [item["split"] for item in report["splits"]] == ["train", "development"]
    assert report["hidden_evaluated"] is False
    assert report["paid_services_used"] is False
    assert report["tool_revision"] == "scene-facts-public-coverage-v2"
    assert report["evaluator_revision"] == FIDELITY_EVALUATOR_REVISION
    assert report["selected_splits"] == ["train", "development"]
    assert report["target_derivation"] == "deterministic_public_contract_adapter"
    assert report["evaluation_surface"] == "postprocessed_typed_graph"


def test_cli_rejects_hidden_split() -> None:
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--split", "hidden"])

    assert error.value.code == 2


def test_cli_defaults_to_both_public_splits_and_writes_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: list[DatasetSplit] = []

    def empty_public_split(split: DatasetSplit):  # type: ignore[no-untyped-def]
        requested.append(split)
        return iter(())

    monkeypatch.setattr(benchmark, "generate_split", empty_public_split)

    assert benchmark.main([]) == 0
    report = json.loads(capsys.readouterr().out)

    assert requested == [DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT]
    assert report["total"] == 0
    assert report["hidden_evaluated"] is False


def test_development_report_is_exact_aggregate_only_and_deterministic(tmp_path) -> None:
    output = tmp_path / "coverage.json"

    assert (
        benchmark.main(
            [
                "--split",
                "development",
                "--token-budget",
                "64",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    first = output.read_text(encoding="utf-8")
    assert (
        benchmark.main(["--split", "development", "--token-budget", "64", "--output", str(output)])
        == 0
    )
    second = output.read_text(encoding="utf-8")
    report = json.loads(first)

    assert first == second
    assert report["total"] == 512
    assert report["eligible"] <= report["total"]
    assert report["exact"] <= report["eligible"]
    assert report["wire_token_estimates"]["count"] == report["eligible"]
    assert report["wire_token_estimates"]["maximum"] <= 64
    assert sum(item["total"] for item in report["categories"]) == report["total"]
    assert report["splits"][0]["split"] == "development"

    sample = next(generate_split(DatasetSplit.DEVELOPMENT))
    assert sample.record_id not in first
    assert sample.passage not in first
    assert all(term not in first for term in sample.privacy_terms)
