from __future__ import annotations

import json
import stat
from dataclasses import asdict
from pathlib import Path

import pytest

import bookforge.fidelity_endpoint_evaluation as endpoint_evaluation
from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    PopulationContract,
    summarize_evaluations,
)
from bookforge.fidelity_endpoint_evaluation import (
    _claim_hidden_evaluation,
    _load_records,
    _secure_regular,
    evaluate_endpoint_records,
)
from bookforge.fidelity_evaluation import concept_vocabulary, evaluate_surface


def test_endpoint_evaluation_retains_metrics_but_not_private_text() -> None:
    records = _load_records(Path("tests/fixtures/story-fidelity-smoke.jsonl"))
    by_passage = {record.passage: record.target.as_wire() for record in records}
    vocabulary = concept_vocabulary(records)
    expected = summarize_evaluations(
        [
            evaluate_surface(
                record,
                record.target.as_wire(),
                surface="raw",
                concept_vocabulary=vocabulary,
            )
            for record in records
        ]
    )
    population = PopulationContract(
        split=expected.split,
        records=expected.records,
        pairs=expected.counterfactual_pairs,
        record_ids_sha256=expected.record_ids_sha256,
        category_record_counts=expected.category_record_counts,
    )

    summary = evaluate_endpoint_records(
        records,
        predict=by_passage.__getitem__,
        population=population,
    )

    serialized = json.dumps(asdict(summary))
    assert summary.record_ids_sha256 == population.record_ids_sha256
    assert all(record.passage not in serialized for record in records)
    assert all(record.target.as_wire() not in serialized for record in records)


def test_hidden_evaluation_claim_is_global_per_candidate_and_engine(tmp_path: Path) -> None:
    state_root = tmp_path / "hidden-state"
    identity = CandidateIdentity(
        candidate_id="candidate-one",
        candidate_manifest_sha256="1" * 64,
        engine_sha256="2" * 64,
        model_revision="sha256:" + "2" * 64,
    )
    population = PopulationContract(
        split="hidden",
        records=512,
        pairs=256,
        record_ids_sha256="3" * 64,
        category_record_counts={"attributes": 1},
        content_sha256="4" * 64,
    )

    state = _claim_hidden_evaluation(
        identity=identity,
        population=population,
        plan={"approval": "exact"},
        state_root=state_root,
    )

    assert stat.S_IMODE(state_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((state / "intent.json").stat().st_mode) == 0o600
    with pytest.raises(RuntimeError, match="already consumed"):
        _claim_hidden_evaluation(
            identity=identity,
            population=population,
            plan={"approval": "exact"},
            state_root=state_root,
        )


def test_hidden_evaluation_claim_requires_absolute_state_root() -> None:
    identity = CandidateIdentity(
        candidate_id="candidate-one",
        candidate_manifest_sha256="1" * 64,
        engine_sha256="2" * 64,
        model_revision="sha256:" + "2" * 64,
    )
    population = PopulationContract(
        split="hidden",
        records=512,
        pairs=256,
        record_ids_sha256="3" * 64,
        category_record_counts={"attributes": 1},
        content_sha256="4" * 64,
    )

    with pytest.raises(ValueError, match="must be absolute"):
        _claim_hidden_evaluation(
            identity=identity,
            population=population,
            plan={"approval": "exact"},
            state_root=Path("relative-hidden-state"),
        )


def test_private_inputs_require_exact_mode_and_reject_symlinks(tmp_path: Path) -> None:
    private = tmp_path / "hidden.jsonl"
    private.write_text("{}\n")
    private.chmod(0o644)
    with pytest.raises(ValueError, match="mode 0600"):
        _secure_regular(private, "private hidden split", exact_mode=0o600)

    private.chmod(0o600)
    link = tmp_path / "hidden-link.jsonl"
    link.symlink_to(private)
    with pytest.raises(ValueError, match="symbolic link"):
        _secure_regular(link, "private hidden split", exact_mode=0o600)


def test_loopback_predictor_bypasses_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"SETTING: cave"}}]}'

    class Opener:
        called = False

        def open(self, *_args: object, **_kwargs: object) -> Response:
            self.called = True
            return Response()

    opener = Opener()
    monkeypatch.setattr(endpoint_evaluation, "_NO_PROXY_OPENER", opener)
    monkeypatch.setattr(
        endpoint_evaluation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("global proxy-aware urlopen was used"),
    )

    predict = endpoint_evaluation._endpoint_predictor("http://127.0.0.1:11436", "llm", 1)

    assert predict("A moth waits.") == "SETTING: cave"
    assert opener.called is True
