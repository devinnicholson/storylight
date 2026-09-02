# ruff: noqa: E402
from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    population_contract_from_manifest,
)
from bookforge.fidelity_schema import DatasetSplit
from training.jax_fidelity import baseline_development as baseline
from training.jax_fidelity.development_eligibility import (
    DevelopmentEligibilityError,
    validate_baseline_evidence_chain,
)
from training.jax_fidelity.integrity import canonical_json_bytes, sha256_file

DATASET = ROOT / "datasets/story-fidelity-v1/manifest.json"
DEVELOPMENT = ROOT / "datasets/story-fidelity-v1/development.jsonl"
DATASET_SHA256 = sha256_file(DATASET)
DEVELOPMENT_SHA256 = sha256_file(DEVELOPMENT)
DATASET_V2 = ROOT / "datasets/story-fidelity-v2/manifest.json"
DATASET_V2_SHA256 = sha256_file(DATASET_V2)


def _accepted_identity(tmp_path: Path) -> tuple[Path, str, Path, str, CandidateIdentity]:
    engine = tmp_path / "accepted/llm.engine"
    engine.parent.mkdir()
    engine.write_bytes(b"accepted TensorRT engine")
    engine_sha256 = sha256_file(engine)
    candidate_id = f"accepted-baseline-{engine_sha256[:20]}"
    manifest = tmp_path / "accepted/candidate.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": "complete",
                "identity_type": "accepted-baseline",
                "candidate_id": candidate_id,
                "model_revision": f"sha256:{engine_sha256}",
                "engine_sha256": engine_sha256,
                "engine_path": "engines/llm/llm.engine",
                "source_dataset_manifest_sha256": DATASET_SHA256,
                "file_count": 1,
                "total_bytes": engine.stat().st_size,
                "files": [
                    {
                        "path": "engines/llm/llm.engine",
                        "bytes": engine.stat().st_size,
                        "sha256": engine_sha256,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = sha256_file(manifest)
    return (
        manifest,
        manifest_sha256,
        engine,
        engine_sha256,
        CandidateIdentity(
            candidate_id=candidate_id,
            candidate_manifest_sha256=manifest_sha256,
            engine_sha256=engine_sha256,
            model_revision=f"sha256:{engine_sha256}",
        ),
    )


def _baseline_report(path: Path, identity: CandidateIdentity) -> str:
    population = population_contract_from_manifest(
        DATASET,
        expected_manifest_sha256=DATASET_SHA256,
        split=DatasetSplit.DEVELOPMENT,
    )
    summary = {
        "surface": "raw",
        "split": "development",
        "records": population.records,
        "record_ids_sha256": population.record_ids_sha256,
        "category_record_counts": dict(population.category_record_counts),
        "schema_valid_rate": 1.0,
        "privacy_pass_rate": 0.74609375,
        "semantic_atom_recall": 0.587109375,
        "exact_example_pass_rate": 0.0,
        "category_pass_rates": {
            category: 0.0 for category in population.category_record_counts
        },
        "counterfactual_pairs": population.pairs,
        "counterfactual_sensitivity": 0.0,
        "unsupported_concept_rate": 0.0,
        "pii_leaks": 0,
        "privacy_term_leaks": 0,
        "source_echoes": 0,
        "injection_leaks": 0,
        "forbidden_hits": 0,
    }
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "story-fidelity-evaluation-v1",
                "split": "development",
                "candidate_identity": asdict(identity),
                "dataset_manifest_sha256": DATASET_SHA256,
                "custody_receipt_sha256": None,
                "privacy": {"passages_recorded": False, "outputs_recorded": False},
                "summary": summary,
            }
        )
    )
    return sha256_file(path)


def _materialize_arguments(tmp_path: Path) -> tuple[dict[str, object], Path, str]:
    manifest, manifest_sha, engine, engine_sha, identity = _accepted_identity(tmp_path)
    source = tmp_path / "source-report.json"
    source_sha = _baseline_report(source, identity)
    arguments: dict[str, object] = {
        "dataset_manifest_path": DATASET,
        "dataset_manifest_sha256": DATASET_SHA256,
        "development_records_path": DEVELOPMENT,
        "development_records_sha256": DEVELOPMENT_SHA256,
        "accepted_manifest_path": manifest,
        "accepted_manifest_sha256": manifest_sha,
        "accepted_engine_path": engine,
        "accepted_engine_sha256": engine_sha,
        "output_directory": tmp_path / "baseline-evidence",
        "source_report_path": source,
        "source_report_sha256": source_sha,
        "base_url": "http://127.0.0.1:11435",
        "model": "llm",
        "timeout_seconds": 12.0,
    }
    return arguments, source, source_sha


def test_recovery_is_immutable_checksum_bound_and_completion_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, source, source_sha = _materialize_arguments(tmp_path)
    token = baseline.approval_token(
        mode="recover",
        accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
        accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
        dataset_manifest_sha256=DATASET_SHA256,
        development_records_sha256=DEVELOPMENT_SHA256,
        source_binding_sha256=source_sha,
    )
    monkeypatch.setenv(baseline.APPROVAL_ENVIRONMENT, token)

    result = baseline.materialize_baseline(**arguments)

    output = Path(arguments["output_directory"])
    report = output / "baseline-development.json"
    completion = output / "completion.json"
    assert report.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(report.stat().st_mode) == 0o400
    assert completion.stat().st_mtime_ns >= report.stat().st_mtime_ns
    assert result["mode"] == "recover"
    assert result["report_sha256"] == source_sha
    validated = baseline.validate_baseline_bundle(
        output,
        expected_completion_sha256=str(result["completion_sha256"]),
        dataset_manifest_path=DATASET,
        dataset_manifest_sha256=DATASET_SHA256,
        development_records_sha256=DEVELOPMENT_SHA256,
        accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
        accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
    )
    assert validated["report_sha256"] == source_sha
    with pytest.raises(FileExistsError, match="overwrite"):
        baseline.materialize_baseline(**arguments)


def test_recovery_rejects_a_redefined_historical_baseline(tmp_path: Path) -> None:
    arguments, source, _ = _materialize_arguments(tmp_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["summary"]["semantic_atom_recall"] = 0.99
    source.write_bytes(canonical_json_bytes(document))

    with pytest.raises(baseline.BaselineDevelopmentError, match="headline metrics"):
        baseline.validate_baseline_report(
            source,
            expected_sha256=sha256_file(source),
            identity=CandidateIdentity(
                candidate_id=f"accepted-baseline-{str(arguments['accepted_engine_sha256'])[:20]}",
                candidate_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
                engine_sha256=str(arguments["accepted_engine_sha256"]),
                model_revision=f"sha256:{arguments['accepted_engine_sha256']}",
            ),
            dataset_manifest_path=DATASET,
            dataset_manifest_sha256=DATASET_SHA256,
        )


def test_baseline_signatures_preserve_v1_and_register_v2_exactly(
    tmp_path: Path,
) -> None:
    assert dict(
        baseline._BASELINE_SIGNATURES_BY_DATASET_MANIFEST_SHA256[DATASET_SHA256]
    ) == {
        "schema_valid_rate": 1.0,
        "semantic_atom_recall": 0.587109375,
        "exact_example_pass_rate": 0.0,
        "privacy_pass_rate": 0.74609375,
    }
    assert dict(
        baseline._BASELINE_SIGNATURES_BY_DATASET_MANIFEST_SHA256[DATASET_V2_SHA256]
    ) == {
        "schema_valid_rate": 1.0,
        "semantic_atom_recall": 0.537109375,
        "exact_example_pass_rate": 0.0,
        "privacy_pass_rate": 1.0,
    }
    assert set(baseline._BASELINE_SIGNATURES_BY_DATASET_MANIFEST_SHA256) == {
        DATASET_SHA256,
        DATASET_V2_SHA256,
    }
    with pytest.raises(
        baseline.BaselineDevelopmentError,
        match="no frozen baseline signature is registered",
    ):
        baseline._headline_signature("d" * 64)

    _, _, _, _, identity = _accepted_identity(tmp_path)
    report = tmp_path / "v2-baseline.json"
    _baseline_report(report, identity)
    document = json.loads(report.read_text(encoding="utf-8"))
    population = population_contract_from_manifest(
        DATASET_V2,
        expected_manifest_sha256=DATASET_V2_SHA256,
        split=DatasetSplit.DEVELOPMENT,
    )
    document["dataset_manifest_sha256"] = DATASET_V2_SHA256
    summary = document["summary"]
    summary["records"] = population.records
    summary["record_ids_sha256"] = population.record_ids_sha256
    summary["category_record_counts"] = dict(population.category_record_counts)
    summary["category_pass_rates"] = {
        category: 0.0 for category in population.category_record_counts
    }
    summary["counterfactual_pairs"] = population.pairs
    report.write_bytes(canonical_json_bytes(document))

    with pytest.raises(
        baseline.BaselineDevelopmentError,
        match="baseline headline metrics differ",
    ):
        baseline.validate_baseline_report(
            report,
            expected_sha256=sha256_file(report),
            identity=identity,
            dataset_manifest_path=DATASET_V2,
            dataset_manifest_sha256=DATASET_V2_SHA256,
        )


def test_materialization_requires_exact_approval_before_writing(tmp_path: Path) -> None:
    arguments, _, _ = _materialize_arguments(tmp_path)

    with pytest.raises(RuntimeError, match="exact baseline development approval"):
        baseline.materialize_baseline(**arguments)
    assert not Path(arguments["output_directory"]).exists()


def test_regeneration_uses_only_the_loopback_endpoint_and_validates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, source, _ = _materialize_arguments(tmp_path)
    arguments["source_report_path"] = None
    arguments["source_report_sha256"] = None
    endpoint_sha = baseline._endpoint_binding("http://127.0.0.1:11435", "llm", 12.0)
    token = baseline.approval_token(
        mode="regenerate",
        accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
        accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
        dataset_manifest_sha256=DATASET_SHA256,
        development_records_sha256=DEVELOPMENT_SHA256,
        source_binding_sha256=endpoint_sha,
    )
    monkeypatch.setenv(baseline.APPROVAL_ENVIRONMENT, token)
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        report = Path(command[command.index("--output") + 1])
        report.write_bytes(source.read_bytes())

    result = baseline.materialize_baseline(
        **arguments,
        runner=runner,
        endpoint_probe=lambda _base_url, _engine: "a" * 64,
    )

    assert result["mode"] == "regenerate"
    assert result["source_binding_sha256"] == endpoint_sha
    assert result["runtime_attestation_sha256"] == "a" * 64
    assert len(commands) == 1
    assert commands[0][commands[0].index("--base-url") + 1] == "http://127.0.0.1:11435"
    assert "--execute" in commands[0]
    with pytest.raises(baseline.BaselineDevelopmentError, match="accepted endpoint"):
        baseline._endpoint_binding("https://example.com", "llm", 12.0)


def test_bundle_rejects_report_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments, _, source_sha = _materialize_arguments(tmp_path)
    token = baseline.approval_token(
        mode="recover",
        accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
        accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
        dataset_manifest_sha256=DATASET_SHA256,
        development_records_sha256=DEVELOPMENT_SHA256,
        source_binding_sha256=source_sha,
    )
    monkeypatch.setenv(baseline.APPROVAL_ENVIRONMENT, token)
    result = baseline.materialize_baseline(**arguments)
    output = Path(arguments["output_directory"])
    os.chmod(output, 0o700)
    report = output / "baseline-development.json"
    os.chmod(report, 0o600)
    report.write_bytes(b"{}\n")

    with pytest.raises(baseline.BaselineDevelopmentError):
        baseline.validate_baseline_bundle(
            output,
            expected_completion_sha256=str(result["completion_sha256"]),
            dataset_manifest_path=DATASET,
            dataset_manifest_sha256=DATASET_SHA256,
            development_records_sha256=DEVELOPMENT_SHA256,
            accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
            accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
        )


def test_bundle_rejects_recovery_source_not_bound_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _, source_sha = _materialize_arguments(tmp_path)
    monkeypatch.setenv(
        baseline.APPROVAL_ENVIRONMENT,
        baseline.approval_token(
            mode="recover",
            accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
            accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
            dataset_manifest_sha256=DATASET_SHA256,
            development_records_sha256=DEVELOPMENT_SHA256,
            source_binding_sha256=source_sha,
        ),
    )
    baseline.materialize_baseline(**arguments)
    output = Path(arguments["output_directory"])
    os.chmod(output, 0o700)
    completion_path = output / "completion.json"
    os.chmod(completion_path, 0o600)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["source_binding_sha256"] = "f" * 64
    completion_path.write_bytes(canonical_json_bytes(completion))

    with pytest.raises(baseline.BaselineDevelopmentError, match="lineage changed"):
        baseline.validate_baseline_bundle(
            output,
            expected_completion_sha256=sha256_file(completion_path),
            dataset_manifest_path=DATASET,
            dataset_manifest_sha256=DATASET_SHA256,
            development_records_sha256=DEVELOPMENT_SHA256,
            accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
            accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
        )


def test_regeneration_removes_partial_evidence_when_runtime_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, source, _ = _materialize_arguments(tmp_path)
    arguments["source_report_path"] = None
    arguments["source_report_sha256"] = None
    endpoint_sha = baseline._endpoint_binding("http://127.0.0.1:11435", "llm", 12.0)
    monkeypatch.setenv(
        baseline.APPROVAL_ENVIRONMENT,
        baseline.approval_token(
            mode="regenerate",
            accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
            accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
            dataset_manifest_sha256=DATASET_SHA256,
            development_records_sha256=DEVELOPMENT_SHA256,
            source_binding_sha256=endpoint_sha,
        ),
    )
    probes = iter(("a" * 64, "b" * 64))

    def runner(command: list[str], **_kwargs: object) -> None:
        Path(command[command.index("--output") + 1]).write_bytes(source.read_bytes())

    with pytest.raises(baseline.BaselineDevelopmentError, match="changed during regeneration"):
        baseline.materialize_baseline(
            **arguments,
            runner=runner,
            endpoint_probe=lambda _base_url, _engine: next(probes),
        )

    assert not Path(arguments["output_directory"]).exists()
    assert not list(tmp_path.glob(".baseline-evidence.partial-*"))


def test_development_eligibility_requires_the_trusted_baseline_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _, source_sha = _materialize_arguments(tmp_path)
    monkeypatch.setenv(
        baseline.APPROVAL_ENVIRONMENT,
        baseline.approval_token(
            mode="recover",
            accepted_manifest_sha256=str(arguments["accepted_manifest_sha256"]),
            accepted_engine_sha256=str(arguments["accepted_engine_sha256"]),
            dataset_manifest_sha256=DATASET_SHA256,
            development_records_sha256=DEVELOPMENT_SHA256,
            source_binding_sha256=source_sha,
        ),
    )
    result = baseline.materialize_baseline(**arguments)
    root = Path(arguments["output_directory"])
    chain = {
        "baseline_report_path": root / "baseline-development.json",
        "baseline_report_sha256": source_sha,
        "baseline_completion_path": root / "completion.json",
        "baseline_completion_sha256": str(result["completion_sha256"]),
        "baseline_candidate_manifest_sha256": str(
            arguments["accepted_manifest_sha256"]
        ),
        "baseline_engine_sha256": str(arguments["accepted_engine_sha256"]),
        "dataset_manifest_path": DATASET,
        "dataset_manifest_sha256": DATASET_SHA256,
        "development_records_sha256": DEVELOPMENT_SHA256,
    }

    completion = validate_baseline_evidence_chain(**chain)
    assert completion["report_sha256"] == source_sha

    unrelated = tmp_path / "unrelated-baseline.json"
    unrelated.write_bytes((root / "baseline-development.json").read_bytes())
    with pytest.raises(DevelopmentEligibilityError, match="declared baseline bundle"):
        validate_baseline_evidence_chain(
            **{**chain, "baseline_report_path": unrelated}
        )
    with pytest.raises(DevelopmentEligibilityError, match="approved SHA-256"):
        validate_baseline_evidence_chain(
            **{**chain, "baseline_completion_sha256": "0" * 64}
        )
