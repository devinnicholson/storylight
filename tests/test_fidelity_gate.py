from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from bookforge.fidelity_benchmark import CandidateIdentity, FidelitySummary, RuntimeEvidence
from bookforge.fidelity_gate import build_gate_artifact, write_gate_artifact
from bookforge.fidelity_manifest import FidelityDatasetManifest, sha256_path
from bookforge.fidelity_schema import DatasetSplit

REVIEW_POPULATION_SHA256 = (
    "26e34c0d27fb0ed32008156c2e3663a12222073d60a10123a7080f36a9c11703"
)
REVIEW_RECORD_IDS_SHA256 = (
    "e6612fe1281b5d5d539b4cd30897740c1ea6e1a2805407ac3455708863ee86c6"
)
REVIEW_PASSAGES_SHA256 = (
    "0505e0f6132ec5df69ad8a9f3ec0c5c66c00ebc4431c48db94c1d5137d7dc3d9"
)


def _write(path: Path, document: object) -> str:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return sha256_path(path)


def _summary(
    manifest: FidelityDatasetManifest, *, split: DatasetSplit, exact: float
) -> FidelitySummary:
    population = manifest.splits[split]
    return FidelitySummary(
        surface="raw",
        split=split.value,
        records=population.records,
        record_ids_sha256=population.record_ids_sha256,
        category_record_counts=population.categories,
        schema_valid_rate=1,
        privacy_pass_rate=1,
        semantic_atom_recall=0.99,
        exact_example_pass_rate=exact,
        category_pass_rates={name: 1.0 for name in population.categories},
        counterfactual_pairs=population.pairs,
        counterfactual_sensitivity=0.99,
        unsupported_concept_rate=0,
        pii_leaks=0,
        privacy_term_leaks=0,
        source_echoes=0,
        injection_leaks=0,
        forbidden_hits=0,
    )


def _evaluation(
    *,
    manifest_sha256: str,
    identity: CandidateIdentity,
    summary: FidelitySummary,
    custody_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "story-fidelity-evaluation-v1",
        "split": summary.split,
        "candidate_identity": asdict(identity),
        "dataset_manifest_sha256": manifest_sha256,
        "custody_receipt_sha256": custody_sha256,
        "privacy": {"passages_recorded": False, "outputs_recorded": False},
        "summary": asdict(summary),
    }


def _evidence(tmp_path: Path) -> dict[str, object]:
    manifest_path = Path("datasets/story-fidelity-v1/manifest.json")
    manifest_sha256 = sha256_path(manifest_path)
    manifest = FidelityDatasetManifest.model_validate_json(manifest_path.read_text())
    engine_sha = "a" * 64
    candidate_manifest_path = tmp_path / "candidate.json"
    candidate_manifest_sha256 = _write(
        candidate_manifest_path,
        {
            "candidate_id": "candidate-one",
            "engine_sha256": engine_sha,
            "model_revision": f"sha256:{engine_sha}",
            "training_run_id": "lora-train-gate-test",
            "source_config_sha256": "e" * 64,
            "source_dataset_manifest_sha256": manifest_sha256,
        },
    )
    candidate = CandidateIdentity(
        candidate_id="candidate-one",
        candidate_manifest_sha256=candidate_manifest_sha256,
        engine_sha256=engine_sha,
        model_revision=f"sha256:{engine_sha}",
    )
    baseline = CandidateIdentity(
        candidate_id="accepted-baseline",
        candidate_manifest_sha256="b" * 64,
        engine_sha256="c" * 64,
        model_revision="sha256:" + "c" * 64,
    )
    custody = "d" * 64
    documents = {
        "candidate_hidden": _evaluation(
            manifest_sha256=manifest_sha256,
            identity=candidate,
            summary=_summary(manifest, split=DatasetSplit.HIDDEN, exact=0.97),
            custody_sha256=custody,
        ),
        "baseline_hidden": _evaluation(
            manifest_sha256=manifest_sha256,
            identity=baseline,
            summary=_summary(manifest, split=DatasetSplit.HIDDEN, exact=0.96),
            custody_sha256=custody,
        ),
        "candidate_development": _evaluation(
            manifest_sha256=manifest_sha256,
            identity=candidate,
            summary=_summary(manifest, split=DatasetSplit.DEVELOPMENT, exact=0.97),
            custody_sha256=None,
        ),
        "baseline_development": _evaluation(
            manifest_sha256=manifest_sha256,
            identity=baseline,
            summary=_summary(manifest, split=DatasetSplit.DEVELOPMENT, exact=0.90),
            custody_sha256=None,
        ),
        "runtime": {
            "schema_version": "story-fidelity-runtime-v1",
            "candidate_identity": asdict(candidate),
            "runtime": asdict(
                RuntimeEvidence(
                    maximum_output_tokens=64,
                    p50_seconds=1.5,
                    p95_seconds=1.8,
                    maximum_seconds=2.1,
                    p95_regression_fraction=0.03,
                    unified_memory_peak_gb=3.9,
                    available_memory_mib=900,
                    planner_ready_seconds=75,
                    projector_flow_passed=True,
                    restoration_demonstrated=True,
                )
            ),
        },
        "contest": {
            "schema_version": "story-fidelity-contest-suite-v1",
            "candidate_identity": asdict(candidate),
            "passed": True,
            "cases": 20,
            "semantic_checks": 80,
            "forbidden_checks": 5,
            "case_ids_sha256": "a" * 64,
            "source_report_sha256": "b" * 64,
        },
        "human_review": {
            "schema_version": "story-fidelity-human-review-v1",
            "candidate_identity": asdict(candidate),
            "passed": True,
            "review_protocol": "locked-adversarial-v1",
            "records_reviewed": 25,
            "dataset_manifest_sha256": manifest_sha256,
            "population_contract_sha256": REVIEW_POPULATION_SHA256,
            "record_ids_sha256": REVIEW_RECORD_IDS_SHA256,
            "passage_hashes_sha256": REVIEW_PASSAGES_SHA256,
            "failed_record_ids": [],
            "source_decisions_sha256": "d" * 64,
            "reviewer_attestation": "I_REVIEWED_25_LOCKED_ADVERSARIAL_EXAMPLES",
            "retains_passages": False,
            "retains_model_outputs": False,
        },
        "jetson_shadow": {
            "schema_version": "1.0",
            "stage": "jetson-shadow",
            "producer": "bookforge-jetson-shadow-recorder",
            "run_id": "fidelity-gate-test",
            "training_run_id": "lora-train-gate-test",
            "config_sha256": "e" * 64,
            "dataset_manifest_sha256": manifest_sha256,
            "status": "succeeded",
            "inputs": {"int4-export": "f" * 64},
            "candidate_id": candidate.candidate_id,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "candidate_identity": asdict(candidate),
            "shadow_status": "passed",
            "evidence_sha256": {
                "runtime": "",
                "candidate_manifest": candidate_manifest_sha256,
                "hidden_summary": "0" * 64,
            },
        },
    }
    documents["jetson_shadow"]["evidence_sha256"]["runtime"] = _write(
        tmp_path / "runtime.json", documents["runtime"]
    )
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name, document in documents.items():
        if name == "jetson_shadow":
            continue
        paths[name] = tmp_path / f"{name}.json"
        digests[name] = _write(paths[name], document)
    documents["jetson_shadow"]["evidence_sha256"]["hidden_summary"] = digests[
        "candidate_hidden"
    ]
    paths["jetson_shadow"] = tmp_path / "jetson_shadow.json"
    digests["jetson_shadow"] = _write(
        paths["jetson_shadow"], documents["jetson_shadow"]
    )
    return {
        "run_id": "fidelity-gate-test",
        "config_sha256": "e" * 64,
        "dataset_manifest_path": manifest_path,
        "dataset_manifest_sha256": manifest_sha256,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        **{f"{name}_path": path for name, path in paths.items()},
        **{f"{name}_sha256": digest for name, digest in digests.items()},
    }


def test_trusted_gate_reconstructs_decision_from_separate_populations(tmp_path: Path) -> None:
    artifact = build_gate_artifact(**_evidence(tmp_path))  # type: ignore[arg-type]

    assert artifact["producer"] == "bookforge-fidelity-gate-builder"
    assert artifact["training_run_id"] == "lora-train-gate-test"
    assert artifact["status"] == "passed"
    assert artifact["decision"]["passed"] is True  # type: ignore[index]
    assert artifact["inputs"] == {"jetson-shadow": sha256_path(tmp_path / "jetson_shadow.json")}
    assert set(artifact["evidence_sha256"]) == {  # type: ignore[arg-type]
        "baseline_development_summary",
        "baseline_hidden_summary",
        "candidate_development_summary",
        "candidate_hidden_summary",
        "candidate_manifest",
        "contest",
        "dataset_manifest",
        "human_review",
        "runtime",
    }


def test_gate_rejects_shadow_bound_to_another_hidden_summary(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    shadow_path = evidence["jetson_shadow_path"]
    shadow = json.loads(shadow_path.read_text())
    shadow["evidence_sha256"]["hidden_summary"] = "0" * 64
    evidence["jetson_shadow_sha256"] = _write(shadow_path, shadow)

    with pytest.raises(ValueError, match="dependency bindings"):
        build_gate_artifact(**evidence)  # type: ignore[arg-type]


def test_gate_rejects_candidate_summary_from_another_engine(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    path = evidence["candidate_development_path"]
    assert isinstance(path, Path)
    document = json.loads(path.read_text())
    document["candidate_identity"]["engine_sha256"] = "f" * 64
    document["candidate_identity"]["model_revision"] = "sha256:" + "f" * 64
    evidence["candidate_development_sha256"] = _write(path, document)

    with pytest.raises(ValueError, match="another candidate"):
        build_gate_artifact(**evidence)  # type: ignore[arg-type]


def test_gate_artifact_is_write_once(tmp_path: Path) -> None:
    artifact = build_gate_artifact(**_evidence(tmp_path))  # type: ignore[arg-type]
    output = tmp_path / "gate.json"

    digest = write_gate_artifact(output, artifact)

    assert digest == sha256_path(output)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_gate_artifact(output, artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("producer", "untrusted-shadow-builder"),
        ("inputs", {"int4-export": "not-a-sha256"}),
        (
            "evidence_sha256",
            {
                "runtime": "0" * 64,
                "candidate_manifest": "0" * 64,
                "hidden_summary": "0" * 64,
            },
        ),
    ],
)
def test_gate_rejects_untrusted_shadow_bindings(tmp_path: Path, field: str, value: object) -> None:
    evidence = _evidence(tmp_path)
    path = evidence["jetson_shadow_path"]
    assert isinstance(path, Path)
    document = json.loads(path.read_text())
    document[field] = value
    evidence["jetson_shadow_sha256"] = _write(path, document)

    with pytest.raises(ValueError, match="Jetson shadow evidence"):
        build_gate_artifact(**evidence)  # type: ignore[arg-type]
