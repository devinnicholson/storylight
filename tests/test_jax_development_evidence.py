# ruff: noqa: E402
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.jax_fidelity.development_eligibility import (
    DevelopmentEligibilityError,
    validate_development_evidence_chain,
)
from training.jax_fidelity.integrity import canonical_sha256, sha256_file
from training.jax_fidelity.prediction_evidence import (
    PredictionEvidenceError,
    validate_prediction_completion,
)

CANDIDATE_ID = "fidelity-0123456789abcdefabcd"
CONFIG_SHA256 = "1" * 64
DATASET_SHA256 = "2" * 64
RECORDS_SHA256 = "3" * 64
CANDIDATE_MANIFEST_SHA256 = "4" * 64
CHECKPOINT_MANIFEST_SHA256 = "5" * 64
CHECKPOINT_CONTENT_SHA256 = "6" * 64
REPORT_SHA256 = "7" * 64
CONFIG = ROOT / "experiments/jax-fidelity-lab/config.json"


def _write_json(path: Path, document: object) -> str:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _prediction_release(
    tmp_path: Path,
    *,
    config_sha256: str = CONFIG_SHA256,
    records_sha256: str = RECORDS_SHA256,
) -> tuple[Path, str, Path, str]:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"record_id": f"development-{index:04d}", "raw": "{}"}) + "\n"
            for index in range(512)
        ),
        encoding="utf-8",
    )
    predictions_sha256 = sha256_file(predictions)
    completion = tmp_path / "prediction-completion.json"
    completion_sha256 = _write_json(
        completion,
        {
            "schema_version": "1.0",
            "status": "succeeded",
            "backend": "modal-l40s-cuda",
            "run_id": "prediction-test-01",
            "candidate_id": CANDIDATE_ID,
            "input_manifest_sha256": "8" * 64,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": DATASET_SHA256,
            "development_records_sha256": records_sha256,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "checkpoint_content_sha256": CHECKPOINT_CONTENT_SHA256,
            "batch_size": 8,
            "maximum_output_tokens": 64,
            "predictions": 512,
            "split": "development",
            "hidden_evaluated": False,
            "passages_retained": False,
            "files": [
                {"path": "intent.json", "bytes": 1, "sha256": "9" * 64},
                {
                    "path": "predictions.jsonl",
                    "bytes": predictions.stat().st_size,
                    "sha256": predictions_sha256,
                },
            ],
        },
    )
    return predictions, predictions_sha256, completion, completion_sha256


def _evaluation_completion(
    tmp_path: Path,
    *,
    prediction_completion_sha256: str,
    predictions_sha256: str,
) -> tuple[Path, str]:
    path = tmp_path / "evaluation-completion.json"
    digest = _write_json(
        path,
        {
            "schema_version": "1.0",
            "status": "succeeded",
            "stage": "development-evaluation",
            "surface": "raw",
            "candidate_id": CANDIDATE_ID,
            "config_sha256": CONFIG_SHA256,
            "dataset_manifest_sha256": DATASET_SHA256,
            "development_records_sha256": RECORDS_SHA256,
            "predictions_sha256": predictions_sha256,
            "prediction_completion_sha256": prediction_completion_sha256,
            "evaluation_input_sha256": canonical_sha256(
                {
                    "development_records_sha256": RECORDS_SHA256,
                    "predictions_sha256": predictions_sha256,
                    "prediction_completion_sha256": prediction_completion_sha256,
                }
            ),
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "checkpoint_content_sha256": CHECKPOINT_CONTENT_SHA256,
            "report_sha256": REPORT_SHA256,
            "predictions": 512,
            "hidden_evaluated": False,
        },
    )
    return path, digest


def test_prediction_completion_binds_candidate_config_dataset_and_bytes(
    tmp_path: Path,
) -> None:
    predictions, predictions_sha, completion, completion_sha = _prediction_release(tmp_path)

    document = validate_prediction_completion(
        completion,
        expected_sha256=completion_sha,
        predictions_path=predictions,
        expected_predictions_sha256=predictions_sha,
        expected_candidate_id=CANDIDATE_ID,
        expected_config_sha256=CONFIG_SHA256,
        expected_dataset_manifest_sha256=DATASET_SHA256,
        expected_development_records_sha256=RECORDS_SHA256,
    )

    assert document["checkpoint_content_sha256"] == CHECKPOINT_CONTENT_SHA256
    with pytest.raises(PredictionEvidenceError, match="another candidate_id"):
        validate_prediction_completion(
            completion,
            expected_sha256=completion_sha,
            predictions_path=predictions,
            expected_predictions_sha256=predictions_sha,
            expected_candidate_id="fidelity-ffffffffffffffffffff",
        )


def test_development_evidence_chain_rejects_swapped_predictions_or_report(
    tmp_path: Path,
) -> None:
    predictions, predictions_sha, prediction_completion, prediction_completion_sha = (
        _prediction_release(tmp_path)
    )
    evaluation_completion, evaluation_completion_sha = _evaluation_completion(
        tmp_path,
        prediction_completion_sha256=prediction_completion_sha,
        predictions_sha256=predictions_sha,
    )
    arguments = {
        "predictions_path": predictions,
        "predictions_sha256": predictions_sha,
        "prediction_completion_path": prediction_completion,
        "prediction_completion_sha256": prediction_completion_sha,
        "evaluation_completion_path": evaluation_completion,
        "evaluation_completion_sha256": evaluation_completion_sha,
        "candidate_id": CANDIDATE_ID,
        "config_sha256": CONFIG_SHA256,
        "dataset_manifest_sha256": DATASET_SHA256,
        "candidate_report_sha256": REPORT_SHA256,
    }

    prediction, evaluation = validate_development_evidence_chain(**arguments)
    assert prediction["candidate_id"] == evaluation["candidate_id"] == CANDIDATE_ID

    swapped = tmp_path / "swapped-predictions.jsonl"
    swapped.write_bytes(predictions.read_bytes().replace(b'"raw": "{}"', b'"raw": "changed"'))
    with pytest.raises(DevelopmentEligibilityError, match="approved SHA-256"):
        validate_development_evidence_chain(**{**arguments, "predictions_path": swapped})
    with pytest.raises(DevelopmentEligibilityError, match="another prediction"):
        validate_development_evidence_chain(**{**arguments, "candidate_report_sha256": "a" * 64})


def test_evaluation_approval_binds_records_predictions_and_completion(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    records_sha = sha256_file(records)
    config_sha = sha256_file(CONFIG)
    predictions, predictions_sha, prediction_completion, prediction_completion_sha = (
        _prediction_release(
            tmp_path,
            config_sha256=config_sha,
            records_sha256=records_sha,
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.jax_fidelity.evaluate",
            "--config",
            str(CONFIG),
            "--records",
            str(records),
            "--records-sha256",
            records_sha,
            "--predictions",
            str(predictions),
            "--predictions-sha256",
            predictions_sha,
            "--prediction-completion",
            str(prediction_completion),
            "--prediction-completion-sha256",
            prediction_completion_sha,
            "--surface",
            "raw",
            "--output",
            str(tmp_path / "report.json"),
            "--completion",
            str(tmp_path / "evaluation.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    lineage_sha = canonical_sha256(
        {
            "development_records_sha256": records_sha,
            "predictions_sha256": predictions_sha,
            "prediction_completion_sha256": prediction_completion_sha,
        }
    )

    assert plan["run_id"].endswith(lineage_sha[:12])
    assert plan["approval_token"].endswith(lineage_sha)
