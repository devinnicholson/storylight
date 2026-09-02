"""Validate terminal evidence from the finite development prediction worker."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .integrity import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")
_COMPLETION_FIELDS = {
    "schema_version",
    "status",
    "backend",
    "run_id",
    "candidate_id",
    "input_manifest_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "development_records_sha256",
    "candidate_manifest_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_content_sha256",
    "batch_size",
    "maximum_output_tokens",
    "predictions",
    "split",
    "hidden_evaluated",
    "passages_retained",
    "files",
}
_BOUND_HASHES = (
    "input_manifest_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "development_records_sha256",
    "candidate_manifest_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_content_sha256",
)


class PredictionEvidenceError(ValueError):
    """Prediction completion or output bytes violate their immutable contract."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionEvidenceError("prediction completion is not valid JSON") from error
    if not isinstance(value, dict):
        raise PredictionEvidenceError("prediction completion must contain one JSON object")
    return value


def _file_rows(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    files = document.get("files")
    if not isinstance(files, list):
        raise PredictionEvidenceError("prediction completion has no file declarations")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PredictionEvidenceError("prediction file declaration is malformed")
        name = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            name not in {"intent.json", "predictions.jsonl"}
            or name in rows
            or type(size) is not int
            or size < 1
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise PredictionEvidenceError("prediction file declaration is unsafe")
        rows[name] = row
    if set(rows) != {"intent.json", "predictions.jsonl"}:
        raise PredictionEvidenceError("prediction completion has an incomplete file set")
    return rows


def _verify_predictions(path: Path) -> None:
    record_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise PredictionEvidenceError(
                        f"prediction JSONL line {line_number} is incomplete"
                    )
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or set(value) != {"record_id", "raw"}
                    or not isinstance(value["record_id"], str)
                    or not value["record_id"].startswith("development-")
                    or not isinstance(value["raw"], str)
                    or value["record_id"] in record_ids
                ):
                    raise PredictionEvidenceError(
                        f"prediction JSONL line {line_number} violates the output schema"
                    )
                record_ids.add(value["record_id"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionEvidenceError("predictions are not valid JSONL") from error
    if len(record_ids) != 512:
        raise PredictionEvidenceError("predictions are not the complete development population")


def validate_prediction_completion(
    path: Path | str,
    *,
    expected_sha256: str,
    predictions_path: Path | str,
    expected_predictions_sha256: str,
    expected_candidate_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_development_records_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an exact, byte-verified development-prediction completion."""

    source = Path(path)
    prediction_output = Path(predictions_path)
    if _SHA256.fullmatch(expected_sha256) is None or sha256_file(source) != expected_sha256:
        raise PredictionEvidenceError("prediction completion differs from its trusted SHA-256")
    if (
        _SHA256.fullmatch(expected_predictions_sha256) is None
        or sha256_file(prediction_output) != expected_predictions_sha256
    ):
        raise PredictionEvidenceError("predictions differ from their approved SHA-256")

    document = _json_object(source)
    if (
        set(document) != _COMPLETION_FIELDS
        or document.get("schema_version") != "1.0"
        or document.get("status") != "succeeded"
        or document.get("backend") != "modal-l40s-cuda"
        or not isinstance(document.get("run_id"), str)
        or _RUN_ID.fullmatch(document["run_id"]) is None
        or not isinstance(document.get("candidate_id"), str)
        or _CANDIDATE_ID.fullmatch(document["candidate_id"]) is None
        or document.get("split") != "development"
        or document.get("predictions") != 512
        or document.get("hidden_evaluated") is not False
        or document.get("passages_retained") is not False
        or type(document.get("batch_size")) is not int
        or not 1 <= document["batch_size"] <= 16
        or type(document.get("maximum_output_tokens")) is not int
        or not 1 <= document["maximum_output_tokens"] <= 256
        or any(
            not isinstance(document.get(name), str) or _SHA256.fullmatch(document[name]) is None
            for name in _BOUND_HASHES
        )
    ):
        raise PredictionEvidenceError("prediction completion identity contract changed")

    expected_bindings = {
        "candidate_id": expected_candidate_id,
        "config_sha256": expected_config_sha256,
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "development_records_sha256": expected_development_records_sha256,
    }
    for name, expected in expected_bindings.items():
        if expected is not None and document.get(name) != expected:
            raise PredictionEvidenceError(f"prediction completion has another {name}")

    rows = _file_rows(document)
    prediction_row = rows["predictions.jsonl"]
    if (
        prediction_row["sha256"] != expected_predictions_sha256
        or prediction_row["bytes"] != prediction_output.stat().st_size
    ):
        raise PredictionEvidenceError("prediction completion is not bound to prediction bytes")
    _verify_predictions(prediction_output)
    return document
