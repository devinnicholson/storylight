"""Run the frozen evaluator over separately produced, content-addressed predictions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .commands import shell_join
from .configuration import load_config
from .integrity import canonical_json_bytes, canonical_sha256, sha256_file
from .prediction_evidence import (
    PredictionEvidenceError,
    validate_prediction_completion,
)
from .runtime import approval_token, require_approval, run_checked


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _report(path: Path, *, surface: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("development evaluator did not produce valid JSON") from error
    summary = document.get("summary") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "captured_at", "privacy", "summary"}
        or document.get("schema_version") != "1.0"
        or not isinstance(document.get("captured_at"), str)
        or not isinstance(summary, dict)
        or summary.get("surface") != ("renderer" if surface == "renderer-safe" else surface)
        or summary.get("split") != "development"
        or summary.get("records") != 512
        or document.get("privacy")
        != {"passages_recorded": False, "outputs_recorded": False}
    ):
        raise RuntimeError("development evaluator produced the wrong population or surface")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--prediction-completion", type=Path, required=True)
    parser.add_argument("--prediction-completion-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument(
        "--surface",
        choices=("raw", "postprocessed", "renderer-safe"),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if sha256_file(args.records) != args.records_sha256:
        raise SystemExit("evaluation records SHA-256 mismatch")
    if sha256_file(args.predictions) != args.predictions_sha256:
        raise SystemExit("prediction JSONL SHA-256 mismatch")
    try:
        prediction = validate_prediction_completion(
            args.prediction_completion,
            expected_sha256=args.prediction_completion_sha256,
            predictions_path=args.predictions,
            expected_predictions_sha256=args.predictions_sha256,
            expected_config_sha256=config.sha256,
            expected_development_records_sha256=args.records_sha256,
        )
    except PredictionEvidenceError as error:
        raise SystemExit(str(error)) from error
    command = [
        "python3",
        "-m",
        "bookforge.fidelity_benchmark",
        "--records",
        str(args.records),
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.output),
        "--surface",
        args.surface,
    ]
    evaluation_input_sha256 = canonical_sha256(
        {
            "development_records_sha256": args.records_sha256,
            "predictions_sha256": args.predictions_sha256,
            "prediction_completion_sha256": args.prediction_completion_sha256,
        }
    )
    run_id = f"evaluate-{args.surface}-{config.sha256[:12]}-{evaluation_input_sha256[:12]}"
    token = approval_token(
        stage="evaluate",
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=evaluation_input_sha256,
    )
    print(
        json.dumps(
            {"run_id": run_id, "command": shell_join(command), "approval_token": token},
            indent=2,
            sort_keys=True,
        )
    )
    if not args.execute:
        return
    require_approval(token)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("development report already exists")
    if args.completion.exists() or args.completion.is_symlink():
        raise FileExistsError("development evaluation completion already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bookforge-evaluation-", dir=args.output.parent) as raw:
        temporary_report = Path(raw) / "report.json"
        executed_command = list(command)
        executed_command[executed_command.index("--output") + 1] = str(temporary_report)
        run_checked(executed_command, cwd=Path.cwd())
        _report(temporary_report, surface=args.surface)
        _write_once(args.output, temporary_report.read_bytes())

    completion = {
        "schema_version": "1.0",
        "status": "succeeded",
        "stage": "development-evaluation",
        "surface": "renderer" if args.surface == "renderer-safe" else args.surface,
        "candidate_id": prediction["candidate_id"],
        "config_sha256": prediction["config_sha256"],
        "dataset_manifest_sha256": prediction["dataset_manifest_sha256"],
        "development_records_sha256": prediction["development_records_sha256"],
        "predictions_sha256": args.predictions_sha256,
        "prediction_completion_sha256": args.prediction_completion_sha256,
        "evaluation_input_sha256": evaluation_input_sha256,
        "candidate_manifest_sha256": prediction["candidate_manifest_sha256"],
        "checkpoint_manifest_sha256": prediction["checkpoint_manifest_sha256"],
        "checkpoint_content_sha256": prediction["checkpoint_content_sha256"],
        "report_sha256": sha256_file(args.output),
        "predictions": 512,
        "hidden_evaluated": False,
    }
    _write_once(args.completion, canonical_json_bytes(completion))


if __name__ == "__main__":
    main()
