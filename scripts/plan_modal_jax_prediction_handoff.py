#!/usr/bin/env python3
"""Plan one no-reupload Modal prediction handoff without remote mutation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from infra.gcp.jax.prediction_handoff import (  # noqa: E402
    approval_token,
    build_prediction_reference_manifest,
    manifest_sha256,
    prediction_approval_token,
    sha256_file,
)


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required plan input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"required plan input is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"required plan input must be a JSON object: {path}")
    return value


def build_plan(
    *,
    source_input_manifest_path: Path,
    merge_completion_path: Path,
    candidate_manifest_path: Path,
    checkpoint_manifest_path: Path,
    dataset_manifest_path: Path,
    development_records_path: Path,
    prediction_run_id: str,
) -> dict[str, Any]:
    source = _json_object(source_input_manifest_path)
    completion = _json_object(merge_completion_path)
    candidate = _json_object(candidate_manifest_path)
    checkpoint = _json_object(checkpoint_manifest_path)
    source_run_id = source.get("run_id")
    merge_run_id = completion.get("merge_run_id")
    if not isinstance(source_run_id, str) or not isinstance(merge_run_id, str):
        raise ValueError("plan inputs do not declare source and merge run IDs")
    source_sha = sha256_file(source_input_manifest_path)
    completion_sha = sha256_file(merge_completion_path)
    target = build_prediction_reference_manifest(
        source_run_id=source_run_id,
        source_input_manifest_sha256=source_sha,
        source_input_manifest=source,
        merge_run_id=merge_run_id,
        merge_completion_sha256=completion_sha,
        merge_completion=completion,
        candidate_manifest=candidate,
        checkpoint_manifest=checkpoint,
        prediction_run_id=prediction_run_id,
    )
    bindings = target["bindings"]
    if (
        sha256_file(dataset_manifest_path) != bindings["dataset_manifest_sha256"]
        or sha256_file(development_records_path) != bindings["development_records_sha256"]
        or sha256_file(candidate_manifest_path) != bindings["candidate_manifest_sha256"]
        or sha256_file(checkpoint_manifest_path)
        != completion.get("merged_hf_manifest_sha256")
    ):
        raise ValueError("local planning receipts differ from the referenced Modal bytes")
    dataset = _json_object(dataset_manifest_path)
    splits = dataset.get("splits")
    development = splits.get("development") if isinstance(splits, dict) else None
    hidden = splits.get("hidden") if isinstance(splits, dict) else None
    if (
        not isinstance(development, dict)
        or development.get("public") is not True
        or development.get("path") != "development.jsonl"
        or development.get("records") != 512
        or development.get("sha256") != bindings["development_records_sha256"]
        or not isinstance(hidden, dict)
        or hidden.get("public") is not False
        or hidden.get("path") is not None
    ):
        raise ValueError("planning dataset is not the public development population")
    target_sha = manifest_sha256(target)
    token_arguments = {
        "source_run_id": source_run_id,
        "source_input_manifest_sha256": source_sha,
        "merge_run_id": merge_run_id,
        "merge_completion_sha256": completion_sha,
        "prediction_run_id": prediction_run_id,
        "candidate_id": bindings["candidate_id"],
        "candidate_manifest_sha256": bindings["candidate_manifest_sha256"],
        "checkpoint_manifest_sha256": bindings["checkpoint_manifest_sha256"],
        "checkpoint_content_sha256": bindings["checkpoint_content_sha256"],
        "target_manifest_sha256": target_sha,
    }
    return {
        "schema_version": "1.0",
        "status": "plan-only",
        "producer": "bookforge-modal-jax-prediction-handoff-planner",
        **token_arguments,
        "target_manifest": target,
        "approval_token": approval_token(**token_arguments),
        "source_input_volume": "bookforge-jax-fidelity-inputs",
        "source_input_prefix": source_run_id,
        "merge_release_volume": "bookforge-jax-fidelity-release",
        "merge_release_prefix": f"merged/{merge_run_id}",
        "target_input_prefix": f"prediction/{prediction_run_id}",
        "prediction_batch_size": 4,
        "prediction_approval_token": prediction_approval_token(
            run_id=prediction_run_id,
            bindings=bindings,
            input_manifest_sha256=target_sha,
            batch_size=4,
        ),
        "checkpoint_bytes_reuploaded": 0,
        "remote_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input-manifest", type=Path, required=True)
    parser.add_argument("--merge-completion", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--development-records", type=Path, required=True)
    parser.add_argument("--prediction-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite prediction handoff plan: {args.output}")
    plan = build_plan(
        source_input_manifest_path=args.source_input_manifest,
        merge_completion_path=args.merge_completion,
        candidate_manifest_path=args.candidate_manifest,
        checkpoint_manifest_path=args.checkpoint_manifest,
        dataset_manifest_path=args.dataset_manifest,
        development_records_path=args.development_records,
        prediction_run_id=args.prediction_run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
