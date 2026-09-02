"""Validate final-only SFT masks and token budgets with the pinned tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .configuration import load_config
from .formatting import maxtext_sft_segments, prompt_contract_sha256
from .integrity import (
    canonical_json_bytes,
    sha256_file,
    verify_artifact_manifest,
)
from .prepare import PAIR_DEDUPLICATION_POLICY


def _approved_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its approved SHA-256")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return document


def validate_prepared_training(
    *,
    config_path: Path,
    prepared_path: Path,
    prepared_sha256: str,
    preparation_manifest_path: Path,
    preparation_manifest_sha256: str,
    tokenizer_path: Path,
    tokenizer_manifest_path: Path,
    tokenizer_manifest_sha256: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    if sha256_file(prepared_path) != prepared_sha256:
        raise ValueError("prepared training bytes differ from their approved SHA-256")
    preparation = _approved_json(
        preparation_manifest_path,
        preparation_manifest_sha256,
        "preparation manifest",
    )
    tokenizer_manifest = _approved_json(
        tokenizer_manifest_path,
        tokenizer_manifest_sha256,
        "tokenizer manifest",
    )
    verify_artifact_manifest(tokenizer_path, tokenizer_manifest)
    if (
        preparation.get("policy") != PAIR_DEDUPLICATION_POLICY
        or preparation.get("prepared_sha256") != prepared_sha256
        or preparation.get("prompt_contract_sha256") != prompt_contract_sha256()
        or preparation.get("assistant_turns_per_record") != 1
        or preparation.get("pair_adjacency_preserved") is not True
    ):
        raise ValueError("preparation manifest does not describe the approved v2 population")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        legacy=False,
        extra_special_tokens={},
    )
    maximum_prompt = 0
    maximum_completion = 0
    maximum_total = 0
    records = 0
    for line_number, line in enumerate(
        prepared_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        row = json.loads(line)
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list):
            raise ValueError(f"prepared record {line_number} has no messages")
        segments = maxtext_sft_segments(tokenizer, messages)
        if [is_prompt for _, is_prompt in segments] != [True, False]:
            raise ValueError(f"prepared record {line_number} does not have one completion")
        counts = [
            len(
                tokenizer(
                    text,
                    truncation=False,
                    max_length=config.production["input_budget_tokens"],
                )["input_ids"]
            )
            for text, _ in segments
        ]
        prompt_tokens, completion_tokens = counts
        total_tokens = sum(counts)
        if (
            prompt_tokens > config.production["input_budget_tokens"]
            or completion_tokens > config.production["completion_budget_tokens"]
            or total_tokens > config.training["max_target_length"]
        ):
            raise ValueError(f"prepared record {line_number} exceeds a token budget")
        maximum_prompt = max(maximum_prompt, prompt_tokens)
        maximum_completion = max(maximum_completion, completion_tokens)
        maximum_total = max(maximum_total, total_tokens)
        records += 1
    if records != preparation.get("prepared_records"):
        raise ValueError("validated records differ from the preparation manifest")
    return {
        "schema_version": "bookforge-jax-prepared-validation-v1",
        "status": "passed",
        "config_sha256": config.sha256,
        "prepared_sha256": prepared_sha256,
        "preparation_manifest_sha256": preparation_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "prompt_contract_sha256": prompt_contract_sha256(),
        "records": records,
        "assistant_turns_per_record": 1,
        "maximum_prompt_tokens": maximum_prompt,
        "maximum_completion_tokens": maximum_completion,
        "maximum_total_tokens": maximum_total,
        "input_budget_tokens": config.production["input_budget_tokens"],
        "completion_budget_tokens": config.production["completion_budget_tokens"],
        "max_target_length": config.training["max_target_length"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--preparation-manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"prepared validation output already exists: {args.output}")
    document = validate_prepared_training(
        config_path=args.config,
        prepared_path=args.prepared,
        prepared_sha256=args.prepared_sha256,
        preparation_manifest_path=args.preparation_manifest,
        preparation_manifest_sha256=args.preparation_manifest_sha256,
        tokenizer_path=args.tokenizer,
        tokenizer_manifest_path=args.tokenizer_manifest,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
