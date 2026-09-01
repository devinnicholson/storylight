"""Generate deterministic development predictions from one local merged checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from bookforge.tensorrt_slot_client import _slot_messages

from .configuration import load_config
from .integrity import (
    canonical_json_bytes,
    sha256_file,
    validate_dataset_manifest,
    verify_artifact_manifest,
)
from .runtime import approval_token, require_approval


class PredictionError(RuntimeError):
    """The candidate prediction run violated its immutable input contract."""


def _records(path: Path) -> list[FidelityRecord]:
    rows = [
        FidelityRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 512 or {row.split for row in rows} != {DatasetSplit.DEVELOPMENT}:
        raise PredictionError("candidate predictions require all 512 development records")
    return rows


def _checkpoint_manifest(
    checkpoint: Path,
    manifest_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_file(manifest_path) != expected_sha256:
        raise PredictionError("checkpoint manifest differs from its approved SHA-256")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionError("checkpoint manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise PredictionError("checkpoint manifest must contain one JSON object")
    verify_artifact_manifest(checkpoint, manifest)
    return manifest


def _write_once(path: Path, content: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _generate(
    records: list[FidelityRecord],
    *,
    checkpoint: Path,
    batch_size: int,
    max_output_tokens: int,
) -> list[dict[str, str]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise PredictionError("torch and transformers are required for prediction") from error

    if not torch.cuda.is_available():
        raise PredictionError("candidate prediction requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    torch.manual_seed(20260901)
    results: list[dict[str, str]] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                _slot_messages(record.passage),
                tokenize=False,
                add_generation_prompt=True,
            )
            for record in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_output_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[1, 106, 50],
                use_cache=True,
            )
        prompt_width = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results.extend(
            {"record_id": record.record_id, "raw": output.strip()}
            for record, output in zip(batch, outputs, strict=True)
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16:
        parser.error("--batch-size must be between 1 and 16")
    if args.output.exists() or args.completion.exists():
        raise PredictionError("prediction output and completion are write-once")

    config = load_config(args.config)
    dataset = validate_dataset_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
        required_split_records=config.dataset["required_split_records"],
    )
    if sha256_file(args.records) != args.records_sha256:
        raise PredictionError("development record SHA-256 changed")
    records = _records(args.records)
    checkpoint_manifest = _checkpoint_manifest(
        args.checkpoint,
        args.checkpoint_manifest,
        args.checkpoint_manifest_sha256,
    )
    run_id = (
        f"candidate-predict-{config.sha256[:12]}-"
        f"{args.checkpoint_manifest_sha256[:12]}"
    )
    token = approval_token(
        stage="candidate-predict",
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=args.checkpoint_manifest_sha256,
    )
    plan = {
        "schema_version": "1.0",
        "run_id": run_id,
        "records": len(records),
        "records_sha256": args.records_sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
        "batch_size": args.batch_size,
        "maximum_output_tokens": config.production["completion_budget_tokens"],
        "approval_token": token,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return
    require_approval(token)
    predictions = _generate(
        records,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        max_output_tokens=config.production["completion_budget_tokens"],
    )
    output = b"".join(canonical_json_bytes(row) for row in predictions)
    _write_once(args.output, output)
    completion = {
        **plan,
        "approval_token": None,
        "status": "succeeded",
        "predictions_sha256": sha256_file(args.output),
        "predictions": len(predictions),
        "retains_passages": False,
        "retains_development_outputs": True,
    }
    _write_once(args.completion, canonical_json_bytes(completion))


if __name__ == "__main__":
    main()
