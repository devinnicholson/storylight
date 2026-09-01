"""Plan or explicitly execute one pinned MaxText LoRA run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .commands import build_train_command, shell_join
from .configuration import load_config
from .integrity import artifact_binding, sha256_file, validate_dataset_manifest
from .manifests import complete_run, stable_run_id, start_run
from .runtime import (
    ExecutionRefused,
    approval_token,
    require_approval,
    run_checked,
    validate_maxtext_checkout,
)
from .verify_runtime import validate_runtime, validate_runtime_lock


def _maxtext_hardware() -> str:
    platform = os.environ.get("JAX_PLATFORMS", "tpu").split(",", 1)[0].strip().lower()
    if platform == "tpu":
        return "tpu"
    if platform in {"cuda", "gpu"}:
        return "gpu"
    raise SystemExit("JAX_PLATFORMS must select tpu or cuda for MaxText training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--prepared-train-jsonl", type=Path, required=True)
    parser.add_argument("--prepared-train-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--maxtext-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = validate_dataset_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
        required_split_records=config.dataset["required_split_records"],
    )
    if sha256_file(args.prepared_train_jsonl) != args.prepared_train_sha256:
        raise SystemExit("prepared training JSONL SHA-256 mismatch")
    if args.output_directory.exists():
        raise SystemExit("output directory already exists; training outputs are write-once")
    inputs = {
        "prepared_train": {
            "sha256": args.prepared_train_sha256,
            "bytes": args.prepared_train_jsonl.stat().st_size,
        },
        "base_checkpoint": artifact_binding(args.base_checkpoint),
        "tokenizer_checkpoint": artifact_binding(args.hf_tokenizer_checkpoint),
    }
    stage = "lora-smoke" if args.smoke else "lora-train"
    run_id = stable_run_id(
        stage=stage,
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset.manifest_sha256,
    )
    command = build_train_command(
        config,
        maxtext_checkpoint=args.base_checkpoint,
        hf_tokenizer_checkpoint=args.hf_tokenizer_checkpoint,
        prepared_train_jsonl=args.prepared_train_jsonl,
        output_directory=args.output_directory,
        run_name=run_id,
        hardware=_maxtext_hardware(),
        smoke=args.smoke,
    )
    token = approval_token(
        stage=stage,
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=args.prepared_train_sha256,
    )
    start_run(
        args.run_directory,
        run_id=run_id,
        stage=stage,
        config_sha256=config.sha256,
        dataset_manifest_sha256=dataset.manifest_sha256,
        command=command,
        metadata={"inputs": inputs, "smoke": args.smoke},
    )
    plan = {"run_id": run_id, "command": shell_join(command), "approval_token": token}
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return
    try:
        require_approval(token)
    except ExecutionRefused as error:
        complete_run(
            args.run_directory,
            run_id=run_id,
            status="rejected",
            artifacts=[],
            evidence={"error": str(error), "inputs": inputs},
        )
        raise
    try:
        checkout = validate_maxtext_checkout(args.maxtext_root, config)
        validate_runtime()
        runtime_lock = Path(
            os.environ.get("BOOKFORGE_JAX_RUNTIME_LOCK", "/opt/bookforge/runtime.lock.json")
        )
        if not runtime_lock.is_file():
            raise RuntimeError("full installed dependency lock is missing")
        validate_runtime_lock(runtime_lock)
        run_checked(command, cwd=checkout)
        artifacts = sorted(path for path in args.output_directory.rglob("*") if path.is_file())
        if not artifacts:
            raise RuntimeError("MaxText completed without writing checkpoint artifacts")
    except Exception as error:
        complete_run(
            args.run_directory,
            run_id=run_id,
            status="failed",
            artifacts=[],
            evidence={"error_type": type(error).__name__, "error": str(error), "inputs": inputs},
        )
        raise
    complete_run(
        args.run_directory,
        run_id=run_id,
        status="succeeded",
        artifacts=artifacts,
        evidence={
            "inputs": inputs,
            "runtime_lock": {
                "path": str(runtime_lock.resolve()),
                "sha256": sha256_file(runtime_lock),
            },
        },
    )


if __name__ == "__main__":
    main()
