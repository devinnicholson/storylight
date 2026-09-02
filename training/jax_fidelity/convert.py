"""Plan or explicitly execute one pinned checkpoint-conversion stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .checkpoint_evidence import parse_max_kl_divergence
from .commands import (
    build_hf_to_maxtext_command,
    build_logit_check_command,
    build_maxtext_to_hf_command,
    shell_join,
)
from .configuration import load_config
from .integrity import (
    artifact_manifest,
    canonical_json_bytes,
    sha256_file,
    verify_conversion_manifest,
)
from .manifests import complete_run, stable_run_id, start_run
from .runtime import (
    ExecutionRefused,
    approval_token,
    require_approval,
    run_checked,
    run_checked_capture,
    validate_maxtext_checkout,
)


def _write_once_json(path: Path, document: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def _write_once_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("hf-to-maxtext", "maxtext-to-hf", "logit-check"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--hf-checkpoint", type=Path)
    parser.add_argument("--maxtext-checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--maxtext-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.direction == "hf-to-maxtext":
        if args.hf_checkpoint is None or args.output_directory is None:
            parser.error("hf-to-maxtext requires --hf-checkpoint and --output-directory")
        command = build_hf_to_maxtext_command(
            config, hf_checkpoint=args.hf_checkpoint, output_directory=args.output_directory
        )
        artifact_roots = {"hf_checkpoint": args.hf_checkpoint}
    elif args.direction == "maxtext-to-hf":
        if (
            args.base_checkpoint is None
            or args.adapter_checkpoint is None
            or args.hf_checkpoint is None
            or args.output_directory is None
        ):
            parser.error(
                "maxtext-to-hf requires --base-checkpoint, --adapter-checkpoint, "
                "--hf-checkpoint, and --output-directory"
            )
        command = build_maxtext_to_hf_command(
            config,
            base_checkpoint=args.base_checkpoint,
            lora_checkpoint=args.adapter_checkpoint,
            hf_tokenizer_checkpoint=args.hf_checkpoint,
            output_directory=args.output_directory,
        )
        artifact_roots = {
            "adapter_checkpoint": args.adapter_checkpoint,
            "base_checkpoint": args.base_checkpoint,
            "hf_checkpoint": args.hf_checkpoint,
        }
    else:
        if args.maxtext_checkpoint is None or args.hf_checkpoint is None:
            parser.error("logit-check requires --maxtext-checkpoint and --hf-checkpoint")
        command = build_logit_check_command(
            config,
            maxtext_checkpoint=args.maxtext_checkpoint,
            hf_checkpoint=args.hf_checkpoint,
            adapter_checkpoint=args.adapter_checkpoint,
        )
        artifact_roots = {
            "hf_checkpoint": args.hf_checkpoint,
            "maxtext_checkpoint": args.maxtext_checkpoint,
        }
        if args.adapter_checkpoint is not None:
            artifact_roots["adapter_checkpoint"] = args.adapter_checkpoint

    verify_conversion_manifest(
        args.input_manifest,
        expected_manifest_sha256=args.input_manifest_sha256,
        artifact_roots=artifact_roots,
    )

    run_id = stable_run_id(
        stage=args.direction,
        config_sha256=config.sha256,
        dataset_manifest_sha256=args.input_manifest_sha256,
    )
    token = approval_token(
        stage=args.direction,
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=args.input_manifest_sha256,
    )
    if args.direction != "logit-check" and args.output_directory.exists():
        raise SystemExit("conversion output directory already exists; outputs are write-once")
    start_run(
        args.run_directory,
        run_id=run_id,
        stage=args.direction,
        config_sha256=config.sha256,
        dataset_manifest_sha256=args.input_manifest_sha256,
        command=command,
        metadata={
            "conversion_input_manifest_sha256": args.input_manifest_sha256,
            "direction": args.direction,
        },
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
    try:
        require_approval(token)
    except ExecutionRefused as error:
        complete_run(
            args.run_directory,
            run_id=run_id,
            status="rejected",
            artifacts=[],
            evidence={"error": str(error), "input_manifest_sha256": args.input_manifest_sha256},
        )
        raise
    try:
        checkout = validate_maxtext_checkout(args.maxtext_root, config)
        stage_directory = args.run_directory.resolve() / run_id
        evidence_path = stage_directory / "conversion-evidence.json"
        if args.direction == "logit-check":
            output = run_checked_capture(command, cwd=checkout)
            log_path = args.run_directory.resolve() / run_id / "logit-check.log"
            _write_once_bytes(log_path, output.encode())
            maximum_kl = parse_max_kl_divergence(output)
            evidence: dict[str, object] = {
                "schema_version": "1.0",
                "status": "succeeded",
                "direction": args.direction,
                "input_manifest_sha256": args.input_manifest_sha256,
                "forward_kl_divergence": maximum_kl,
                "maximum_kl_divergence": config.conversion["max_kl_divergence"],
                "comparison": (
                    "adapted-maxtext-vs-merged-hf"
                    if args.adapter_checkpoint is not None
                    else "base-maxtext-vs-base-hf"
                ),
                "log_sha256": sha256_file(log_path),
            }
            artifacts = [evidence_path, log_path]
        else:
            run_checked(command, cwd=checkout)
            normalization_receipt: Path | None = None
            if args.direction == "maxtext-to-hf":
                from .hf_generation_normalization import normalize_generation_config

                raw_output = args.output_directory.with_name(
                    f".{args.output_directory.name}.{run_id}.raw"
                )
                if raw_output.exists() or raw_output.is_symlink():
                    raise RuntimeError("raw MaxText export staging directory already exists")
                os.replace(args.output_directory, raw_output)
                normalization_receipt = stage_directory / "generation-normalization.json"
                normalize_generation_config(
                    source_checkpoint=raw_output,
                    original_checkpoint=args.hf_checkpoint,
                    destination=args.output_directory,
                    receipt_path=normalization_receipt,
                )
                shutil.rmtree(raw_output)
            output_manifest = artifact_manifest(args.output_directory)
            evidence = {
                "schema_version": "1.0",
                "status": "succeeded",
                "direction": args.direction,
                "input_manifest_sha256": args.input_manifest_sha256,
                "output_manifest": output_manifest,
            }
            if normalization_receipt is not None:
                evidence["generation_normalization_receipt_sha256"] = sha256_file(
                    normalization_receipt
                )
            artifacts = sorted(
                path for path in args.output_directory.rglob("*") if path.is_file()
            ) + [evidence_path]
            if normalization_receipt is not None:
                artifacts.append(normalization_receipt)
        _write_once_json(evidence_path, evidence)
    except Exception as error:
        complete_run(
            args.run_directory,
            run_id=run_id,
            status="failed",
            artifacts=[],
            evidence={
                "error_type": type(error).__name__,
                "error": str(error),
                "input_manifest_sha256": args.input_manifest_sha256,
            },
        )
        raise
    complete_run(
        args.run_directory,
        run_id=run_id,
        status="succeeded",
        artifacts=artifacts,
        evidence=evidence,
    )


if __name__ == "__main__":
    main()
