"""Validate HF -> MaxText -> merged HF conversion evidence without model downloads."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import EOS_TOKEN_IDS, ExperimentConfig, load_config
from .integrity import DatasetIntegrityError, verify_artifact_manifest

REQUIRED_BOOLEAN_CHECKS = (
    "tensor_names",
    "tensor_shapes",
    "tokenizer",
    "special_tokens",
    "ple_weights",
    "kv_sharing",
    "gemma4_metadata",
)
REQUIRED_LINEAGE_HASHES = (
    "hf_to_maxtext_completion_sha256",
    "smoke_completion_sha256",
    "maxtext_to_hf_completion_sha256",
    "logit_completion_sha256",
)
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class RoundtripError(ValueError):
    """Checkpoint conversion evidence does not satisfy the release seam."""


def contract_document(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "base_model": {
            "id": config.production["model_id"],
            "revision": config.production["model_revision"],
        },
        "maxtext": {
            "revision": config.versions["maxtext_revision"],
            "model_name": config.production["maxtext_model_name"],
            "scan_layers": False,
            "use_multimodal": False,
        },
        "requirements": {
            "boolean_checks": list(REQUIRED_BOOLEAN_CHECKS),
            "eos_token_ids": list(EOS_TOKEN_IDS),
            "max_forward_kl_divergence": config.conversion["max_kl_divergence"],
            "artifact_manifest_required": True,
        },
    }


def validate_roundtrip_evidence(
    config: ExperimentConfig,
    evidence: Mapping[str, Any],
    *,
    exported_checkpoint: Path | str,
) -> None:
    """Fail unless every architecture and numerical seam has explicit evidence."""

    if evidence.get("schema_version") != "1.0":
        raise RoundtripError("roundtrip evidence schema_version must be 1.0")
    if evidence.get("contract") != contract_document(config):
        raise RoundtripError("roundtrip evidence does not match the pinned contract")
    checks = evidence.get("checks")
    if not isinstance(checks, dict):
        raise RoundtripError("roundtrip evidence must contain checks")
    failed = [name for name in REQUIRED_BOOLEAN_CHECKS if checks.get(name) is not True]
    if failed:
        raise RoundtripError(f"roundtrip checks did not pass: {failed}")
    if checks.get("eos_token_ids") != list(EOS_TOKEN_IDS):
        raise RoundtripError("roundtrip EOS token set changed")
    divergence = checks.get("forward_kl_divergence")
    if type(divergence) not in (int, float):
        raise RoundtripError("forward KL divergence is missing")
    if float(divergence) < 0 or float(divergence) > config.conversion["max_kl_divergence"]:
        raise RoundtripError("forward KL divergence exceeds the 0.03 gate")
    lineage = evidence.get("lineage")
    if not isinstance(lineage, dict):
        raise RoundtripError("roundtrip evidence has no terminal lineage")
    if any(
        not isinstance(lineage.get(name), str)
        or _SHA256.fullmatch(lineage[name]) is None
        for name in REQUIRED_LINEAGE_HASHES
    ):
        raise RoundtripError("roundtrip terminal lineage is incomplete")
    smoke_run_id = lineage.get("smoke_run_id")
    if not isinstance(smoke_run_id, str) or not smoke_run_id.startswith("lora-smoke-"):
        raise RoundtripError("roundtrip evidence has no successful LoRA smoke identity")
    manifest = evidence.get("exported_checkpoint_manifest")
    if not isinstance(manifest, dict):
        raise RoundtripError("exported checkpoint artifact manifest is missing")
    try:
        verify_artifact_manifest(exported_checkpoint, manifest)
    except DatasetIntegrityError as error:
        raise RoundtripError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--exported-checkpoint", type=Path)
    parser.add_argument("--print-contract", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.print_contract:
        print(json.dumps(contract_document(config), indent=2, sort_keys=True))
        return
    if args.evidence is None or args.exported_checkpoint is None:
        raise SystemExit("--evidence and --exported-checkpoint are required for validation")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_roundtrip_evidence(config, evidence, exported_checkpoint=args.exported_checkpoint)
    print("roundtrip evidence passed")


if __name__ == "__main__":
    main()
