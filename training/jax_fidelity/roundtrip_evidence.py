"""Record the real HF -> MaxText -> smoke -> HF compatibility seam."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from .checkpoint_evidence import inspect_hf_roundtrip
from .configuration import load_config
from .integrity import canonical_json_bytes, sha256_file
from .roundtrip_smoke import contract_document, validate_roundtrip_evidence

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class RoundtripRecorderError(ValueError):
    """Terminal conversion evidence is incomplete or inconsistent."""


def _approved_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None or sha256_file(path) != expected_sha256:
        raise RoundtripRecorderError(f"{label} differs from its approved SHA-256")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RoundtripRecorderError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RoundtripRecorderError(f"{label} must contain one JSON object")
    return value


def _conversion(
    path: Path,
    expected_sha256: str,
    *,
    direction: str,
) -> dict[str, Any]:
    completion = _approved_json(path, expected_sha256, f"{direction} completion")
    evidence = completion.get("evidence")
    if (
        completion.get("status") != "succeeded"
        or not isinstance(evidence, dict)
        or evidence.get("direction") != direction
        or not completion.get("artifacts")
    ):
        raise RoundtripRecorderError(f"{direction} did not produce successful terminal evidence")
    return completion


def _smoke(path: Path, expected_sha256: str) -> dict[str, Any]:
    completion = _approved_json(path, expected_sha256, "LoRA smoke completion")
    run_id = completion.get("run_id")
    evidence = completion.get("evidence")
    if (
        completion.get("status") != "succeeded"
        or not isinstance(run_id, str)
        or not run_id.startswith("lora-smoke-")
        or not completion.get("artifacts")
        or not isinstance(evidence, dict)
        or not isinstance(evidence.get("runtime_lock"), dict)
    ):
        raise RoundtripRecorderError("the five-step LoRA smoke has no trusted completion")
    return completion


def build_roundtrip_evidence(
    *,
    config_path: Path,
    base_checkpoint: Path,
    exported_checkpoint: Path,
    hf_to_maxtext_completion: Path,
    hf_to_maxtext_completion_sha256: str,
    smoke_completion: Path,
    smoke_completion_sha256: str,
    maxtext_to_hf_completion: Path,
    maxtext_to_hf_completion_sha256: str,
    logit_completion: Path,
    logit_completion_sha256: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    hf_to_maxtext = _conversion(
        hf_to_maxtext_completion,
        hf_to_maxtext_completion_sha256,
        direction="hf-to-maxtext",
    )
    smoke = _smoke(smoke_completion, smoke_completion_sha256)
    maxtext_to_hf = _conversion(
        maxtext_to_hf_completion,
        maxtext_to_hf_completion_sha256,
        direction="maxtext-to-hf",
    )
    logit = _conversion(
        logit_completion,
        logit_completion_sha256,
        direction="logit-check",
    )
    logit_evidence = logit["evidence"]
    divergence = logit_evidence.get("forward_kl_divergence")
    if type(divergence) not in (int, float):
        raise RoundtripRecorderError("logit completion has no measured KL divergence")
    if float(divergence) > config.conversion["max_kl_divergence"]:
        raise RoundtripRecorderError("measured KL divergence exceeds the pinned gate")
    if logit_evidence.get("comparison") != "adapted-maxtext-vs-merged-hf":
        raise RoundtripRecorderError(
            "logit evidence does not compare equivalent adapted model states"
        )

    inspection = inspect_hf_roundtrip(
        config,
        base_checkpoint=base_checkpoint,
        exported_checkpoint=exported_checkpoint,
    )
    exported_manifest = inspection.pop("exported_checkpoint_manifest")
    checks = {
        name: inspection.pop(name)
        for name in (
            "tensor_names",
            "tensor_shapes",
            "tokenizer",
            "special_tokens",
            "ple_weights",
            "kv_sharing",
            "gemma4_metadata",
            "eos_token_ids",
        )
    }
    checks["forward_kl_divergence"] = float(divergence)
    checks["logit_comparison"] = "adapted-maxtext-vs-merged-hf"
    document = {
        "schema_version": "1.0",
        "contract": contract_document(config),
        "checks": checks,
        "inspection": inspection,
        "exported_checkpoint_manifest": exported_manifest,
        "lineage": {
            "hf_to_maxtext_completion_sha256": hf_to_maxtext_completion_sha256,
            "smoke_completion_sha256": smoke_completion_sha256,
            "smoke_run_id": smoke["run_id"],
            "maxtext_to_hf_completion_sha256": maxtext_to_hf_completion_sha256,
            "logit_completion_sha256": logit_completion_sha256,
            "logit_run_id": logit["run_id"],
            "hf_to_maxtext_run_id": hf_to_maxtext["run_id"],
            "maxtext_to_hf_run_id": maxtext_to_hf["run_id"],
        },
    }
    validate_roundtrip_evidence(config, document, exported_checkpoint=exported_checkpoint)
    return document


def _write_once(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--exported-checkpoint", type=Path, required=True)
    for name in (
        "hf-to-maxtext-completion",
        "smoke-completion",
        "maxtext-to-hf-completion",
        "logit-completion",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_roundtrip_evidence(
        config_path=args.config,
        base_checkpoint=args.base_checkpoint,
        exported_checkpoint=args.exported_checkpoint,
        hf_to_maxtext_completion=args.hf_to_maxtext_completion,
        hf_to_maxtext_completion_sha256=args.hf_to_maxtext_completion_sha256,
        smoke_completion=args.smoke_completion,
        smoke_completion_sha256=args.smoke_completion_sha256,
        maxtext_to_hf_completion=args.maxtext_to_hf_completion,
        maxtext_to_hf_completion_sha256=args.maxtext_to_hf_completion_sha256,
        logit_completion=args.logit_completion,
        logit_completion_sha256=args.logit_completion_sha256,
    )
    _write_once(args.output, document)
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
