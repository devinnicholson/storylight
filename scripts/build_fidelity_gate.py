#!/usr/bin/env python3
"""Build one checksum-bound Story Fidelity promotion gate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bookforge.fidelity_gate import build_gate_artifact, write_gate_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    for name in (
        "candidate-hidden",
        "baseline-hidden",
        "candidate-development",
        "baseline-development",
        "runtime",
        "contest",
        "human-review",
        "jetson-shadow",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifact = build_gate_artifact(
        run_id=args.run_id,
        config_sha256=args.config_sha256,
        dataset_manifest_path=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        candidate_hidden_path=args.candidate_hidden,
        candidate_hidden_sha256=args.candidate_hidden_sha256,
        baseline_hidden_path=args.baseline_hidden,
        baseline_hidden_sha256=args.baseline_hidden_sha256,
        candidate_development_path=args.candidate_development,
        candidate_development_sha256=args.candidate_development_sha256,
        baseline_development_path=args.baseline_development,
        baseline_development_sha256=args.baseline_development_sha256,
        runtime_path=args.runtime,
        runtime_sha256=args.runtime_sha256,
        contest_path=args.contest,
        contest_sha256=args.contest_sha256,
        human_review_path=args.human_review,
        human_review_sha256=args.human_review_sha256,
        jetson_shadow_path=args.jetson_shadow,
        jetson_shadow_sha256=args.jetson_shadow_sha256,
    )
    digest = write_gate_artifact(args.output, artifact)
    print(json.dumps({"gate_artifact_sha256": digest, **artifact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
