#!/usr/bin/env python3
"""Create a fail-closed full-remat versus no-remat benchmark receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.jax_fidelity.remat_ab import (  # noqa: E402
    RunEvidence,
    compare_remat_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--baseline-release", type=Path, required=True)
    parser.add_argument("--baseline-billing", type=Path, required=True)
    parser.add_argument("--baseline-app-id", required=True)
    parser.add_argument("--candidate-release", type=Path, action="append", required=True)
    parser.add_argument("--candidate-billing", type=Path, action="append", required=True)
    parser.add_argument("--candidate-app-id", action="append", required=True)
    parser.add_argument("--candidate-label", action="append", required=True)
    parser.add_argument("--safety-factor", type=float, default=1.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_count = len(args.candidate_release)
    if not (
        len(args.candidate_billing)
        == len(args.candidate_app_id)
        == len(args.candidate_label)
        == candidate_count
    ):
        parser.error("candidate release, billing, app ID, and label counts must match")

    receipt = compare_remat_runs(
        baseline_config_path=args.baseline_config,
        candidate_config_path=args.candidate_config,
        baseline=RunEvidence(
            label="full-remat-warm-cache",
            release_directory=args.baseline_release,
            billing_path=args.baseline_billing,
            provider_app_id=args.baseline_app_id,
        ),
        candidates=[
            RunEvidence(label, release, billing, app_id)
            for label, release, billing, app_id in zip(
                args.candidate_label,
                args.candidate_release,
                args.candidate_billing,
                args.candidate_app_id,
                strict=True,
            )
        ],
        safety_factor=args.safety_factor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
