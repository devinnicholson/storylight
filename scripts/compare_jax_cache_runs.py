#!/usr/bin/env python3
"""Create a fail-closed cold-versus-warm JAX cache benchmark receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.jax_fidelity.cache_ab import compare_cache_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("cold", "warm"):
        parser.add_argument(f"--{prefix}-run", type=Path, required=True)
        parser.add_argument(f"--{prefix}-completion", type=Path, required=True)
        parser.add_argument(f"--{prefix}-cache", type=Path, required=True)
        parser.add_argument(f"--{prefix}-billing", type=Path, required=True)
        parser.add_argument(f"--{prefix}-app-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = compare_cache_runs(
        cold_run_path=args.cold_run,
        cold_completion_path=args.cold_completion,
        cold_cache_path=args.cold_cache,
        cold_billing_path=args.cold_billing,
        cold_app_id=args.cold_app_id,
        warm_run_path=args.warm_run,
        warm_completion_path=args.warm_completion,
        warm_cache_path=args.warm_cache,
        warm_billing_path=args.warm_billing,
        warm_app_id=args.warm_app_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
