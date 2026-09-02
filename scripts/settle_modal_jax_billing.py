#!/usr/bin/env python3
"""Settle one JAX attempt from a freshly captured Modal billing report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from infra.gcp.jax.modal_reconciliation import (  # noqa: E402
    settle_attempt_from_modal_cli,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--provider-app-id", required=True)
    parser.add_argument("--provider-app-description", required=True)
    args = parser.parse_args()

    environment = os.environ.copy()
    executable_directory = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = os.pathsep.join(
        part
        for part in (executable_directory, environment.get("PATH", ""))
        if part
    )
    settled = settle_attempt_from_modal_cli(
        args.ledger,
        attempt_id=args.attempt_id,
        provider_app_id=args.provider_app_id,
        provider_app_description=args.provider_app_description,
        environment=environment,
    )
    settlement = settled["billing"]["settlements"][0]
    print(
        json.dumps(
            {
                "attempt_id": settled["attempt_id"],
                "billing_report_artifact": settlement["billing_report_artifact"],
                "billing_report_sha256": settlement["billing_report_sha256"],
                "provider_app_cost_usd": settlement["provider_app_cost_usd"],
                "provider_app_id": settlement["provider_app_id"],
                "status": settlement["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
