#!/usr/bin/env python3
"""Build checksum-bound cost or paid-resource closure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from storylight.fidelity_closure_evidence import (  # noqa: E402
    build_cost_reconciliation,
    build_paid_resource_inventory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise ValueError(f"{label} differs from its approved SHA-256")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return document


def _write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cost = subparsers.add_parser("cost")
    cost.add_argument("--run-id", required=True)
    cost.add_argument("--gcp-source", type=Path, required=True)
    cost.add_argument("--gcp-source-sha256", required=True)
    cost.add_argument("--modal-source", type=Path, required=True)
    cost.add_argument("--modal-source-sha256", required=True)
    cost.add_argument("--output", type=Path, required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--run-id", required=True)
    for resource in (
        "vertex-jobs",
        "cloud-run-services",
        "modal-tasks",
        "modal-functions",
    ):
        inventory.add_argument(f"--{resource}-source", type=Path, required=True)
        inventory.add_argument(f"--{resource}-source-sha256", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "cost":
        document = build_cost_reconciliation(
            arguments.run_id,
            {
                "gcp": (
                    _source(
                        arguments.gcp_source,
                        arguments.gcp_source_sha256,
                        "GCP cost source",
                    ),
                    arguments.gcp_source_sha256,
                ),
                "modal": (
                    _source(
                        arguments.modal_source,
                        arguments.modal_source_sha256,
                        "Modal cost source",
                    ),
                    arguments.modal_source_sha256,
                ),
            },
        )
    else:
        sources = {}
        for resource in (
            "vertex_jobs",
            "cloud_run_services",
            "modal_tasks",
            "modal_functions",
        ):
            path = getattr(arguments, f"{resource}_source")
            source_sha256 = getattr(arguments, f"{resource}_source_sha256")
            sources[resource] = (
                _source(path, source_sha256, f"{resource} source snapshot"),
                source_sha256,
            )
        document = build_paid_resource_inventory(arguments.run_id, sources)
    _write_once(arguments.output, document)
    print(json.dumps({"output": str(arguments.output), "sha256": _sha256(arguments.output)}))


if __name__ == "__main__":
    main()
