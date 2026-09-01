#!/usr/bin/env python3
"""Build candidate-bound contest or locked human-review promotion evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bookforge.fidelity_benchmark import candidate_identity_from_manifest
from bookforge.fidelity_manifest import sha256_path
from bookforge.fidelity_review_evidence import (
    build_contest_evidence,
    build_human_review_evidence,
)


def json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def candidate_identity(path: Path, expected_sha256: str):
    if sha256_path(path) != expected_sha256:
        raise ValueError("candidate manifest differs from its approved SHA-256")
    return candidate_identity_from_manifest(
        json_object(path), manifest_sha256=expected_sha256
    )


def write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for name in ("contest", "human-review"):
        child = subparsers.add_parser(name)
        child.add_argument("--candidate-manifest", type=Path, required=True)
        child.add_argument("--candidate-manifest-sha256", required=True)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--source-sha256", required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "human-review":
            child.add_argument("--attestation", required=True)
            child.add_argument("--population-contract", type=Path, required=True)
            child.add_argument("--population-contract-sha256", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    identity = candidate_identity(
        args.candidate_manifest, args.candidate_manifest_sha256
    )
    if sha256_path(args.source) != args.source_sha256:
        raise ValueError("source evidence differs from its approved SHA-256")
    source = json_object(args.source)
    if args.command == "contest":
        document = build_contest_evidence(
            identity,
            source,
            source_report_sha256=args.source_sha256,
        )
    else:
        decisions = source.get("decisions")
        if (
            source.get("schema_version") != "story-fidelity-human-review-decisions-v1"
            or source.get("candidate_manifest_sha256")
            != args.candidate_manifest_sha256
            or source.get("population_contract_sha256")
            != args.population_contract_sha256
            or not isinstance(decisions, list)
        ):
            raise ValueError("human-review source must contain a decisions list")
        if sha256_path(args.population_contract) != args.population_contract_sha256:
            raise ValueError("population contract differs from its approved SHA-256")
        document = build_human_review_evidence(
            identity,
            decisions,
            population=json_object(args.population_contract),
            population_contract_sha256=args.population_contract_sha256,
            source_decisions_sha256=args.source_sha256,
            attestation=args.attestation,
        )
    write_once(args.output, document)
    print(json.dumps({"output": str(args.output), "sha256": sha256_path(args.output)}))


if __name__ == "__main__":
    main()
