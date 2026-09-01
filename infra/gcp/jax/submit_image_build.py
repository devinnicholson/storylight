#!/usr/bin/env python3
"""Submit one reserved, non-retrying Cloud Build from an exact materialized plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from build_image_plan import approval_token

APPROVAL_ENVIRONMENT = "BOOKFORGE_GCP_JAX_IMAGE_BUILD_APPROVAL"


def _write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def submit(
    *,
    plan_path: Path,
    materialized_context: Path,
    state_directory: Path,
    runner=subprocess.run,
) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("mode") != "plan-only":
        raise ValueError("image build plan is malformed")
    source_sha = plan.get("source_sha256")
    builder = plan.get("builder_image")
    if not isinstance(source_sha, str) or not isinstance(builder, str):
        raise ValueError("image build identity is missing")
    command = plan.get("provider_build_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("provider build command is malformed")
    token = approval_token(source_sha, builder, command)
    if plan.get("approval_token") != token or os.environ.get(APPROVAL_ENVIRONMENT) != token:
        raise RuntimeError("exact one-purpose image build approval is required")
    if not materialized_context.is_dir() or materialized_context.is_symlink():
        raise ValueError("materialized build context must be a regular directory")
    rows = plan.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("image plan has no source files")
    calculated_source_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if calculated_source_sha != source_sha:
        raise ValueError("image plan source hash changed")
    expected: dict[str, tuple[object, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("image source entry is malformed")
        expected[row["path"]] = (row.get("bytes"), row.get("sha256"))
    actual_paths = {
        path.relative_to(materialized_context).as_posix()
        for path in materialized_context.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        raise ValueError("materialized build context file set changed after planning")
    for relative, (expected_bytes, expected_sha) in expected.items():
        path = materialized_context / relative
        if (
            path.is_symlink()
            or path.stat().st_size != expected_bytes
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha
        ):
            raise ValueError(f"materialized build context changed after planning: {relative}")
    intent = state_directory / f"image-{source_sha}.submission-intent.json"
    receipt = state_directory / f"image-{source_sha}.submission-receipt.json"
    if intent.exists() or receipt.exists():
        raise RuntimeError("image source already has build state; retry is forbidden")
    _write_once(
        intent,
        {
            "schema_version": "1.0",
            "producer": "bookforge-gcp-jax-image-submitter",
            "source_sha256": source_sha,
            "status": "submission-intent-recorded",
            "retry_allowed": False,
        },
    )
    rendered = [
        item.replace("{MATERIALIZED_CONTEXT}", str(materialized_context.resolve()))
        for item in command
    ]
    completed = runner(rendered, check=True, capture_output=True, text=True, timeout=1_860)
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or result.get("status") != "SUCCESS":
        raise RuntimeError("Cloud Build did not return an unambiguous success receipt")
    _write_once(receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialized-context", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("image build submitter is inert without --execute")
    print(
        json.dumps(
            submit(
                plan_path=args.plan,
                materialized_context=args.materialized_context,
                state_directory=args.state_directory,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
