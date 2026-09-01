#!/usr/bin/env python3
"""Create a content-addressed, plan-only build recipe for the Vertex JAX image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PROJECT_ID = "your-gcp-project"
REGION = "us-east1"
REPOSITORY = "bookforge-jax"
IMAGE = "trainer"
_DIGEST_FROM = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}(?:\s|$)", re.MULTILINE)
_DIGEST_IMAGE = re.compile(r"^[a-z0-9][a-z0-9./_:-]*@sha256:[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(root: Path, *, builder_image: str) -> dict[str, object]:
    """Bind every tracked source byte used by the Docker context without building it."""

    root = root.resolve()
    dockerfile = root / "training/jax_fidelity/Dockerfile"
    if not dockerfile.is_file() or dockerfile.is_symlink():
        raise ValueError("JAX Dockerfile must be a regular file")
    if _DIGEST_FROM.search(dockerfile.read_text(encoding="utf-8")) is None:
        raise ValueError("JAX Dockerfile base image is not digest-pinned")
    if _DIGEST_IMAGE.fullmatch(builder_image) is None:
        raise ValueError("Cloud Build Docker builder must be digest-pinned")
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    if not tracked:
        raise ValueError("repository has no tracked build inputs")
    rows: list[dict[str, object]] = []
    for relative in sorted(tracked):
        path = root / relative
        if not path.is_file() or path.is_symlink() or ".git" in path.parts:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    source_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tag = source_sha256[:20]
    tagged_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}:{tag}"
    return {
        "schema_version": "1.0",
        "mode": "plan-only",
        "project": PROJECT_ID,
        "region": REGION,
        "source_sha256": source_sha256,
        "source_files": rows,
        "dockerfile": "training/jax_fidelity/Dockerfile",
        "builder_image": builder_image,
        "tagged_image_uri": tagged_uri,
        "automatic_retries": 0,
        "build_command": [
            "gcloud",
            "builds",
            "submit",
            ".",
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            "--config=infra/gcp/jax/image-cloudbuild.yaml",
            f"--substitutions=_IMAGE_URI={tagged_uri},_DOCKER_BUILDER_IMAGE={builder_image}",
        ],
        "digest_resolution_command": [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            tagged_uri,
            f"--project={PROJECT_ID}",
            "--format=value(image_summary.digest)",
        ],
        "promotion_rule": "job_plan.py accepts only tagged_image_uri@sha256:<64 lowercase hex>",
        "remote_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--builder-image", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        build_plan(args.root, builder_image=args.builder_image), indent=2, sort_keys=True
    ) + "\n"
    if args.output:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
