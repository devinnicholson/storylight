"""Fail before training when installed direct dependencies differ from the lock."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path

from .integrity import canonical_json_bytes, canonical_sha256

PACKAGE_NAMES = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "flax": "flax",
    "optax": "optax",
    "orbax_checkpoint": "orbax-checkpoint",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "transformers": "transformers",
    "maxtext_release": "maxtext",
}


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_lock_document() -> dict[str, object]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_name(raw_name)
        previous = packages.setdefault(name, distribution.version)
        if previous != distribution.version:
            raise RuntimeError(f"multiple installed versions found for {name}")
    rows = [{"name": name, "version": packages[name]} for name in sorted(packages)]
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "packages": rows,
        "packages_sha256": canonical_sha256(rows),
    }


def write_runtime_lock(path: Path | str) -> None:
    destination = Path(path)
    encoded = canonical_json_bytes(runtime_lock_document())
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if destination.read_bytes() != encoded:
            raise RuntimeError(f"refusing to replace runtime lock: {destination}") from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def validate_runtime_lock(path: Path | str) -> None:
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = runtime_lock_document()
    if expected != actual:
        raise RuntimeError("full installed dependency set differs from the image runtime lock")


def validate_runtime(versions_path: Path | str | None = None) -> None:
    path = Path(versions_path) if versions_path else Path(__file__).with_name("versions.json")
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual_python = ".".join(platform.python_version_tuple())
    if actual_python != expected["python"]:
        raise RuntimeError(f"Python drift: expected {expected['python']}, found {actual_python}")
    mismatches: list[str] = []
    for key, package in PACKAGE_NAMES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package}=missing")
            continue
        if actual != expected[key]:
            mismatches.append(f"{package}={actual} (expected {expected[key]})")
    if mismatches:
        raise RuntimeError("runtime dependency drift: " + "; ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    validate_runtime()
    if args.write_lock is not None:
        write_runtime_lock(args.write_lock)
    if args.lock is not None:
        validate_runtime_lock(args.lock)
    print("pinned JAX runtime passed")


if __name__ == "__main__":
    main()
