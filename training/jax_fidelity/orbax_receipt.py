"""Discover and bind one exact MaxText Orbax ``items`` checkpoint leaf."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .integrity import artifact_manifest, canonical_json_bytes, verify_artifact_manifest


class OrbaxReceiptError(ValueError):
    """An Orbax output did not contain one unambiguous checkpoint leaf."""


def terminal_checkpoint_step(completed_steps: int) -> int:
    """Map a positive optimizer-step count to MaxText's zero-based checkpoint step."""

    if type(completed_steps) is not int or completed_steps < 1:
        raise OrbaxReceiptError("completed_steps must be a positive integer")
    return completed_steps - 1


def _safe_directory(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir() or path.is_symlink():
        raise OrbaxReceiptError(f"{label} is not a regular directory: {path}")
    for child in resolved.rglob("*"):
        if child.is_symlink():
            raise OrbaxReceiptError(f"{label} contains a symbolic link: {child}")
    return resolved


def discover_orbax_items(root: Path | str, *, expected_step: int) -> Path:
    """Return the only nonempty ``<expected_step>/items`` directory below ``root``.

    MaxText may add run and checkpoint directories around an Orbax checkpoint. The
    step directory and ``items`` leaf are stable. Refusing zero or multiple matches
    prevents a stale checkpoint from being selected by path ordering.
    """

    if type(expected_step) is not int or expected_step < 0:
        raise OrbaxReceiptError("expected_step must be a non-negative integer")
    output_root = _safe_directory(Path(root), label="Orbax output root")
    candidates: list[Path] = []
    for candidate in sorted(output_root.rglob("items")):
        if candidate.parent.name != str(expected_step):
            continue
        safe = _safe_directory(candidate, label="Orbax items leaf")
        if any(path.is_file() for path in safe.rglob("*")):
            candidates.append(safe)
    if len(candidates) != 1:
        relative = [path.relative_to(output_root).as_posix() for path in candidates]
        raise OrbaxReceiptError(
            f"expected one Orbax step {expected_step} items leaf, found {len(candidates)}: "
            f"{relative}"
        )
    return candidates[0]


def orbax_leaf_receipt(
    root: Path | str,
    leaf: Path | str,
    *,
    expected_step: int,
    role: str,
) -> dict[str, Any]:
    """Describe the selected leaf without retaining machine-specific absolute paths."""

    if not role or not role.replace("-", "").isalnum():
        raise OrbaxReceiptError("Orbax receipt role must be a bounded slug")
    output_root = _safe_directory(Path(root), label="Orbax output root")
    selected = _safe_directory(Path(leaf), label="Orbax items leaf")
    try:
        relative = selected.relative_to(output_root)
    except ValueError as error:
        raise OrbaxReceiptError("Orbax items leaf escaped its output root") from error
    if selected.name != "items" or selected.parent.name != str(expected_step):
        raise OrbaxReceiptError("Orbax items leaf does not match the expected step")
    return {
        "schema_version": "1.0",
        "format": "maxtext-orbax-items",
        "role": role,
        "expected_step": expected_step,
        "relative_path": relative.as_posix(),
        "artifact_manifest": artifact_manifest(selected),
    }


def verify_orbax_leaf_receipt(
    root: Path | str,
    receipt: dict[str, Any],
    *,
    expected_step: int,
    role: str,
) -> Path:
    """Resolve and verify a previously recorded receipt beneath ``root``."""

    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("format") != "maxtext-orbax-items"
        or receipt.get("role") != role
        or receipt.get("expected_step") != expected_step
    ):
        raise OrbaxReceiptError("Orbax receipt identity changed")
    raw_relative = receipt.get("relative_path")
    if not isinstance(raw_relative, str):
        raise OrbaxReceiptError("Orbax receipt has no relative path")
    relative = Path(raw_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise OrbaxReceiptError("Orbax receipt path is unsafe")
    output_root = _safe_directory(Path(root), label="Orbax output root")
    selected = output_root / relative
    expected = receipt.get("artifact_manifest")
    if not isinstance(expected, dict):
        raise OrbaxReceiptError("Orbax receipt has no artifact manifest")
    verify_artifact_manifest(selected, expected)
    return selected.resolve()


def write_orbax_leaf_receipt(path: Path | str, receipt: dict[str, Any]) -> None:
    """Write one immutable receipt after verifying that it is JSON serializable."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(receipt)
    json.loads(encoded)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
