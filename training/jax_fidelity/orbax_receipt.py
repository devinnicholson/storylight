"""Discover and bind one exact MaxText Orbax ``items`` checkpoint leaf."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

from .integrity import (
    artifact_manifest,
    canonical_json_bytes,
    sha256_file,
    verify_artifact_manifest,
)


class OrbaxReceiptError(ValueError):
    """An Orbax output did not contain one unambiguous checkpoint leaf."""


def _orbax_key_path(serialized: object, entry: object) -> tuple[str | int, ...]:
    """Decode one Orbax tree path and cross-check both metadata representations."""

    if not isinstance(serialized, str) or not isinstance(entry, dict):
        raise OrbaxReceiptError("Orbax tree metadata entry is malformed")
    key_metadata = entry.get("key_metadata")
    if not isinstance(key_metadata, list) or not key_metadata:
        raise OrbaxReceiptError("Orbax tree metadata has no key path")
    path: list[str | int] = []
    for component in key_metadata:
        key = component.get("key") if isinstance(component, dict) else None
        if type(key) not in (str, int) or key == "":
            raise OrbaxReceiptError("Orbax tree metadata key path is malformed")
        path.append(key)

    try:
        decoded = ast.literal_eval(serialized)
    except (SyntaxError, ValueError) as error:
        raise OrbaxReceiptError("Orbax serialized tree path is malformed") from error
    if not isinstance(decoded, tuple) or tuple(path) != decoded:
        raise OrbaxReceiptError("Orbax serialized and structured tree paths disagree")
    return tuple(path)


def _lora_side(path: tuple[str | int, ...]) -> tuple[tuple[str | int, ...], str] | None:
    """Return a stable module identity and LoRA side for known MaxText layouts."""

    leaf = str(path[-1]).lower()
    exact = {
        "lora_a": "a",
        "lora_b": "b",
        "lora_a.kernel": "a",
        "lora_b.kernel": "b",
    }
    if leaf in exact:
        return path[:-1], exact[leaf]
    if len(path) > 1 and leaf == "kernel":
        parent = str(path[-2]).lower()
        if parent in ("lora_a", "lora_b"):
            return path[:-2], parent[-1]
    for suffix, side in (("_lora_a", "a"), ("_lora_b", "b")):
        if leaf.endswith(suffix) and len(leaf) > len(suffix):
            return (*path[:-1], leaf[: -len(suffix)]), side
    return None


def _is_model_parameter_path(path: tuple[str | int, ...]) -> bool:
    """Return whether an on-disk leaf belongs to the Linen model-parameter tree."""

    return len(path) > 2 and path[:2] == ("params", "params")


def lora_checkpoint_evidence(
    items: Path | str,
    *,
    expected_rank: int,
    expected_pair_count: int | None = None,
    expected_step: int | None = None,
    approved_maxtext_patch_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate paired LoRA tensors from an Orbax ``items/_METADATA`` file.

    Orbax's OCDBT filenames are content-addressed and do not expose parameter
    names. ``_METADATA`` is the authoritative tree map, so the gate parses its
    structured key metadata, checks it against Orbax's serialized tuple key, and
    validates every LoRA A/B shape. A step-only checkpoint is therefore rejected
    before conversion can produce a no-op merged model.
    """

    if type(expected_rank) is not int or expected_rank < 1:
        raise OrbaxReceiptError("expected LoRA rank must be a positive integer")
    if expected_pair_count is not None and (
        type(expected_pair_count) is not int or expected_pair_count < 1
    ):
        raise OrbaxReceiptError("expected LoRA pair count must be a positive integer")
    if expected_step is not None and (type(expected_step) is not int or expected_step < 0):
        raise OrbaxReceiptError("expected checkpoint step must be a non-negative integer")
    if approved_maxtext_patch_sha256 is not None and (
        not isinstance(approved_maxtext_patch_sha256, str)
        or len(approved_maxtext_patch_sha256) != 64
        or any(character not in "0123456789abcdef" for character in approved_maxtext_patch_sha256)
    ):
        raise OrbaxReceiptError("approved MaxText patch must be a lowercase SHA-256")
    root = _safe_directory(Path(items), label="Orbax items leaf")
    if expected_step is not None and (
        root.name != "items" or root.parent.name != str(expected_step)
    ):
        raise OrbaxReceiptError("Orbax adapter leaf does not match the approved terminal step")
    metadata_path = root / "_METADATA"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise OrbaxReceiptError("Orbax items leaf has no regular _METADATA file")
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrbaxReceiptError("Orbax _METADATA is not valid JSON") from error
    if not isinstance(document, dict):
        raise OrbaxReceiptError("Orbax _METADATA must contain one JSON object")
    tree = document.get("tree_metadata")
    if not isinstance(tree, dict) or not tree:
        raise OrbaxReceiptError("Orbax _METADATA has no tree metadata")

    leaves: dict[tuple[str | int, ...], dict[str, Any]] = {}
    unrecognized_lora_paths: list[tuple[str | int, ...]] = []
    has_step = False
    for serialized, entry in tree.items():
        path = _orbax_key_path(serialized, entry)
        if path in leaves:
            raise OrbaxReceiptError("Orbax tree metadata contains a duplicate key path")
        value_metadata = entry.get("value_metadata")
        if not isinstance(value_metadata, dict):
            raise OrbaxReceiptError("Orbax tree value metadata is malformed")
        leaves[path] = value_metadata
        has_step = has_step or path == ("step",)
        if any("lora" in str(component).lower() for component in path) and _lora_side(path) is None:
            unrecognized_lora_paths.append(path)
    if not has_step:
        raise OrbaxReceiptError("Orbax adapter checkpoint has no root step tensor")
    if unrecognized_lora_paths:
        raise OrbaxReceiptError("Orbax adapter checkpoint has unrecognized LoRA tensor paths")

    pairs: dict[tuple[str | int, ...], dict[str, tuple[int, ...]]] = {}
    optimizer_lora_tensor_count = 0
    for path, value_metadata in leaves.items():
        identity = _lora_side(path)
        if identity is None:
            continue
        if not _is_model_parameter_path(path):
            optimizer_lora_tensor_count += 1
            continue
        module, side = identity
        if (
            value_metadata.get("value_type") != "jax.Array"
            or value_metadata.get("skip_deserialize") is not False
        ):
            raise OrbaxReceiptError("LoRA leaf is not a restorable JAX array")
        raw_shape = value_metadata.get("write_shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) < 2
            or any(type(dimension) is not int or dimension < 1 for dimension in raw_shape)
        ):
            raise OrbaxReceiptError("LoRA leaf has no valid nonempty write shape")
        module_sides = pairs.setdefault(module, {})
        if side in module_sides:
            raise OrbaxReceiptError("Orbax adapter checkpoint has duplicate LoRA tensor sides")
        module_sides[side] = tuple(raw_shape)

    if not pairs:
        raise OrbaxReceiptError("Orbax adapter checkpoint contains no LoRA tensors")
    evidence_pairs: list[dict[str, Any]] = []
    for module in sorted(pairs, key=lambda value: tuple(str(part) for part in value)):
        sides = pairs[module]
        if set(sides) != {"a", "b"}:
            raise OrbaxReceiptError("Orbax adapter checkpoint has an unpaired LoRA tensor")
        a_shape = sides["a"]
        b_shape = sides["b"]
        if a_shape[-1] != expected_rank or b_shape[0] != expected_rank:
            raise OrbaxReceiptError("Orbax LoRA tensor shape does not match the approved rank")
        evidence_pairs.append(
            {
                "module_path": "/".join(str(component) for component in module),
                "a_shape": list(a_shape),
                "b_shape": list(b_shape),
            }
        )

    if expected_pair_count is not None and len(evidence_pairs) != expected_pair_count:
        raise OrbaxReceiptError(
            "Orbax LoRA pair count does not match the approved model topology"
        )

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "format": "maxtext-orbax-lora-tree",
        "metadata_sha256": sha256_file(metadata_path),
        "tree_leaf_count": len(leaves),
        "lora_tensor_count": len(evidence_pairs) * 2,
        "lora_pair_count": len(evidence_pairs),
        "optimizer_lora_tensor_count": optimizer_lora_tensor_count,
        "rank": expected_rank,
        "pairs": evidence_pairs,
    }
    if expected_pair_count is not None:
        evidence["expected_lora_pair_count"] = expected_pair_count
    if expected_step is not None:
        evidence["checkpoint_step"] = expected_step
        evidence["checkpoint_step_binding"] = "directory-name-plus-root-step-leaf"
    if approved_maxtext_patch_sha256 is not None:
        evidence["approved_maxtext_patch_sha256"] = approved_maxtext_patch_sha256
    return evidence


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
