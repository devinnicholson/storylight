#!/usr/bin/env python3
"""Verify and atomically install a Cloud-exported Gemma 4 ONNX checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
EXPORT_ID = "gemma4-e2b-it-int4-awq-v010"
EDGELLM_VERSION = "v0.10.0"
EDGELLM_REVISION = "71dd1bae032e70771265917ec74d3ff4cad07a10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(payload: dict[str, Any], key: str, expected: Any) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"manifest {key!r} does not match the pinned Bookforge value")


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest contains an invalid file path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or value != candidate.as_posix():
        raise ValueError(f"manifest contains an unsafe file path: {value!r}")
    return Path(*candidate.parts)


def verify_bundle(bundle_root: Path) -> tuple[dict[str, Any], list[Path]]:
    bundle_root = bundle_root.resolve()
    manifest_path = bundle_root / "export.manifest.json"
    onnx_root = bundle_root / "onnx"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read export manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("export manifest is not a JSON object")

    for key, expected in (
        ("schema_version", 1),
        ("result", "complete"),
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
        ("export_id", EXPORT_ID),
        ("quantization", "int4_awq"),
        ("components", ["thinker"]),
        ("externalized_weights", ["int4_ffn"]),
        ("tensorrt_edge_llm_version", EDGELLM_VERSION),
        ("tensorrt_edge_llm_revision", EDGELLM_REVISION),
        ("mtp_included", False),
        ("engine_built_in_cloud", False),
    ):
        _require(manifest, key, expected)

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("export manifest has no files")
    if manifest.get("file_count") != len(files):
        raise ValueError("export manifest file count does not match its file list")

    verified: list[Path] = []
    declared: set[Path] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("export manifest contains an invalid file entry")
        relative = _safe_relative_path(item.get("path"))
        if relative in declared:
            raise ValueError(f"export manifest repeats {relative.as_posix()}")
        declared.add(relative)
        path = onnx_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"export file is missing or unsafe: {relative.as_posix()}")
        size = path.stat().st_size
        if item.get("bytes") != size:
            raise ValueError(f"export size mismatch: {relative.as_posix()}")
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"export checksum is invalid: {relative.as_posix()}")
        if _sha256(path) != expected_sha:
            raise ValueError(f"export checksum mismatch: {relative.as_posix()}")
        total_bytes += size
        verified.append(relative)

    actual = {
        path.relative_to(onnx_root)
        for path in onnx_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != declared:
        raise ValueError("ONNX directory contains files not bound by the completion manifest")
    if manifest.get("total_bytes") != total_bytes:
        raise ValueError("export manifest total byte count does not match its files")
    for required in (Path("llm/config.json"), Path("llm/model.onnx")):
        if required not in declared:
            raise ValueError(f"export manifest omitted required file: {required.as_posix()}")
    if not any(path.parent == Path("llm") and path.suffix == ".safetensors" for path in declared):
        raise ValueError("export manifest omitted externalized INT4 weights")
    return manifest, verified


def install_bundle(bundle_root: Path, destination: Path) -> dict[str, Any]:
    manifest, verified = verify_bundle(bundle_root)
    source = bundle_root.resolve() / "onnx"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(
            f"checkpoint destination already exists; refusing to overwrite: {destination}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=False)
        for item in manifest["files"]:
            relative = _safe_relative_path(item["path"])
            if _sha256(temporary / relative) != item["sha256"]:
                raise ValueError(f"installed checksum mismatch: {relative.as_posix()}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "result": "installed",
        "destination": str(destination),
        "export_run_id": manifest.get("export_run_id"),
        "file_count": len(verified),
        "total_bytes": manifest["total_bytes"],
    }


def _default_destination() -> Path:
    root = Path.home() / ".local/share/bookforge/tensorrt-edgellm-v0.10.0"
    return root / "models/gemma4-e2b-it-int4-awq-v010/onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--destination", type=Path, default=_default_destination())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest, files = verify_bundle(args.bundle_root)
        result = {
            "result": "verified",
            "export_run_id": manifest.get("export_run_id"),
            "file_count": len(files),
            "total_bytes": manifest["total_bytes"],
        }
    else:
        result = install_bundle(args.bundle_root, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
