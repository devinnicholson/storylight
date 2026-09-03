#!/usr/bin/env python3
"""Fetch and verify one completed Modal JAX release without deleting remote state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import cast

VOLUME_NAME = "bookforge-jax-fidelity-release"
EXPORT_VOLUME_NAME = "bookforge-tensorrt-edge-llm-fidelity"
ROOT = Path(__file__).parents[1]
JETSON_BUILDER = ROOT / "deploy/jetson/build-trained-planner-candidate.sh"
JETSON_INSTALLER = ROOT / "deploy/jetson/install-trained-planner-candidate.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modal_get(remote_path: str, destination: Path) -> None:
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote_path, str(destination)],
        check=True,
        timeout=600,
    )


def _modal_export_get(remote_path: str, destination: Path) -> None:
    subprocess.run(
        ["modal", "volume", "get", EXPORT_VOLUME_NAME, remote_path, str(destination)],
        check=True,
        timeout=600,
    )


def _completion(path: Path, run_id: str, expected_sha256: str) -> dict[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError("release completion differs from the trusted completion SHA-256")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("completion manifest must be an object")
    if value.get("run_id") != run_id or value.get("status") != "succeeded":
        raise ValueError("release is not the requested successful run")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release completion manifest has no files")
    return value


def _verify_portable_package(root: Path) -> None:
    required = {
        "adapter.manifest.json",
        "package.manifest.json",
        "runtime.lock.json",
        "training/run.json",
        "training/completion.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "completion.json"
    }
    if not required.issubset(actual):
        raise ValueError("portable training package is incomplete")
    package = json.loads((root / "package.manifest.json").read_text(encoding="utf-8"))
    rows = package.get("files") if isinstance(package, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("portable package manifest has no files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("portable package file entry is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("portable package file path is unsafe or duplicated")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"portable package artifact failed verification: {relative}")
        declared.add(relative.as_posix())
    package_actual = {
        path for path in actual if not path.startswith("provider/")
    } - {"package.manifest.json"}
    if package_actual != declared:
        raise ValueError("portable package contains undeclared or missing files")


def fetch_release(
    run_id: str, expected_completion_sha256: str, destination: Path
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError("destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bookforge-modal-release-", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        manifest_path = temporary / "completion.json"
        _modal_get(f"{run_id}/completion.json", manifest_path)
        manifest = _completion(manifest_path, run_id, expected_completion_sha256)
        download_root = temporary / "download"
        download_root.mkdir()
        _modal_get(run_id, download_root)
        payload = download_root / run_id
        if not payload.is_dir() or payload.is_symlink():
            raise ValueError("Modal release directory download had an unexpected shape")
        remote_completion = payload / "completion.json"
        if remote_completion.is_file() and _sha256(remote_completion) != expected_completion_sha256:
            raise ValueError("release directory completion differs from the trusted completion")
        declared: set[str] = set()
        for row in cast(list[object], manifest["files"]):
            if not isinstance(row, dict):
                raise ValueError("release file entry must be an object")
            relative = row.get("path")
            expected_sha = row.get("sha256")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in declared
            ):
                raise ValueError("release file path is unsafe or duplicated")
            declared.add(relative)
            candidate = payload / relative
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != row.get("bytes")
                or _sha256(candidate) != expected_sha
            ):
                raise ValueError(f"release artifact failed verification: {relative}")
        actual = {
            path.relative_to(payload).as_posix()
            for path in payload.rglob("*")
            if path.is_file() and path != remote_completion
        }
        if actual != declared:
            raise ValueError("release contains undeclared or missing files")
        _verify_portable_package(payload)
        shutil.copy2(manifest_path, payload / "completion.json")
        # Keep the temporary directory on the destination filesystem and move
        # the verified tree into place atomically. Large checkpoints should not
        # require a transient second copy or twice their on-disk capacity.
        os.replace(payload, destination)
    return manifest


def _export_verifier():
    path = ROOT / "deploy/jetson/download-gcs-fidelity-tensorrt-candidate.py"
    spec = importlib.util.spec_from_file_location("bookforge_modal_export_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Jetson export verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_export_bundle


def fetch_export(
    *,
    candidate_id: str,
    source_release_manifest_sha256: str,
    expected_export_manifest_sha256: str,
    destination: Path,
) -> dict[str, object]:
    """Fetch one Modal ONNX export into the exact local Jetson-builder layout."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists")
    prefix = f"{candidate_id}/{source_release_manifest_sha256[:20]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bookforge-modal-export-", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        manifest = temporary / "export.manifest.json"
        _modal_export_get(f"{prefix}/export.manifest.json", manifest)
        if _sha256(manifest) != expected_export_manifest_sha256:
            raise ValueError("Modal export manifest checksum changed")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("candidate_id") != candidate_id
            or document.get("source_release_manifest_sha256") != source_release_manifest_sha256
            or document.get("status") != "succeeded"
        ):
            raise ValueError("Modal export lineage differs from the requested candidate")
        rows = document.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Modal export manifest has no files")
        bundle = temporary / "bundle"
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Modal export file entry is malformed")
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Modal export file path is unsafe")
            target = bundle / "onnx" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _modal_export_get(f"{prefix}/onnx/{relative.as_posix()}", target)
            if (
                not target.is_file()
                or row.get("bytes") != target.stat().st_size
                or row.get("sha256") != _sha256(target)
            ):
                raise ValueError(f"Modal export artifact failed verification: {relative}")
        shutil.copy2(manifest, bundle / "export.manifest.json")
        _export_verifier()(bundle, expected_export_manifest_sha256)
        shutil.copytree(bundle, destination)
    return {
        "schema_version": "1.0",
        "status": "fetched-and-verified",
        "candidate_id": candidate_id,
        "source_release_manifest_sha256": source_release_manifest_sha256,
        "export_manifest_sha256": expected_export_manifest_sha256,
        "destination": str(destination),
        "jetson_builder_compatible": True,
        "remote_mutation": False,
    }


def build_jetson_candidate(
    *,
    export_bundle: Path,
    export_manifest_sha256: str,
    candidate_output: Path,
    runner=subprocess.run,
) -> dict[str, object]:
    """Build the fetched ONNX on Jetson and prove the result passes the installer."""

    if candidate_output.exists() or candidate_output.is_symlink():
        raise FileExistsError("candidate output already exists")
    runner(
        [
            str(JETSON_BUILDER),
            "--export-bundle",
            str(export_bundle),
            "--export-manifest-sha256",
            export_manifest_sha256,
            "--output",
            str(candidate_output),
        ],
        check=True,
        timeout=2_100,
    )
    manifest = candidate_output / "candidate.manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError("Jetson candidate builder did not produce a safe manifest")
    manifest_sha256 = _sha256(manifest)
    runner(
        [
            str(JETSON_INSTALLER),
            "--bundle",
            str(candidate_output),
            "--expected-manifest-sha256",
            manifest_sha256,
            "--verify-only",
        ],
        check=True,
        timeout=120,
    )
    return {
        "schema_version": "1.0",
        "status": "built-and-installer-verified",
        "candidate_bundle": str(candidate_output),
        "candidate_manifest_sha256": manifest_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--candidate-id")
    parser.add_argument("--source-release-manifest-sha256")
    parser.add_argument("--expected-export-manifest-sha256")
    parser.add_argument("--expected-completion-sha256")
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.candidate_id is not None:
        if not args.source_release_manifest_sha256 or not args.expected_export_manifest_sha256:
            raise SystemExit(
                "Modal export fetch requires --source-release-manifest-sha256 and "
                "--expected-export-manifest-sha256"
            )
        if not args.execute:
            print(
                json.dumps(
                    {
                        "mode": "plan-only",
                        "remote": (
                            f"{EXPORT_VOLUME_NAME}/{args.candidate_id}/"
                            f"{args.source_release_manifest_sha256[:20]}"
                        ),
                        "destination": str(args.destination),
                        "remote_mutation": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        result = fetch_export(
            candidate_id=args.candidate_id,
            source_release_manifest_sha256=args.source_release_manifest_sha256,
            expected_export_manifest_sha256=args.expected_export_manifest_sha256,
            destination=args.destination,
        )
        if args.candidate_output is not None:
            result["jetson_candidate"] = build_jetson_candidate(
                export_bundle=args.destination,
                export_manifest_sha256=args.expected_export_manifest_sha256,
                candidate_output=args.candidate_output,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.candidate_output is not None:
        raise SystemExit("--candidate-output is valid only for a Modal TensorRT export")
    if not args.expected_completion_sha256:
        raise SystemExit("--expected-completion-sha256 is required for a Modal training release")
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "plan-only",
                    "remote": f"{VOLUME_NAME}/{args.run_id}",
                    "destination": str(args.destination),
                    "remote_mutation": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            fetch_release(
                args.run_id,
                args.expected_completion_sha256,
                args.destination,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
