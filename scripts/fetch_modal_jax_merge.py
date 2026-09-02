#!/usr/bin/env python3
"""Fetch and verify one immutable full-adapter merged-HF candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from training.jax_fidelity.integrity import canonical_sha256, sha256_file, verify_artifact_manifest
from training.jax_fidelity.merged_candidate import validate_merged_candidate_manifest

VOLUME_NAME = "bookforge-jax-fidelity-release"
REMOTE_ROOT = "merged"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _modal_get(remote: str, destination: Path) -> None:
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote, str(destination)],
        check=True,
        timeout=900,
    )


def _completion(path: Path, *, merge_run_id: str, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("merged release completion checksum changed")
    document = _json_object(path)
    if (
        document.get("schema_version") != "1.0"
        or document.get("status") != "succeeded"
        or document.get("backend") != "modal-l40s"
        or document.get("release_type") != "provisional-merged-hf-development-candidate"
        or document.get("merge_run_id") != merge_run_id
        or document.get("development_evaluated") is not False
        or document.get("release_authorized") is not False
    ):
        raise ValueError("merged release completion identity or eligibility changed")
    return document


def _verify_files(root: Path, rows: object) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("merged release completion has no files")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("merged release file declaration is malformed")
        relative = Path(row["path"])
        name = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or name in declared:
            raise ValueError("merged release path is unsafe or duplicated")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"merged release file failed verification: {name}")
        declared.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "completion.json"
    }
    if actual != declared:
        raise ValueError("merged release contains undeclared or missing files")


def _verify_payload(root: Path, completion: dict[str, Any], *, config_path: Path) -> None:
    _verify_files(root, completion.get("files"))
    merged_hf = root / "merged-hf"
    merged_manifest_path = root / "merged-hf.manifest.json"
    candidate_manifest_path = root / "candidate.manifest.json"
    source_bindings_path = root / "evidence/source-bindings.json"
    conversion_run_path = root / "evidence/maxtext-to-hf.run.json"
    conversion_completion_path = root / "evidence/maxtext-to-hf.completion.json"
    if sha256_file(merged_manifest_path) != completion.get("merged_hf_manifest_sha256"):
        raise ValueError("merged HF manifest checksum changed")
    merged_manifest = _json_object(merged_manifest_path)
    verify_artifact_manifest(merged_hf, merged_manifest)
    if (
        canonical_sha256(merged_manifest) != completion.get("checkpoint_manifest_sha256")
        or merged_manifest.get("content_sha256") != completion.get("checkpoint_content_sha256")
        or sha256_file(candidate_manifest_path) != completion.get("candidate_manifest_sha256")
    ):
        raise ValueError("merged checkpoint binding changed")
    candidate = validate_merged_candidate_manifest(
        candidate_manifest_path,
        merged_hf,
        config_path=config_path,
        expected_manifest_sha256=str(completion["candidate_manifest_sha256"]),
        expected_config_sha256=str(completion["config_sha256"]),
        expected_dataset_manifest_sha256=str(completion["dataset_manifest_sha256"]),
        expected_candidate_id=str(completion["candidate_id"]),
    )
    if candidate.get("training_run_id") != completion.get("training_run_id"):
        raise ValueError("candidate training run binding changed")
    source_bindings = _json_object(source_bindings_path)
    for name in (
        "input_manifest_sha256",
        "hf_snapshot_manifest_sha256",
        "roundtrip_completion_sha256",
        "base_orbax_receipt_sha256",
        "base_orbax_manifest_sha256",
        "training_release_completion_sha256",
        "adapter_manifest_sha256",
        "training_run_sha256",
        "training_completion_sha256",
        "config_sha256",
        "dataset_manifest_sha256",
        "training_run_id",
    ):
        if source_bindings.get(name) != completion.get(name):
            raise ValueError(f"merged source binding changed: {name}")
    if sha256_file(conversion_run_path) != source_bindings.get("conversion_run_sha256"):
        raise ValueError("conversion run checksum changed")
    if sha256_file(conversion_completion_path) != source_bindings.get(
        "conversion_completion_sha256"
    ):
        raise ValueError("conversion completion checksum changed")
    conversion = _json_object(conversion_completion_path)
    if (
        conversion.get("status") != "succeeded"
        or conversion.get("run_id") != source_bindings.get("conversion_run_id")
        or conversion.get("run_manifest_sha256") != source_bindings.get("conversion_run_sha256")
        or not conversion.get("artifacts")
        or not conversion.get("evidence")
    ):
        raise ValueError("conversion terminal evidence changed")


def fetch_merge(
    *,
    merge_run_id: str,
    completion_sha256: str,
    config_path: Path,
    destination: Path,
) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("merge destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{merge_run_id}"
    with tempfile.TemporaryDirectory(
        prefix="bookforge-modal-merge-", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        completion_path = temporary / "completion.json"
        _modal_get(f"{remote}/completion.json", completion_path)
        completion = _completion(
            completion_path,
            merge_run_id=merge_run_id,
            expected_sha256=completion_sha256,
        )
        payload = temporary / "payload"
        _modal_get(remote, payload)
        _verify_payload(payload, completion, config_path=config_path)
        shutil.copy2(completion_path, payload / "completion.json")
        shutil.copytree(payload, destination)
    return {
        "schema_version": "1.0",
        "status": "fetched-and-verified",
        "merge_run_id": merge_run_id,
        "candidate_id": completion["candidate_id"],
        "completion_sha256": completion_sha256,
        "candidate_manifest_sha256": completion["candidate_manifest_sha256"],
        "destination": str(destination),
        "prediction_stager_compatible": True,
        "remote_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-run-id", required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "plan-only",
                    "remote": f"{VOLUME_NAME}/{REMOTE_ROOT}/{args.merge_run_id}",
                    "completion_sha256": args.completion_sha256,
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
            fetch_merge(
                merge_run_id=args.merge_run_id,
                completion_sha256=args.completion_sha256,
                config_path=args.config,
                destination=args.destination,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
