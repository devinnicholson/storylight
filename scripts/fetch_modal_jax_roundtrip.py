#!/usr/bin/env python3
"""Fetch and verify one immutable Modal JAX roundtrip release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

VOLUME_NAME = "bookforge-jax-fidelity-release"
REMOTE_ROOT = "roundtrip"
EXPECTED_BACKEND = "modal-l4x2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modal_get(remote_path: str, destination: Path) -> None:
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote_path, str(destination)],
        check=True,
        timeout=900,
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _completion(path: Path, *, run_id: str, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("Modal roundtrip completion checksum changed")
    document = _json_object(path)
    if (
        document.get("schema_version") != "1.0"
        or document.get("status") != "succeeded"
        or document.get("backend") != EXPECTED_BACKEND
        or document.get("run_id") != run_id
    ):
        raise ValueError("Modal roundtrip completion identity changed")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Modal roundtrip completion has no files")
    return document


def _verify_files(root: Path, rows: list[object]) -> None:
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Modal roundtrip file declaration is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("Modal roundtrip file path is unsafe or duplicated")
        declared.add(relative.as_posix())
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"Modal roundtrip artifact failed verification: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "completion.json"
    }
    if actual != declared:
        raise ValueError("Modal roundtrip release has undeclared or missing files")


def _verify_contract(root: Path, completion: dict[str, Any]) -> None:
    from training.jax_fidelity.configuration import load_config
    from training.jax_fidelity.integrity import sha256_file, verify_artifact_manifest
    from training.jax_fidelity.orbax_receipt import (
        terminal_checkpoint_step,
        verify_orbax_leaf_receipt,
    )
    from training.jax_fidelity.roundtrip_smoke import validate_roundtrip_evidence

    config_path = root / "inputs/config.json"
    if sha256_file(config_path) != completion.get("config_sha256"):
        raise ValueError("Modal roundtrip packaged config hash changed")
    config = load_config(config_path)
    base_manifest_path = root / "base-orbax.manifest.json"
    smoke_manifest_path = root / "smoke-adapter.manifest.json"
    merged_manifest_path = root / "merged-hf.manifest.json"
    for path, field in (
        (base_manifest_path, "base_orbax_manifest_sha256"),
        (smoke_manifest_path, "smoke_adapter_manifest_sha256"),
        (merged_manifest_path, "merged_hf_manifest_sha256"),
    ):
        if sha256_file(path) != completion.get(field):
            raise ValueError(f"Modal roundtrip packaged manifest hash changed: {path.name}")
    base_receipt_path = root / "evidence/base-orbax.receipt.json"
    smoke_receipt_path = root / "evidence/smoke-orbax.receipt.json"
    roundtrip_path = root / "evidence/roundtrip.json"
    for path, field in (
        (base_receipt_path, "base_orbax_receipt_sha256"),
        (smoke_receipt_path, "smoke_orbax_receipt_sha256"),
        (roundtrip_path, "roundtrip_evidence_sha256"),
    ):
        if sha256_file(path) != completion.get(field):
            raise ValueError(f"Modal roundtrip evidence hash changed: {path.name}")
    base_receipt = _json_object(base_receipt_path)
    smoke_receipt = _json_object(smoke_receipt_path)
    base_leaf = verify_orbax_leaf_receipt(
        root / "base-orbax",
        base_receipt,
        expected_step=0,
        role="base-maxtext",
    )
    smoke_leaf = verify_orbax_leaf_receipt(
        root / "smoke-adapter",
        smoke_receipt,
        expected_step=terminal_checkpoint_step(config.training["smoke_steps"]),
        role="smoke-lora",
    )
    verify_artifact_manifest(base_leaf, _json_object(base_manifest_path))
    verify_artifact_manifest(smoke_leaf, _json_object(smoke_manifest_path))
    verify_artifact_manifest(root / "merged-hf", _json_object(merged_manifest_path))
    validate_roundtrip_evidence(
        config,
        _json_object(roundtrip_path),
        exported_checkpoint=root / "merged-hf",
    )


def fetch_roundtrip(
    *,
    run_id: str,
    expected_completion_sha256: str,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("roundtrip destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{run_id}"
    with tempfile.TemporaryDirectory(
        prefix="bookforge-modal-roundtrip-",
        dir=destination.parent,
    ) as raw:
        temporary = Path(raw)
        completion_path = temporary / "completion.json"
        _modal_get(f"{remote}/completion.json", completion_path)
        completion = _completion(
            completion_path,
            run_id=run_id,
            expected_sha256=expected_completion_sha256,
        )
        payload = temporary / "payload"
        _modal_get(remote, payload)
        if (payload / "completion.json").is_file():
            if _sha256(payload / "completion.json") != expected_completion_sha256:
                raise ValueError("roundtrip directory completion differs from trusted completion")
        else:
            shutil.copyfile(completion_path, payload / "completion.json")
        _verify_files(payload, completion["files"])
        _verify_contract(payload, completion)
        shutil.copytree(payload, destination)
    return {
        "schema_version": "1.0",
        "status": "fetched-and-verified",
        "run_id": run_id,
        "completion_sha256": expected_completion_sha256,
        "destination": str(destination),
        "remote_mutation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "plan-only",
                    "remote": f"{VOLUME_NAME}/{REMOTE_ROOT}/{args.run_id}",
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
            fetch_roundtrip(
                run_id=args.run_id,
                expected_completion_sha256=args.expected_completion_sha256,
                destination=args.destination,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
