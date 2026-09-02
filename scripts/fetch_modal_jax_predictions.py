#!/usr/bin/env python3
"""Fetch one checksum-approved public development prediction release from Modal."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

VOLUME_NAME = "bookforge-jax-fidelity-release"
REMOTE_ROOT = "prediction"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[a-z][a-z0-9-]{7,62}\Z")
_CANDIDATE_ID = re.compile(r"fidelity-[0-9a-f]{20}\Z")


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
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _completion(path: Path, *, run_id: str, expected_sha256: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None or _sha256(path) != expected_sha256:
        raise ValueError("Modal prediction completion checksum changed")
    document = _json_object(path)
    hash_names = (
        "input_manifest_sha256",
        "config_sha256",
        "dataset_manifest_sha256",
        "development_records_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
    )
    if (
        document.get("schema_version") != "1.0"
        or document.get("status") != "succeeded"
        or document.get("backend") != "modal-l40s-cuda"
        or document.get("run_id") != run_id
        or document.get("split") != "development"
        or document.get("predictions") != 512
        or document.get("hidden_evaluated") is not False
        or document.get("passages_retained") is not False
        or not isinstance(document.get("candidate_id"), str)
        or _CANDIDATE_ID.fullmatch(document["candidate_id"]) is None
        or type(document.get("batch_size")) is not int
        or not 1 <= document["batch_size"] <= 16
        or any(
            not isinstance(document.get(name), str) or _SHA256.fullmatch(document[name]) is None
            for name in hash_names
        )
    ):
        raise ValueError("Modal prediction completion identity or privacy contract changed")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Modal prediction completion has no files")
    return document


def _verify_files(root: Path, rows: list[object]) -> None:
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Modal prediction file declaration is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("Modal prediction file path is unsafe or duplicated")
        declared.add(relative.as_posix())
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError(f"Modal prediction artifact failed verification: {relative}")
    if declared != {"intent.json", "predictions.jsonl"}:
        raise ValueError(
            "Modal prediction release contains files outside the public output contract"
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "completion.json"
    }
    if actual != declared:
        raise ValueError("Modal prediction release has undeclared or missing files")


def _verify_predictions(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"prediction JSONL line {line_number} is incomplete")
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or set(value) != {"record_id", "raw"}
                or not isinstance(value["record_id"], str)
                or not value["record_id"].startswith("development-")
                or not isinstance(value["raw"], str)
            ):
                raise ValueError(f"prediction JSONL line {line_number} violates the output schema")
            rows.append(value)
    if len(rows) != 512 or len({row["record_id"] for row in rows}) != 512:
        raise ValueError("prediction JSONL is not the complete development population")


def _verify_intent(path: Path, completion: dict[str, Any]) -> None:
    intent = _json_object(path)
    binding_names = (
        "candidate_id",
        "config_sha256",
        "dataset_manifest_sha256",
        "development_records_sha256",
        "candidate_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_content_sha256",
    )
    expected_bindings = {name: completion.get(name) for name in binding_names}
    if intent != {
        "schema_version": "1.0",
        "status": "prediction-intent-recorded",
        "retry_allowed": False,
        "run_id": completion["run_id"],
        "candidate_id": completion["candidate_id"],
        "input_manifest_sha256": completion["input_manifest_sha256"],
        "bindings": expected_bindings,
        "batch_size": completion["batch_size"],
    }:
        raise ValueError("Modal prediction intent differs from its terminal completion")


def fetch_predictions(
    *,
    run_id: str,
    expected_completion_sha256: str,
    destination: Path,
) -> dict[str, object]:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("prediction run ID must be an immutable lowercase slug")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("prediction destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{run_id}"
    with tempfile.TemporaryDirectory(
        prefix="bookforge-modal-prediction-",
        dir=destination.parent,
    ) as raw:
        temporary = Path(raw)
        trusted_completion = temporary / "trusted-completion.json"
        _modal_get(f"{remote}/completion.json", trusted_completion)
        completion = _completion(
            trusted_completion,
            run_id=run_id,
            expected_sha256=expected_completion_sha256,
        )
        payload = temporary / "payload"
        _modal_get(remote, payload)
        remote_completion = payload / "completion.json"
        if (
            not remote_completion.is_file()
            or _sha256(remote_completion) != expected_completion_sha256
        ):
            raise ValueError("prediction directory completion differs from the trusted completion")
        _verify_files(payload, completion["files"])
        _verify_intent(payload / "intent.json", completion)
        _verify_predictions(payload / "predictions.jsonl")
        shutil.copytree(payload, destination)
    return {
        "schema_version": "1.0",
        "status": "fetched-and-verified",
        "run_id": run_id,
        "candidate_id": completion["candidate_id"],
        "completion_sha256": expected_completion_sha256,
        "predictions_sha256": next(
            row["sha256"] for row in completion["files"] if row["path"] == "predictions.jsonl"
        ),
        "destination": str(destination),
        "hidden_evaluated": False,
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
                    "schema_version": "1.0",
                    "status": "plan-only",
                    "remote": f"{VOLUME_NAME}/{REMOTE_ROOT}/{args.run_id}",
                    "completion_sha256": args.expected_completion_sha256,
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
            fetch_predictions(
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
