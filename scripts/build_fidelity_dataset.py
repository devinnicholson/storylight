#!/usr/bin/env python3
"""Build, rotate, recover, and verify the Story Fidelity Lab dataset."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any

from storylight.fidelity_dataset import (
    DATASET_ID,
    HIDDEN_DERIVATION,
    HIDDEN_KEY_BYTES,
    HIDDEN_KEY_PREFIX,
    DatasetSplit,
    decode_hidden_key,
    generate_split,
    hidden_key_fingerprint,
    records_jsonl,
    validate_split_isolation,
)
from storylight.fidelity_manifest import (
    FidelityDatasetManifest,
    build_dataset_manifest,
    sha256_bytes,
    sha256_path,
    validate_manifest,
)

CUSTODY_VERSION = "story-fidelity-custody-v1"
ROTATION_PREFIX = "ROTATE_STORY_FIDELITY_V1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_KEY = re.compile(r"[A-Za-z0-9_-]{64}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _atomic_publish(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link output: {path}")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"refusing non-regular output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _secure_regular(path: Path, label: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(f"{label} must not be accessible by group or other users")
    return metadata


def _outside_repository(path: Path, label: str) -> None:
    try:
        path.expanduser().resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return
    raise ValueError(f"{label} must be stored outside the repository")


def _versioned_key() -> str:
    encoded = base64.urlsafe_b64encode(secrets.token_bytes(HIDDEN_KEY_BYTES)).rstrip(b"=")
    return HIDDEN_KEY_PREFIX + encoded.decode("ascii")


def _initialize_key(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite hidden key: {path}")
    _write_new_file(path, (_versioned_key() + "\n").encode(), 0o600)


def _migrate_legacy_key(path: Path, value: str) -> str:
    if _LEGACY_KEY.fullmatch(value) is None:
        raise ValueError("legacy hidden key does not use the expected 384-bit base64url format")
    try:
        material = base64.b64decode(value, altchars=b"-_", validate=True)
    except ValueError as error:
        raise ValueError("legacy hidden key is not valid base64url") from error
    if len(material) != HIDDEN_KEY_BYTES:
        raise ValueError("legacy hidden key does not contain 384 bits")
    migrated = HIDDEN_KEY_PREFIX + value
    decode_hidden_key(migrated)
    _atomic_publish(path, (migrated + "\n").encode(), 0o600)
    return migrated


def _read_key(path: Path, *, migrate_legacy: bool) -> str:
    _secure_regular(path, "hidden key file")
    value = path.read_text(encoding="utf-8").strip()
    try:
        decode_hidden_key(value)
    except ValueError:
        if not migrate_legacy:
            raise ValueError(
                "hidden key is legacy or invalid; use --migrate-legacy-key only for the "
                "original 384-bit base64url key"
            ) from None
        value = _migrate_legacy_key(path, value)
    return value


def _canonical_json(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _custody_receipt(
    *,
    key_fingerprint: str,
    hidden_sha256: str,
    manifest_sha256: str,
    manifest: FidelityDatasetManifest,
) -> dict[str, Any]:
    return {
        "schema_version": CUSTODY_VERSION,
        "dataset_id": DATASET_ID,
        "key_fingerprint_sha256": key_fingerprint,
        "hidden_sha256": hidden_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "generator_source_sha256": manifest.generator_source_sha256,
        "generator_config_sha256": manifest.generator_config_sha256,
        "generator_runtime": manifest.generator_runtime.model_dump(mode="json"),
        "hidden_derivation": HIDDEN_DERIVATION,
    }


def _load_receipt(path: Path) -> dict[str, Any]:
    _secure_regular(path, "custody receipt")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"custody receipt is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("custody receipt must be a JSON object")
    expected_fields = {
        "schema_version",
        "dataset_id",
        "key_fingerprint_sha256",
        "hidden_sha256",
        "dataset_manifest_sha256",
        "generator_source_sha256",
        "generator_config_sha256",
        "generator_runtime",
        "hidden_derivation",
    }
    if set(value) != expected_fields:
        raise ValueError("custody receipt fields do not match the versioned contract")
    if value["schema_version"] != CUSTODY_VERSION or value["dataset_id"] != DATASET_ID:
        raise ValueError("custody receipt identity does not match this dataset")
    for field in (
        "key_fingerprint_sha256",
        "hidden_sha256",
        "dataset_manifest_sha256",
        "generator_source_sha256",
        "generator_config_sha256",
    ):
        if not isinstance(value[field], str) or _SHA256.fullmatch(value[field]) is None:
            raise ValueError(f"custody receipt {field} is not a lowercase SHA-256")
    return value


def _validate_receipt(
    receipt_path: Path,
    manifest_path: Path,
    hidden_path: Path,
    *,
    key: str | None = None,
) -> None:
    receipt = _load_receipt(receipt_path)
    manifest = FidelityDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected = _custody_receipt(
        key_fingerprint=(
            hidden_key_fingerprint(key)
            if key is not None
            else str(receipt.get("key_fingerprint_sha256", ""))
        ),
        hidden_sha256=sha256_path(hidden_path),
        manifest_sha256=sha256_path(manifest_path),
        manifest=manifest,
    )
    if receipt != expected:
        raise ValueError("custody receipt does not match the key, hidden split, and manifest")


def _existing_digest(path: Path, *, private: bool) -> str | None:
    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link custody target: {path}")
    if not path.exists():
        return None
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"refusing non-regular custody target: {path}")
    if private:
        _secure_regular(path, f"private custody target {path.name}")
    return sha256_path(path)


def _rotation_token(old_anchor: str, new_manifest_sha256: str, hidden_sha256: str) -> str:
    return f"{ROTATION_PREFIX}:{old_anchor}:{new_manifest_sha256}:{hidden_sha256}"


def _current_anchor(manifest_path: Path, paths: dict[str, tuple[Path, bool]]) -> str:
    manifest_digest = _existing_digest(manifest_path, private=False)
    if manifest_digest is not None:
        return manifest_digest
    state = {
        name: _existing_digest(path, private=private)
        for name, (path, private) in sorted(paths.items())
        if path != manifest_path
    }
    return hashlib.sha256(_canonical_json(state)).hexdigest()


def _acquire_lock(parent: Path) -> tuple[int, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".{DATASET_ID}.build.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"another dataset build holds {path}") from error
    return descriptor, path


def build(arguments: argparse.Namespace) -> None:
    required = {
        "--hidden-key-file": arguments.hidden_key_file,
        "--private-hidden-output": arguments.private_hidden_output,
        "--custody-receipt": arguments.custody_receipt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"dataset builds require {', '.join(missing)}")
    key_path = arguments.hidden_key_file.expanduser()
    hidden_path = arguments.private_hidden_output.expanduser()
    receipt_path = arguments.custody_receipt.expanduser()
    for path, label in (
        (key_path, "hidden key"),
        (hidden_path, "private hidden output"),
        (receipt_path, "custody receipt"),
    ):
        _outside_repository(path, label)
    if len({key_path.resolve(), hidden_path.resolve(), receipt_path.resolve()}) != 3:
        raise ValueError("key, hidden output, and custody receipt must use distinct paths")

    if arguments.initialize_hidden_key:
        _initialize_key(key_path)
    key = _read_key(key_path, migrate_legacy=arguments.migrate_legacy_key)
    output_dir = arguments.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    smoke_path = arguments.smoke_output.resolve()
    paths = {
        "train": (output_dir / "train.jsonl", False),
        "development": (output_dir / "development.jsonl", False),
        "hidden": (hidden_path, True),
        "smoke": (smoke_path, False),
        "receipt": (receipt_path, True),
        "manifest": (manifest_path, False),
    }

    lock_descriptor, lock_path = _acquire_lock(output_dir.parent)
    try:
        split_records = {
            DatasetSplit.TRAIN: list(generate_split(DatasetSplit.TRAIN)),
            DatasetSplit.DEVELOPMENT: list(generate_split(DatasetSplit.DEVELOPMENT)),
            DatasetSplit.HIDDEN: list(generate_split(DatasetSplit.HIDDEN, hidden_key=key)),
        }
        validate_split_isolation(record for records in split_records.values() for record in records)
        payloads = {
            "train": records_jsonl(split_records[DatasetSplit.TRAIN]).encode(),
            "development": records_jsonl(split_records[DatasetSplit.DEVELOPMENT]).encode(),
            "hidden": records_jsonl(split_records[DatasetSplit.HIDDEN]).encode(),
            "smoke": records_jsonl(split_records[DatasetSplit.DEVELOPMENT][:32]).encode(),
        }
        split_hashes = {
            DatasetSplit.TRAIN: sha256_bytes(payloads["train"]),
            DatasetSplit.DEVELOPMENT: sha256_bytes(payloads["development"]),
            DatasetSplit.HIDDEN: sha256_bytes(payloads["hidden"]),
        }
        manifest = build_dataset_manifest(split_records, split_hashes)
        payloads["manifest"] = manifest.canonical_json().encode()
        manifest_sha256 = sha256_bytes(payloads["manifest"])
        receipt = _custody_receipt(
            key_fingerprint=hidden_key_fingerprint(key),
            hidden_sha256=split_hashes[DatasetSplit.HIDDEN],
            manifest_sha256=manifest_sha256,
            manifest=manifest,
        )
        payloads["receipt"] = _canonical_json(receipt)

        with tempfile.TemporaryDirectory(
            prefix=f".{DATASET_ID}.staged-", dir=output_dir.parent
        ) as temporary_name:
            staged_root = Path(temporary_name)
            staged_hidden = staged_root / "private-hidden.jsonl"
            _write_new_file(staged_root / "train.jsonl", payloads["train"], 0o600)
            _write_new_file(staged_root / "development.jsonl", payloads["development"], 0o600)
            _write_new_file(staged_root / "manifest.json", payloads["manifest"], 0o600)
            _write_new_file(staged_hidden, payloads["hidden"], 0o600)
            validate_manifest(staged_root / "manifest.json", private_hidden_path=staged_hidden)

        old_anchor = _current_anchor(manifest_path, paths)
        changed_existing = [
            name
            for name, (path, private) in paths.items()
            if (existing := _existing_digest(path, private=private)) is not None
            and existing != sha256_bytes(payloads[name])
        ]
        expected_rotation = _rotation_token(
            old_anchor,
            manifest_sha256,
            split_hashes[DatasetSplit.HIDDEN],
        )
        if changed_existing and arguments.rotation_token != expected_rotation:
            raise ValueError(
                "existing custody content differs; rerun only after review with "
                f"--rotation-token {expected_rotation}"
            )
        if not changed_existing and arguments.rotation_token is not None:
            raise ValueError("rotation token was supplied but no existing content would change")

        for name in ("train", "development", "smoke", "hidden", "receipt", "manifest"):
            path, private = paths[name]
            if _existing_digest(path, private=private) == sha256_bytes(payloads[name]):
                continue
            _atomic_publish(path, payloads[name], 0o600 if private else 0o644)

        validate_manifest(manifest_path, private_hidden_path=hidden_path)
        _validate_receipt(receipt_path, manifest_path, hidden_path, key=key)
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)
    print(f"Built {DATASET_ID}: {manifest_path}")
    print(f"Private hidden records: {hidden_path}")
    print(f"Private custody receipt: {receipt_path}")


def verify(arguments: argparse.Namespace) -> None:
    manifest_path = arguments.output_dir.resolve() / "manifest.json"
    if arguments.public_only:
        if any(
            value is not None
            for value in (
                arguments.private_hidden_output,
                arguments.custody_receipt,
                arguments.hidden_key_file,
            )
        ):
            raise ValueError("--public-only does not accept private custody paths")
        validate_manifest(manifest_path)
        print(f"Verified public dataset: {manifest_path}")
        return
    if arguments.private_hidden_output is None or arguments.custody_receipt is None:
        raise ValueError(
            "custody verification requires --private-hidden-output and --custody-receipt; "
            "use --public-only to verify only committed files"
        )
    hidden_path = arguments.private_hidden_output.expanduser()
    receipt_path = arguments.custody_receipt.expanduser()
    if not hidden_path.exists() or hidden_path.is_symlink():
        raise ValueError(f"supplied private hidden path is missing or unsafe: {hidden_path}")
    validate_manifest(manifest_path, private_hidden_path=hidden_path)
    key = (
        _read_key(arguments.hidden_key_file.expanduser(), migrate_legacy=False)
        if arguments.hidden_key_file is not None
        else None
    )
    _validate_receipt(receipt_path, manifest_path, hidden_path, key=key)
    print(f"Verified public and private custody state: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/story-fidelity-v2"))
    parser.add_argument("--private-hidden-output", type=Path)
    parser.add_argument("--hidden-key-file", type=Path)
    parser.add_argument("--custody-receipt", type=Path)
    parser.add_argument("--initialize-hidden-key", action="store_true")
    parser.add_argument("--migrate-legacy-key", action="store_true")
    parser.add_argument("--rotation-token")
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path("tests/fixtures/story-fidelity-smoke.jsonl"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--public-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.public_only and not arguments.verify_only:
        parser.error("--public-only requires --verify-only")
    if arguments.verify_only:
        verify(arguments)
    else:
        build(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
