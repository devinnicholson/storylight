#!/usr/bin/env python3
"""Safely publish checksum-bound trained-planner tooling on a Jetson."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath

BASE = Path("/usr/local/lib/bookforge/trained-planner-tooling")
TRUSTED_STATE = Path("/var/lib/bookforge-trusted")
STATE = TRUSTED_STATE / "trained-planner"
CANDIDATE_ROOT = TRUSTED_STATE / "trained-planner-candidates"
LEGACY_ACTIVE_STATE = Path("/var/lib/bookforge") / "trained-planner/active.env"
UNIT = Path("/etc/systemd/user/bookforge-trained-planner-candidate@.service")
MAX_FILES = 64
MAX_BYTES = 16 * 1024 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest repeats JSON key: {key}")
        result[key] = value
    return result


def safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid manifest path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
        raise ValueError(f"unsafe manifest path: {value!r}")
    return Path(*parsed.parts)


def verify_bundle(bundle: Path, expected_sha256: str, source_commit: str) -> dict[str, object]:
    root_info = bundle.lstat()
    if bundle.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("tooling bundle must be a non-symlink directory")
    manifest_path = bundle / "tooling.manifest.json"
    info = manifest_path.lstat()
    if manifest_path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError("tooling.manifest.json must be a regular non-symlink")
    if sha256(manifest_path) != expected_sha256:
        raise ValueError("tooling manifest checksum does not match the out-of-band digest")
    manifest = json.loads(manifest_path.read_text(), object_pairs_hook=reject_duplicates)
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("artifact_type") != "bookforge-jetson-trained-planner-tooling"
        or manifest.get("source_commit") != source_commit
        or not isinstance(manifest.get("source_commit_verified"), bool)
        or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("source_manifest_sha256", "")))
    ):
        raise ValueError("tooling manifest identity is invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_FILES:
        raise ValueError("tooling manifest has an invalid file list")
    declared: set[str] = {"tooling.manifest.json"}
    normalized: list[dict[str, object]] = []
    total = 0
    for record in raw_files:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "mode", "sha256"}:
            raise ValueError("tooling file record is invalid")
        relative = safe_relative(record["path"])
        relative_text = relative.as_posix()
        if relative_text in declared:
            raise ValueError("tooling manifest repeats a path")
        declared.add(relative_text)
        size, mode, digest = record["bytes"], record["mode"], record["sha256"]
        if (
            not isinstance(size, int)
            or size <= 0
            or mode not in (0o444, 0o555)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise ValueError(f"invalid metadata for {relative_text}")
        path = bundle / relative
        path_info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(path_info.st_mode):
            raise ValueError(f"tooling payload is not a regular non-symlink: {relative_text}")
        if path_info.st_size != size or sha256(path) != digest:
            raise ValueError(f"tooling payload checksum mismatch: {relative_text}")
        total += size
        normalized.append(record)
    if total > MAX_BYTES:
        raise ValueError("tooling bundle exceeds the size limit")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != declared:
        raise ValueError("tooling bundle has undeclared, missing, or linked files")
    expected_directories = {
        parent.as_posix()
        for declared_path in declared
        for parent in safe_relative(declared_path).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_directories != expected_directories:
        raise ValueError("tooling bundle has undeclared or missing directories")
    if hashlib.sha256(canonical(normalized)).hexdigest() != manifest["source_manifest_sha256"]:
        raise ValueError("tooling source manifest checksum is invalid")
    return manifest


def validate_root_ancestry(
    path: Path, *, lstat: Callable[[Path], os.stat_result] = os.lstat
) -> None:
    if not path.is_absolute():
        raise ValueError(f"privileged path is not absolute: {path}")
    for ancestor in reversed((path, *path.parents)):
        info = lstat(ancestor)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or info.st_mode & 0o022
        ):
            raise ValueError(f"unsafe privileged directory ancestry: {ancestor}")


def secure_directory(path: Path, mode: int) -> None:
    validate_root_ancestry(path.parent)
    with suppress(FileExistsError):
        os.mkdir(path, mode)
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or info.st_mode & 0o022
    ):
        raise ValueError(f"unsafe privileged directory: {path}")
    os.chmod(path, mode)


def write_atomic(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, 0, 0)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def copy_declared_bundle(source: Path, destination: Path, manifest: dict[str, object]) -> None:
    destination.mkdir(mode=0o700)

    def copy_file(source_path: Path, target: Path, maximum_bytes: int) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_descriptor = os.open(source_path, flags)
        try:
            source_info = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_info.st_mode) or source_info.st_size > maximum_bytes:
                raise ValueError(f"staged tooling source is invalid: {source_path}")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target_descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            copied = 0
            try:
                while block := os.read(source_descriptor, 1024 * 1024):
                    copied += len(block)
                    if copied > maximum_bytes:
                        raise ValueError("tooling source grew beyond its declared bound")
                    view = memoryview(block)
                    while view:
                        written = os.write(target_descriptor, view)
                        view = view[written:]
                os.fsync(target_descriptor)
            finally:
                os.close(target_descriptor)
        finally:
            os.close(source_descriptor)

    copy_file(
        source / "tooling.manifest.json",
        destination / "tooling.manifest.json",
        MAX_BYTES,
    )
    for record in manifest["files"]:
        relative = safe_relative(record["path"])
        copy_file(source / relative, destination / relative, int(record["bytes"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--approval-token")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.expected_manifest_sha256):
        raise SystemExit("invalid --expected-manifest-sha256")
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_commit):
        raise SystemExit("invalid --source-commit")
    bundle = args.bundle.expanduser().absolute()
    manifest = verify_bundle(bundle, args.expected_manifest_sha256, args.source_commit)
    source_manifest = str(manifest["source_manifest_sha256"])
    version = f"{args.source_commit[:12]}-{source_manifest[:20]}"
    token = (
        "INSTALL_BOOKFORGE_TRAINED_PLANNER_TOOLING:"
        f"{args.source_commit}:{args.expected_manifest_sha256}"
    )
    plan = {
        "schema_version": "1.0",
        "artifact_type": "bookforge-jetson-tooling-install-plan",
        "source_commit": args.source_commit,
        "source_commit_verified": manifest["source_commit_verified"],
        "source_manifest_sha256": source_manifest,
        "tooling_manifest_sha256": args.expected_manifest_sha256,
        "version": version,
        "version_path": str(BASE / "versions" / version),
        "approval_token": token,
        "service_restart": False,
        "deployable": manifest["source_commit_verified"] is True,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return
    if os.geteuid() != 0:
        raise SystemExit("--execute must run as root")
    if manifest["source_commit_verified"] is not True:
        raise SystemExit("refusing to install tooling that was not clean at its declared commit")
    if args.approval_token != token:
        raise SystemExit("approval token does not match the verified staged update")
    if LEGACY_ACTIVE_STATE.exists() or LEGACY_ACTIVE_STATE.is_symlink():
        raise SystemExit(
            "legacy trained-planner promotion state requires explicit reconciliation"
        )

    secure_directory(Path("/usr/local/lib/bookforge"), 0o755)
    secure_directory(BASE, 0o755)
    secure_directory(BASE / "versions", 0o555)
    lock_path = BASE / ".install.lock"
    lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        lock_info = os.fstat(lock)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != 0
            or lock_info.st_gid != 0
            or lock_info.st_mode & 0o077
        ):
            raise ValueError("unsafe tooling installation lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage = Path(tempfile.mkdtemp(prefix=".incoming-", dir=BASE))
        try:
            copy_declared_bundle(bundle, stage / "payload", manifest)
            staged = stage / "payload"
            staged_manifest = verify_bundle(
                staged, args.expected_manifest_sha256, args.source_commit
            )
            if staged_manifest != manifest:
                raise ValueError("staged manifest changed during the guarded copy")
            for record in manifest["files"]:
                target = staged / safe_relative(record["path"])
                os.chown(target, 0, 0)
                os.chmod(target, int(record["mode"]))
            manifest_target = staged / "tooling.manifest.json"
            os.chown(manifest_target, 0, 0)
            os.chmod(manifest_target, 0o444)
            for directory in sorted(
                (path for path in staged.rglob("*") if path.is_dir()), reverse=True
            ):
                os.chown(directory, 0, 0)
                os.chmod(directory, 0o555)
            os.chown(staged, 0, 0)
            os.chmod(staged, 0o555)
            destination = BASE / "versions" / version
            if destination.exists() or destination.is_symlink():
                installed = verify_bundle(
                    destination, args.expected_manifest_sha256, args.source_commit
                )
                if installed != manifest:
                    raise ValueError("existing immutable tooling version does not match")
            else:
                os.replace(staged, destination)
            shutil.rmtree(stage, ignore_errors=True)

            # Runtime story packs live in /var/lib/bookforge, which is intentionally
            # writable by the service account. Privileged model state must not be a
            # child of that replaceable tree.
            secure_directory(TRUSTED_STATE, 0o755)
            secure_directory(CANDIDATE_ROOT, 0o755)
            # The current provenance receipt is public metadata consumed by the
            # sudo/read-only preflight. Sensitive state remains in 0700 children.
            secure_directory(STATE, 0o755)
            for child in ("evidence", "history", "rollback", "tooling-provenance"):
                secure_directory(STATE / child, 0o700)
            secure_directory(Path("/etc/bookforge"), 0o755)
            secure_directory(Path("/etc/bookforge/trained-planner"), 0o755)
            secure_directory(UNIT.parent, 0o755)

            unit_source = destination / "systemd/bookforge-trained-planner-candidate@.service"
            if UNIT.exists() or UNIT.is_symlink():
                unit_info = UNIT.lstat()
                if (
                    UNIT.is_symlink()
                    or not stat.S_ISREG(unit_info.st_mode)
                    or unit_info.st_uid != 0
                    or unit_info.st_gid != 0
                ):
                    raise ValueError("existing candidate unit is not a safe root-owned file")
            write_atomic(UNIT, unit_source.read_bytes(), 0o444)
            current = BASE / "current"
            if current.exists() or current.is_symlink():
                current_info = current.lstat()
                if (
                    not current.is_symlink()
                    or current_info.st_uid != 0
                    or current_info.st_gid != 0
                    or current.resolve().parent != (BASE / "versions").resolve()
                ):
                    raise ValueError("existing tooling current pointer is unsafe")
            link_tmp = BASE / f".current.{os.getpid()}"
            os.symlink(Path("versions") / version, link_tmp)
            os.lchown(link_tmp, 0, 0)
            os.replace(link_tmp, current)
            receipt_plan = {key: value for key, value in plan.items() if key != "approval_token"}
            receipt = {
                **receipt_plan,
                "artifact_type": "bookforge-jetson-tooling-install-receipt",
                "unit_sha256": sha256(UNIT),
            }
            receipt_bytes = canonical(receipt)
            version_receipt = STATE / "tooling-provenance" / f"{version}.json"
            if version_receipt.exists() or version_receipt.is_symlink():
                receipt_info = version_receipt.lstat()
                if (
                    version_receipt.is_symlink()
                    or not stat.S_ISREG(receipt_info.st_mode)
                    or receipt_info.st_uid != 0
                    or receipt_info.st_gid != 0
                    or version_receipt.read_bytes() != receipt_bytes
                ):
                    raise ValueError("existing immutable tooling receipt does not match")
            else:
                write_atomic(version_receipt, receipt_bytes, 0o444)
            current_receipt = STATE / "tooling-current.json"
            if current_receipt.exists() or current_receipt.is_symlink():
                current_receipt_info = current_receipt.lstat()
                if (
                    current_receipt.is_symlink()
                    or not stat.S_ISREG(current_receipt_info.st_mode)
                    or current_receipt_info.st_uid != 0
                    or current_receipt_info.st_gid != 0
                ):
                    raise ValueError("existing current tooling receipt is unsafe")
            write_atomic(current_receipt, receipt_bytes, 0o444)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    finally:
        os.close(lock)
    print(json.dumps({**plan, "installed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
