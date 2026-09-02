#!/usr/bin/env python3
"""Create and restore-test an encrypted, off-device fidelity custody backup."""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import pwd
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "bookforge-fidelity-custody-backup-v1"
KEYCHAIN_SERVICE = "com.bookforge.story-fidelity-v1.custody-backup"
DATASET_MANIFEST_SHA256 = "e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de"
CONFIG_SHA256 = "db6b3788aa555f89624f30d05a827f1911c0d62e5376e3aced40333dff833fd0"
PRIVATE_SHA256 = {
    "hidden.jsonl": "ef32a7ca378d8239352936b08665c3b45fe12dab47b8d5a1bf2356859f8db61c",
    "hidden.key": "0597ce504ae95b47d44e5454d7fce644bff92cc1976130222eb32bf6cd6cf4cd",
    "custody.json": "c0ecfb48ce0135b23a1eaf9a37ee6db57f4a1e854d394a46983a67bc4563a9f4",
}
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")
_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")
_FORBIDDEN_RECEIPT_KEYS = {"password", "passphrase", "secret", "key_material"}


class BackupError(RuntimeError):
    """A safe, user-facing custody backup failure."""


@dataclass(frozen=True)
class SourceArtifact:
    archive_name: str
    path: Path
    sha256: str
    private: bool


@dataclass(frozen=True)
class BackupConfiguration:
    repository: Path
    custody_root: Path
    backup_root: Path
    ssh_key: Path
    jetson_host: str
    jetson_user: str
    host_key_alias: str
    remote_directory: str
    check_only: bool
    recovery_key_ceremony: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lstat_regular(path: Path, *, label: str, private: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BackupError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise BackupError(f"{label} must be a regular file, not a link: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if private and mode != 0o600:
        raise BackupError(f"{label} must have mode 0600, not {mode:04o}: {path}")


def _validate_artifact(artifact: SourceArtifact) -> None:
    _lstat_regular(
        artifact.path,
        label=f"custody source {artifact.archive_name}",
        private=artifact.private,
    )
    actual = _sha256(artifact.path)
    if actual != artifact.sha256:
        raise BackupError(
            f"custody source {artifact.archive_name} changed: expected "
            f"{artifact.sha256}, got {actual}"
        )


def _run(
    command: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    label: str,
    suppress_output: bool = False,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            input=input_bytes,
            capture_output=True,
            check=False,
            close_fds=True,
            env=environment,
        )
    except OSError as error:
        raise BackupError(f"could not start {label}") from error
    if result.returncode != 0:
        if suppress_output:
            raise BackupError(f"{label} failed; no secret output was retained")
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        detail = detail[-2000:] if detail else "no diagnostic output"
        raise BackupError(f"{label} failed: {detail}")
    return result.stdout


def _command_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BackupError(f"required command is unavailable: {name}")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _secure_directory(path: Path, *, repository: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise BackupError(f"backup directory must be absolute: {path}")
    if path.is_symlink():
        raise BackupError(f"backup directory may not be a symbolic link: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = path.resolve(strict=True)
    if _inside(resolved, repository):
        raise BackupError("encrypted custody backups must remain outside the repository")
    if not resolved.is_dir() or resolved.is_symlink():
        raise BackupError(f"backup directory is unsafe: {resolved}")
    resolved.chmod(0o700)
    if stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise BackupError(f"backup directory could not be restricted to mode 0700: {resolved}")
    return resolved


def _source_artifacts(config: BackupConfiguration) -> tuple[SourceArtifact, ...]:
    return (
        *(
            SourceArtifact(name, config.custody_root / name, digest, True)
            for name, digest in PRIVATE_SHA256.items()
        ),
        SourceArtifact(
            "manifest.json",
            config.repository / "datasets/story-fidelity-v1/manifest.json",
            DATASET_MANIFEST_SHA256,
            False,
        ),
        SourceArtifact(
            "config.json",
            config.repository / "experiments/jax-fidelity-lab/config.json",
            CONFIG_SHA256,
            False,
        ),
    )


def _verification_environment(repository: Path) -> dict[str, str]:
    """Provide only the local import path and non-secret process basics."""

    return {
        "HOME": str(Path.home()),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(repository / "src"),
    }


def _verification_interpreter(repository: Path) -> str:
    candidate = repository / ".venv/bin/python"
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise BackupError("the repository Python environment is missing") from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(candidate, os.X_OK):
        raise BackupError("the repository Python environment is not executable")
    return str(candidate)


def _validate_configuration(config: BackupConfiguration) -> tuple[SourceArtifact, ...]:
    if sys.platform != "darwin":
        raise BackupError("custody sparseimage creation must run on macOS")
    if os.geteuid() == 0:
        raise BackupError("run custody backup as the signed-in user, never with sudo")
    if not _HOST.fullmatch(config.jetson_host):
        raise BackupError("Jetson host must be a DNS name or IPv4 address")
    if not _HOST.fullmatch(config.host_key_alias):
        raise BackupError("SSH host-key alias is invalid")
    if not _USER.fullmatch(config.jetson_user):
        raise BackupError("Jetson user is invalid")
    expected_remote = f"/home/{config.jetson_user}/.local/share/bookforge/custody-backups"
    if config.remote_directory != expected_remote:
        raise BackupError(
            "remote custody directory must be the fixed user-owned Bookforge backup path"
        )
    if config.repository.is_symlink() or not config.repository.is_dir():
        raise BackupError("repository path is missing or unsafe")
    if config.custody_root.is_symlink() or not config.custody_root.is_dir():
        raise BackupError("private custody root is missing or unsafe")
    if stat.S_IMODE(config.custody_root.stat().st_mode) != 0o700:
        raise BackupError("private custody root must have mode 0700")
    if _inside(config.custody_root.resolve(), config.repository.resolve()):
        raise BackupError("private custody must remain outside the repository")
    _lstat_regular(config.ssh_key, label="SSH identity", private=False)
    if stat.S_IMODE(config.ssh_key.stat().st_mode) & 0o077:
        raise BackupError("SSH identity must not be accessible by group or other users")
    for command in (
        "hdiutil",
        "osascript",
        "security",
        "ssh",
        "scp",
    ):
        _command_path(command)
    artifacts = _source_artifacts(config)
    for artifact in artifacts:
        _validate_artifact(artifact)
    verifier = config.repository / "scripts/build_fidelity_dataset.py"
    _lstat_regular(verifier, label="custody verifier", private=False)
    _run(
        [
            _verification_interpreter(config.repository),
            str(verifier),
            "--verify-only",
            "--output-dir",
            str(config.repository / "datasets/story-fidelity-v1"),
            "--private-hidden-output",
            str(config.custody_root / "hidden.jsonl"),
            "--hidden-key-file",
            str(config.custody_root / "hidden.key"),
            "--custody-receipt",
            str(config.custody_root / "custody.json"),
        ],
        label="private custody verification",
        environment=_verification_environment(config.repository),
    )
    return artifacts


def _keychain_passphrase(account: str) -> bytes:
    security = _command_path("security")
    lookup = subprocess.run(
        [
            security,
            "find-generic-password",
            "-a",
            account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        close_fds=True,
    )
    if lookup.returncode == 0:
        value = lookup.stdout.rstrip(b"\r\n")
        if len(value) < 32:
            raise BackupError("the existing custody Keychain item is unexpectedly short")
        return value

    value = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=")
    framework_path = ctypes.util.find_library("Security")
    if framework_path is None:
        raise BackupError("macOS Security framework is unavailable")
    framework = ctypes.CDLL(framework_path)
    add_password = framework.SecKeychainAddGenericPassword
    add_password.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    add_password.restype = ctypes.c_int32
    service = KEYCHAIN_SERVICE.encode()
    account_bytes = account.encode()
    service_buffer = ctypes.create_string_buffer(service)
    account_buffer = ctypes.create_string_buffer(account_bytes)
    value_buffer = ctypes.create_string_buffer(value)
    status = add_password(
        None,
        len(service),
        service_buffer,
        len(account_bytes),
        account_buffer,
        len(value),
        value_buffer,
        None,
    )
    if status != 0:
        raise BackupError(f"custody Keychain creation failed with OSStatus {status}")
    return value


def _confirm_offline_recovery(passphrase: bytes) -> None:
    try:
        value = passphrase.decode("ascii")
    except UnicodeDecodeError as error:
        raise BackupError("recovery key is not ASCII") from error
    if re.fullmatch(r"[A-Za-z0-9_-]{64}", value) is None:
        raise BackupError("recovery key has an unexpected format")
    script = "\n".join(
        (
            f'set recoveryKey to "{value}"',
            (
                'display dialog "Copy this recovery key to offline storage. Keep it away '
                'from this Mac and Jetson." default answer recoveryKey buttons '
                '{"Cancel", "I saved it offline"} default button "I saved it offline" '
                'cancel button "Cancel" with title "Bookforge custody recovery"'
            ),
            (
                'set confirmedKey to text returned of (display dialog "Retype the '
                'recovery key to verify your offline copy." default answer "" with hidden '
                'answer buttons {"Cancel", "Confirm"} default button "Confirm" cancel '
                'button "Cancel" with title "Bookforge custody recovery")'
            ),
            "if confirmedKey is not recoveryKey then",
            '    display alert "The recovery key did not match. No backup will be created."',
            "    error number -128",
            "end if",
            'return "confirmed"',
        )
    )
    result = _run(
        [_command_path("osascript"), "-"],
        input_bytes=script.encode("utf-8"),
        label="offline recovery-key ceremony",
        suppress_output=True,
    )
    if result.strip() != b"confirmed":
        raise BackupError("offline recovery-key ceremony returned an invalid confirmation")


def _local_account() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
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


def _copy_verified(artifact: SourceArtifact, destination: Path) -> None:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(artifact.path, source_flags)
    destination_fd = -1
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise BackupError(f"custody source changed type: {artifact.archive_name}")
        if artifact.private and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise BackupError(f"custody source changed mode: {artifact.archive_name}")
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            destination_flags |= os.O_NOFOLLOW
        destination_fd = os.open(destination, destination_flags, 0o600)
        digest = hashlib.sha256()
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fchmod(destination_fd, 0o600)
        os.fsync(destination_fd)
        if digest.hexdigest() != artifact.sha256:
            raise BackupError(f"custody source changed while copying: {artifact.archive_name}")
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)


def _hdiutil(passphrase: bytes, arguments: Sequence[str], *, label: str) -> bytes:
    return _run(
        [_command_path("hdiutil"), *arguments],
        input_bytes=passphrase + b"\n",
        label=label,
    )


def _attach_image(image: Path, mountpoint: Path, passphrase: bytes, *, readonly: bool) -> None:
    arguments = ["attach", "-quiet", "-nobrowse"]
    if readonly:
        arguments.append("-readonly")
    arguments.extend(["-mountpoint", str(mountpoint), "-stdinpass", str(image)])
    _hdiutil(passphrase, arguments, label="encrypted image attach")


def _detach_image(mountpoint: Path) -> None:
    _run(
        [_command_path("hdiutil"), "detach", "-quiet", str(mountpoint)],
        label="encrypted image detach",
    )


def _metadata_payload(artifacts: Sequence[SourceArtifact], created_at: str) -> bytes:
    document = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "encryption": "AES-256",
        "key_custody": "macOS Keychain",
        "artifacts": [
            {
                "name": artifact.archive_name,
                "sha256": artifact.sha256,
                "mode": "0600",
            }
            for artifact in artifacts
        ],
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _populate_image(
    mountpoint: Path, artifacts: Sequence[SourceArtifact], created_at: str
) -> Mapping[str, str]:
    expected: dict[str, str] = {}
    for artifact in artifacts:
        _copy_verified(artifact, mountpoint / artifact.archive_name)
        expected[artifact.archive_name] = artifact.sha256
    metadata = _metadata_payload(artifacts, created_at)
    metadata_path = mountpoint / "backup-metadata.json"
    _write_exclusive(metadata_path, metadata)
    expected[metadata_path.name] = hashlib.sha256(metadata).hexdigest()
    checksum_payload = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(expected.items())
    ).encode()
    _write_exclusive(mountpoint / "SHA256SUMS", checksum_payload)
    os.sync()
    return expected


def _verify_image_contents(mountpoint: Path, expected: Mapping[str, str]) -> None:
    checksum_lines: list[str] = []
    for name, digest in sorted(expected.items()):
        path = mountpoint / name
        _lstat_regular(path, label=f"restored {name}", private=True)
        actual = _sha256(path)
        if actual != digest:
            raise BackupError(f"restored custody artifact changed: {name}")
        checksum_lines.append(f"{digest}  {name}\n")
    checksum_path = mountpoint / "SHA256SUMS"
    _lstat_regular(checksum_path, label="restored SHA256SUMS", private=True)
    if checksum_path.read_text(encoding="utf-8") != "".join(checksum_lines):
        raise BackupError("restored SHA256SUMS does not match the expected custody set")


def _mount_verify(
    image: Path,
    passphrase: bytes,
    expected: Mapping[str, str],
    temporary_root: Path,
    label: str,
) -> None:
    mountpoint = temporary_root / f"mount-{label}"
    mountpoint.mkdir(mode=0o700)
    attached = False
    try:
        _attach_image(image, mountpoint, passphrase, readonly=True)
        attached = True
        _verify_image_contents(mountpoint, expected)
    finally:
        if attached:
            _detach_image(mountpoint)


def _ssh_options(config: BackupConfiguration) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        f"HostKeyAlias={config.host_key_alias}",
        "-i",
        str(config.ssh_key),
    ]


def _ssh_script(
    config: BackupConfiguration,
    script: str,
    arguments: Sequence[str],
    *,
    label: str,
) -> bytes:
    return _run(
        [
            _command_path("ssh"),
            *_ssh_options(config),
            f"{config.jetson_user}@{config.jetson_host}",
            "/bin/bash",
            "-s",
            "--",
            *arguments,
        ],
        input_bytes=script.encode(),
        label=label,
    )


_REMOTE_PREPARE = r"""set -eu
umask 077
remote_directory=$1
partial_path=$2
final_path=$3
remote_user=$4
expected_directory="/home/${remote_user}/.local/share/bookforge/custody-backups"
[ "$remote_directory" = "$expected_directory" ] || exit 64
case "$partial_path" in "$remote_directory"/.*.partial.sparseimage) ;; *) exit 64 ;; esac
case "$final_path" in "$remote_directory"/story-fidelity-v1-*.sparseimage) ;; *) exit 64 ;; esac
[ ! -L "$remote_directory" ] || exit 65
install -d -m 0700 "$remote_directory"
[ "$(stat -c '%U:%G:%a' "$remote_directory")" = "${remote_user}:${remote_user}:700" ] || exit 65
[ ! -e "$partial_path" ] && [ ! -L "$partial_path" ] || exit 73
[ ! -e "$final_path" ] && [ ! -L "$final_path" ] || exit 73
"""

_REMOTE_PUBLISH = r"""set -eu
partial_path=$1
final_path=$2
expected_sha256=$3
[ -f "$partial_path" ] && [ ! -L "$partial_path" ] || exit 65
actual_sha256=$(sha256sum "$partial_path" | cut -d ' ' -f 1)
[ "$actual_sha256" = "$expected_sha256" ] || exit 65
chmod 0600 "$partial_path"
mv "$partial_path" "$final_path"
[ -f "$final_path" ] && [ ! -L "$final_path" ] || exit 65
mode=$(stat -c '%a' "$final_path")
[ "$mode" = 600 ] || exit 65
printf '%s\n%s\n' "$actual_sha256" "$mode"
"""

_REMOTE_CLEAN_PARTIAL = r"""set -eu
partial_path=$1
remote_directory=$2
case "$partial_path" in "$remote_directory"/.*.partial.sparseimage) ;; *) exit 64 ;; esac
if [ -f "$partial_path" ] && [ ! -L "$partial_path" ]; then
  rm -f "$partial_path"
fi
"""


def _remote_prepare(config: BackupConfiguration, partial_path: str, final_path: str) -> None:
    _ssh_script(
        config,
        _REMOTE_PREPARE,
        [config.remote_directory, partial_path, final_path, config.jetson_user],
        label="Jetson custody destination preparation",
    )


def _copy_ciphertext_to_remote(
    config: BackupConfiguration, archive: Path, partial_path: str
) -> None:
    _run(
        [
            _command_path("scp"),
            *_ssh_options(config),
            "-p",
            str(archive),
            f"{config.jetson_user}@{config.jetson_host}:{partial_path}",
        ],
        label="encrypted custody image upload",
    )


def _remote_publish(
    config: BackupConfiguration,
    partial_path: str,
    final_path: str,
    expected_sha256: str,
) -> None:
    result = (
        _ssh_script(
            config,
            _REMOTE_PUBLISH,
            [partial_path, final_path, expected_sha256],
            label="Jetson ciphertext verification",
        )
        .decode("utf-8", errors="strict")
        .splitlines()
    )
    if result != [expected_sha256, "600"]:
        raise BackupError("Jetson returned unexpected ciphertext verification evidence")


def _cleanup_remote_partial(config: BackupConfiguration, partial_path: str) -> None:
    with contextlib.suppress(BackupError):
        _ssh_script(
            config,
            _REMOTE_CLEAN_PARTIAL,
            [partial_path, config.remote_directory],
            label="Jetson partial cleanup",
        )


def _copy_ciphertext_from_remote(
    config: BackupConfiguration, final_path: str, destination: Path
) -> None:
    _run(
        [
            _command_path("scp"),
            *_ssh_options(config),
            "-p",
            f"{config.jetson_user}@{config.jetson_host}:{final_path}",
            str(destination),
        ],
        label="encrypted custody image restore",
    )


def _assert_non_secret_receipt(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(token in normalized for token in _FORBIDDEN_RECEIPT_KEYS):
                raise BackupError("receipt attempted to persist secret material")
            _assert_non_secret_receipt(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_non_secret_receipt(nested)


def _write_receipt(path: Path, document: Mapping[str, Any]) -> None:
    _assert_non_secret_receipt(document)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    _write_exclusive(path, payload, 0o600)


def _receipt_document(
    *,
    config: BackupConfiguration,
    backup_id: str,
    created_at: str,
    archive: Path,
    archive_sha256: str,
    remote_path: str,
    artifacts: Sequence[SourceArtifact],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "backup_id": backup_id,
        "created_at": created_at,
        "source_verification": {
            "status": "passed",
            "artifacts": [
                {
                    "name": artifact.archive_name,
                    "sha256": artifact.sha256,
                    "mode": "0600" if artifact.private else "repository-controlled",
                }
                for artifact in artifacts
            ],
        },
        "encryption": {
            "container": "Apple sparse disk image",
            "cipher": "AES-256",
            "credential_store": "macOS Keychain",
            "keychain_service": KEYCHAIN_SERVICE,
            "keychain_account": _local_account(),
            "independent_recovery": {
                "method": "operator-transcribed offline paper copy",
                "confirmed": True,
            },
        },
        "local_archive": {
            "path": str(archive),
            "sha256": archive_sha256,
            "mode": "0600",
            "read_only_restore_verified": True,
        },
        "jetson_archive": {
            "host": config.jetson_host,
            "host_key_alias": config.host_key_alias,
            "user": config.jetson_user,
            "path": remote_path,
            "sha256": archive_sha256,
            "mode": "0600",
            "ciphertext_only": True,
            "round_trip_restore_verified": True,
        },
    }


def run_backup(config: BackupConfiguration) -> Path | None:
    artifacts = _validate_configuration(config)
    if config.check_only:
        print("Exact private custody inputs and public bindings are verified.")
        print(f"Planned encrypted destination: {config.remote_directory}")
        print("Check-only mode did not access Keychain, create an image, or contact Jetson.")
        return None
    if not config.recovery_key_ceremony:
        raise BackupError(
            "execution requires --recovery-key-ceremony so the archive is recoverable "
            "without this Mac"
        )

    backup_root = _secure_directory(config.backup_root, repository=config.repository.resolve())
    created = datetime.now(UTC)
    created_at = created.isoformat()
    backup_id = f"story-fidelity-v1-{created:%Y%m%dT%H%M%SZ}-{secrets.token_hex(6)}"
    archive = backup_root / f"{backup_id}.sparseimage"
    receipt = backup_root / f"{backup_id}.receipt.json"
    if archive.exists() or archive.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise BackupError("new custody backup path already exists")

    passphrase = _keychain_passphrase(_local_account())
    _confirm_offline_recovery(passphrase)
    remote_final = f"{config.remote_directory}/{archive.name}"
    remote_partial = f"{config.remote_directory}/.{backup_id}.partial.sparseimage"
    remote_is_partial = False
    try:
        with tempfile.TemporaryDirectory(prefix="bookforge-custody-backup-") as temporary:
            temporary_root = Path(temporary)
            staging_image = temporary_root / archive.name
            _hdiutil(
                passphrase,
                [
                    "create",
                    "-quiet",
                    "-size",
                    "64m",
                    "-type",
                    "SPARSE",
                    "-fs",
                    "APFS",
                    "-volname",
                    "BookforgeCustodyV1",
                    "-encryption",
                    "AES-256",
                    "-stdinpass",
                    str(staging_image),
                ],
                label="encrypted custody image creation",
            )
            _lstat_regular(staging_image, label="staged encrypted custody image", private=False)
            staging_image.chmod(0o600)

            writable_mount = temporary_root / "mount-write"
            writable_mount.mkdir(mode=0o700)
            attached = False
            try:
                _attach_image(staging_image, writable_mount, passphrase, readonly=False)
                attached = True
                expected = _populate_image(writable_mount, artifacts, created_at)
            finally:
                if attached:
                    _detach_image(writable_mount)

            _mount_verify(
                staging_image,
                passphrase,
                expected,
                temporary_root,
                "local",
            )
            os.replace(staging_image, archive)
            archive.chmod(0o600)
            archive_sha256 = _sha256(archive)

            _remote_prepare(config, remote_partial, remote_final)
            remote_is_partial = True
            _copy_ciphertext_to_remote(config, archive, remote_partial)
            _remote_publish(config, remote_partial, remote_final, archive_sha256)
            remote_is_partial = False

            restored = temporary_root / f"restored-{archive.name}"
            _copy_ciphertext_from_remote(config, remote_final, restored)
            _lstat_regular(restored, label="round-trip restored ciphertext", private=False)
            if _sha256(restored) != archive_sha256:
                raise BackupError("round-trip restored ciphertext SHA-256 changed")
            _mount_verify(
                restored,
                passphrase,
                expected,
                temporary_root,
                "jetson-roundtrip",
            )

        document = _receipt_document(
            config=config,
            backup_id=backup_id,
            created_at=created_at,
            archive=archive,
            archive_sha256=archive_sha256,
            remote_path=remote_final,
            artifacts=artifacts,
        )
        _write_receipt(receipt, document)
    finally:
        if remote_is_partial:
            _cleanup_remote_partial(config, remote_partial)
        passphrase = b""

    print("Custody backup and Jetson round-trip restore verification completed.")
    print(f"Encrypted archive: {archive}")
    print(f"Ciphertext SHA-256: {archive_sha256}")
    print(f"Jetson ciphertext: {remote_final}")
    print(f"Non-secret receipt: {receipt}")
    return receipt


def _parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custody-root",
        type=Path,
        default=home / "Library/Application Support/Bookforge/fidelity/story-fidelity-v1",
    )
    parser.add_argument(
        "--backup-directory",
        type=Path,
        default=home / "Library/Application Support/Bookforge/fidelity/backups",
    )
    parser.add_argument("--ssh-key", type=Path, default=home / ".ssh/bookforge_jetson")
    parser.add_argument("--jetson-host", required=True)
    parser.add_argument("--jetson-user", required=True)
    parser.add_argument("--host-key-alias", required=True)
    parser.add_argument(
        "--recovery-key-ceremony",
        action="store_true",
        help="show a private macOS recovery dialog and require offline-copy confirmation",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify exact local inputs without Keychain, image, network, or writes",
    )
    parser.set_defaults(repository=repository)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = BackupConfiguration(
        repository=args.repository.resolve(),
        custody_root=args.custody_root.expanduser(),
        backup_root=args.backup_directory.expanduser(),
        ssh_key=args.ssh_key.expanduser(),
        jetson_host=args.jetson_host,
        jetson_user=args.jetson_user,
        host_key_alias=args.host_key_alias,
        remote_directory=(f"/home/{args.jetson_user}/.local/share/bookforge/custody-backups"),
        check_only=args.check_only,
        recovery_key_ceremony=args.recovery_key_ceremony,
    )
    try:
        run_backup(config)
    except BackupError as error:
        print(f"Custody backup refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
