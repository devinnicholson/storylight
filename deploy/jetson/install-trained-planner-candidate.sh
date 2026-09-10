#!/usr/bin/env bash
# Verify and atomically install one immutable trained TensorRT planner candidate.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CANDIDATE_ROOT="/var/lib/storylight-trusted/trained-planner-candidates"
readonly UNIT_SOURCE="${STORYLIGHT_CANDIDATE_UNIT_SOURCE:-$SCRIPT_DIR/systemd/storylight-trained-planner-candidate@.service}"
readonly UNIT_TARGET="/etc/systemd/user/storylight-trained-planner-candidate@.service"
readonly PYTHON="python3"
bundle=""
expected_manifest_sha256=""
target_user="${STORYLIGHT_SERVICE_USER:-${SUDO_USER:-}}"
approval_token=""
dry_run=0
verify_only=0
installed_layout=0

usage() {
  cat <<'EOF'
Usage: sudo install-trained-planner-candidate.sh OPTIONS

Required:
  --bundle PATH                       Candidate bundle or installed directory.
  --expected-manifest-sha256 SHA256   Out-of-band candidate.manifest.json digest.

Options:
  --user USER          Storylight service user (required for installation).
  --approval-token TOKEN  Exact token printed by --dry-run.
  --verify-only        Verify the bundle without writing.
  --installed-layout   Permit the installer-created .manifest.sha256 marker.
  --dry-run            Verify and print the destination without writing.
  -h, --help           Show this help.

The bundle must contain candidate.manifest.json and exactly the files declared
by that manifest. Installation never overwrites an existing candidate.
EOF
}

while (($#)); do
  case "$1" in
    --bundle)
      [[ $# -ge 2 ]] || { printf '%s\n' '--bundle requires a value' >&2; exit 64; }
      bundle="$2"
      shift 2
      ;;
    --expected-manifest-sha256)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--expected-manifest-sha256 requires a value' >&2
        exit 64
      }
      expected_manifest_sha256="$2"
      shift 2
      ;;
    --user)
      [[ $# -ge 2 ]] || { printf '%s\n' '--user requires a value' >&2; exit 64; }
      target_user="$2"
      shift 2
      ;;
    --approval-token)
      [[ $# -ge 2 ]] || { printf '%s\n' '--approval-token requires a value' >&2; exit 64; }
      approval_token="$2"
      shift 2
      ;;
    --verify-only) verify_only=1; shift ;;
    --installed-layout) installed_layout=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ -z "$bundle" ]] || [[ ! "$expected_manifest_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  usage >&2
  exit 64
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Python is required to validate the candidate manifest.\n' >&2
  exit 69
fi
readonly EXPECTED_APPROVAL_TOKEN="INSTALL_STORYLIGHT_TRAINED_PLANNER_CANDIDATE:${target_user}:${expected_manifest_sha256}"
if ((verify_only == 0 && dry_run == 0)) && [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Candidate installation requires the exact one-purpose token printed by --dry-run.\n' >&2
  exit 77
fi
if ((verify_only == 0 && dry_run == 0)); then
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    printf 'Run candidate installation with sudo.\n' >&2
    exit 64
  fi
  if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
    || ! id "$target_user" >/dev/null 2>&1; then
    printf 'Select a valid Storylight service user with --user.\n' >&2
    exit 65
  fi
  if [[ ! -r "$UNIT_SOURCE" || -L "$UNIT_SOURCE" ]] \
    || [[ "$(stat -c '%U:%G:%a' "$UNIT_SOURCE")" != "root:root:644" \
      && "$(stat -c '%U:%G:%a' "$UNIT_SOURCE")" != "root:root:444" ]]; then
    printf 'The root-owned candidate systemd template is missing.\n' >&2
    exit 69
  fi
fi

"$PYTHON" - "$bundle" "$expected_manifest_sha256" "$CANDIDATE_ROOT" \
  "$verify_only" "$dry_run" "$installed_layout" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import atexit
import fcntl
from pathlib import Path, PurePosixPath

source_bundle = Path(sys.argv[1]).expanduser()
if source_bundle.is_symlink():
    raise ValueError("candidate bundle root may not be a symbolic link")
bundle = source_bundle.resolve()
expected_manifest_sha256 = sys.argv[2]
candidate_root = Path(sys.argv[3])
verify_only = sys.argv[4] == "1"
dry_run = sys.argv[5] == "1"
installed_layout = sys.argv[6] == "1"
cleanup_paths: list[Path] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("candidate manifest has an invalid file path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe candidate path: {value!r}")
    return Path(*path.parts)


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"candidate manifest repeats JSON key: {key}")
        result[key] = value
    return result


def cleanup() -> None:
    for path in cleanup_paths:
        shutil.rmtree(path, ignore_errors=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_untrusted_bundle(source: Path, destination: Path) -> None:
    source_stat = source.stat(follow_symlinks=False)
    if not stat.S_ISDIR(source_stat.st_mode) or source_stat.st_mode & 0o022:
        raise ValueError("candidate bundle root must be a non-group-writable directory")
    counters = {"files": 0, "bytes": 0}

    def copy_directory(source_fd: int, target: Path) -> None:
        entries = sorted(os.scandir(source_fd), key=lambda entry: entry.name)
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            if entry_stat.st_mode & 0o022:
                raise ValueError(f"candidate source has an unsafe writable mode: {entry.name}")
            if stat.S_ISDIR(entry_stat.st_mode):
                target_directory = target / entry.name
                target_directory.mkdir(mode=0o700)
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=source_fd,
                )
                try:
                    opened_stat = os.fstat(child_fd)
                    if (opened_stat.st_dev, opened_stat.st_ino) != (
                        entry_stat.st_dev,
                        entry_stat.st_ino,
                    ) or opened_stat.st_mode & 0o022:
                        raise ValueError("candidate source changed while it was staged")
                    copy_directory(child_fd, target_directory)
                    fsync_directory(target_directory)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ValueError(f"candidate source contains a non-regular file: {entry.name}")
            source_file_fd = os.open(
                entry.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=source_fd,
            )
            target_file = target / entry.name
            try:
                opened_stat = os.fstat(source_file_fd)
                if (
                    not stat.S_ISREG(opened_stat.st_mode)
                    or opened_stat.st_mode & 0o022
                    or (opened_stat.st_dev, opened_stat.st_ino)
                    != (entry_stat.st_dev, entry_stat.st_ino)
                ):
                    raise ValueError("candidate source changed while it was staged")
                counters["files"] += 1
                counters["bytes"] += opened_stat.st_size
                if counters["files"] > 4096 or counters["bytes"] > 64 * 1024**3:
                    raise ValueError("candidate bundle exceeds the bounded staging limits")
                target_fd = os.open(
                    target_file,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
                try:
                    while block := os.read(source_file_fd, 8 * 1024 * 1024):
                        view = memoryview(block)
                        while view:
                            written = os.write(target_fd, view)
                            view = view[written:]
                    os.fsync(target_fd)
                finally:
                    os.close(target_fd)
            finally:
                os.close(source_file_fd)

    source_fd = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        opened_root = os.fstat(source_fd)
        if (
            (opened_root.st_dev, opened_root.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
            or opened_root.st_mode & 0o022
        ):
            raise ValueError("candidate bundle root changed while it was staged")
        copy_directory(source_fd, destination)
        fsync_directory(destination)
    finally:
        os.close(source_fd)


atexit.register(cleanup)
if not verify_only and not dry_run:
    if candidate_root.exists():
        root_stat = candidate_root.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != 0
            or root_stat.st_gid != 0
            or root_stat.st_mode & 0o022
        ):
            raise ValueError("candidate root is not a safe root-owned directory")
    else:
        candidate_root.mkdir(parents=True, mode=0o755)
    os.chown(candidate_root, 0, 0)
    os.chmod(candidate_root, 0o755)
    staging = Path(tempfile.mkdtemp(prefix=".incoming.partial-", dir=candidate_root))
    cleanup_paths.append(staging)
    copy_untrusted_bundle(bundle, staging)
    bundle = staging

manifest_path = bundle / "candidate.manifest.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise ValueError("candidate.manifest.json is missing or unsafe")
if sha256(manifest_path) != expected_manifest_sha256:
    raise ValueError("candidate manifest checksum mismatch")
try:
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"candidate manifest is unreadable: {error}") from error
if not isinstance(manifest, dict):
    raise ValueError("candidate manifest must be a JSON object")

required = {
    "schema_version": "1.0",
    "result": "complete",
    "model_id": "google/gemma-4-E2B-it",
    "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
    "cache_contract_revision": "semantic-v18-tensorrt-slot-privacy",
    "planner_protocol": "four-slot-v1",
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise ValueError(f"candidate manifest {key!r} does not match the pinned value")

candidate_id = manifest.get("candidate_id")
if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,95}", candidate_id):
    raise ValueError("candidate manifest contains an invalid candidate_id")
model_revision = manifest.get("model_revision")
if not isinstance(model_revision, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", model_revision):
    raise ValueError("candidate manifest contains an invalid model_revision")
engine_path = safe_path(manifest.get("engine_path"))
if engine_path != Path("engines/llm/llm.engine"):
    raise ValueError("candidate engine_path is not the fixed TensorRT slot engine path")

files = manifest.get("files")
if not isinstance(files, list) or not files:
    raise ValueError("candidate manifest has no files")
if manifest.get("file_count") != len(files):
    raise ValueError("candidate manifest file_count does not match files")
declared: set[Path] = set()
total_bytes = 0
for entry in files:
    if not isinstance(entry, dict):
        raise ValueError("candidate manifest contains an invalid file entry")
    relative = safe_path(entry.get("path"))
    if relative in declared:
        raise ValueError(f"candidate manifest repeats {relative.as_posix()}")
    declared.add(relative)
    source = bundle / relative
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"candidate file is missing or unsafe: {relative.as_posix()}")
    size = source.stat().st_size
    if entry.get("bytes") != size:
        raise ValueError(f"candidate file size mismatch: {relative.as_posix()}")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
        raise ValueError(f"candidate file checksum is invalid: {relative.as_posix()}")
    if sha256(source) != expected_sha:
        raise ValueError(f"candidate file checksum mismatch: {relative.as_posix()}")
    total_bytes += size
if engine_path not in declared:
    raise ValueError("candidate manifest does not declare the TensorRT engine")
if manifest.get("total_bytes") != total_bytes:
    raise ValueError("candidate manifest total_bytes does not match files")
actual_engine_sha256 = sha256(bundle / engine_path)
if manifest.get("engine_sha256") != actual_engine_sha256:
    raise ValueError("candidate manifest engine_sha256 does not match the engine")
if model_revision != f"sha256:{actual_engine_sha256}":
    raise ValueError("candidate model_revision is not the TensorRT engine digest")

actual: set[Path] = set()
for root, directories, filenames in os.walk(bundle, followlinks=False):
    root_path = Path(root)
    for name in directories:
        if (root_path / name).is_symlink():
            raise ValueError("candidate bundle contains a symlink directory")
    for name in filenames:
        path = root_path / name
        if path.is_symlink():
            raise ValueError("candidate bundle contains a symlink file")
        relative = path.relative_to(bundle)
        if relative == Path("candidate.manifest.json"):
            continue
        if installed_layout and relative == Path(".manifest.sha256"):
            if path.read_text(encoding="ascii").strip() != expected_manifest_sha256:
                raise ValueError("installed manifest marker does not match")
            continue
        actual.add(relative)
if actual != declared:
    raise ValueError("candidate bundle contains undeclared or missing files")

destination = candidate_root / candidate_id
print(f"candidate_id={candidate_id}")
print(f"model_revision={model_revision}")
print(f"engine_sha256={actual_engine_sha256}")
print(f"destination={destination}")
if verify_only:
    print("Candidate verification complete; no files changed.")
    raise SystemExit(0)
if dry_run:
    print("Dry run complete; no files or services changed.")
    raise SystemExit(0)
if destination.exists():
    raise FileExistsError(f"refusing to overwrite installed candidate: {destination}")

temporary = bundle
try:
    marker = temporary / ".manifest.sha256"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(expected_manifest_sha256 + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    for root, directories, filenames in os.walk(temporary, followlinks=False):
        root_path = Path(root)
        os.chown(root_path, 0, 0)
        os.chmod(root_path, 0o555)
        for name in directories:
            path = root_path / name
            os.chown(path, 0, 0)
            os.chmod(path, 0o555)
        for name in filenames:
            path = root_path / name
            os.chown(path, 0, 0)
            os.chmod(path, 0o444)
    lock_path = candidate_root / ".install.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(lock_descriptor, 0o600)
        os.fchown(lock_descriptor, 0, 0)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite installed candidate: {destination}")
        os.replace(temporary, destination)
        fsync_directory(candidate_root)
        cleanup_paths.remove(temporary)
    finally:
        os.close(lock_descriptor)
except BaseException:
    raise
print(f"Installed immutable candidate: {destination}")
PY

if ((verify_only == 1 || dry_run == 1)); then
  if ((dry_run == 1)); then
    printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  fi
  exit 0
fi

install -d -o root -g root -m 0755 /etc/systemd/user
install -o root -g root -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"
target_uid="$(id -u "$target_user")"
runuser -u "$target_user" -- env \
  XDG_RUNTIME_DIR="/run/user/${target_uid}" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${target_uid}/bus" \
  systemctl --user daemon-reload
printf 'Candidate installed side by side; no service was enabled or started.\n'
