#!/usr/bin/env bash
# Snapshot the exact accepted TensorRT engine into an immutable baseline identity bundle.

set -euo pipefail
IFS=$'\n\t'

accepted_engine="${STORYLIGHT_ACCEPTED_ENGINE:-$HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine}"
dataset_manifest_sha256=""
output=""

usage() {
  cat <<'EOF'
Usage: emit-accepted-planner-identity.sh OPTIONS

Required:
  --source-dataset-manifest-sha256 SHA256
  --output PATH

Options:
  --accepted-engine PATH  Override the fixed accepted engine path (tests/recovery only).
  -h, --help              Show this help.

The output is a new read-only candidate-style bundle. Routing and services are never changed.
EOF
}

while (($#)); do
  case "$1" in
    --source-dataset-manifest-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--source-dataset-manifest-sha256 requires a value' >&2; exit 64; }
      dataset_manifest_sha256="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { printf '%s\n' '--output requires a value' >&2; exit 64; }
      output="$2"
      shift 2
      ;;
    --accepted-engine)
      [[ $# -ge 2 ]] || { printf '%s\n' '--accepted-engine requires a value' >&2; exit 64; }
      accepted_engine="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run baseline identity capture as the Storylight service user, not root.\n' >&2
  exit 64
fi
if [[ ! "$dataset_manifest_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ -z "$output" || "$output" != /* ]]; then
  usage >&2
  exit 64
fi

python3 - "$accepted_engine" "$dataset_manifest_sha256" "$output" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).expanduser()
dataset_manifest_sha256 = sys.argv[2]
output = Path(sys.argv[3])
if source.is_symlink() or not source.is_file():
    raise ValueError("accepted engine is missing or unsafe")
if output.exists() or output.is_symlink():
    raise FileExistsError(f"refusing to overwrite baseline identity: {output}")
output.parent.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
try:
    engine = temporary / "engines/llm/llm.engine"
    engine.parent.mkdir(parents=True)
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or not before.st_size:
            raise ValueError("accepted engine is not a non-empty regular file")
        target_fd = os.open(
            engine,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            while block := os.read(source_fd, 8 * 1024 * 1024):
                digest.update(block)
                copied += len(block)
                view = memoryview(block)
                while view:
                    view = view[os.write(target_fd, view) :]
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)
    engine_sha256 = digest.hexdigest()
    after = source.stat(follow_symlinks=False)
    if copied != before.st_size or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise ValueError("accepted engine changed while its identity was captured")
    if sha256(source) != engine_sha256 or sha256(engine) != engine_sha256:
        raise ValueError("accepted engine changed while its identity was verified")

    candidate_id = f"accepted-baseline-{engine_sha256[:20]}"
    manifest = {
        "schema_version": "1.0",
        "result": "complete",
        "identity_type": "accepted-baseline",
        "candidate_id": candidate_id,
        "model_id": "google/gemma-4-E2B-it",
        "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model_revision": f"sha256:{engine_sha256}",
        "engine_sha256": engine_sha256,
        "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        "cache_contract_revision": "semantic-v18-tensorrt-slot-privacy",
        "planner_protocol": "four-slot-v1",
        "engine_path": "engines/llm/llm.engine",
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "file_count": 1,
        "total_bytes": copied,
        "files": [
            {
                "path": "engines/llm/llm.engine",
                "bytes": copied,
                "sha256": engine_sha256,
            }
        ],
    }
    manifest_path = temporary / "candidate.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = sha256(manifest_path)
    for directory, directories, filenames in os.walk(temporary):
        os.chmod(directory, 0o500)
        for name in directories:
            os.chmod(Path(directory) / name, 0o500)
        for name in filenames:
            os.chmod(Path(directory) / name, 0o400)
    os.replace(temporary, output)
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
print(f"baseline_candidate_id={candidate_id}")
print(f"baseline_model_revision=sha256:{engine_sha256}")
print(f"baseline_engine_sha256={engine_sha256}")
print(f"baseline_manifest_sha256={manifest_sha256}")
print(f"baseline_manifest={output / 'candidate.manifest.json'}")
PY
