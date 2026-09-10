#!/usr/bin/env bash
# Build one tuned Gemma 4 export into an isolated, installable TensorRT candidate.

set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly VERIFIER="$SCRIPT_DIR/download-gcs-fidelity-tensorrt-candidate.py"
readonly DEFAULT_BUILDER="$SCRIPT_DIR/build-gemma4-tensorrt-edge-engine.sh"
readonly BUILD_TIMEOUT_SECONDS=1800
export_bundle=""
export_manifest_sha256=""
output=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: build-trained-planner-candidate.sh OPTIONS

Required:
  --export-bundle PATH                 Verified fidelity TensorRT export.
  --export-manifest-sha256 SHA256      Out-of-band export.manifest.json digest.
  --output PATH                        New candidate bundle destination.

Options:
  --dry-run     Verify inputs and print isolated build paths without changing files.
  -h, --help    Show this help.

Run this wrapper as the Storylight user, never with elevated privileges. A failed content-addressed
build is retained as terminal evidence and is never overwritten or promoted.
EOF
}

while (($#)); do
  case "$1" in
    --export-bundle)
      [[ $# -ge 2 ]] || { printf '%s\n' '--export-bundle requires a value' >&2; exit 64; }
      export_bundle="$2"
      shift 2
      ;;
    --export-manifest-sha256)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--export-manifest-sha256 requires a value' >&2
        exit 64
      }
      export_manifest_sha256="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { printf '%s\n' '--output requires a value' >&2; exit 64; }
      output="$2"
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run the trained candidate builder as the Storylight user, not root.\n' >&2
  exit 64
fi
if [[ -z "$export_bundle" || -z "$output" ]] \
  || [[ ! "$export_manifest_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  usage >&2
  exit 64
fi
if [[ ! -f "$VERIFIER" || -L "$VERIFIER" ]]; then
  printf 'The fidelity export verifier is missing or unsafe.\n' >&2
  exit 69
fi

verification_json="$(python3 "$VERIFIER" \
  --local-bundle "$export_bundle" \
  --expected-manifest-sha256 "$export_manifest_sha256" \
  --verify-only)"
readonly verification_json

manifest_field() {
  python3 -c \
    'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' \
    "$verification_json" "$1"
}
candidate_id="$(manifest_field candidate_id)"
source_files_sha256="$(manifest_field source_files_content_sha256)"
source_release_sha256="$(manifest_field source_release_manifest_sha256)"
config_sha256="$(manifest_field config_sha256)"
dataset_manifest_sha256="$(manifest_field dataset_manifest_sha256)"
training_run_id="$(manifest_field training_run_id)"
readonly candidate_id source_files_sha256 source_release_sha256 config_sha256
readonly dataset_manifest_sha256 training_run_id
readonly install_root="${STORYLIGHT_EDGELLM_ROOT:-$HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0}"
readonly build_root="${STORYLIGHT_TRAINED_CANDIDATE_BUILD_ROOT:-$install_root/trained-candidate-builds}"
readonly model_root="$build_root/$candidate_id/${export_manifest_sha256:0:20}/model"
readonly builder="${STORYLIGHT_TRAINED_CANDIDATE_BUILDER:-$DEFAULT_BUILDER}"

if [[ -e "$model_root" || -L "$model_root" || -e "$output" || -L "$output" ]]; then
  printf 'Refusing to overwrite candidate build state or output.\n' >&2
  exit 73
fi
if [[ ! -x "$builder" || -L "$builder" ]]; then
  printf 'The TensorRT engine builder is missing, non-executable, or unsafe: %s\n' "$builder" >&2
  exit 69
fi

python3 - "$model_root" "$output" <<'PY'
import sys
from pathlib import Path

model = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
for parent, child in ((model, output), (output, model)):
    try:
        child.relative_to(parent)
    except ValueError:
        continue
    raise ValueError("candidate output and isolated model root may not contain one another")
PY

if ((dry_run == 1)); then
  printf 'candidate_id=%s\n' "$candidate_id"
  printf 'source_model_revision=sha256:%s\n' "$source_files_sha256"
  printf 'model_root=%s\n' "$model_root"
  printf 'output=%s\n' "$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$output")"
  printf 'Dry run complete; no files changed and no TensorRT build started.\n'
  exit 0
fi

timeout_bin="${STORYLIGHT_TIMEOUT_BIN:-}"
if [[ -z "$timeout_bin" ]]; then
  timeout_bin="$(command -v timeout || true)"
fi
if [[ -z "$timeout_bin" ]] || [[ ! -x "$timeout_bin" || -L "$timeout_bin" ]]; then
  printf 'GNU timeout is required to bound the TensorRT engine build.\n' >&2
  exit 69
fi

python3 - "$export_bundle" "$model_root" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve() / "onnx"
target = Path(sys.argv[2]).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
try:
    shutil.copytree(source, temporary / "onnx", symlinks=False)
    shutil.copyfile(
        source.parent / "export.manifest.json",
        temporary / "export.manifest.json",
        follow_symlinks=False,
    )
    os.replace(temporary, target)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
PY

python3 "$VERIFIER" \
  --local-bundle "$model_root" \
  --expected-manifest-sha256 "$export_manifest_sha256" \
  --verify-only >/dev/null

"$timeout_bin" --signal=TERM --kill-after=30s "$BUILD_TIMEOUT_SECONDS" \
  env STORYLIGHT_EDGELLM_MODEL_ROOT="$model_root" "$builder"

readonly engine="$model_root/engines/llm/llm.engine"
if [[ ! -s "$engine" || -L "$engine" ]]; then
  printf 'The isolated TensorRT build did not produce a safe engine.\n' >&2
  exit 70
fi

python3 - \
  "$model_root" \
  "$output" \
  "$candidate_id" \
  "$export_manifest_sha256" \
  "$source_files_sha256" \
  "$source_release_sha256" \
  "$config_sha256" \
  "$dataset_manifest_sha256" \
  "$training_run_id" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

model_root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).expanduser().resolve()
candidate_id = sys.argv[3]
export_manifest_sha256 = sys.argv[4]
source_files_sha256 = sys.argv[5]
source_release_sha256 = sys.argv[6]
config_sha256 = sys.argv[7]
dataset_manifest_sha256 = sys.argv[8]
training_run_id = sys.argv[9]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


output.parent.mkdir(parents=True, exist_ok=True)
if output.exists() or output.is_symlink():
    raise FileExistsError(f"refusing to overwrite candidate bundle: {output}")
temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
try:
    for name in ("onnx", "engines"):
        source_root = model_root / name
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError(f"candidate model root lacks safe {name} output")
        for path in source_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("candidate model root may not contain symbolic links")
            if path.is_file():
                relative = path.relative_to(model_root)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target, follow_symlinks=False)

    engine = temporary / "engines/llm/llm.engine"
    if not engine.is_file() or not engine.stat().st_size:
        raise ValueError("candidate bundle lacks its fixed TensorRT engine")
    files = []
    for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "result": "complete",
        "candidate_id": candidate_id,
        "model_id": "google/gemma-4-E2B-it",
        "base_model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model_revision": f"sha256:{sha256(engine)}",
        "tensorrt_edge_llm_revision": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        "cache_contract_revision": "semantic-v18-tensorrt-slot-privacy",
        "planner_protocol": "four-slot-v1",
        "engine_path": "engines/llm/llm.engine",
        "engine_sha256": sha256(engine),
        "source_export_manifest_sha256": export_manifest_sha256,
        "source_release_manifest_sha256": source_release_sha256,
        "source_files_content_sha256": source_files_sha256,
        "source_config_sha256": config_sha256,
        "source_dataset_manifest_sha256": dataset_manifest_sha256,
        "training_run_id": training_run_id,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    manifest_path = temporary / "candidate.manifest.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    manifest_sha256 = sha256(manifest_path)
    os.replace(temporary, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
print(f"candidate_id={candidate_id}")
print(f"candidate_manifest_sha256={manifest_sha256}")
print(f"candidate_bundle={output}")
PY
