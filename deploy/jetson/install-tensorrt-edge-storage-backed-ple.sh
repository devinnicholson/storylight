#!/usr/bin/env bash
# Install Bookforge's exact NVMe-backed Gemma 4 PLE runtime optimization.

set -euo pipefail

readonly EDGELLM_REVISION="71dd1bae032e70771265917ec74d3ff4cad07a10"
readonly INSTALL_ROOT="${BOOKFORGE_EDGELLM_ROOT:-$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0}"
readonly SOURCE_DIR="$INSTALL_ROOT/src"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly VENV_DIR="$INSTALL_ROOT/venv"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly OVERRIDE_ROOT="$SCRIPT_DIR/tensorrt-edge-llm-overrides"
readonly PREPROCESSOR_REL="cpp/runtime/preprocess/gemma4EmbeddingPreprocessor.cpp"
readonly HEADER_REL="cpp/runtime/preprocess/gemma4EmbeddingPreprocessor.h"
readonly UPSTREAM_PREPROCESSOR_SHA="270828e2d573d42ee7fbb8b4b6a60ef8506bdc32c4f97a208474114349e085da"
readonly UPSTREAM_HEADER_SHA="017f8f80e93bff0572036cf732c4a01a603d0ca53f45c17c8de1c163e9abfb1d"
readonly EVIDENCE_DIR="$INSTALL_ROOT/evidence/storage-backed-ple"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this installer as the Bookforge user, not root.\n' >&2
  exit 64
fi
if [[ ! -d "$SOURCE_DIR/.git" ]] || [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) != "$EDGELLM_REVISION" ]]; then
  printf 'Refusing to patch an unpinned TensorRT Edge-LLM checkout.\n' >&2
  exit 65
fi
for required in "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.cpp" \
  "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.h" "$BUILD_DIR/build.ninja" \
  "$VENV_DIR/bin/cmake"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required storage-backed PLE input is missing: %s\n' "$required" >&2
    exit 69
  fi
done

verify_replaceable() {
  local target="$1"
  local replacement="$2"
  local upstream_sha="$3"
  local current_sha replacement_sha
  current_sha=$(sha256sum "$target" | cut -d ' ' -f 1)
  replacement_sha=$(sha256sum "$replacement" | cut -d ' ' -f 1)
  if [[ "$current_sha" != "$upstream_sha" && "$current_sha" != "$replacement_sha" ]]; then
    printf 'Refusing to overwrite an unexpected local source change: %s\n' "$target" >&2
    exit 73
  fi
}

verify_replaceable "$SOURCE_DIR/$PREPROCESSOR_REL" \
  "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.cpp" "$UPSTREAM_PREPROCESSOR_SHA"
verify_replaceable "$SOURCE_DIR/$HEADER_REL" \
  "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.h" "$UPSTREAM_HEADER_SHA"

install -m 0644 "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.cpp" \
  "$SOURCE_DIR/$PREPROCESSOR_REL"
install -m 0644 "$OVERRIDE_ROOT/runtime/preprocess/gemma4EmbeddingPreprocessor.h" \
  "$SOURCE_DIR/$HEADER_REL"

export PATH="/usr/local/cuda/bin:$VENV_DIR/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build "$BUILD_DIR" --target llm_inference _edgellm_runtime -j1
test -x "$BUILD_DIR/examples/llm/llm_inference"
find "$BUILD_DIR" -type f -name '*_edgellm_runtime*.so' -print -quit | grep -q .

mkdir -p "$EVIDENCE_DIR"
{
  printf 'edgellm_revision=%s\n' "$EDGELLM_REVISION"
  printf 'installed_at=%s\n' "$(date --iso-8601=seconds)"
  sha256sum "$SOURCE_DIR/$PREPROCESSOR_REL" "$SOURCE_DIR/$HEADER_REL" \
    "$BUILD_DIR/examples/llm/llm_inference"
} >"$EVIDENCE_DIR/install.txt"
printf 'Storage-backed Gemma 4 PLE runtime is installed. Evidence: %s\n' "$EVIDENCE_DIR/install.txt"
