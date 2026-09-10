#!/usr/bin/env bash
# Build the pinned, external-weight Gemma 4 E2B TensorRT engine on Jetson.

set -euo pipefail

readonly EDGELLM_REVISION="71dd1bae032e70771265917ec74d3ff4cad07a10"
readonly INSTALL_ROOT="${STORYLIGHT_EDGELLM_ROOT:-$HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0}"
readonly SOURCE_DIR="$INSTALL_ROOT/src"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly MODEL_ROOT="${STORYLIGHT_EDGELLM_MODEL_ROOT:-$INSTALL_ROOT/models/gemma4-e2b-it-int4-awq-v010}"
readonly CHECKPOINT_DIR="$MODEL_ROOT/onnx/llm"
readonly ENGINE_DIR="$MODEL_ROOT/engines/llm"
readonly EVIDENCE_DIR="$MODEL_ROOT/engine-build-evidence"
readonly GEMMA_MODEL="${STORYLIGHT_GEMMA_MODEL:-gemma3:1b-it-q4_K_M}"
readonly OLLAMA_BIN="${STORYLIGHT_OLLAMA_BIN:-$HOME/.local/opt/ollama-v0.32.15/bin/ollama}"
readonly LLM_BUILD="$BUILD_DIR/examples/llm/llm_build"
readonly EDGELLM_PLUGIN="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
monitor_pid=""

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this engine builder as the Storylight user, not root.\n' >&2
  exit 64
fi
if [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) != "$EDGELLM_REVISION" ]]; then
  printf 'Refusing unpinned TensorRT Edge-LLM source.\n' >&2
  exit 65
fi
for required in "$LLM_BUILD" "$EDGELLM_PLUGIN" "$CHECKPOINT_DIR/config.json" \
  "$CHECKPOINT_DIR/model.onnx" "$OLLAMA_BIN"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required Gemma 4 TensorRT input is missing: %s\n' "$required" >&2
    exit 69
  fi
done
if ! find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
  printf 'Externalized Gemma 4 INT4 weights are missing.\n' >&2
  exit 69
fi
curl -fsS http://127.0.0.1:8080/readyz >/dev/null

mkdir -p "$ENGINE_DIR" "$EVIDENCE_DIR"
rm -f "$ENGINE_DIR/llm.engine"

restore_runtime() {
  curl -fsS -X POST http://127.0.0.1:11434/api/chat \
    -H 'content-type: application/json' \
    --data "{\"model\":\"$GEMMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ready as JSON.\"}],\"stream\":false,\"format\":{\"type\":\"object\",\"properties\":{\"ready\":{\"type\":\"boolean\"}},\"required\":[\"ready\"]},\"keep_alive\":\"-1m\",\"options\":{\"temperature\":0,\"num_ctx\":4096,\"num_predict\":16}}" \
    >"$EVIDENCE_DIR/gemma3-restore.json" || true
}
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" >/dev/null 2>&1; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
  restore_runtime
}
trap cleanup EXIT INT TERM

"$OLLAMA_BIN" stop "$GEMMA_MODEL" >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then break; fi
  sleep 1
done
if "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
  printf 'Gemma 3 did not unload before the Gemma 4 engine build.\n' >&2
  exit 70
fi

free -b >"$EVIDENCE_DIR/free-before-build.txt"
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 >"$EVIDENCE_DIR/tegrastats.log" 2>&1 &
  monitor_pid=$!
fi

build_started_ns=$(date +%s%N)
export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
set +e
{
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$LLM_BUILD" \
      --onnxDir "$CHECKPOINT_DIR" \
      --engineDir "$ENGINE_DIR" \
      --maxBatchSize 1 \
      --maxInputLen 1280 \
      --maxKVCacheCapacity 1536
  else
    "$LLM_BUILD" \
      --onnxDir "$CHECKPOINT_DIR" \
      --engineDir "$ENGINE_DIR" \
      --maxBatchSize 1 \
      --maxInputLen 1280 \
      --maxKVCacheCapacity 1536
  fi
} 2>&1 | tee "$EVIDENCE_DIR/build.log"
build_status=${PIPESTATUS[0]}
set -e
build_finished_ns=$(date +%s%N)
printf 'wall_nanoseconds=%s\nexit_status=%s\n' \
  "$((build_finished_ns - build_started_ns))" "$build_status" \
  >"$EVIDENCE_DIR/time.txt"
free -b >"$EVIDENCE_DIR/free-after-build.txt"
if [[ "$build_status" -ne 0 ]]; then
  printf 'TensorRT engine build failed with status %s; see %s.\n' \
    "$build_status" "$EVIDENCE_DIR/build.log" >&2
  exit "$build_status"
fi

test -s "$ENGINE_DIR/llm.engine"
find "$ENGINE_DIR" -maxdepth 2 -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$EVIDENCE_DIR/sha256sums.txt"
printf 'Gemma 4 TensorRT engine is ready for shadow testing at %s.\n' "$ENGINE_DIR"
