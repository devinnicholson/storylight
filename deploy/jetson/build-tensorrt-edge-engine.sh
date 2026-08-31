#!/usr/bin/env bash
# Build the pinned Bookforge TensorRT Edge-LLM control engine on Jetson.

set -euo pipefail

readonly EDGELLM_REVISION="71dd1bae032e70771265917ec74d3ff4cad07a10"
readonly INSTALL_ROOT="${BOOKFORGE_EDGELLM_ROOT:-$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0}"
readonly SOURCE_DIR="$INSTALL_ROOT/src"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly MODEL_ROOT="${BOOKFORGE_EDGELLM_MODEL_ROOT:-$INSTALL_ROOT/models/qwen2.5-0.5b-instruct-awq-v010}"
readonly CHECKPOINT_DIR="$MODEL_ROOT/onnx/llm"
readonly ENGINE_DIR="$MODEL_ROOT/engines/llm"
readonly EVIDENCE_DIR="$MODEL_ROOT/engine-build-evidence"
readonly GEMMA_MODEL="${BOOKFORGE_GEMMA_MODEL:-gemma3:1b-it-q4_K_M}"
readonly OLLAMA_BIN="${BOOKFORGE_OLLAMA_BIN:-$HOME/.local/opt/ollama-v0.32.15/bin/ollama}"
readonly LLM_BUILD="$BUILD_DIR/examples/llm/llm_build"
readonly EDGELLM_PLUGIN="$BUILD_DIR/libNvInfer_edgellm_plugin.so"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this engine builder as the Bookforge user, not root.\n' >&2
  exit 64
fi
if [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) != "$EDGELLM_REVISION" ]]; then
  printf 'Refusing unpinned TensorRT Edge-LLM source.\n' >&2
  exit 65
fi
for required in "$LLM_BUILD" "$EDGELLM_PLUGIN" "$CHECKPOINT_DIR/config.json" \
  "$CHECKPOINT_DIR/model.onnx" "$CHECKPOINT_DIR/model.onnx.data" \
  "$CHECKPOINT_DIR/embedding.safetensors" "$OLLAMA_BIN"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required TensorRT engine input is missing: %s\n' "$required" >&2
    exit 69
  fi
done
curl -fsS http://127.0.0.1:8080/readyz >/dev/null

mkdir -p "$ENGINE_DIR" "$EVIDENCE_DIR"
rm -f "$ENGINE_DIR/llm.engine"

tegrastats_pid=""
restore_runtime() {
  if [[ -n "$tegrastats_pid" ]]; then
    kill "$tegrastats_pid" 2>/dev/null || true
    wait "$tegrastats_pid" 2>/dev/null || true
  fi
  curl -fsS -X POST http://127.0.0.1:11434/api/chat \
    -H 'content-type: application/json' \
    --data "{\"model\":\"$GEMMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ready as JSON.\"}],\"stream\":false,\"format\":{\"type\":\"object\",\"properties\":{\"ready\":{\"type\":\"boolean\"}},\"required\":[\"ready\"]},\"keep_alive\":\"-1m\",\"options\":{\"temperature\":0,\"num_ctx\":4096,\"num_predict\":16}}" \
    >"$EVIDENCE_DIR/gemma-restore.json" || true
}
trap restore_runtime EXIT INT TERM

# The 8 GB Orin Nano cannot safely retain Gemma while TensorRT builds another
# resident engine. The service stays up; only its loaded model is released.
"$OLLAMA_BIN" stop "$GEMMA_MODEL" >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
    break
  fi
  sleep 1
done
if "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
  printf 'Gemma did not unload before the TensorRT engine build.\n' >&2
  exit 70
fi

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 --logfile "$EVIDENCE_DIR/tegrastats.log" &
  tegrastats_pid=$!
fi

build_started_ns=$(date +%s%N)
export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
"$LLM_BUILD" \
  --onnxDir "$CHECKPOINT_DIR" \
  --engineDir "$ENGINE_DIR" \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 1536
build_finished_ns=$(date +%s%N)
printf 'wall_nanoseconds=%s\n' "$((build_finished_ns - build_started_ns))" \
  >"$EVIDENCE_DIR/time.txt"

test -s "$ENGINE_DIR/llm.engine"
find "$ENGINE_DIR" -maxdepth 2 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$EVIDENCE_DIR/sha256sums.txt"
curl -fsS http://127.0.0.1:8080/readyz >"$EVIDENCE_DIR/api-ready.json"

printf 'TensorRT Edge-LLM engine is ready at %s.\n' "$ENGINE_DIR"
