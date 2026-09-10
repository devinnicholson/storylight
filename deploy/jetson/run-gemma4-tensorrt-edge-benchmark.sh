#!/usr/bin/env bash
# Shadow benchmark Gemma 4 TensorRT and always restore production Gemma 3.

set -euo pipefail

readonly INSTALL_ROOT="${STORYLIGHT_EDGELLM_ROOT:-$HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0}"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly MODEL_ROOT="$INSTALL_ROOT/models/gemma4-e2b-it-int4-awq-v010"
readonly ENGINE_DIR="$MODEL_ROOT/engines/llm"
readonly CHECKPOINT_DIR="$MODEL_ROOT/onnx/llm"
readonly EVIDENCE_DIR="$MODEL_ROOT/inference-evidence"
readonly BENCHMARK_SCRIPT="${STORYLIGHT_EDGELLM_BENCHMARK_SCRIPT:-/opt/storylight/deploy/jetson/benchmark-tensorrt-edge-llm.py}"
readonly STORYLIGHT_PYTHON="${STORYLIGHT_PYTHON:-/opt/storylight/.venv/bin/python}"
readonly GEMMA_MODEL="${STORYLIGHT_GEMMA_MODEL:-gemma3:1b-it-q4_K_M}"
readonly OLLAMA_BIN="${STORYLIGHT_OLLAMA_BIN:-$HOME/.local/opt/ollama-v0.32.15/bin/ollama}"
readonly LLM_INFERENCE="$BUILD_DIR/examples/llm/llm_inference"
readonly EDGELLM_PLUGIN="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
readonly PROMPT_PROFILE="${STORYLIGHT_EDGELLM_PROMPT_PROFILE:-repair}"
readonly SUITE="${STORYLIGHT_EDGELLM_SUITE:-contest}"
readonly CASE_ID="${STORYLIGHT_EDGELLM_CASE_ID:-}"
readonly REPORT_PATH="$EVIDENCE_DIR/benchmark-$PROMPT_PROFILE-$SUITE${CASE_ID:+-$CASE_ID}.json"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this shadow benchmark as the Storylight user, not root.\n' >&2
  exit 64
fi
for required in "$LLM_INFERENCE" "$EDGELLM_PLUGIN" "$ENGINE_DIR/llm.engine" \
  "$CHECKPOINT_DIR/config.json" "$BENCHMARK_SCRIPT" "$STORYLIGHT_PYTHON" "$OLLAMA_BIN"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required Gemma 4 shadow input is missing: %s\n' "$required" >&2
    exit 69
  fi
done
curl -fsS http://127.0.0.1:8080/readyz >/dev/null

mkdir -p "$EVIDENCE_DIR/work"
restore_runtime() {
  curl -fsS -X POST http://127.0.0.1:11434/api/chat \
    -H 'content-type: application/json' \
    --data "{\"model\":\"$GEMMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Return ready as JSON.\"}],\"stream\":false,\"format\":{\"type\":\"object\",\"properties\":{\"ready\":{\"type\":\"boolean\"}},\"required\":[\"ready\"]},\"keep_alive\":\"-1m\",\"options\":{\"temperature\":0,\"num_ctx\":4096,\"num_predict\":16}}" \
    >"$EVIDENCE_DIR/gemma3-restore.json" || true
}
trap restore_runtime EXIT INT TERM

"$OLLAMA_BIN" stop "$GEMMA_MODEL" >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then break; fi
  sleep 1
done
if "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
  printf 'Gemma 3 did not unload before the Gemma 4 shadow benchmark.\n' >&2
  exit 70
fi

export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
export EDGELLM_GEMMA4_PLE_STORAGE_BACKED="${EDGELLM_GEMMA4_PLE_STORAGE_BACKED:-1}"
benchmark_args=(
  --binary "$LLM_INFERENCE"
  --engine-dir "$ENGINE_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --work-dir "$EVIDENCE_DIR/work"
  --output "$REPORT_PATH"
  --warmup 1
  --prompt-profile "$PROMPT_PROFILE"
  --suite "$SUITE"
  --candidate-model google/gemma-4-E2B-it
  --candidate-revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
)
if [[ -n "$CASE_ID" ]]; then
  benchmark_args+=(--case-id "$CASE_ID")
fi
"$STORYLIGHT_PYTHON" "$BENCHMARK_SCRIPT" "${benchmark_args[@]}"

test -s "$REPORT_PATH"
sha256sum "$REPORT_PATH" >"$EVIDENCE_DIR/sha256sums-production.txt"
printf 'Gemma 4 shadow evidence is ready at %s; Gemma 3 remains production.\n' "$REPORT_PATH"
