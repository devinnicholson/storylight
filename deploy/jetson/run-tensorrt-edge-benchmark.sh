#!/usr/bin/env bash
# Benchmark the pinned TensorRT Edge-LLM control without competing with Gemma.

set -euo pipefail

readonly INSTALL_ROOT="${BOOKFORGE_EDGELLM_ROOT:-$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0}"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly MODEL_ROOT="$INSTALL_ROOT/models/qwen2.5-0.5b-instruct-awq-v010"
readonly ENGINE_DIR="$MODEL_ROOT/engines/llm"
readonly CHECKPOINT_DIR="$MODEL_ROOT/onnx/llm"
readonly EVIDENCE_DIR="$MODEL_ROOT/inference-evidence"
readonly BENCHMARK_SCRIPT="${BOOKFORGE_EDGELLM_BENCHMARK_SCRIPT:-/opt/bookforge/deploy/jetson/benchmark-tensorrt-edge-llm.py}"
readonly BOOKFORGE_PYTHON="${BOOKFORGE_PYTHON:-/opt/bookforge/.venv/bin/python}"
readonly GEMMA_MODEL="${BOOKFORGE_GEMMA_MODEL:-gemma3:1b-it-q4_K_M}"
readonly OLLAMA_BIN="${BOOKFORGE_OLLAMA_BIN:-$HOME/.local/opt/ollama-v0.32.15/bin/ollama}"
readonly LLM_INFERENCE="$BUILD_DIR/examples/llm/llm_inference"
readonly EDGELLM_PLUGIN="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
readonly PROMPT_PROFILE="${BOOKFORGE_EDGELLM_PROMPT_PROFILE:-production}"
readonly REPORT_PATH="$EVIDENCE_DIR/benchmark-$PROMPT_PROFILE.json"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this benchmark as the Bookforge user, not root.\n' >&2
  exit 64
fi
for required in "$LLM_INFERENCE" "$EDGELLM_PLUGIN" "$ENGINE_DIR/llm.engine" \
  "$CHECKPOINT_DIR/config.json" "$BENCHMARK_SCRIPT" "$BOOKFORGE_PYTHON" "$OLLAMA_BIN"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required TensorRT benchmark input is missing: %s\n' "$required" >&2
    exit 69
  fi
done
curl -fsS http://127.0.0.1:8080/readyz >/dev/null

mkdir -p "$EVIDENCE_DIR/work"
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

"$OLLAMA_BIN" stop "$GEMMA_MODEL" >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
    break
  fi
  sleep 1
done
if "$OLLAMA_BIN" ps | grep -Fq "$GEMMA_MODEL"; then
  printf 'Gemma did not unload before the TensorRT benchmark.\n' >&2
  exit 70
fi

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 250 --logfile "$EVIDENCE_DIR/tegrastats.log" &
  tegrastats_pid=$!
fi

export EDGELLM_PLUGIN_PATH="$EDGELLM_PLUGIN"
"$BOOKFORGE_PYTHON" "$BENCHMARK_SCRIPT" \
  --binary "$LLM_INFERENCE" \
  --engine-dir "$ENGINE_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --work-dir "$EVIDENCE_DIR/work" \
  --output "$REPORT_PATH" \
  --warmup 1 \
  --prompt-profile "$PROMPT_PROFILE"

test -s "$REPORT_PATH"
sha256sum "$REPORT_PATH" >"$EVIDENCE_DIR/sha256sums-$PROMPT_PROFILE.txt"
printf 'TensorRT Edge-LLM benchmark evidence is ready at %s.\n' "$REPORT_PATH"
