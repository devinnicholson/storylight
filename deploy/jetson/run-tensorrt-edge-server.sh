#!/usr/bin/env bash
# Start NVIDIA's experimental TensorRT Edge-LLM OpenAI-compatible server locally.

set -euo pipefail

readonly INSTALL_ROOT="${BOOKFORGE_EDGELLM_ROOT:-$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0}"
readonly SOURCE_DIR="$INSTALL_ROOT/src"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly VENV_DIR="$INSTALL_ROOT/venv"
readonly ENGINE_DIR="${1:-${BOOKFORGE_EDGELLM_ENGINE_DIR:-}}"
readonly PORT="${BOOKFORGE_EDGELLM_SERVER_PORT:-11435}"
readonly PYBIND_DIR="$BUILD_DIR/pybind"
readonly PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run the TensorRT Edge-LLM server as the Bookforge user, not root.\n' >&2
  exit 64
fi

if [[ -z "$ENGINE_DIR" ]]; then
  printf 'Usage: %s /absolute/path/to/engine-directory\n' "$0" >&2
  exit 64
fi

if [[ "$ENGINE_DIR" != /* ]] || [[ ! -s "$ENGINE_DIR/llm.engine" ]]; then
  printf 'A valid absolute TensorRT engine directory is required: %s\n' "$ENGINE_DIR" >&2
  exit 66
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf 'BOOKFORGE_EDGELLM_SERVER_PORT must be between 1024 and 65535.\n' >&2
  exit 64
fi

if [[ ! -x "$VENV_DIR/bin/python" ]] || [[ ! -s "$PLUGIN_PATH" ]]; then
  printf 'Install TensorRT Edge-LLM before starting its server.\n' >&2
  exit 69
fi

if ! find "$PYBIND_DIR" -maxdepth 1 -type f -name '*_edgellm_runtime*.so' -print -quit \
  | grep -q .; then
  printf 'The TensorRT Edge-LLM Python binding is missing. Re-run the installer.\n' >&2
  exit 69
fi

# NVIDIA's server is an evaluation component. Binding only to loopback prevents
# accidental LAN exposure while the candidate remains behind Bookforge's gate.
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export EDGELLM_PYBIND_DIR="$PYBIND_DIR"
export EDGELLM_PLUGIN_PATH="$PLUGIN_PATH"
export LD_LIBRARY_PATH="$BUILD_DIR:$BUILD_DIR/cpp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$INSTALL_ROOT"
exec "$VENV_DIR/bin/python" -m experimental.server \
  --model "$ENGINE_DIR" \
  --host 127.0.0.1 \
  --port "$PORT"
