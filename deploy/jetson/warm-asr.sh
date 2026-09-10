#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${STORYLIGHT_VENV_PATH:-${REPO_ROOT}/.venv}"
ENV_FILE="${STORYLIGHT_ENV_FILE:-/etc/storylight/storylight.env}"
SERVICE_USER="${STORYLIGHT_SERVICE_USER:-${USER:-$(id -un)}}"
SERVICE_NAME="storylight@${SERVICE_USER}.service"

if systemctl is-active --quiet "$SERVICE_NAME"; then
  printf 'Stop %s before building the shared TensorRT engine.\n' "$SERVICE_NAME" >&2
  exit 1
fi
if [[ ! -r "$ENV_FILE" ]]; then
  printf 'Storylight environment is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ "${STORYLIGHT_ASR_BACKEND:-}" != "whisper_trt" ]]; then
  printf 'Set STORYLIGHT_ASR_BACKEND=whisper_trt in %s first.\n' "$ENV_FILE" >&2
  exit 1
fi

export HOME="${STORYLIGHT_ASR_HOME:-${STORYLIGHT_CACHE_DIR}/home}"
mkdir -p "$HOME" "${STORYLIGHT_CACHE_DIR}/whisper_trt"
chmod 700 "$HOME" "${STORYLIGHT_CACHE_DIR}/whisper_trt"

exec "${VENV_PATH}/bin/python" -m storylight.asr_warmup
