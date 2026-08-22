#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${BOOKFORGE_VENV_PATH:-${REPO_ROOT}/.venv}"
ENV_FILE="${BOOKFORGE_ENV_FILE:-/etc/bookforge/bookforge.env}"
SERVICE_USER="${BOOKFORGE_SERVICE_USER:-${USER:-$(id -un)}}"
SERVICE_NAME="bookforge@${SERVICE_USER}.service"

if systemctl is-active --quiet "$SERVICE_NAME"; then
  printf 'Stop %s before building the shared TensorRT engine.\n' "$SERVICE_NAME" >&2
  exit 1
fi
if [[ ! -r "$ENV_FILE" ]]; then
  printf 'Bookforge environment is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ "${BOOKFORGE_ASR_BACKEND:-}" != "whisper_trt" ]]; then
  printf 'Set BOOKFORGE_ASR_BACKEND=whisper_trt in %s first.\n' "$ENV_FILE" >&2
  exit 1
fi

export HOME="${BOOKFORGE_ASR_HOME:-${BOOKFORGE_CACHE_DIR}/home}"
mkdir -p "$HOME" "${BOOKFORGE_CACHE_DIR}/whisper_trt"
chmod 700 "$HOME" "${BOOKFORGE_CACHE_DIR}/whisper_trt"

exec "${VENV_PATH}/bin/python" -m bookforge.asr_warmup
