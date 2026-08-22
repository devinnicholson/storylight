#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${BOOKFORGE_VENV_PATH:-${REPO_ROOT}/.venv}"
SERVICE_USER="${BOOKFORGE_SERVICE_USER:-${USER:-$(id -un)}}"
SERVICE_NAME="bookforge@${SERVICE_USER}.service"
OUTPUT="${BOOKFORGE_PRIVACY_OUTPUT:-/var/lib/bookforge/evidence/privacy-latest.json}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  printf 'Bookforge virtual environment is missing: %s\n' "$VENV_PATH" >&2
  exit 1
fi

SERVICE_PID="$(systemctl show --property MainPID --value "$SERVICE_NAME" 2>/dev/null || true)"
if [[ ! "$SERVICE_PID" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Bookforge service is not running: %s\n' "$SERVICE_NAME" >&2
  exit 1
fi

exec "${VENV_PATH}/bin/python" -m bookforge.privacy_audit \
  --pid "$SERVICE_PID" \
  --output "$OUTPUT"
