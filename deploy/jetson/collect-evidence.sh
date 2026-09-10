#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${STORYLIGHT_VENV_PATH:-${REPO_ROOT}/.venv}"
SERVICE_USER="${STORYLIGHT_SERVICE_USER:-${USER:-$(id -un)}}"
SERVICE_NAME="storylight@${SERVICE_USER}.service"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${STORYLIGHT_EVIDENCE_OUTPUT:-/var/lib/storylight/evidence/hardware-${TIMESTAMP}.json}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  printf 'Storylight virtual environment is missing: %s\n' "$VENV_PATH" >&2
  exit 1
fi

SERVICE_PID="$(systemctl show --property MainPID --value "$SERVICE_NAME" 2>/dev/null || true)"
if [[ ! "$SERVICE_PID" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Storylight service is not running: %s\n' "$SERVICE_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT")"
exec "${VENV_PATH}/bin/python" -m storylight.hardware_acceptance \
  --output "$OUTPUT" \
  --service-pid "$SERVICE_PID" \
  "$@"
