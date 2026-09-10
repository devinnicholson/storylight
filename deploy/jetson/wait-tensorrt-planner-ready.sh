#!/usr/bin/env bash
# Bound systemd startup on the loopback TensorRT readiness endpoint.

set -euo pipefail

readonly PORT="${STORYLIGHT_EDGELLM_SERVER_PORT:-11435}"
readonly ATTEMPTS="${STORYLIGHT_TENSORRT_READY_ATTEMPTS:-90}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
  printf 'STORYLIGHT_EDGELLM_SERVER_PORT must be between 1024 and 65535.\n' >&2
  exit 64
fi
if [[ ! "$ATTEMPTS" =~ ^[0-9]+$ ]] || ((ATTEMPTS < 1 || ATTEMPTS > 180)); then
  printf 'STORYLIGHT_TENSORRT_READY_ATTEMPTS must be between 1 and 180.\n' >&2
  exit 64
fi

for _ in $(seq 1 "$ATTEMPTS"); do
  if curl --fail --silent --max-time 1 \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    printf 'TensorRT planner is ready on loopback.\n'
    exit 0
  fi
  sleep 1
done

printf 'TensorRT planner did not become ready within %s seconds.\n' "$ATTEMPTS" >&2
exit 1
