#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${BOOKFORGE_GKE_NAMESPACE:-bookforge}"
SERVICE="${BOOKFORGE_GKE_SERVICE:-bookforge-anticipatory}"
LOCAL_PORT="${BOOKFORGE_GKE_LOCAL_PORT:-18082}"
JETSON_PORT="${BOOKFORGE_JETSON_ANTICIPATORY_PORT:-18082}"
JETSON_HOST="${BOOKFORGE_JETSON_HOST:-}"
JETSON_USER="${BOOKFORGE_JETSON_USER:-operator}"
JETSON_KEY="${BOOKFORGE_JETSON_KEY:-}"
HOST_KEY_ALIAS="${BOOKFORGE_JETSON_HOST_KEY_ALIAS:-}"

if [[ -z "${JETSON_HOST}" || -z "${JETSON_KEY}" ]]; then
  printf '%s\n' \
    "Set BOOKFORGE_JETSON_HOST and BOOKFORGE_JETSON_KEY." \
    "Example host: the Jetson's stable Tailscale IP or MagicDNS name." \
    "The key remains on this workstation and is never copied to GKE."
  exit 2
fi
if [[ ! "${LOCAL_PORT}" =~ ^[0-9]+$ || ! "${JETSON_PORT}" =~ ^[0-9]+$ ]] || \
   (( LOCAL_PORT < 1024 || LOCAL_PORT > 65535 || JETSON_PORT < 1024 || JETSON_PORT > 65535 )); then
  echo "Bridge ports must be integers between 1024 and 65535." >&2
  exit 2
fi
if [[ ! -f "${JETSON_KEY}" ]]; then
  echo "Jetson SSH key was not found: ${JETSON_KEY}" >&2
  exit 2
fi
for command in curl kubectl ssh; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command}" >&2
    exit 1
  fi
done

PORT_FORWARD_LOG="$(mktemp /tmp/bookforge-gke-port-forward.XXXXXX)"
PORT_FORWARD_PID=""
cleanup() {
  exit_code=$?
  if [[ -n "${PORT_FORWARD_PID}" ]]; then
    kill "${PORT_FORWARD_PID}" >/dev/null 2>&1 || true
    wait "${PORT_FORWARD_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${PORT_FORWARD_LOG}"
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

kubectl -n "${NAMESPACE}" port-forward \
  "service/${SERVICE}" "${LOCAL_PORT}:8080" >"${PORT_FORWARD_LOG}" 2>&1 &
PORT_FORWARD_PID=$!

ready=0
for _ in {1..40}; do
  if ! kill -0 "${PORT_FORWARD_PID}" >/dev/null 2>&1; then
    echo "GKE port-forward exited before becoming ready:" >&2
    sed -n '1,80p' "${PORT_FORWARD_LOG}" >&2
    exit 1
  fi
  if curl -fsS --max-time 1 "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if (( ready != 1 )); then
  echo "GKE anticipatory service did not become reachable within 20 seconds." >&2
  sed -n '1,80p' "${PORT_FORWARD_LOG}" >&2
  exit 1
fi

ssh_options=(
  -N
  -T
  -i "${JETSON_KEY}"
  -o BatchMode=yes
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
)
if [[ -n "${HOST_KEY_ALIAS}" ]]; then
  ssh_options+=(-o "HostKeyAlias=${HOST_KEY_ALIAS}")
fi

printf '%s\n' \
  "Private bridge ready." \
  "  workstation: http://127.0.0.1:${LOCAL_PORT}" \
  "  Jetson:      http://127.0.0.1:${JETSON_PORT}" \
  "Keep this terminal open during the bounded experiment. Ctrl-C removes both tunnels."

ssh "${ssh_options[@]}" \
  -R "127.0.0.1:${JETSON_PORT}:127.0.0.1:${LOCAL_PORT}" \
  "${JETSON_USER}@${JETSON_HOST}"
