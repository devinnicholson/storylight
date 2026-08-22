#!/usr/bin/env bash

set -euo pipefail

KIOSK_URL="${BOOKFORGE_KIOSK_URL:-http://127.0.0.1:8080/projector}"
PROFILE_DIR="${BOOKFORGE_CHROMIUM_PROFILE:-${XDG_STATE_HOME:-${HOME}/.local/state}/bookforge/chromium}"
STARTUP_TIMEOUT="${BOOKFORGE_KIOSK_STARTUP_TIMEOUT:-90}"

if [[ ! "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]]; then
  printf 'BOOKFORGE_KIOSK_STARTUP_TIMEOUT must be an integer number of seconds.\n' >&2
  exit 2
fi

if [[ -n "${BOOKFORGE_CHROMIUM_BIN:-}" ]]; then
  CHROMIUM_BIN="$BOOKFORGE_CHROMIUM_BIN"
elif command -v chromium >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium-browser)"
else
  printf 'Chromium was not found. Set BOOKFORGE_CHROMIUM_BIN to its absolute path.\n' >&2
  exit 1
fi

if [[ ! -x "$CHROMIUM_BIN" ]]; then
  printf 'Chromium executable is not runnable: %s\n' "$CHROMIUM_BIN" >&2
  exit 1
fi

mkdir -p "$PROFILE_DIR"

READY_URL="${BOOKFORGE_READY_URL:-${KIOSK_URL%%/projector*}/readyz}"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --max-time 2 "$READY_URL" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    printf 'Bookforge did not become ready at %s within %s seconds.\n' "$READY_URL" "$STARTUP_TIMEOUT" >&2
    exit 1
  fi
  sleep 1
done

exec "$CHROMIUM_BIN" \
  --kiosk \
  --app="$KIOSK_URL" \
  --no-first-run \
  --no-default-browser-check \
  --disable-background-networking \
  --disable-component-update \
  --disable-domain-reliability \
  --disable-sync \
  --metrics-recording-only \
  --disable-session-crashed-bubble \
  --overscroll-history-navigation=0 \
  --user-data-dir="$PROFILE_DIR"
