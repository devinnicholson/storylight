#!/usr/bin/env bash

set -euo pipefail

KIOSK_URL="${BOOKFORGE_KIOSK_URL:-http://127.0.0.1:8080/projector}"
PROFILE_DIR="${BOOKFORGE_CHROMIUM_PROFILE:-${XDG_STATE_HOME:-${HOME}/.local/state}/bookforge/chromium}"

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

exec "$CHROMIUM_BIN" \
  --kiosk \
  --app="$KIOSK_URL" \
  --no-first-run \
  --disable-session-crashed-bubble \
  --overscroll-history-navigation=0 \
  --user-data-dir="$PROFILE_DIR"
