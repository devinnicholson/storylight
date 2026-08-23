#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIOSK_URL="${BOOKFORGE_KIOSK_URL:-http://127.0.0.1:8080/projector?pack=latest&session=bookforge-live&live=1}"
PROFILE_DIR="${BOOKFORGE_CHROMIUM_PROFILE:-${XDG_STATE_HOME:-${HOME}/.local/state}/bookforge/chromium}"
STARTUP_TIMEOUT="${BOOKFORGE_KIOSK_STARTUP_TIMEOUT:-90}"

if [[ ! "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]]; then
  printf 'BOOKFORGE_KIOSK_STARTUP_TIMEOUT must be an integer number of seconds.\n' >&2
  exit 2
fi

"${SCRIPT_DIR}/check-kiosk-session.sh" || exit 78

if ! command -v systemd-inhibit >/dev/null 2>&1; then
  printf 'systemd-inhibit is required for a reversible projector sleep inhibitor.\n' >&2
  exit 1
fi

launch_browser() {
  exec systemd-inhibit \
    --what=sleep \
    --who="Bookforge projector kiosk" \
    --why="Keep the active projector session awake while Bookforge is presenting" \
    --mode=block \
    "$@"
}

if [[ -n "${BOOKFORGE_BROWSER_BIN:-}" ]]; then
  BROWSER_BIN="$BOOKFORGE_BROWSER_BIN"
elif [[ -n "${BOOKFORGE_CHROMIUM_BIN:-}" ]]; then
  BROWSER_BIN="$BOOKFORGE_CHROMIUM_BIN"
  BROWSER_KIND=chromium
elif command -v chromium >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium-browser)"
elif command -v firefox >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v firefox)"
elif command -v firefox-esr >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v firefox-esr)"
else
  printf 'A supported browser was not found. Set BOOKFORGE_BROWSER_BIN to an absolute Chromium or Firefox path.\n' >&2
  exit 1
fi

if [[ ! -x "$BROWSER_BIN" ]]; then
  printf 'Browser executable is not runnable: %s\n' "$BROWSER_BIN" >&2
  exit 1
fi

if [[ -z "${BROWSER_KIND:-}" ]]; then
  case "${BROWSER_BIN##*/}" in
    firefox|firefox-esr)
      BROWSER_KIND=firefox
      ;;
    chromium|chromium-browser|google-chrome|google-chrome-stable)
      BROWSER_KIND=chromium
      ;;
    *)
      printf 'Unsupported browser executable: %s. Use Chromium or Firefox.\n' "$BROWSER_BIN" >&2
      exit 1
      ;;
  esac
fi

READY_URL="${BOOKFORGE_READY_URL:-${KIOSK_URL%%/projector*}/readyz}"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --max-time 2 "$READY_URL" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    printf 'Bookforge did not become ready at %s within %s seconds.\n' "$READY_URL" "$STARTUP_TIMEOUT" >&2
    exit 1
  fi
  sleep 1
done

if [[ "$BROWSER_KIND" == firefox ]]; then
  launch_browser "$BROWSER_BIN" \
    --kiosk \
    --private-window "$KIOSK_URL"
fi

mkdir -p "$PROFILE_DIR"

launch_browser "$BROWSER_BIN" \
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
