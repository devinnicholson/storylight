#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIOSK_URL="${STORYLIGHT_KIOSK_URL:-http://127.0.0.1:8080/projector?pack=latest&session=storylight-live&live=1}"
PROFILE_DIR="${STORYLIGHT_CHROMIUM_PROFILE:-${XDG_STATE_HOME:-${HOME}/.local/state}/storylight/chromium}"
FIREFOX_PROFILE_DIR="${STORYLIGHT_FIREFOX_PROFILE:-${XDG_STATE_HOME:-${HOME}/.local/state}/storylight/firefox}"
STORYLIGHT_FIREFOX_BIN="${HOME}/.local/opt/firefox-storylight/firefox"
STARTUP_TIMEOUT="${STORYLIGHT_KIOSK_STARTUP_TIMEOUT:-90}"

if [[ ! "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]]; then
  printf 'STORYLIGHT_KIOSK_STARTUP_TIMEOUT must be an integer number of seconds.\n' >&2
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
    --who="Storylight projector kiosk" \
    --why="Keep the active projector session awake while Storylight is presenting" \
    --mode=block \
    "$@"
}

if [[ -n "${STORYLIGHT_BROWSER_BIN:-}" ]]; then
  BROWSER_BIN="$STORYLIGHT_BROWSER_BIN"
elif [[ -n "${STORYLIGHT_CHROMIUM_BIN:-}" ]]; then
  BROWSER_BIN="$STORYLIGHT_CHROMIUM_BIN"
  BROWSER_KIND=chromium
elif command -v chromium >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium-browser)"
elif [[ -x "$STORYLIGHT_FIREFOX_BIN" ]]; then
  # Prefer the verified native archive over Ubuntu's Snap wrapper. The Snap
  # launcher moves Firefox into a separate scope and exits, which makes a
  # supervising systemd service restart forever while the browser is alive.
  BROWSER_BIN="$STORYLIGHT_FIREFOX_BIN"
elif command -v firefox >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v firefox)"
elif command -v firefox-esr >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v firefox-esr)"
else
  printf 'A supported browser was not found. Set STORYLIGHT_BROWSER_BIN to an absolute Chromium or Firefox path.\n' >&2
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

READY_URL="${STORYLIGHT_READY_URL:-${KIOSK_URL%%/projector*}/readyz}"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --max-time 2 "$READY_URL" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    printf 'Storylight did not become ready at %s within %s seconds.\n' "$READY_URL" "$STARTUP_TIMEOUT" >&2
    exit 1
  fi
  sleep 1
done

if [[ "$BROWSER_KIND" == firefox ]]; then
  FIREFOX_INSTALL_DIR="$(dirname -- "$(readlink -f -- "$BROWSER_BIN")")"
  FIREFOX_POLICY_SOURCE="${SCRIPT_DIR}/firefox-policies.json"
  FIREFOX_POLICY_TARGET="${FIREFOX_INSTALL_DIR}/distribution/policies.json"
  if [[ -f "$FIREFOX_POLICY_SOURCE" && -w "$FIREFOX_INSTALL_DIR" ]]; then
    mkdir -p "${FIREFOX_INSTALL_DIR}/distribution"
    install -m 0644 "$FIREFOX_POLICY_SOURCE" "$FIREFOX_POLICY_TARGET"
  fi
  if [[ ! -r "$FIREFOX_POLICY_TARGET" ]] \
    || ! grep -Eq '"SkipTermsOfUse"[[:space:]]*:[[:space:]]*true' "$FIREFOX_POLICY_TARGET"; then
    printf 'Firefox kiosk policy is missing: %s. Re-run the pinned Firefox installer.\n' "$FIREFOX_POLICY_TARGET" >&2
    exit 1
  fi
  mkdir -p "$FIREFOX_PROFILE_DIR"
  chmod 700 "$FIREFOX_PROFILE_DIR"
  # A dedicated deterministic kiosk profile prevents Firefox's first-run,
  # privacy-policy, default-browser, and telemetry prompts from ever covering
  # the projection. Do not reuse or modify the operator's normal profile.
  printf '%s\n' \
    'user_pref("browser.aboutwelcome.enabled", false);' \
    'user_pref("browser.aboutwelcome.didSeeFinalScreen", true);' \
    'user_pref("browser.startup.firstrunSkipsHomepage", true);' \
    'user_pref("browser.startup.homepage_override.mstone", "ignore");' \
    'user_pref("browser.shell.checkDefaultBrowser", false);' \
    'user_pref("datareporting.policy.dataSubmissionEnabled", false);' \
    'user_pref("datareporting.healthreport.uploadEnabled", false);' \
    'user_pref("toolkit.telemetry.enabled", false);' \
    'user_pref("toolkit.telemetry.unified", false);' \
    'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);' \
    'user_pref("app.shield.optoutstudies.enabled", false);' \
    'user_pref("browser.newtabpage.activity-stream.telemetry", false);' \
    >"${FIREFOX_PROFILE_DIR}/user.js"
  chmod 600 "${FIREFOX_PROFILE_DIR}/user.js"
  export MOZ_WEBRENDER="${MOZ_WEBRENDER:-1}"
  export MOZ_X11_EGL="${MOZ_X11_EGL:-1}"
  export MOZ_DISABLE_DEFAULT_BROWSER_AGENT=1
  launch_browser "$BROWSER_BIN" \
    --profile "$FIREFOX_PROFILE_DIR" \
    --new-instance \
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
