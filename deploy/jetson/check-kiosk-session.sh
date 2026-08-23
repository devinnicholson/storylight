#!/usr/bin/env bash
# Read-only, fail-closed readiness check for the physical projector session.

set -euo pipefail

fail() {
  printf 'Kiosk session is not presentation-ready: %s\n' "$1" >&2
  exit 1
}

for tool in loginctl xset; do
  command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done

CURRENT_UID="$(id -u)"

session_property() {
  loginctl show-session "$SESSION_ID" --property="$1" --value 2>/dev/null
}

find_graphical_session() {
  local candidate candidate_uid candidate_type candidate_active candidate_remote

  if [[ -n "${BOOKFORGE_KIOSK_SESSION_ID:-}" ]]; then
    printf '%s\n' "$BOOKFORGE_KIOSK_SESSION_ID"
    return
  fi
  if [[ -n "${XDG_SESSION_ID:-}" ]]; then
    printf '%s\n' "$XDG_SESSION_ID"
    return
  fi

  while read -r candidate candidate_uid _; do
    [[ -n "$candidate" && "$candidate_uid" == "$CURRENT_UID" ]] || continue
    candidate_type="$(loginctl show-session "$candidate" --property=Type --value 2>/dev/null || true)"
    candidate_active="$(loginctl show-session "$candidate" --property=Active --value 2>/dev/null || true)"
    candidate_remote="$(loginctl show-session "$candidate" --property=Remote --value 2>/dev/null || true)"
    if [[ "$candidate_type" == x11 && "$candidate_active" == yes && "$candidate_remote" == no ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(loginctl list-sessions --no-legend 2>/dev/null)
}

SESSION_ID="$(find_graphical_session)"
[[ -n "$SESSION_ID" ]] || fail "no active local X11 session belongs to uid $CURRENT_UID"

[[ "$(session_property User)" == "$CURRENT_UID" ]] || fail "session $SESSION_ID belongs to another user"
[[ "$(session_property Type)" == x11 ]] || fail "session $SESSION_ID is not X11"
[[ "$(session_property Remote)" == no ]] || fail "session $SESSION_ID is remote"
[[ "$(session_property Active)" == yes ]] || fail "session $SESSION_ID is not active"
[[ "$(session_property State)" == active ]] || fail "session $SESSION_ID state is not active"
[[ "$(session_property LockedHint)" == no ]] || fail "session $SESSION_ID is locked; unlock it physically and retry"
[[ "$(session_property IdleHint)" == no ]] || fail "session $SESSION_ID is idle; interact with the physical desktop and retry"

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is unset"
[[ -n "${XAUTHORITY:-}" && -r "$XAUTHORITY" ]] || fail "XAUTHORITY is unset or unreadable"

XSET_STATUS="$(xset q 2>/dev/null)" || fail "the X11 display cannot be queried"
grep -q 'Monitor is On' <<<"$XSET_STATUS" || fail "the projector display is off; wake it without bypassing the lock, then retry"

printf 'Kiosk session %s is active, unlocked, non-idle, and visible on %s.\n' "$SESSION_ID" "$DISPLAY"
