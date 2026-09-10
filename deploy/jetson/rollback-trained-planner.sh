#!/usr/bin/env bash
# Restore the exact pre-promotion TensorRT engine and root environment.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE

readonly CONFIG_FILE="/etc/storylight/storylight.env"
readonly STATE_DIR="/var/lib/storylight-trusted/trained-planner"
readonly ACTIVE_STATE="$STATE_DIR/active.env"
readonly CANDIDATE_ROOT="/var/lib/storylight-trusted/trained-planner-candidates"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly INSTALLER="${STORYLIGHT_CANDIDATE_INSTALLER:-$SCRIPT_DIR/install-trained-planner-candidate.sh}"
readonly ACCEPTED_ENGINE_SHA256="95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
target_user="${STORYLIGHT_SERVICE_USER:-${SUDO_USER:-}}"
candidate_id=""
manifest_sha256=""
approval_token=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo rollback-trained-planner.sh OPTIONS

Required:
  --user USER
  --candidate-id ID
  --manifest-sha256 SHA256
  --approval-token TOKEN       Exact token printed by --dry-run.

Options:
  --dry-run                    Verify and print the one-purpose token only.
  -h, --help                   Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || { printf '%s\n' '--user requires a value' >&2; exit 64; }
      target_user="$2"
      shift 2
      ;;
    --candidate-id)
      [[ $# -ge 2 ]] || { printf '%s\n' '--candidate-id requires a value' >&2; exit 64; }
      candidate_id="$2"
      shift 2
      ;;
    --manifest-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--manifest-sha256 requires a value' >&2; exit 64; }
      manifest_sha256="$2"
      shift 2
      ;;
    --approval-token)
      [[ $# -ge 2 ]] || { printf '%s\n' '--approval-token requires a value' >&2; exit 64; }
      approval_token="$2"
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run rollback with sudo, including --dry-run.\n' >&2
  exit 64
fi
if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
  || ! id "$target_user" >/dev/null 2>&1 \
  || [[ ! "$candidate_id" =~ ^[a-z0-9][a-z0-9-]{2,95}$ ]] \
  || [[ ! "$manifest_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  usage >&2
  exit 64
fi
if [[ ! -f "$ACTIVE_STATE" || -L "$ACTIVE_STATE" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$ACTIVE_STATE")" != "root:root:600" ]]; then
  printf 'The root-owned trained planner state is missing or unsafe.\n' >&2
  exit 78
fi

state_field() {
  name="$1"
  sed -n "s/^${name}=//p" "$ACTIVE_STATE"
}

state_candidate="$(state_field CANDIDATE_ID)"
state_manifest="$(state_field MANIFEST_SHA256)"
state_gate_evidence="$(state_field GATE_EVIDENCE_SHA256)"
state_gate_producer="$(state_field GATE_PRODUCER)"
state_model_revision="$(state_field MODEL_REVISION)"
state_engine_sha256="$(state_field ENGINE_SHA256)"
state_accepted_engine_sha256="$(state_field ACCEPTED_ENGINE_SHA256)"
backup_file="$(state_field BACKUP_FILE)"
backup_sha256="$(state_field BACKUP_SHA256)"
kiosk_was_active="$(state_field KIOSK_WAS_ACTIVE)"
if [[ "$state_candidate" != "$candidate_id" ]] \
  || [[ "$state_manifest" != "$manifest_sha256" ]] \
  || [[ ! "$state_gate_evidence" =~ ^[a-f0-9]{64}$ ]] \
  || [[ "$state_gate_producer" != "storylight-fidelity-gate-builder" ]] \
  || [[ ! "$state_model_revision" =~ ^sha256:[a-f0-9]{64}$ ]] \
  || [[ "$state_engine_sha256" != "${state_model_revision#sha256:}" ]] \
  || [[ "$state_accepted_engine_sha256" != "$ACCEPTED_ENGINE_SHA256" ]] \
  || [[ ! "$backup_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ "$kiosk_was_active" != "0" && "$kiosk_was_active" != "1" ]]; then
  printf 'Rollback arguments do not match the durable promotion state.\n' >&2
  exit 78
fi
case "$backup_file" in
  "$STATE_DIR"/rollback/*-"$candidate_id"/storylight.env) ;;
  *) printf 'Durable rollback state contains an unsafe backup path.\n' >&2; exit 78 ;;
esac
if [[ ! -f "$backup_file" || -L "$backup_file" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$backup_file")" != "root:root:600" ]] \
  || [[ "$(sha256sum "$backup_file" | cut -d' ' -f1)" != "$backup_sha256" ]]; then
  printf 'The exact pre-promotion environment backup is missing or changed.\n' >&2
  exit 78
fi
verification="$(
  "$INSTALLER" \
    --bundle "$CANDIDATE_ROOT/$candidate_id" \
    --expected-manifest-sha256 "$manifest_sha256" \
    --verify-only \
    --installed-layout
)"
verified_model_revision="$(printf '%s\n' "$verification" | sed -n 's/^model_revision=//p')"
verified_engine_sha256="$(printf '%s\n' "$verification" | sed -n 's/^engine_sha256=//p')"
if [[ "$verified_model_revision" != "$state_model_revision" ]] \
  || [[ "$verified_engine_sha256" != "$state_engine_sha256" ]]; then
  printf 'Installed candidate identity differs from the durable promotion state.\n' >&2
  exit 78
fi
readonly EXPECTED_APPROVAL_TOKEN="ROLLBACK_STORYLIGHT_TRAINED_PLANNER:${target_user}:${candidate_id}:${manifest_sha256}:${backup_sha256}:${state_gate_evidence}"
if ((dry_run == 1)); then
  printf '%s\n' "$verification"
  printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  printf 'Would restore checksum %s from %s and restart the accepted engine.\n' \
    "$backup_sha256" "$backup_file"
  printf 'Dry run complete; no files or services changed.\n'
  exit 0
fi
if [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Rollback requires the exact one-purpose approval token printed by --dry-run.\n' >&2
  exit 77
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Rollback refuses active swap. Disable build-only swap first.\n' >&2
  exit 70
fi

target_uid="$(id -u "$target_user")"
target_home="$(getent passwd "$target_user" | cut -d: -f6)"
if [[ -z "$target_home" || "$target_home" != /* || ! -d "$target_home" ]]; then
  printf 'Cannot resolve the Storylight service user home directory.\n' >&2
  exit 65
fi
readonly TARGET_UID="$target_uid"
readonly TARGET_HOME="$target_home"
readonly ACCEPTED_ENGINE="$TARGET_HOME/.local/share/storylight/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine"
if [[ ! -s "$ACCEPTED_ENGINE" || -L "$ACCEPTED_ENGINE" ]] \
  || [[ "$(sha256sum "$ACCEPTED_ENGINE" | cut -d' ' -f1)" != "$ACCEPTED_ENGINE_SHA256" ]]; then
  printf 'The exact accepted TensorRT engine is missing or changed.\n' >&2
  exit 78
fi

user_systemctl() {
  runuser -u "$target_user" -- env \
    XDG_RUNTIME_DIR="/run/user/${TARGET_UID}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${TARGET_UID}/bus" \
    systemctl --user "$@"
}

readonly UNIT="storylight-trained-planner-candidate@${candidate_id}.service"
readonly ACCEPTED_UNIT="storylight-tensorrt-planner.service"
readonly GEMMA_UNIT="storylight-gemma.service"
readonly KIOSK_UNIT="storylight-kiosk.service"
readonly PORT_ENV="/etc/storylight/trained-planner/${candidate_id}.env"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
rollback_complete=0

finish_restore() {
  exit_code=$?
  trap - EXIT INT TERM
  if ((rollback_complete == 0)); then
    printf 'Rollback was interrupted; retrying the accepted engine restoration.\n' >&2
    user_systemctl stop "$UNIT" || true
    emergency_tmp="$(mktemp /etc/storylight/.storylight.env.emergency.XXXXXX)"
    install -o root -g root -m 0600 "$backup_file" "$emergency_tmp"
    mv -f "$emergency_tmp" "$CONFIG_FILE"
    user_systemctl stop "$GEMMA_UNIT" || true
    user_systemctl enable "$ACCEPTED_UNIT" >/dev/null 2>&1 || true
    user_systemctl start "$ACCEPTED_UNIT" || true
    systemctl restart "storylight@${target_user}.service" \
      "storylight-controller@${target_user}.service" || true
    if ((kiosk_was_active == 1)); then
      user_systemctl start "$KIOSK_UNIT" || true
    fi
  fi
  exit "$exit_code"
}
trap finish_restore EXIT INT TERM

user_systemctl stop "$KIOSK_UNIT" || true
user_systemctl disable "$UNIT" >/dev/null 2>&1 || true
user_systemctl stop "$UNIT" || true
user_systemctl enable "$ACCEPTED_UNIT" >/dev/null
if [[ -e "$PORT_ENV" ]]; then
  install -d -o root -g root -m 0700 "$STATE_DIR/history"
  mv -f "$PORT_ENV" "$STATE_DIR/history/${TIMESTAMP}-${candidate_id}-port.env"
fi
restore_tmp="$(mktemp /etc/storylight/.storylight.env.rollback.XXXXXX)"
install -o root -g root -m 0600 "$backup_file" "$restore_tmp"
mv -f "$restore_tmp" "$CONFIG_FILE"
if [[ "$(sha256sum "$CONFIG_FILE" | cut -d' ' -f1)" != "$backup_sha256" ]]; then
  printf 'Restored environment checksum does not match the durable backup.\n' >&2
  exit 78
fi
user_systemctl stop "$GEMMA_UNIT" || true
user_systemctl start "$ACCEPTED_UNIT"
for _ in $(seq 1 90); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:11435/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:11435/v1/models >/dev/null
systemctl restart "storylight@${target_user}.service" \
  "storylight-controller@${target_user}.service"
curl --fail --silent --show-error http://127.0.0.1:8080/readyz >/dev/null
curl --fail --silent --show-error http://127.0.0.1:8081/healthz >/dev/null
if ((kiosk_was_active == 1)); then
  user_systemctl start "$KIOSK_UNIT"
  user_systemctl is-active --quiet "$KIOSK_UNIT"
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Swap became active during rollback.\n' >&2
  exit 70
fi

install -d -o root -g root -m 0700 "$STATE_DIR/history"
receipt_tmp="$(mktemp "$STATE_DIR/history/.rollback.XXXXXX")"
sed 's/^PHASE=.*/PHASE=ROLLED_BACK/' "$ACTIVE_STATE" >"$receipt_tmp"
printf 'ROLLED_BACK_AT=%s\n' "$TIMESTAMP" >>"$receipt_tmp"
chown root:root "$receipt_tmp"
chmod 0600 "$receipt_tmp"
mv -f "$receipt_tmp" "$STATE_DIR/history/${TIMESTAMP}-${candidate_id}-rolled-back.env"
mv -f "$ACTIVE_STATE" "$STATE_DIR/history/${TIMESTAMP}-${candidate_id}-active-original.env"
rollback_complete=1
trap - EXIT INT TERM
printf 'Exact accepted TensorRT engine and environment restored for %s.\n' "$candidate_id"
