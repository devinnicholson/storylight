#!/usr/bin/env bash
# Evaluate one installed candidate locally and restore the accepted appliance on every exit.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly INSTALLER="$SCRIPT_DIR/install-trained-planner-candidate.sh"
readonly DEVELOPMENT_GATE="$SCRIPT_DIR/trained-planner-candidate-evaluation.py"
readonly BOOKFORGE_PYTHON="${BOOKFORGE_PYTHON:-/opt/bookforge/.venv/bin/python}"
readonly CONFIG_FILE="/etc/bookforge/bookforge.env"
readonly CANDIDATE_ROOT="/var/lib/bookforge-trusted/trained-planner-candidates"
readonly EVIDENCE_ROOT="/var/lib/bookforge-trusted/trained-planner/evidence"
readonly HIDDEN_STATE_ROOT="/var/lib/bookforge-trusted/trained-planner/hidden-evaluation-state"
readonly ACCEPTED_ENGINE_SHA256="95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"

target_user="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-}}"
candidate_id=""
dataset_manifest=""
dataset_manifest_sha256=""
development_records=""
baseline_development_report=""
baseline_development_report_sha256=""
hidden_records=""
custody_receipt=""
development_output=""
eligibility_output=""
hidden_output=""
window_output=""
approval_token=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo run-trained-planner-candidate-evaluation.sh OPTIONS

Required:
  --user USER
  --candidate-id ID
  --dataset-manifest PATH
  --dataset-manifest-sha256 SHA256
  --development-records PATH
  --baseline-development-report PATH
  --baseline-development-report-sha256 SHA256
  --hidden-records PATH
  --custody-receipt PATH
  --development-output PATH
  --eligibility-output PATH
  --hidden-output PATH
  --window-output PATH

Options:
  --approval-token TOKEN  Exact token printed by --dry-run.
  --dry-run               Verify immutable inputs and print the one-purpose token.
  -h, --help              Show this help.

The hidden records and custody receipt must be mode 0600. A real run stops the kiosk and accepted
planner, serves the immutable candidate only on 127.0.0.1:11436, evaluates development, refuses
hidden evaluation unless the device development gate passes, consumes hidden evaluation once,
and restores the exact accepted engine, API, controller, and kiosk on every exit.
EOF
}

while (($#)); do
  case "$1" in
    --user) target_user="$2"; shift 2 ;;
    --candidate-id) candidate_id="$2"; shift 2 ;;
    --dataset-manifest) dataset_manifest="$2"; shift 2 ;;
    --dataset-manifest-sha256) dataset_manifest_sha256="$2"; shift 2 ;;
    --development-records) development_records="$2"; shift 2 ;;
    --baseline-development-report) baseline_development_report="$2"; shift 2 ;;
    --baseline-development-report-sha256) baseline_development_report_sha256="$2"; shift 2 ;;
    --hidden-records) hidden_records="$2"; shift 2 ;;
    --custody-receipt) custody_receipt="$2"; shift 2 ;;
    --development-output) development_output="$2"; shift 2 ;;
    --eligibility-output) eligibility_output="$2"; shift 2 ;;
    --hidden-output) hidden_output="$2"; shift 2 ;;
    --window-output) window_output="$2"; shift 2 ;;
    --approval-token) approval_token="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
  || ! id "$target_user" >/dev/null 2>&1 \
  || [[ ! "$candidate_id" =~ ^[a-z0-9][a-z0-9-]{2,95}$ ]] \
  || [[ ! "$dataset_manifest_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$baseline_development_report_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  usage >&2
  exit 64
fi
for required in "$INSTALLER" "$DEVELOPMENT_GATE" "$BOOKFORGE_PYTHON"; do
  if [[ ! -x "$required" || -L "$required" ]]; then
    printf 'Required immutable evaluation tool is missing or unsafe: %s\n' "$required" >&2
    exit 69
  fi
done

readonly TARGET_UID="$(id -u "$target_user")"
readonly TARGET_HOME="$(getent passwd "$target_user" | cut -d: -f6)"
readonly CANDIDATE_DIR="$CANDIDATE_ROOT/$candidate_id"
readonly CANDIDATE_MANIFEST="$CANDIDATE_DIR/candidate.manifest.json"
readonly MARKER="$CANDIDATE_DIR/.manifest.sha256"
readonly ACCEPTED_ENGINE="$TARGET_HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine"
readonly UNIT="bookforge-trained-planner-candidate@${candidate_id}.service"
readonly ACCEPTED_UNIT="bookforge-tensorrt-planner.service"
readonly GEMMA_UNIT="bookforge-gemma.service"
readonly KIOSK_UNIT="bookforge-kiosk.service"
readonly PORT_ENV="/etc/bookforge/trained-planner/${candidate_id}.env"

if [[ ! -r "$MARKER" || -L "$MARKER" ]]; then
  printf 'Installed candidate manifest marker is missing or unsafe.\n' >&2
  exit 66
fi
manifest_sha256="$(tr -d '[:space:]' <"$MARKER")"
if [[ ! "$manifest_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  printf 'Installed candidate manifest marker is invalid.\n' >&2
  exit 65
fi
verification="$("$INSTALLER" \
  --bundle "$CANDIDATE_DIR" \
  --expected-manifest-sha256 "$manifest_sha256" \
  --verify-only \
  --installed-layout)"
verified_candidate_id="$(printf '%s\n' "$verification" | sed -n 's/^candidate_id=//p')"
model_revision="$(printf '%s\n' "$verification" | sed -n 's/^model_revision=//p')"
engine_sha256="$(printf '%s\n' "$verification" | sed -n 's/^engine_sha256=//p')"
if [[ "$verified_candidate_id" != "$candidate_id" ]] \
  || [[ ! "$model_revision" =~ ^sha256:[a-f0-9]{64}$ ]] \
  || [[ "$engine_sha256" != "${model_revision#sha256:}" ]]; then
  printf 'Installed candidate identity is invalid.\n' >&2
  exit 65
fi

secure_input() {
  local path="$1"
  local label="$2"
  local mode
  if [[ "$path" != /* || ! -f "$path" || -L "$path" ]] \
    || [[ "$(realpath -e -- "$path")" != "$path" ]]; then
    printf '%s must be an absolute, canonical, non-writable regular file.\n' "$label" >&2
    exit 65
  fi
  mode="$(stat -c '%a' "$path")"
  case "$mode" in
    400|440|444|600|640|644) ;;
    *)
      printf '%s has an unsafe file mode: %s.\n' "$label" "$mode" >&2
      exit 65
      ;;
  esac
}

secure_private_input() {
  local path="$1"
  local label="$2"
  secure_input "$path" "$label"
  local owner_uid
  owner_uid="$(stat -c '%u' "$path")"
  if [[ "$(stat -c '%a' "$path")" != 600 ]] \
    || [[ "$owner_uid" -ne 0 && "$owner_uid" -ne "$TARGET_UID" ]]; then
    printf '%s must be mode 0600 and owned by root or the Bookforge user.\n' "$label" >&2
    exit 65
  fi
}

secure_input "$dataset_manifest" "dataset manifest"
secure_input "$development_records" "development records"
secure_input "$baseline_development_report" "baseline development report"
secure_private_input "$hidden_records" "private hidden records"
secure_private_input "$custody_receipt" "hidden custody receipt"
if [[ "$(sha256sum "$dataset_manifest" | cut -d' ' -f1)" != "$dataset_manifest_sha256" ]] \
  || [[ "$(sha256sum "$baseline_development_report" | cut -d' ' -f1)" \
    != "$baseline_development_report_sha256" ]]; then
  printf 'Dataset or baseline development evidence checksum changed.\n' >&2
  exit 65
fi
if [[ -e "$PORT_ENV" || -L "$PORT_ENV" ]]; then
  printf 'Candidate port override exists; evaluation requires isolated port 11436.\n' >&2
  exit 78
fi

readonly OUTPUT_ROOT="$EVIDENCE_ROOT/$candidate_id"
declare -A seen_outputs=()
for output in "$development_output" "$eligibility_output" "$hidden_output" "$window_output"; do
  if [[ "$output" != "$OUTPUT_ROOT"/* ]] \
    || [[ "$(realpath -m -- "$output")" != "$output" ]] \
    || [[ -e "$output" || -L "$output" ]] \
    || [[ -n "${seen_outputs[$output]:-}" ]]; then
    printf 'Evaluation outputs must be distinct new canonical paths below %s.\n' \
      "$OUTPUT_ROOT" >&2
    exit 64
  fi
  seen_outputs["$output"]=1
done

hidden_record_sha256="$(sha256sum "$hidden_records" | cut -d' ' -f1)"
custody_receipt_sha256="$(sha256sum "$custody_receipt" | cut -d' ' -f1)"
development_records_sha256="$(sha256sum "$development_records" | cut -d' ' -f1)"
identity_digest="$(printf '%s\n%s\n' "$manifest_sha256" "$engine_sha256" | sha256sum | cut -d' ' -f1)"
if [[ -e "$HIDDEN_STATE_ROOT/$identity_digest" || -L "$HIDDEN_STATE_ROOT/$identity_digest" ]]; then
  printf 'This candidate and engine already consumed hidden evaluation.\n' >&2
  exit 73
fi
action_sha256="$({
  printf '%s\0' "$target_user" "$candidate_id" "$manifest_sha256" "$engine_sha256"
  printf '%s\0' "$dataset_manifest_sha256" "$development_records_sha256"
  printf '%s\0' "$baseline_development_report_sha256" "$hidden_record_sha256"
  printf '%s\0' "$custody_receipt_sha256" "$development_output" "$eligibility_output"
  printf '%s\0' "$hidden_output" "$window_output"
} | sha256sum | cut -d' ' -f1)"
readonly EXPECTED_APPROVAL_TOKEN="RUN_BOOKFORGE_TRAINED_PLANNER_CANDIDATE_EVALUATION:${action_sha256}"

if ((dry_run == 1)); then
  printf '%s\n' "$verification"
  printf 'development_output=%s\neligibility_output=%s\nhidden_output=%s\nwindow_output=%s\n' \
    "$development_output" "$eligibility_output" "$hidden_output" "$window_output"
  printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  printf 'Would serve only %s on 127.0.0.1:11436, evaluate locally, and restore %s.\n' \
    "$engine_sha256" "$ACCEPTED_ENGINE_SHA256"
  printf 'Dry run complete; no files or services changed.\n'
  exit 0
fi
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run candidate evaluation with sudo.\n' >&2
  exit 64
fi
if [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Candidate evaluation requires the exact one-purpose token printed by --dry-run.\n' >&2
  exit 77
fi
if [[ ! -f "$CONFIG_FILE" || -L "$CONFIG_FILE" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_FILE")" != root:root:600 ]] \
  || [[ ! -s "$ACCEPTED_ENGINE" || -L "$ACCEPTED_ENGINE" ]] \
  || [[ "$(sha256sum "$ACCEPTED_ENGINE" | cut -d' ' -f1)" != "$ACCEPTED_ENGINE_SHA256" ]]; then
  printf 'Accepted Bookforge configuration or engine identity is unsafe.\n' >&2
  exit 78
fi
if ! grep -Fxq 'BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_slots' "$CONFIG_FILE" \
  || ! grep -Fxq 'BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL=http://127.0.0.1:11435' "$CONFIG_FILE" \
  || ! grep -Fxq \
    "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION=sha256:${ACCEPTED_ENGINE_SHA256}" \
    "$CONFIG_FILE"; then
  printf 'Production is not routed to the exact accepted planner.\n' >&2
  exit 78
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Candidate evaluation refuses active swap.\n' >&2
  exit 70
fi

install -d -o root -g root -m 0700 "$EVIDENCE_ROOT" "$OUTPUT_ROOT" "$HIDDEN_STATE_ROOT"
readonly CONFIG_SHA256_BEFORE="$(sha256sum "$CONFIG_FILE" | cut -d' ' -f1)"
kiosk_was_active=0
restored=0

user_systemctl() {
  runuser -u "$target_user" -- env \
    XDG_RUNTIME_DIR="/run/user/${TARGET_UID}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${TARGET_UID}/bus" \
    systemctl --user "$@"
}

wait_endpoint() {
  local url="$1"
  for _ in $(seq 1 90); do
    if curl --fail --silent --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

restore_accepted() {
  local failed=0
  user_systemctl stop "$KIOSK_UNIT" || failed=1
  user_systemctl stop "$UNIT" || failed=1
  user_systemctl stop "$GEMMA_UNIT" || failed=1
  user_systemctl start "$ACCEPTED_UNIT" || failed=1
  wait_endpoint http://127.0.0.1:11435/v1/models || failed=1
  systemctl restart "bookforge@${target_user}.service" \
    "bookforge-controller@${target_user}.service" || failed=1
  wait_endpoint http://127.0.0.1:8080/readyz || failed=1
  if ((kiosk_was_active == 1)); then
    user_systemctl start "$KIOSK_UNIT" || failed=1
    user_systemctl is-active --quiet "$KIOSK_UNIT" || failed=1
  fi
  if [[ "$(sha256sum "$CONFIG_FILE" 2>/dev/null | cut -d' ' -f1 || true)" \
      != "$CONFIG_SHA256_BEFORE" ]] \
    || [[ "$(sha256sum "$ACCEPTED_ENGINE" 2>/dev/null | cut -d' ' -f1 || true)" \
      != "$ACCEPTED_ENGINE_SHA256" ]] \
    || ! user_systemctl is-active --quiet "$ACCEPTED_UNIT" \
    || ! curl --fail --silent --max-time 2 http://127.0.0.1:11435/v1/models \
      >/dev/null 2>&1; then
    failed=1
  fi
  restored=1
  return "$failed"
}

restore_on_exit() {
  exit_code=$?
  trap - EXIT INT TERM
  if ((restored == 0)) && ! restore_accepted; then
    exit_code=70
  fi
  exit "$exit_code"
}

if ! user_systemctl is-active --quiet "$ACCEPTED_UNIT" \
  || ! user_systemctl is-active --quiet "$KIOSK_UNIT" \
  || ! curl --fail --silent --max-time 2 http://127.0.0.1:11435/v1/models \
    >/dev/null 2>&1; then
  printf 'Accepted TensorRT and kiosk services must be healthy before evaluation.\n' >&2
  exit 70
fi
kiosk_was_active=1
trap restore_on_exit EXIT INT TERM

user_systemctl stop "$KIOSK_UNIT"
user_systemctl stop "$ACCEPTED_UNIT"
user_systemctl stop "$GEMMA_UNIT" || true
for _ in $(seq 1 30); do
  if ! pgrep -u "$TARGET_UID" -f 'experimental.server|ollama serve|firefox-bookforge' \
    >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if pgrep -u "$TARGET_UID" -f 'experimental.server|ollama serve|firefox-bookforge' \
  >/dev/null 2>&1; then
  printf 'The accepted planner, fallback, or kiosk did not drain before evaluation.\n' >&2
  exit 70
fi
available_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || ((available_kib < 4194304)); then
  printf 'Candidate evaluation requires at least 4 GiB available memory.\n' >&2
  exit 70
fi
user_systemctl reset-failed "$UNIT" || true
user_systemctl start "$UNIT"
wait_endpoint http://127.0.0.1:11436/v1/models
if ss -H -ltn | awk \
  '$4 ~ /:11436$/ && $4 !~ /^127[.]0[.]0[.]1:/ && $4 !~ /^\[::1\]:/ {found=1} END {exit !found}'; then
  printf 'Candidate planner exposed a non-loopback listener.\n' >&2
  exit 70
fi

"$BOOKFORGE_PYTHON" -m bookforge.fidelity_endpoint_evaluation \
  --records "$development_records" \
  --manifest "$dataset_manifest" \
  --manifest-sha256 "$dataset_manifest_sha256" \
  --split development \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --serving-engine-sha256 "$engine_sha256" \
  --base-url http://127.0.0.1:11436 \
  --model llm \
  --timeout-seconds 12 \
  --output "$development_output" \
  --execute
development_output_sha256="$(sha256sum "$development_output" | cut -d' ' -f1)"

"$BOOKFORGE_PYTHON" "$DEVELOPMENT_GATE" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --dataset-manifest "$dataset_manifest" \
  --dataset-manifest-sha256 "$dataset_manifest_sha256" \
  --candidate-report "$development_output" \
  --candidate-report-sha256 "$development_output_sha256" \
  --baseline-report "$baseline_development_report" \
  --baseline-report-sha256 "$baseline_development_report_sha256" \
  --output "$eligibility_output"
eligibility_output_sha256="$(sha256sum "$eligibility_output" | cut -d' ' -f1)"

hidden_plan="$("$BOOKFORGE_PYTHON" -m bookforge.fidelity_endpoint_evaluation \
  --records "$hidden_records" \
  --manifest "$dataset_manifest" \
  --manifest-sha256 "$dataset_manifest_sha256" \
  --custody-receipt "$custody_receipt" \
  --split hidden \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --serving-engine-sha256 "$engine_sha256" \
  --base-url http://127.0.0.1:11436 \
  --model llm \
  --timeout-seconds 12 \
  --output "$hidden_output" \
  --hidden-state-root "$HIDDEN_STATE_ROOT")"
hidden_approval="$("$BOOKFORGE_PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin)["approval_token"])' <<<"$hidden_plan")"
if [[ "$hidden_approval" != EVALUATE_PRIVATE_HIDDEN_ONCE:* ]]; then
  printf 'Hidden evaluator did not produce its exact one-shot approval.\n' >&2
  exit 70
fi
BOOKFORGE_HIDDEN_EVAL_APPROVAL="$hidden_approval" \
  "$BOOKFORGE_PYTHON" -m bookforge.fidelity_endpoint_evaluation \
  --records "$hidden_records" \
  --manifest "$dataset_manifest" \
  --manifest-sha256 "$dataset_manifest_sha256" \
  --custody-receipt "$custody_receipt" \
  --split hidden \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --serving-engine-sha256 "$engine_sha256" \
  --base-url http://127.0.0.1:11436 \
  --model llm \
  --timeout-seconds 12 \
  --output "$hidden_output" \
  --hidden-state-root "$HIDDEN_STATE_ROOT" \
  --execute
hidden_output_sha256="$(sha256sum "$hidden_output" | cut -d' ' -f1)"

properties="$(user_systemctl show "$UNIT" \
  --property=MemoryPeak --property=NRestarts --property=OOMKilled)"
memory_peak="$(printf '%s\n' "$properties" | sed -n 's/^MemoryPeak=//p')"
restarts="$(printf '%s\n' "$properties" | sed -n 's/^NRestarts=//p')"
oom="$(printf '%s\n' "$properties" | sed -n 's/^OOMKilled=//p')"
if [[ ! "$memory_peak" =~ ^[1-9][0-9]*$ ]] \
  || [[ "$restarts" != 0 ]] \
  || [[ "$oom" != no ]] \
  || swapon --show --noheadings | grep -q .; then
  printf 'Candidate evaluation observed invalid memory, restart, OOM, or swap telemetry.\n' >&2
  exit 70
fi

if ! restore_accepted; then
  exit 70
fi
trap - EXIT INT TERM
"$BOOKFORGE_PYTHON" - \
  "$window_output" "$candidate_id" "$manifest_sha256" "$engine_sha256" \
  "$dataset_manifest_sha256" "$development_output_sha256" \
  "$eligibility_output_sha256" "$hidden_output_sha256" "$custody_receipt_sha256" \
  "$memory_peak" "$CONFIG_SHA256_BEFORE" "$ACCEPTED_ENGINE_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
document = {
    "schema_version": "1.0",
    "producer": "bookforge-jetson-candidate-evaluation-window",
    "status": "succeeded",
    "candidate_identity": {
        "candidate_id": sys.argv[2],
        "candidate_manifest_sha256": sys.argv[3],
        "engine_sha256": sys.argv[4],
        "model_revision": f"sha256:{sys.argv[4]}",
    },
    "dataset_manifest_sha256": sys.argv[5],
    "evidence_sha256": {
        "development_summary": sys.argv[6],
        "development_gate": sys.argv[7],
        "hidden_summary": sys.argv[8],
        "hidden_custody_receipt": sys.argv[9],
    },
    "candidate_memory_peak_bytes": int(sys.argv[10]),
    "accepted_config_sha256_before_and_after": sys.argv[11],
    "accepted_engine_sha256_before_and_after": sys.argv[12],
    "candidate_endpoint": "http://127.0.0.1:11436",
    "hidden_records_retained": False,
    "model_outputs_retained": False,
    "restoration_demonstrated": True,
}
payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    os.fchmod(stream.fileno(), 0o600)
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
printf 'Candidate evaluation completed and accepted engine %s was restored.\n' \
  "$ACCEPTED_ENGINE_SHA256"
