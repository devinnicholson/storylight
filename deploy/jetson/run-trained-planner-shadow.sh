#!/usr/bin/env bash
# Run a checksum-bound ABBA acceptance window and restore the accepted engine exactly.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly INSTALLER="${BOOKFORGE_CANDIDATE_INSTALLER:-$SCRIPT_DIR/install-trained-planner-candidate.sh}"
readonly EVIDENCE_HELPER="${BOOKFORGE_SHADOW_EVIDENCE_HELPER:-$SCRIPT_DIR/trained-planner-shadow-evidence.py}"
readonly BOOKFORGE_PYTHON="${BOOKFORGE_PYTHON:-/opt/bookforge/.venv/bin/python}"
readonly CONFIG_FILE="/etc/bookforge/bookforge.env"
readonly CANDIDATE_ROOT="/var/lib/bookforge/trained-planner-candidates"
readonly PORT_ROOT="/etc/bookforge/trained-planner"
readonly EVIDENCE_ROOT="/var/lib/bookforge/trained-planner/evidence"
readonly ACCEPTED_ENGINE_SHA256="95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
readonly CACHE_CONTRACT_REVISION="semantic-v18-tensorrt-slot-privacy"
target_user="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-}}"
candidate_id=""
suite="contest"
runtime_output=""
stage_output=""
hidden_report=""
hidden_report_sha256=""
run_id=""
config_sha256=""
dataset_manifest_sha256=""
int4_export_sha256=""
approval_token=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo run-trained-planner-shadow.sh OPTIONS

Required:
  --user USER
  --candidate-id ID
  --hidden-evaluation-report PATH
  --hidden-evaluation-report-sha256 SHA256
  --run-id ID
  --config-sha256 SHA256
  --dataset-manifest-sha256 SHA256
  --int4-export-sha256 SHA256

Options:
  --suite five|contest  Semantic suite (default: contest).
  --runtime-output PATH New gate-compatible runtime evidence JSON path.
  --stage-output PATH   New hash-chained jetson-shadow stage JSON path.
  --approval-token TOKEN  Exact token printed by --dry-run.
  --dry-run             Verify immutable inputs without changing services.
  -h, --help            Show this help.

The hidden input is an aggregate, externally generated report. This script never reads hidden
records or model outputs. A real run measures baseline/candidate/candidate/baseline and restores
the exact accepted engine, environment, API route, controller, and kiosk on every exit.
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
    --suite)
      [[ $# -ge 2 ]] || { printf '%s\n' '--suite requires a value' >&2; exit 64; }
      suite="$2"
      shift 2
      ;;
    --runtime-output)
      [[ $# -ge 2 ]] || { printf '%s\n' '--runtime-output requires a value' >&2; exit 64; }
      runtime_output="$2"
      shift 2
      ;;
    --stage-output)
      [[ $# -ge 2 ]] || { printf '%s\n' '--stage-output requires a value' >&2; exit 64; }
      stage_output="$2"
      shift 2
      ;;
    --hidden-evaluation-report)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--hidden-evaluation-report requires a value' >&2
        exit 64
      }
      hidden_report="$2"
      shift 2
      ;;
    --hidden-evaluation-report-sha256)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--hidden-evaluation-report-sha256 requires a value' >&2
        exit 64
      }
      hidden_report_sha256="$2"
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || { printf '%s\n' '--run-id requires a value' >&2; exit 64; }
      run_id="$2"
      shift 2
      ;;
    --config-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--config-sha256 requires a value' >&2; exit 64; }
      config_sha256="$2"
      shift 2
      ;;
    --dataset-manifest-sha256)
      [[ $# -ge 2 ]] || {
        printf '%s\n' '--dataset-manifest-sha256 requires a value' >&2
        exit 64
      }
      dataset_manifest_sha256="$2"
      shift 2
      ;;
    --int4-export-sha256)
      [[ $# -ge 2 ]] || { printf '%s\n' '--int4-export-sha256 requires a value' >&2; exit 64; }
      int4_export_sha256="$2"
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

if [[ ! "$target_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
  || ! id "$target_user" >/dev/null 2>&1 \
  || [[ ! "$candidate_id" =~ ^[a-z0-9][a-z0-9-]{2,95}$ ]] \
  || [[ "$suite" != "five" && "$suite" != "contest" ]] \
  || [[ ! "$run_id" =~ ^[a-z0-9][a-z0-9-]{2,63}$ ]] \
  || [[ ! "$hidden_report_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$config_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$dataset_manifest_sha256" =~ ^[a-f0-9]{64}$ ]] \
  || [[ ! "$int4_export_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  usage >&2
  exit 64
fi
if [[ ! -x "$BOOKFORGE_PYTHON" || ! -x "$EVIDENCE_HELPER" ]]; then
  printf 'Bookforge Python or the shadow evidence helper is missing.\n' >&2
  exit 69
fi

readonly CANDIDATE_DIR="$CANDIDATE_ROOT/$candidate_id"
readonly MARKER="$CANDIDATE_DIR/.manifest.sha256"
if [[ ! -r "$MARKER" || -L "$MARKER" ]]; then
  printf 'Installed candidate manifest marker is missing or unsafe.\n' >&2
  exit 66
fi
manifest_sha256="$(tr -d '[:space:]' <"$MARKER")"
if [[ ! "$manifest_sha256" =~ ^[a-f0-9]{64}$ ]]; then
  printf 'Installed candidate manifest marker is invalid.\n' >&2
  exit 65
fi
verification="$(
  "$INSTALLER" \
    --bundle "$CANDIDATE_DIR" \
    --expected-manifest-sha256 "$manifest_sha256" \
    --verify-only \
    --installed-layout
)"
model_revision="$(printf '%s\n' "$verification" | sed -n 's/^model_revision=//p')"
engine_sha256="$(printf '%s\n' "$verification" | sed -n 's/^engine_sha256=//p')"
verified_candidate_id="$(printf '%s\n' "$verification" | sed -n 's/^candidate_id=//p')"
if [[ "$verified_candidate_id" != "$candidate_id" ]] \
  || [[ ! "$model_revision" =~ ^sha256:[a-f0-9]{64}$ ]] \
  || [[ "$engine_sha256" != "${model_revision#sha256:}" ]]; then
  printf 'Installed candidate identity is invalid.\n' >&2
  exit 65
fi
"$EVIDENCE_HELPER" validate-candidate \
  --manifest "$CANDIDATE_DIR/candidate.manifest.json" \
  --expected-sha256 "$manifest_sha256" \
  --config-sha256 "$config_sha256" \
  --dataset-manifest-sha256 "$dataset_manifest_sha256" >/dev/null
"$EVIDENCE_HELPER" validate-hidden \
  --report "$hidden_report" \
  --expected-sha256 "$hidden_report_sha256" \
  --candidate-revision "$model_revision" >/dev/null

if [[ -z "$runtime_output" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  runtime_output="$EVIDENCE_ROOT/$candidate_id/shadow-$timestamp.runtime.json"
fi
if [[ -z "$stage_output" ]]; then
  stage_output="${runtime_output%.runtime.json}.stage.json"
fi
if [[ "$runtime_output" == "$stage_output" ]]; then
  printf 'Runtime and stage evidence paths must be different.\n' >&2
  exit 64
fi
for evidence_path in "$runtime_output" "$stage_output"; do
  if [[ "$evidence_path" != /* ]] || [[ -e "$evidence_path" || -L "$evidence_path" ]]; then
    printf 'Shadow outputs must be new absolute paths.\n' >&2
    exit 64
  fi
  case "$evidence_path" in
    "$EVIDENCE_ROOT"/*) ;;
    *)
      printf 'Shadow evidence must remain below %s.\n' "$EVIDENCE_ROOT" >&2
      exit 64
      ;;
  esac
  case "$evidence_path" in
    "$CANDIDATE_DIR"|"$CANDIDATE_DIR"/*)
      printf 'Shadow evidence cannot modify the immutable candidate directory.\n' >&2
      exit 64
      ;;
  esac
done

action_sha256="$(
  printf '%s\0' \
    "$target_user" "$candidate_id" "$manifest_sha256" "$suite" \
    "$runtime_output" "$stage_output" "$hidden_report_sha256" "$run_id" \
    "$config_sha256" "$dataset_manifest_sha256" "$int4_export_sha256" \
    | sha256sum | cut -d' ' -f1
)"
readonly EXPECTED_APPROVAL_TOKEN="RUN_BOOKFORGE_TRAINED_PLANNER_SHADOW:${action_sha256}"

if ((dry_run == 1)); then
  printf '%s\n' "$verification"
  printf 'hidden_evaluation_report_sha256=%s\n' "$hidden_report_sha256"
  printf 'run_id=%s\nconfig_sha256=%s\ndataset_manifest_sha256=%s\nint4_export_sha256=%s\n' \
    "$run_id" "$config_sha256" "$dataset_manifest_sha256" "$int4_export_sha256"
  printf 'runtime_output=%s\nstage_output=%s\n' "$runtime_output" "$stage_output"
  printf 'Required approval token: %s\n' "$EXPECTED_APPROVAL_TOKEN"
  printf 'Would run ABBA on 127.0.0.1:11435, exercise the projector flow, and restore %s.\n' \
    "$ACCEPTED_ENGINE_SHA256"
  printf 'Dry run complete; no files or services changed.\n'
  exit 0
fi
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run the physical shadow acceptance with sudo.\n' >&2
  exit 64
fi
if [[ "$approval_token" != "$EXPECTED_APPROVAL_TOKEN" ]]; then
  printf 'Shadow acceptance requires the exact one-purpose token printed by --dry-run.\n' >&2
  exit 77
fi
if [[ ! -f "$CONFIG_FILE" || -L "$CONFIG_FILE" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_FILE")" != "root:root:600" ]]; then
  printf 'Bookforge environment is missing or unsafe.\n' >&2
  exit 78
fi
if swapon --show --noheadings | grep -q .; then
  printf 'Shadow inference refuses active swap. Disable build-only swap first.\n' >&2
  exit 70
fi

target_uid="$(id -u "$target_user")"
target_home="$(getent passwd "$target_user" | cut -d: -f6)"
if [[ -z "$target_home" || "$target_home" != /* || ! -d "$target_home" ]]; then
  printf 'Cannot resolve the Bookforge service user home directory.\n' >&2
  exit 65
fi
readonly TARGET_UID="$target_uid"
readonly TARGET_HOME="$target_home"
readonly ACCEPTED_ENGINE="$TARGET_HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm/llm.engine"
if [[ ! -s "$ACCEPTED_ENGINE" || -L "$ACCEPTED_ENGINE" ]] \
  || [[ "$(sha256sum "$ACCEPTED_ENGINE" | cut -d' ' -f1)" != "$ACCEPTED_ENGINE_SHA256" ]]; then
  printf 'The exact accepted TensorRT engine is missing or changed.\n' >&2
  exit 78
fi

readonly UNIT="bookforge-trained-planner-candidate@${candidate_id}.service"
readonly ACCEPTED_UNIT="bookforge-tensorrt-planner.service"
readonly GEMMA_UNIT="bookforge-gemma.service"
readonly KIOSK_UNIT="bookforge-kiosk.service"
readonly PORT_ENV="$PORT_ROOT/${candidate_id}.env"
readonly OUTPUT_PARENT="$(dirname -- "$runtime_output")"
install -d -o root -g root -m 0700 "$OUTPUT_PARENT" "$PORT_ROOT"
work_dir="$(mktemp -d "$OUTPUT_PARENT/.${candidate_id}.shadow.partial.XXXXXX")"
chmod 0700 "$work_dir"
readonly WORK_DIR="$work_dir"
readonly ORIGINAL_CONFIG="$WORK_DIR/bookforge.env.accepted"
readonly CANDIDATE_CONFIG="$WORK_DIR/bookforge.env.candidate"
readonly RESTORATION="$WORK_DIR/restoration.json"
install -o root -g root -m 0600 "$CONFIG_FILE" "$ORIGINAL_CONFIG"
readonly ORIGINAL_CONFIG_SHA256="$(sha256sum "$ORIGINAL_CONFIG" | cut -d' ' -f1)"

python3 - "$ORIGINAL_CONFIG" "$CANDIDATE_CONFIG" "$model_revision" \
  "$CACHE_CONTRACT_REVISION" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
revision = sys.argv[3]
contract_revision = sys.argv[4]
replacements = {
    "BOOKFORGE_LIVE_SCENE_PLANNER": "model",
    "BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND": "tensorrt_slots",
    "BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL": "http://127.0.0.1:11435",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_NAME": "llm",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MAX_OUTPUT_TOKENS": "64",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION": revision,
    "BOOKFORGE_LIVE_SCENE_PLANNER_CACHE_CONTRACT_REVISION": contract_revision,
    "BOOKFORGE_LIVE_SCENE_PLANNER_COMPACT_WIRE": "false",
}
remaining = dict(replacements)
updated = []
for line in source.read_text(encoding="utf-8").splitlines():
    name, separator, _ = line.partition("=")
    updated.append(f"{name}={remaining.pop(name)}" if separator and name in remaining else line)
updated.extend(f"{name}={value}" for name, value in remaining.items())
destination.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
chown root:root "$CANDIDATE_CONFIG"
chmod 0600 "$CANDIDATE_CONFIG"

user_systemctl() {
  runuser -u "$target_user" -- env \
    XDG_RUNTIME_DIR="/run/user/${TARGET_UID}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${TARGET_UID}/bus" \
    systemctl --user "$@"
}

atomic_config() {
  source="$1"
  temporary="$(mktemp /etc/bookforge/.bookforge.env.shadow.XXXXXX)"
  install -o root -g root -m 0600 "$source" "$temporary"
  mv -f "$temporary" "$CONFIG_FILE"
}

wait_endpoint() {
  url="$1"
  for _ in $(seq 1 90); do
    if curl --fail --silent --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

restored=0
monitor_pid=""
monitor_flag=""
restore_accepted() {
  restore_failed=0
  if ! user_systemctl stop "$KIOSK_UNIT"; then restore_failed=1; fi
  if ! user_systemctl stop "$UNIT"; then restore_failed=1; fi
  if [[ -e "$PORT_ENV" || -L "$PORT_ENV" ]]; then
    if [[ -L "$PORT_ENV" || "$(stat -c '%U:%G:%a' "$PORT_ENV")" != "root:root:644" ]]; then
      restore_failed=1
    elif ! unlink "$PORT_ENV"; then
      restore_failed=1
    fi
  fi
  if ! atomic_config "$ORIGINAL_CONFIG"; then restore_failed=1; fi
  if ! user_systemctl stop "$GEMMA_UNIT"; then restore_failed=1; fi
  if ! user_systemctl start "$ACCEPTED_UNIT"; then restore_failed=1; fi
  if ! wait_endpoint http://127.0.0.1:11435/v1/models; then restore_failed=1; fi
  if ! systemctl restart "bookforge@${target_user}.service" \
    "bookforge-controller@${target_user}.service"; then
    restore_failed=1
  fi
  if ! wait_endpoint http://127.0.0.1:8080/readyz; then restore_failed=1; fi
  if ! user_systemctl start "$KIOSK_UNIT"; then restore_failed=1; fi
  if ! user_systemctl is-active --quiet "$KIOSK_UNIT"; then restore_failed=1; fi
  engine_after="$(sha256sum "$ACCEPTED_ENGINE" 2>/dev/null | cut -d' ' -f1 || true)"
  config_after="$(sha256sum "$CONFIG_FILE" 2>/dev/null | cut -d' ' -f1 || true)"
  endpoint_ready=false
  unit_active=false
  if curl --fail --silent --max-time 2 http://127.0.0.1:11435/v1/models >/dev/null 2>&1; then
    endpoint_ready=true
  fi
  if user_systemctl is-active --quiet "$ACCEPTED_UNIT"; then unit_active=true; fi
  python3 - "$RESTORATION" "$ACCEPTED_ENGINE_SHA256" "$engine_after" \
    "$ORIGINAL_CONFIG_SHA256" "$config_after" "$endpoint_ready" "$unit_active" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
document = {
    "accepted_engine_sha256": sys.argv[2],
    "engine_sha256_after": sys.argv[3],
    "config_sha256_before": sys.argv[4],
    "config_sha256_after": sys.argv[5],
    "endpoint_ready": sys.argv[6] == "true",
    "unit_active": sys.argv[7] == "true",
}
payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
if path.exists():
    path.unlink()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
  if [[ "$engine_after" != "$ACCEPTED_ENGINE_SHA256" ]] \
    || [[ "$config_after" != "$ORIGINAL_CONFIG_SHA256" ]] \
    || [[ "$endpoint_ready" != true || "$unit_active" != true ]]; then
    restore_failed=1
  fi
  restored=1
  return "$restore_failed"
}

restore_on_exit() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$monitor_flag" && -e "$monitor_flag" ]]; then unlink "$monitor_flag"; fi
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
  if ((restored == 0)); then
    if ! restore_accepted; then exit_code=70; fi
  fi
  exit "$exit_code"
}

if ! user_systemctl is-active --quiet "$ACCEPTED_UNIT" \
  || ! user_systemctl is-active --quiet "$KIOSK_UNIT"; then
  printf 'Accepted TensorRT and kiosk services must be active before physical acceptance.\n' >&2
  exit 70
fi
curl --fail --silent --max-time 2 http://127.0.0.1:8080/readyz >/dev/null
trap restore_on_exit EXIT INT TERM

activate_leg() {
  label="$1"
  user_systemctl stop "$KIOSK_UNIT"
  user_systemctl stop "$UNIT" || true
  user_systemctl stop "$ACCEPTED_UNIT" || true
  user_systemctl stop "$GEMMA_UNIT" || true
  if [[ "$label" == "candidate" ]]; then
    atomic_config "$CANDIDATE_CONFIG"
    port_tmp="$(mktemp "$PORT_ROOT/.${candidate_id}.env.XXXXXX")"
    printf 'BOOKFORGE_EDGELLM_SERVER_PORT=11435\n' >"$port_tmp"
    chown root:root "$port_tmp"
    chmod 0644 "$port_tmp"
    mv -f "$port_tmp" "$PORT_ENV"
    active_unit="$UNIT"
  else
    atomic_config "$ORIGINAL_CONFIG"
    if [[ -e "$PORT_ENV" ]]; then unlink "$PORT_ENV"; fi
    active_unit="$ACCEPTED_UNIT"
  fi
  for _ in $(seq 1 30); do
    if ! pgrep -u "$TARGET_UID" -f 'experimental.server|ollama serve|firefox-bookforge' \
      >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  available_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || ((available_kib < 4194304)); then
    printf 'Planner startup requires at least 4 GiB available memory.\n' >&2
    return 70
  fi
  user_systemctl reset-failed "$active_unit" || true
  ready_started="$(date +%s%N)"
  user_systemctl start "$active_unit"
  wait_endpoint http://127.0.0.1:11435/v1/models
  ready_finished="$(date +%s%N)"
  ACTIVE_READY_SECONDS="$(awk -v start="$ready_started" -v end="$ready_finished" \
    'BEGIN { printf "%.6f", (end - start) / 1000000000 }')"
  ACTIVE_UNIT="$active_unit"
  systemctl restart "bookforge@${target_user}.service" \
    "bookforge-controller@${target_user}.service"
  wait_endpoint http://127.0.0.1:8080/readyz
  user_systemctl start "$KIOSK_UNIT"
  user_systemctl is-active --quiet "$KIOSK_UNIT"
  if swapon --show --noheadings | grep -q .; then
    printf 'Swap became active while the measured planner was starting.\n' >&2
    return 70
  fi
}

monitor_runtime() {
  unit="$1"
  flag="$2"
  while [[ -e "$flag" ]]; do
    available="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    swap_used="$(awk '/SwapTotal/ {total=$2} /SwapFree/ {free=$2} END {print total-free}' /proc/meminfo)"
    properties="$(user_systemctl show "$unit" \
      --property=MemoryPeak --property=NRestarts --property=OOMKilled 2>/dev/null || true)"
    memory_peak="$(printf '%s\n' "$properties" | sed -n 's/^MemoryPeak=//p')"
    restarts="$(printf '%s\n' "$properties" | sed -n 's/^NRestarts=//p')"
    oom="$(printf '%s\n' "$properties" | sed -n 's/^OOMKilled=//p')"
    if [[ ! "$memory_peak" =~ ^[1-9][0-9]*$ ]] \
      || [[ ! "$restarts" =~ ^[0-9]+$ ]] \
      || [[ "$oom" != yes && "$oom" != no ]]; then
      printf 'Required systemd memory/restart/OOM telemetry is unavailable.\n' >&2
      return 70
    fi
    [[ "$oom" == yes ]] && oom=1 || oom=0
    external="$(ss -H -ltn 2>/dev/null | awk \
      '$4 ~ /:1143[56]$/ && $4 !~ /^127[.]0[.]0[.]1:/ && $4 !~ /^\[::1\]:/ {found=1} END {print found+0}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$available" "$memory_peak" "$swap_used" "$external" "$oom" "$restarts"
    sleep 0.5
  done
}

run_leg() {
  index="$1"
  label="$2"
  revision="$3"
  checksum="$4"
  activate_leg "$label"
  if [[ "$label" == "candidate" ]]; then
    engine_path="$CANDIDATE_DIR/engines/llm/llm.engine"
  else
    engine_path="$ACCEPTED_ENGINE"
  fi
  if [[ "$(sha256sum "$engine_path" | cut -d' ' -f1)" != "$checksum" ]]; then
    printf 'ABBA leg engine checksum changed before measurement.\n' >&2
    return 70
  fi
  benchmark="$WORK_DIR/leg-${index}-${label}-benchmark.json"
  monitor="$WORK_DIR/leg-${index}-${label}-monitor.tsv"
  leg="$WORK_DIR/leg-${index}-${label}.json"
  monitor_flag="$WORK_DIR/leg-${index}.monitoring"
  : >"$monitor_flag"
  monitor_runtime "$ACTIVE_UNIT" "$monitor_flag" >"$monitor" &
  monitor_pid=$!
  PYTHONPATH=/opt/bookforge/src "$BOOKFORGE_PYTHON" -m bookforge.planner_benchmark \
    --backend tensorrt_slots \
    --base-url http://127.0.0.1:11435 \
    --model llm \
    --model-revision "$revision" \
    --contract standard \
    --suite "$suite" \
    --context-tokens 4096 \
    --max-output-tokens 64 \
    --planner-timeout-seconds 12 \
    --model-timeout-seconds 12 \
    --output "$benchmark"
  projector_passed=0
  live_scene_passed=0
  if curl --fail --silent --max-time 3 http://127.0.0.1:8080/projector >/dev/null; then
    projector_passed=1
  fi
  live_response="$WORK_DIR/leg-${index}-${label}-live-scene.json"
  if curl --fail --silent --show-error --max-time 15 \
    -H 'content-type: application/json' \
    --data '{"text":"A child reads one bright word and paper fireflies form a path to the library.","visual_style":"luminous layered storybook","seed":20260901,"session_id":"fidelity-shadow"}' \
    http://127.0.0.1:8080/v1/live-scene-planner/prepare >"$live_response" \
    && python3 - "$live_response" "$revision" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
if document.get("revision") != sys.argv[2] or not 0 <= document.get("output_tokens", 65) <= 64:
    raise SystemExit(1)
PY
  then
    live_scene_passed=1
  fi
  unlink "$monitor_flag"
  wait "$monitor_pid"
  monitor_flag=""
  monitor_pid=""
  helper_args=(
    make-leg
    --benchmark "$benchmark"
    --monitor "$monitor"
    --label "$label"
    --order-index "$index"
    --model-revision "$revision"
    --engine-sha256 "$checksum"
    --planner-ready-seconds "$ACTIVE_READY_SECONDS"
    --output "$leg"
  )
  if user_systemctl is-active --quiet "$KIOSK_UNIT"; then helper_args+=(--kiosk-active); fi
  if ((projector_passed == 1)); then helper_args+=(--projector-http-passed); fi
  if ((live_scene_passed == 1)); then helper_args+=(--live-scene-passed); fi
  "$EVIDENCE_HELPER" "${helper_args[@]}"
  LEG_PATHS+=("$leg")
}

declare -a LEG_PATHS=()
readonly ACCEPTED_REVISION="sha256:${ACCEPTED_ENGINE_SHA256}"
run_leg 0 baseline "$ACCEPTED_REVISION" "$ACCEPTED_ENGINE_SHA256"
run_leg 1 candidate "$model_revision" "$engine_sha256"
run_leg 2 candidate "$model_revision" "$engine_sha256"
run_leg 3 baseline "$ACCEPTED_REVISION" "$ACCEPTED_ENGINE_SHA256"

restore_accepted
restored=1
"$EVIDENCE_HELPER" aggregate \
  --leg "${LEG_PATHS[0]}" \
  --leg "${LEG_PATHS[1]}" \
  --leg "${LEG_PATHS[2]}" \
  --leg "${LEG_PATHS[3]}" \
  --candidate-manifest "$CANDIDATE_DIR/candidate.manifest.json" \
  --candidate-manifest-sha256 "$manifest_sha256" \
  --accepted-engine-sha256 "$ACCEPTED_ENGINE_SHA256" \
  --hidden-report "$hidden_report" \
  --hidden-report-sha256 "$hidden_report_sha256" \
  --restoration "$RESTORATION" \
  --run-id "$run_id" \
  --config-sha256 "$config_sha256" \
  --dataset-manifest-sha256 "$dataset_manifest_sha256" \
  --int4-export-sha256 "$int4_export_sha256" \
  --runtime-output "$runtime_output" \
  --stage-output "$stage_output"
unlink "$ORIGINAL_CONFIG" "$CANDIDATE_CONFIG"
artifact_dir="${runtime_output}.artifacts"
if [[ -e "$artifact_dir" || -L "$artifact_dir" ]]; then
  printf 'Shadow artifact directory already exists: %s\n' "$artifact_dir" >&2
  exit 73
fi
mv "$WORK_DIR" "$artifact_dir"
sha256sum "$runtime_output" >"${runtime_output}.sha256"
sha256sum "$stage_output" >"${stage_output}.sha256"
trap - EXIT INT TERM
printf 'ABBA runtime evidence is ready at %s and stage evidence at %s; accepted engine %s was restored exactly.\n' \
  "$runtime_output" "$stage_output" "$ACCEPTED_ENGINE_SHA256"
