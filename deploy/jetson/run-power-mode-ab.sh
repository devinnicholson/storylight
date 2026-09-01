#!/usr/bin/env bash
# Reboot-aware 25W versus MAXN_SUPER planner acceptance for the Orin Nano.

set -euo pipefail

readonly TARGET_USER="${BOOKFORGE_POWER_AB_USER:-operator}"
readonly TRUSTED_ROOT="/var/lib/bookforge-trusted"
readonly STATE_DIR="$TRUSTED_ROOT/power-mode-ab"
readonly EVIDENCE_DIR="$STATE_DIR/evidence"
readonly PHASE_FILE="$STATE_DIR/phase"
readonly BASELINE_SOURCE="/tmp/bookforge-inference-25w-power.json"
readonly BASELINE_TELEMETRY_SOURCE="/tmp/bookforge-25w-tegrastats.log"
readonly MAXN_REPORT="$EVIDENCE_DIR/bookforge-inference-maxn.json"
readonly MAXN_TELEMETRY="$EVIDENCE_DIR/bookforge-maxn-tegrastats.log"
readonly MODEL_REVISION="8648f39d-maxn-resident"
monitor_pid=""

usage() {
  cat <<'EOF'
Usage: sudo run-power-mode-ab.sh COMMAND

Commands:
  prepare-maxn    Preserve the 25W baseline, request MAXN_SUPER, then confirm its reboot.
  benchmark-maxn  After reconnecting, validate MAXN_SUPER and run the fixed five-case benchmark.
  restore-25w     Request the accepted 25W mode, then verify it (reboot only if NVIDIA asks).
  finalize        Validate the restored 25W mode and close the persistent evidence record.
  status          Print the current phase and power mode without changing anything.

An nvpmodel command may ask whether to reboot. Enter YES only after the script has printed the
matching durable phase marker. On the measured JetPack 7.2.1 device, entering MAXN_SUPER required a
reboot while returning to 25W applied immediately. Never benchmark or finalize until the requested
mode is observable with nvpmodel.
EOF
}

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run this power-mode acceptance with sudo.\n' >&2
  exit 64
fi
if ! id "$TARGET_USER" >/dev/null 2>&1; then
  printf 'Bookforge user does not exist: %s\n' "$TARGET_USER" >&2
  exit 65
fi
readonly TARGET_GROUP="$(id -gn "$TARGET_USER")"
if [[ ! -r /etc/nv_tegra_release ]] \
  || ! grep -q '^# R39 (release), REVISION: 2.1' /etc/nv_tegra_release; then
  printf 'This runner is pinned to JetPack 7.2.1 / L4T 39.2.1.\n' >&2
  exit 65
fi
for required in nvpmodel curl tegrastats runuser sha256sum pgrep; do
  if ! command -v "$required" >/dev/null 2>&1; then
    printf 'Required command is missing: %s\n' "$required" >&2
    exit 69
  fi
done

ensure_state_layout() {
  # The benchmark runs as the unprivileged Bookforge user. Keep the phase file root-owned while
  # allowing that user to traverse the parent and write only inside the evidence directory.
  if [[ ! -d /var/lib || -L /var/lib ]] \
    || [[ "$(stat -c '%U:%G:%a' /var/lib)" != "root:root:755" ]]; then
    printf 'The /var/lib trust anchor is unsafe.\n' >&2
    exit 78
  fi
  if [[ -e "$TRUSTED_ROOT" || -L "$TRUSTED_ROOT" ]]; then
    if [[ ! -d "$TRUSTED_ROOT" || -L "$TRUSTED_ROOT" ]] \
      || [[ "$(stat -c '%U:%G:%a' "$TRUSTED_ROOT")" != "root:root:755" ]]; then
      printf 'The Bookforge trusted state root is unsafe.\n' >&2
      exit 78
    fi
  else
    install -d -o root -g root -m 0755 "$TRUSTED_ROOT"
  fi
  if [[ -e "$STATE_DIR" || -L "$STATE_DIR" ]]; then
    if [[ ! -d "$STATE_DIR" || -L "$STATE_DIR" ]] \
      || [[ "$(stat -c '%U:%G:%a' "$STATE_DIR")" != "root:${TARGET_GROUP}:750" ]]; then
      printf 'The power acceptance state directory is unsafe.\n' >&2
      exit 78
    fi
  else
    install -d -o root -g "$TARGET_GROUP" -m 0750 "$STATE_DIR"
  fi
  if [[ -e "$EVIDENCE_DIR" || -L "$EVIDENCE_DIR" ]]; then
    if [[ ! -d "$EVIDENCE_DIR" || -L "$EVIDENCE_DIR" ]] \
      || [[ "$(stat -c '%U:%G:%a' "$EVIDENCE_DIR")" != "${TARGET_USER}:${TARGET_GROUP}:750" ]]; then
      printf 'The power acceptance evidence directory is unsafe.\n' >&2
      exit 78
    fi
  else
    install -d -o "$TARGET_USER" -g "$TARGET_GROUP" -m 0750 "$EVIDENCE_DIR"
  fi
}

cleanup_monitor() {
  if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
}

cleanup_stale_monitor() {
  local stale_pid
  local stale_pattern
  stale_pattern="^(tegrastats|/usr/bin/tegrastats) --interval 500 --logfile ${MAXN_TELEMETRY}$"
  while IFS= read -r stale_pid; do
    [[ -n "$stale_pid" ]] || continue
    printf 'Stopping stale Bookforge telemetry monitor PID %s.\n' "$stale_pid"
    kill "$stale_pid"
    for _ in $(seq 1 20); do
      kill -0 "$stale_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$stale_pid" 2>/dev/null; then
      printf 'Stale telemetry monitor PID %s did not stop; refusing to benchmark.\n' \
        "$stale_pid" >&2
      exit 71
    fi
  done < <(pgrep -f -- "$stale_pattern" || true)
}

power_mode_id() {
  nvpmodel -q 2>/dev/null | awk '/^[0-9]+$/{mode=$1} END{print mode}'
}

power_mode_label() {
  nvpmodel -q 2>/dev/null | sed -n '1p'
}

phase() {
  if [[ -r "$PHASE_FILE" ]]; then
    tr -d '\n' <"$PHASE_FILE"
  else
    printf 'not-started'
  fi
}

write_phase() {
  local next_phase="$1"
  local temporary="$STATE_DIR/.phase.$$"
  printf '%s\n' "$next_phase" >"$temporary"
  chmod 0600 "$temporary"
  mv -f "$temporary" "$PHASE_FILE"
  sync "$PHASE_FILE"
}

require_mode() {
  local expected="$1"
  local actual
  actual="$(power_mode_id)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Expected power mode %s, found %s (%s). Refusing.\n' \
      "$expected" "${actual:-unknown}" "$(power_mode_label)" >&2
    exit 70
  fi
}

require_phase() {
  local expected="$1"
  local actual
  actual="$(phase)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Expected phase %s, found %s. Refusing.\n' "$expected" "$actual" >&2
    exit 70
  fi
}

wait_for_runtime() {
  local attempt
  for attempt in $(seq 1 60); do
    if curl --fail --silent --max-time 1 http://127.0.0.1:8080/readyz >/dev/null \
      && curl --fail --silent --max-time 1 http://127.0.0.1:11434/api/version >/dev/null; then
      return
    fi
    sleep 1
  done
  printf 'Bookforge API or local model runtime did not become ready.\n' >&2
  exit 71
}

prepare_maxn() {
  require_mode 1
  case "$(phase)" in
    not-started|cancelled|complete) ;;
    *)
      printf 'A power-mode acceptance is already active at phase %s.\n' "$(phase)" >&2
      exit 70
      ;;
  esac
  if [[ ! -s "$BASELINE_SOURCE" || ! -s "$BASELINE_TELEMETRY_SOURCE" ]]; then
    printf 'The accepted 25W baseline or telemetry is missing from /tmp; recapture it first.\n' >&2
    exit 69
  fi

  ensure_state_layout
  install -o "$TARGET_USER" -g "$TARGET_GROUP" -m 0640 \
    "$BASELINE_SOURCE" "$EVIDENCE_DIR/bookforge-inference-25w.json"
  install -o "$TARGET_USER" -g "$TARGET_GROUP" -m 0640 \
    "$BASELINE_TELEMETRY_SOURCE" "$EVIDENCE_DIR/bookforge-25w-tegrastats.log"
  sha256sum \
    "$EVIDENCE_DIR/bookforge-inference-25w.json" \
    "$EVIDENCE_DIR/bookforge-25w-tegrastats.log" \
    >"$EVIDENCE_DIR/sha256sums-25w.txt"
  chown "$TARGET_USER:$TARGET_GROUP" "$EVIDENCE_DIR/sha256sums-25w.txt"
  chmod 0640 "$EVIDENCE_DIR/sha256sums-25w.txt"

  write_phase awaiting-maxn-reboot
  printf 'Durable phase: awaiting-maxn-reboot\n'
  printf 'Requesting MAXN_SUPER. When nvpmodel asks, enter YES to perform the first reboot.\n'
  if ! nvpmodel -m 2; then
    write_phase cancelled
    printf 'MAXN_SUPER was not requested; phase reset to cancelled.\n' >&2
    exit 1
  fi
  printf 'If the board did not reboot, stop and inspect nvpmodel before continuing.\n'
}

benchmark_maxn() {
  require_phase awaiting-maxn-reboot
  require_mode 2
  wait_for_runtime
  ensure_state_layout
  cleanup_stale_monitor

  rm -f "$MAXN_REPORT" "$MAXN_TELEMETRY"
  tegrastats --interval 500 --logfile "$MAXN_TELEMETRY" &
  monitor_pid=$!
  trap cleanup_monitor EXIT INT TERM

  runuser -u "$TARGET_USER" -- env PYTHONPATH=/opt/bookforge/src \
    /opt/bookforge/.venv/bin/python -m bookforge.planner_benchmark \
    --contract standard \
    --max-output-tokens 320 \
    --keep-alive=-1m \
    --model-revision "$MODEL_REVISION" \
    --output "$MAXN_REPORT" >/dev/null

  cleanup_monitor
  trap - EXIT INT TERM
  chown "$TARGET_USER:$TARGET_GROUP" "$MAXN_REPORT" "$MAXN_TELEMETRY"
  chmod 0640 "$MAXN_REPORT" "$MAXN_TELEMETRY"

  python3 - "$MAXN_REPORT" "$MAXN_TELEMETRY" \
    "$EVIDENCE_DIR/bookforge-maxn-summary.json" <<'PY'
import json
import pathlib
import re
import statistics
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
telemetry = pathlib.Path(sys.argv[2]).read_text()
power = [int(value) for value in re.findall(r"VDD_IN (\d+)mW", telemetry)]
gpu = [int(value) for value in re.findall(r"GR3D_FREQ (\d+)%", telemetry)]
temperature = [float(value) for value in re.findall(r"gpu@([0-9.]+)C", telemetry)]
if not power or not gpu or not temperature:
    raise SystemExit("tegrastats evidence is incomplete")
summary = {
    "schema_version": "1.0",
    "mode": "MAXN_SUPER",
    "mode_id": 2,
    "planner": report["contracts"][0]["summary"],
    "telemetry_samples": len(power),
    "power_mean_w": round(statistics.fmean(power) / 1000, 3),
    "power_max_w": round(max(power) / 1000, 3),
    "gpu_mean_percent": round(statistics.fmean(gpu), 3),
    "gpu_max_percent": max(gpu),
    "gpu_temp_max_c": max(temperature),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
  chown "$TARGET_USER:$TARGET_GROUP" "$EVIDENCE_DIR/bookforge-maxn-summary.json"
  chmod 0640 "$EVIDENCE_DIR/bookforge-maxn-summary.json"
  sha256sum "$MAXN_REPORT" "$MAXN_TELEMETRY" \
    "$EVIDENCE_DIR/bookforge-maxn-summary.json" \
    >"$EVIDENCE_DIR/sha256sums-maxn.txt"
  chown "$TARGET_USER:$TARGET_GROUP" "$EVIDENCE_DIR/sha256sums-maxn.txt"
  chmod 0640 "$EVIDENCE_DIR/sha256sums-maxn.txt"
  write_phase maxn-benchmarked
  printf 'MAXN benchmark is durable. Next run: sudo %s restore-25w\n' "$0"
}

restore_25w() {
  require_phase maxn-benchmarked
  require_mode 2
  write_phase awaiting-25w-verification
  printf 'Durable phase: awaiting-25w-verification\n'
  printf 'Requesting 25W. If nvpmodel asks to reboot, enter YES; otherwise verify immediately.\n'
  if ! nvpmodel -m 1; then
    write_phase restore-cancelled
    printf '25W restore was not requested; phase set to restore-cancelled.\n' >&2
    exit 1
  fi
  printf 'Run finalize now if mode 1 is active, or after reconnecting if a reboot occurred.\n'
}

finalize() {
  case "$(phase)" in
    awaiting-25w-verification|awaiting-25w-reboot|restore-cancelled) ;;
    *)
      printf 'Expected a pending 25W restore, found %s. Refusing.\n' "$(phase)" >&2
      exit 70
      ;;
  esac
  require_mode 1
  wait_for_runtime
  {
    printf '{\n'
    printf '  "schema_version": "1.0",\n'
    printf '  "result": "complete",\n'
    printf '  "restored_mode": "25W",\n'
    printf '  "restored_mode_id": 1,\n'
    printf '  "bookforge_ready": true,\n'
    printf '  "gemma_runtime_ready": true\n'
    printf '}\n'
  } >"$EVIDENCE_DIR/final-state.json"
  chown "$TARGET_USER:$TARGET_GROUP" "$EVIDENCE_DIR/final-state.json"
  chmod 0640 "$EVIDENCE_DIR/final-state.json"
  write_phase complete
  printf 'Power-mode A/B complete; 25W is restored. Evidence: %s\n' "$EVIDENCE_DIR"
}

case "${1:-}" in
  prepare-maxn) prepare_maxn ;;
  benchmark-maxn) benchmark_maxn ;;
  restore-25w) restore_25w ;;
  finalize) finalize ;;
  status)
    printf 'phase=%s\n' "$(phase)"
    printf 'mode=%s\n' "$(power_mode_label)"
    printf 'mode_id=%s\n' "$(power_mode_id)"
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
