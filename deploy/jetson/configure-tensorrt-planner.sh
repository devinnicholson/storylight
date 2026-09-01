#!/usr/bin/env bash
# Atomically promote the accepted resident TensorRT planner with Ollama rollback.

set -euo pipefail

readonly CONFIG_FILE="/etc/bookforge/bookforge.env"
readonly PYTHON="/opt/bookforge/.venv/bin/python"
readonly UNIT_SOURCE="/opt/bookforge/deploy/jetson/systemd/bookforge-tensorrt-planner.service"
TARGET_USER="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-}}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: sudo ./deploy/jetson/configure-tensorrt-planner.sh [options]

Options:
  --user USER  Select the non-root Bookforge service user.
  --dry-run    Validate the engine and print non-secret changes without writing.
  -h, --help   Show this help.

Promotes the hardware-accepted Gemma 4 TensorRT slot planner on loopback. The
smaller Gemma 3 Ollama service is retained as the automatic crash fallback.
EOF
}

while (($#)); do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || { printf '%s\n' '--user requires a value' >&2; exit 2; }
      TARGET_USER="$2"
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  printf 'Run this configuration helper with sudo.\n' >&2
  exit 1
fi
if [[ ! "$TARGET_USER" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || ! id "$TARGET_USER" >/dev/null 2>&1; then
  printf 'Select a valid Bookforge service user with --user.\n' >&2
  exit 2
fi
if [[ ! -f "$CONFIG_FILE" || -L "$CONFIG_FILE" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_FILE")" != "root:root:600" ]]; then
  printf 'Bookforge environment is missing or has unsafe ownership, mode, or type.\n' >&2
  exit 1
fi
if [[ ! -x "$PYTHON" || ! -r "$UNIT_SOURCE" ]]; then
  printf 'The staged Bookforge runtime or TensorRT unit is missing.\n' >&2
  exit 1
fi

TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ -z "$TARGET_HOME" || ! -d "$TARGET_HOME" ]]; then
  printf 'Cannot resolve the home directory for %s.\n' "$TARGET_USER" >&2
  exit 2
fi

user_systemctl() {
  runuser -u "$TARGET_USER" -- env \
    XDG_RUNTIME_DIR="/run/user/${TARGET_UID}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${TARGET_UID}/bus" \
    systemctl --user "$@"
}

readonly ENGINE_DIR="$TARGET_HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0/models/gemma4-e2b-it-int4-awq-v010/engines/llm"
readonly ENGINE_FILE="$ENGINE_DIR/llm.engine"
if [[ ! -s "$ENGINE_FILE" ]]; then
  printf 'The accepted Gemma 4 TensorRT engine is missing: %s\n' "$ENGINE_FILE" >&2
  exit 1
fi
ENGINE_SHA256="$(sha256sum "$ENGINE_FILE" | cut -d' ' -f1)"
if [[ ! "$ENGINE_SHA256" =~ ^[a-f0-9]{64}$ ]]; then
  printf 'Could not fingerprint the TensorRT engine.\n' >&2
  exit 1
fi

"$PYTHON" -c 'import bookforge.api, bookforge.tensorrt_slot_client'
readonly REVISION="sha256:${ENGINE_SHA256}"
readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly BACKUP_FILE="${CONFIG_FILE}.before-tensorrt-${TIMESTAMP}"

"$PYTHON" - "$CONFIG_FILE" "$REVISION" "$DRY_RUN" "$BACKUP_FILE" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

config_path = Path(sys.argv[1])
revision = sys.argv[2]
dry_run = sys.argv[3] == "1"
backup_path = Path(sys.argv[4])
replacements = {
    "BOOKFORGE_LIVE_SCENE_PLANNER": "model",
    "BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND": "tensorrt_slots",
    "BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL": "http://127.0.0.1:11435",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_NAME": "llm",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MAX_OUTPUT_TOKENS": "64",
    "BOOKFORGE_LIVE_SCENE_PLANNER_FALLBACK_READY_SECONDS": "5",
    "BOOKFORGE_LIVE_SCENE_PLANNER_TIMEOUT_SECONDS": "12",
    "BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION": revision,
    "BOOKFORGE_LIVE_SCENE_PLANNER_COMPACT_WIRE": "false",
}
lines = config_path.read_text().splitlines()
remaining = dict(replacements)
updated: list[str] = []
for line in lines:
    name, separator, _ = line.partition("=")
    if separator and name in remaining:
        updated.append(f"{name}={remaining.pop(name)}")
    else:
        updated.append(line)
for name, value in remaining.items():
    updated.append(f"{name}={value}")

print("Bookforge TensorRT planner configuration:")
for name in sorted(replacements):
    print(f"  {name}={replacements[name]}")
if dry_run:
    print("Dry run complete; no files or services changed.")
    raise SystemExit(0)

shutil.copy2(config_path, backup_path)
os.chown(backup_path, 0, 0)
os.chmod(backup_path, 0o600)
descriptor, temporary_name = tempfile.mkstemp(
    dir=config_path.parent,
    prefix=".bookforge.env.tensorrt.",
)
try:
    with os.fdopen(descriptor, "w") as stream:
        stream.write("\n".join(updated) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chown(temporary_name, 0, 0)
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, config_path)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
print(f"Rollback copy: {backup_path}")
PY

if ((DRY_RUN == 1)); then
  exit 0
fi

configured=1
kiosk_was_active=0
rollback() {
  exit_code=$?
  trap - EXIT
  if ((exit_code != 0 && configured == 1)); then
    printf 'TensorRT promotion failed; restoring the previous configuration.\n' >&2
    install -o root -g root -m 0600 "$BACKUP_FILE" "$CONFIG_FILE"
    tensorrt_want="$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-tensorrt-planner.service"
    if [[ -L "$tensorrt_want" ]]; then
      unlink "$tensorrt_want"
    fi
    user_systemctl stop bookforge-tensorrt-planner.service || true
    user_systemctl start bookforge-gemma.service || true
    if ((kiosk_was_active == 1)); then
      user_systemctl start bookforge-kiosk.service || true
    fi
    ln -sfn ../bookforge-gemma.service \
      "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-gemma.service"
    chown -h "$TARGET_UID:$TARGET_GID" \
      "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-gemma.service"
    systemctl restart "bookforge@${TARGET_USER}.service" || true
  fi
  exit "$exit_code"
}
trap rollback EXIT

install -d -o "$TARGET_UID" -g "$TARGET_GID" -m 0700 \
  "$TARGET_HOME/.config/systemd/user" \
  "$TARGET_HOME/.config/systemd/user/default.target.wants"
install -o "$TARGET_UID" -g "$TARGET_GID" -m 0644 "$UNIT_SOURCE" \
  "$TARGET_HOME/.config/systemd/user/bookforge-tensorrt-planner.service"
ln -sfn ../bookforge-tensorrt-planner.service \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-tensorrt-planner.service"
chown -h "$TARGET_UID:$TARGET_GID" \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-tensorrt-planner.service"
gemma_want="$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-gemma.service"
if [[ -L "$gemma_want" ]] \
  && [[ "$(readlink "$gemma_want")" == "../bookforge-gemma.service" ]]; then
  unlink "$gemma_want"
fi

user_systemctl daemon-reload
if user_systemctl is-active --quiet bookforge-kiosk.service; then
  kiosk_was_active=1
fi

# The long-lived kiosk and a loaded Ollama worker each held roughly 1.7 GiB on
# the accepted device. Drain both before CUDA-graph capture, then relaunch a
# fresh kiosk only after TensorRT and the API are ready.
user_systemctl stop bookforge-kiosk.service
user_systemctl stop bookforge-gemma.service
for _ in $(seq 1 30); do
  if ! pgrep -u "$TARGET_UID" -f \
    'ollama serve|llama-server|firefox-bookforge' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if pgrep -u "$TARGET_UID" -f \
  'ollama serve|llama-server|firefox-bookforge' >/dev/null 2>&1; then
  printf 'Ollama or the projector browser did not drain before TensorRT startup.\n' >&2
  exit 1
fi
available_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || ((available_kib < 4194304)); then
  printf 'TensorRT requires at least 4 GiB available before CUDA-graph capture.\n' >&2
  exit 1
fi

user_systemctl reset-failed bookforge-tensorrt-planner.service || true
user_systemctl start bookforge-tensorrt-planner.service

for _ in $(seq 1 90); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:11435/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:11435/v1/models >/dev/null

systemctl restart "bookforge@${TARGET_USER}.service" "bookforge-controller@${TARGET_USER}.service"
for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8080/readyz >/dev/null 2>&1 \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8081/healthz >/dev/null 2>&1; then
    if ((kiosk_was_active == 1)); then
      user_systemctl start bookforge-kiosk.service
      user_systemctl is-active --quiet bookforge-kiosk.service
    fi
    configured=0
    printf 'Bookforge TensorRT planner is resident; API and controller are ready.\n'
    exit 0
  fi
  sleep 1
done
printf 'Bookforge API did not become ready after TensorRT promotion.\n' >&2
exit 1
