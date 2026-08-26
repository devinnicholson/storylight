#!/usr/bin/env bash
# Install the already-staged Bookforge checkout as a standalone Jetson service.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET_USER="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-${USER:-}}}"
DRY_RUN=0
DEPLOY_RENDERER=0
IMPORT_MODAL_PROFILE=0

usage() {
  cat <<'EOF'
Usage: sudo ./deploy/jetson/install-standalone.sh [options]

Options:
  --user USER             Select the non-root Bookforge service user.
  --import-modal-profile  Copy the active user's Modal token into the root-only service env.
  --deploy-renderer       Update the authenticated, scale-to-zero Modal app definition.
  --dry-run               Validate the staged runtime without changing files or services.

Installs Bookforge's API and paired phone gateway services from /opt/bookforge.
It never changes Wi-Fi, JetPack, power mode, storage, display, or login settings.
Existing environment files and pairing secrets are never overwritten.
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
    --import-modal-profile) IMPORT_MODAL_PROFILE=1; shift ;;
    --deploy-renderer) DEPLOY_RENDERER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  printf 'Run this installer with sudo.\n' >&2
  exit 1
fi
if [[ "$(uname -m)" != "aarch64" || ! -r /etc/nv_tegra_release ]]; then
  printf 'Refusing installation: this is not an aarch64 Jetson.\n' >&2
  exit 1
fi
if ! id "$TARGET_USER" >/dev/null 2>&1; then
  printf 'Unknown service user: %s\n' "$TARGET_USER" >&2
  exit 1
fi
TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ -z "$TARGET_HOME" || ! -d "$TARGET_HOME" ]]; then
  printf 'Cannot resolve the home directory for %s.\n' "$TARGET_USER" >&2
  exit 1
fi
if [[ "$REPO_ROOT" != "/opt/bookforge" ]]; then
  printf 'Run the staged installer at /opt/bookforge/deploy/jetson/install-standalone.sh.\n' >&2
  exit 1
fi

readonly PYTHON=/opt/bookforge/.venv/bin/python
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing Bookforge virtual environment: %s\n' "$PYTHON" >&2
  exit 1
fi
"$PYTHON" -c 'import bookforge.api, bookforge.controller_gateway, modal'

if ((DRY_RUN == 1)); then
  printf 'Standalone preflight passed for user %s; no files or services changed.\n' "$TARGET_USER"
  exit 0
fi

install -d -o root -g root -m 0755 /etc/bookforge
if [[ ! -e /etc/bookforge/bookforge.env ]]; then
  install -o root -g root -m 0600 \
    "$SCRIPT_DIR/bookforge.standalone.env.example" /etc/bookforge/bookforge.env
  printf 'Created /etc/bookforge/bookforge.env; add Modal credentials before starting.\n'
fi

if [[ ! -e /etc/bookforge/controller.env ]]; then
  pairing_token="$(openssl rand -hex 32)"
  if [[ ! "$pairing_token" =~ ^[a-f0-9]{64}$ ]]; then
    printf 'Pairing-token generation failed.\n' >&2
    exit 1
  fi
  install -o root -g root -m 0600 /dev/null /etc/bookforge/controller.env
  {
    printf 'BOOKFORGE_CONTROLLER_BACKEND_URL=http://127.0.0.1:8080\n'
    printf 'BOOKFORGE_CONTROLLER_PAIRING_TOKEN=%s\n' "$pairing_token"
    printf 'BOOKFORGE_CONTROLLER_SESSION_TTL_SECONDS=43200\n'
    printf 'BOOKFORGE_CONTROLLER_MAXIMUM_SESSIONS=8\n'
    printf 'BOOKFORGE_CONTROLLER_COOKIE_SECURE=false\n'
    printf 'BOOKFORGE_CONTROLLER_PUBLIC_NAME=Bookforge\n'
  } > /etc/bookforge/controller.env
fi

if ((IMPORT_MODAL_PROFILE == 1)); then
  modal_profile="$TARGET_HOME/.modal.toml"
  if [[ ! -r "$modal_profile" ]]; then
    printf 'Cannot read the Modal profile at %s.\n' "$modal_profile" >&2
    exit 1
  fi
  "$PYTHON" - "$modal_profile" /etc/bookforge/bookforge.env <<'PY'
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

profile_path = Path(sys.argv[1])
environment_path = Path(sys.argv[2])
with profile_path.open("rb") as stream:
    configuration = tomllib.load(stream)
active = [value for value in configuration.values() if isinstance(value, dict) and value.get("active")]
if len(active) != 1:
    raise SystemExit("Expected exactly one active Modal profile")
token_id = active[0].get("token_id", "")
token_secret = active[0].get("token_secret", "")
token_pattern = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
if not token_pattern.fullmatch(token_id) or not token_pattern.fullmatch(token_secret):
    raise SystemExit("The active Modal profile contains malformed credentials")

lines = environment_path.read_text().splitlines()
replacements = {
    "MODAL_TOKEN_ID": token_id,
    "MODAL_TOKEN_SECRET": token_secret,
}
found = set()
for index, line in enumerate(lines):
    name, separator, _ = line.partition("=")
    if separator and name in replacements:
        lines[index] = f"{name}={replacements[name]}"
        found.add(name)
if found != replacements.keys():
    raise SystemExit("The Bookforge environment lacks Modal credential placeholders")

descriptor, temporary_name = tempfile.mkstemp(
    dir=environment_path.parent,
    prefix=".bookforge.env.",
)
try:
    with os.fdopen(descriptor, "w") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, environment_path)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY
fi

install -o root -g root -m 0644 \
  "$SCRIPT_DIR/systemd/bookforge@.service" /etc/systemd/system/bookforge@.service
install -o root -g root -m 0644 \
  "$SCRIPT_DIR/systemd/bookforge-controller@.service" \
  /etc/systemd/system/bookforge-controller@.service

# These user services start with the normal graphical login. The installer does
# not enable autologin or linger, and therefore does not weaken login security.
install -d -o "$TARGET_UID" -g "$TARGET_GID" -m 0700 \
  "$TARGET_HOME/.config/systemd/user" "$TARGET_HOME/.config/bookforge"
install -o "$TARGET_UID" -g "$TARGET_GID" -m 0644 \
  "$SCRIPT_DIR/systemd/bookforge-gemma.service" \
  "$TARGET_HOME/.config/systemd/user/bookforge-gemma.service"
install -o "$TARGET_UID" -g "$TARGET_GID" -m 0644 \
  "$SCRIPT_DIR/systemd/bookforge-kiosk.service" \
  "$TARGET_HOME/.config/systemd/user/bookforge-kiosk.service"
if [[ ! -e "$TARGET_HOME/.config/bookforge/kiosk.env" ]]; then
  install -o "$TARGET_UID" -g "$TARGET_GID" -m 0600 \
    "$SCRIPT_DIR/kiosk.env.example" "$TARGET_HOME/.config/bookforge/kiosk.env"
fi
install -d -o "$TARGET_UID" -g "$TARGET_GID" -m 0700 \
  "$TARGET_HOME/.config/systemd/user/default.target.wants"
ln -sfn ../bookforge-gemma.service \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-gemma.service"
ln -sfn ../bookforge-kiosk.service \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-kiosk.service"
chown -h "$TARGET_UID:$TARGET_GID" \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-gemma.service" \
  "$TARGET_HOME/.config/systemd/user/default.target.wants/bookforge-kiosk.service"

systemctl daemon-reload
systemctl enable "bookforge@${TARGET_USER}.service"
systemctl enable "bookforge-controller@${TARGET_USER}.service"

modal_token_id="$(sed -n 's/^MODAL_TOKEN_ID=//p' /etc/bookforge/bookforge.env)"
modal_token_secret="$(sed -n 's/^MODAL_TOKEN_SECRET=//p' /etc/bookforge/bookforge.env)"
if [[ ${#modal_token_id} -lt 8 || ${#modal_token_secret} -lt 8 ]]; then
  printf 'Services enabled but not started: add Modal credentials to ' >&2
  printf '/etc/bookforge/bookforge.env, then start both units.\n' >&2
  exit 3
fi

if ((DEPLOY_RENDERER == 1)); then
  env MODAL_TOKEN_ID="$modal_token_id" MODAL_TOKEN_SECRET="$modal_token_secret" \
    /opt/bookforge/.venv/bin/modal deploy /opt/bookforge/deploy/modal_fast_scene.py
fi

user_runtime_dir="/run/user/${TARGET_UID}"
if [[ ! -S "$user_runtime_dir/bus" ]]; then
  printf 'Services are enabled, but %s has no active graphical login.\n' "$TARGET_USER" >&2
  printf 'Sign in on the projector, then rerun this installer.\n' >&2
  exit 4
fi
runuser -u "$TARGET_USER" -- env \
  XDG_RUNTIME_DIR="$user_runtime_dir" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=${user_runtime_dir}/bus" \
  systemctl --user daemon-reload
runuser -u "$TARGET_USER" -- env \
  XDG_RUNTIME_DIR="$user_runtime_dir" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=${user_runtime_dir}/bus" \
  systemctl --user restart bookforge-gemma.service

gemma_ready=0
for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    gemma_ready=1
    break
  fi
  sleep 1
done
if ((gemma_ready == 0)); then
  printf 'Local Gemma did not become ready within 30 seconds.\n' >&2
  exit 1
fi

systemctl restart "bookforge@${TARGET_USER}.service"
systemctl restart "bookforge-controller@${TARGET_USER}.service"

standalone_ready=0
for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8080/readyz >/dev/null 2>&1 \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8081/healthz >/dev/null 2>&1; then
    standalone_ready=1
    break
  fi
  sleep 1
done
if ((standalone_ready == 0)); then
  printf 'Bookforge API and controller did not become ready within 30 seconds.\n' >&2
  systemctl --no-pager --full status \
    "bookforge@${TARGET_USER}.service" \
    "bookforge-controller@${TARGET_USER}.service" >&2 || true
  exit 1
fi

printf 'Standalone Bookforge is ready. Show the private pairing URL with:\n'
printf '  sudo /opt/bookforge/deploy/jetson/show-controller-pairing.sh\n'
printf 'Gemma and the projector start after %s signs in graphically.\n' "$TARGET_USER"
printf 'Start or refresh the projector now with: systemctl --user restart bookforge-kiosk.service\n'
printf 'For stable cable-free discovery, run once: sudo /opt/bookforge/deploy/jetson/configure-portable-network.sh\n'
