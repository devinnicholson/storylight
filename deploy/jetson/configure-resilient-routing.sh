#!/usr/bin/env bash
# Atomically enable GCP-first live-scene routing while preserving provider secrets.

set -euo pipefail

readonly CONFIG_FILE="/etc/bookforge/bookforge.env"
readonly PYTHON="/opt/bookforge/.venv/bin/python"
CLOUD_RUN_URL=""
TARGET_USER="${BOOKFORGE_SERVICE_USER:-${SUDO_USER:-}}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: sudo ./deploy/jetson/configure-resilient-routing.sh [options]

Options:
  --user USER          Select the non-root Bookforge service user.
  --cloud-run-url URL  Add the private Cloud Run RTX service ahead of Vertex.
  --dry-run            Validate and print the non-secret changes without writing.
  -h, --help           Show this help.

With no Cloud Run URL, the route is managed Vertex followed by authenticated
Modal. Existing Modal credentials and unrelated Bookforge settings are preserved.
EOF
}

while (($#)); do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || { printf '%s\n' '--user requires a value' >&2; exit 2; }
      TARGET_USER="$2"
      shift 2
      ;;
    --cloud-run-url)
      [[ $# -ge 2 ]] || { printf '%s\n' '--cloud-run-url requires a value' >&2; exit 2; }
      CLOUD_RUN_URL="${2%/}"
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
if [[ ! -f "$CONFIG_FILE" ]]; then
  printf 'Missing Bookforge environment: %s\n' "$CONFIG_FILE" >&2
  exit 1
fi
if [[ -L "$CONFIG_FILE" || "$(stat -c '%U:%G:%a' "$CONFIG_FILE")" != "root:root:600" ]]; then
  printf 'Bookforge environment has unsafe ownership, mode, or file type.\n' >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing Bookforge Python runtime: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! "$TARGET_USER" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || ! id "$TARGET_USER" >/dev/null 2>&1; then
  printf 'Select a valid Bookforge service user with --user.\n' >&2
  exit 2
fi
TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ -z "$TARGET_HOME" || ! -d "$TARGET_HOME" ]]; then
  printf 'Cannot resolve the home directory for %s.\n' "$TARGET_USER" >&2
  exit 2
fi
if [[ -n "$CLOUD_RUN_URL" && ! "$CLOUD_RUN_URL" =~ ^https://[^[:space:]]+$ ]]; then
  printf 'The Cloud Run URL must be empty or use HTTPS.\n' >&2
  exit 2
fi

"$PYTHON" -c 'import bookforge.provider_router, bookforge.vertex_scene_provider, google.auth'

"$PYTHON" - "$CONFIG_FILE" "$CLOUD_RUN_URL" "$DRY_RUN" <<'PY'
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

config_path = Path(sys.argv[1])
cloud_run_url = sys.argv[2]
dry_run = sys.argv[3] == "1"

replacements = {
    "BOOKFORGE_ASSET_MODAL_COMMAND": "/opt/bookforge/.venv/bin/modal",
    "BOOKFORGE_LIVE_SCENE_BACKEND": "gcp_resilient",
    "BOOKFORGE_LIVE_SCENE_VERTEX_PROJECT_ID": "your-gcp-project",
    "BOOKFORGE_LIVE_SCENE_VERTEX_LOCATION": "global",
    "BOOKFORGE_LIVE_SCENE_VERTEX_MODEL": "gemini-2.5-flash-image",
    "BOOKFORGE_LIVE_SCENE_VERTEX_TIMEOUT_SECONDS": "90",
    "BOOKFORGE_LIVE_SCENE_VERTEX_SESSION_COST_CAP_USD": "0.50",
    "BOOKFORGE_LIVE_SCENE_VERTEX_ESTIMATED_IMAGE_USD": "0.05",
    "BOOKFORGE_LIVE_SCENE_ROUTING_PROBE_TIMEOUT_SECONDS": "2",
    "BOOKFORGE_LIVE_SCENE_ROUTING_FAILURE_COOLDOWN_SECONDS": "300",
    "BOOKFORGE_LIVE_SCENE_ENABLE_PREVIEW": "false",
    "BOOKFORGE_LIVE_SCENE_ENABLE_MOTION": "false",
    "BOOKFORGE_LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT": "false",
    "BOOKFORGE_LIVE_SCENE_FIDELITY_MODE": "deferred",
}
if cloud_run_url:
    replacements.update(
        {
            "BOOKFORGE_LIVE_SCENE_GCP_URL": cloud_run_url,
            "BOOKFORGE_LIVE_SCENE_GCP_AUDIENCE": cloud_run_url,
            "BOOKFORGE_LIVE_SCENE_GCP_IMPERSONATE_SERVICE_ACCOUNT": (
                "bookforge-renderer@your-gcp-project.iam.gserviceaccount.com"
            ),
            "BOOKFORGE_LIVE_SCENE_GCP_GPU": "RTX_PRO_6000",
        }
    )

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

print("Bookforge resilient routing configuration:")
for name in sorted(replacements):
    print(f"  {name}={replacements[name]}")
if dry_run:
    print("Dry run complete; no files or services changed.")
    raise SystemExit(0)

timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
backup_path = config_path.with_name(f"{config_path.name}.before-resilient-{timestamp}")
shutil.copy2(config_path, backup_path)
os.chown(backup_path, 0, 0)
os.chmod(backup_path, 0o600)

descriptor, temporary_name = tempfile.mkstemp(
    dir=config_path.parent,
    prefix=".bookforge.env.resilient.",
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

# The hardened system service has an isolated HOME under CacheDirectory and
# cannot read the interactive user's ~/.config/gcloud. Copy only ADC, with
# private permissions, so google-auth works without exposing the rest of HOME.
source_adc="$TARGET_HOME/.config/gcloud/application_default_credentials.json"
service_gcloud_dir="/var/cache/bookforge/home/.config/gcloud"
if [[ -f "$source_adc" ]]; then
  install -d -o "$TARGET_UID" -g "$TARGET_GID" -m 0700 \
    /var/cache/bookforge/home \
    /var/cache/bookforge/home/.config \
    "$service_gcloud_dir"
  install -o "$TARGET_UID" -g "$TARGET_GID" -m 0600 \
    "$source_adc" "$service_gcloud_dir/application_default_credentials.json"
  printf 'Installed ADC into the private Bookforge service home.\n'
else
  printf 'ADC is not configured; the router will safely use Modal fallback.\n' >&2
fi

systemctl restart "bookforge@${TARGET_USER}.service" "bookforge-controller@${TARGET_USER}.service"
for _ in $(seq 1 30); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8080/readyz >/dev/null; then
    printf 'Bookforge resilient API is ready.\n'
    exit 0
  fi
  sleep 1
done
printf 'Bookforge API did not become ready after the routing switch.\n' >&2
exit 1
