#!/usr/bin/env bash
# Make Storylight discoverable and reconnectable without advertising USB/bridge addresses.

set -euo pipefail

DRY_RUN=0
CONNECTION_NAME=""

usage() {
  cat <<'EOF'
Usage: sudo ./deploy/jetson/configure-portable-network.sh [options]

Options:
  --connection NAME  Select the saved Wi-Fi connection (defaults to the active one).
  --dry-run          Print the resolved interface/connection without changing anything.

This keeps Wi-Fi autoconnect enabled, disables Wi-Fi power saving, enables mDNS on the
active connection, and limits Avahi advertisements to the Wi-Fi interface over IPv4.
It does not store or change Wi-Fi credentials, addresses, routes, or DNS servers.
EOF
}

while (($#)); do
  case "$1" in
    --connection)
      [[ $# -ge 2 ]] || { printf '%s\n' '--connection requires a value' >&2; exit 2; }
      CONNECTION_NAME="$2"
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
for tool in nmcli python3 systemctl; do
  command -v "$tool" >/dev/null 2>&1 || { printf 'Missing required tool: %s\n' "$tool" >&2; exit 1; }
done

WIFI_INTERFACE="$(nmcli -t -f DEVICE,TYPE,STATE device status \
  | awk -F: '$2 == "wifi" && $3 ~ /^connected/ {print $1; exit}')"
if [[ ! "$WIFI_INTERFACE" =~ ^[A-Za-z0-9_.:-]{1,32}$ ]]; then
  printf 'No active Wi-Fi interface was found.\n' >&2
  exit 1
fi
if [[ -z "$CONNECTION_NAME" ]]; then
  CONNECTION_NAME="$(nmcli -g GENERAL.CONNECTION device show "$WIFI_INTERFACE")"
fi
if [[ -z "$CONNECTION_NAME" || "$CONNECTION_NAME" == --* || "$CONNECTION_NAME" == *$'\n'* ]]; then
  printf 'The active Wi-Fi connection name is unsafe or empty.\n' >&2
  exit 1
fi

printf 'Storylight Wi-Fi interface: %s\n' "$WIFI_INTERFACE"
printf 'Storylight Wi-Fi connection: %s\n' "$CONNECTION_NAME"
if ((DRY_RUN == 1)); then
  printf 'Dry run complete; no settings changed.\n'
  exit 0
fi

nmcli connection modify "$CONNECTION_NAME" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  connection.autoconnect-retries 0 \
  connection.mdns 2 \
  802-11-wireless.powersave 2
nmcli device reapply "$WIFI_INTERFACE"

readonly AVAHI_CONFIG=/etc/avahi/avahi-daemon.conf
readonly AVAHI_BACKUP=/etc/avahi/avahi-daemon.conf.storylight-backup
if [[ ! -f "$AVAHI_CONFIG" ]]; then
  printf 'Avahi configuration is missing: %s\n' "$AVAHI_CONFIG" >&2
  exit 1
fi
if [[ ! -e "$AVAHI_BACKUP" ]]; then
  install -o root -g root -m 0644 "$AVAHI_CONFIG" "$AVAHI_BACKUP"
fi
python3 - "$AVAHI_CONFIG" "$WIFI_INTERFACE" <<'PY'
import os
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
interface = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
desired = {"allow-interfaces": interface, "use-ipv6": "no"}
rendered: list[str] = []
in_server = False
found_server = False
seen: set[str] = set()

def append_missing() -> None:
    for key, value in desired.items():
        if key not in seen:
            rendered.append(f"{key}={value}")
            seen.add(key)

for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if in_server:
            append_missing()
        in_server = stripped.casefold() == "[server]"
        found_server = found_server or in_server
        rendered.append(line)
        continue
    if in_server:
        candidate = stripped.lstrip("#").strip()
        key, separator, _ = candidate.partition("=")
        key = key.strip().casefold()
        if separator and key in desired:
            if key not in seen:
                rendered.append(f"{key}={desired[key]}")
                seen.add(key)
            continue
    rendered.append(line)
if in_server:
    append_missing()
if not found_server:
    rendered.extend(["", "[server]", *[f"{key}={value}" for key, value in desired.items()]])

mode = stat.S_IMODE(path.stat().st_mode)
descriptor, temporary_name = tempfile.mkstemp(prefix=".avahi-storylight.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(rendered) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, mode)
    os.replace(temporary_name, path)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY

systemctl enable ssh.service avahi-daemon.service >/dev/null
systemctl restart avahi-daemon.service

device_name="$(hostname -s)"
printf 'Portable discovery configured: %s.local advertises only %s over IPv4.\n' \
  "$device_name" "$WIFI_INTERFACE"
printf 'Backup: %s\n' "$AVAHI_BACKUP"
