#!/usr/bin/env bash
# Print the current one-time controller pairing URL from the root-owned secret.

set -euo pipefail

readonly CONTROLLER_ENV="${BOOKFORGE_CONTROLLER_ENV:-/etc/bookforge/controller.env}"

if [[ ! -r "$CONTROLLER_ENV" ]]; then
  printf 'Cannot read %s. Run with sudo on the Jetson.\n' "$CONTROLLER_ENV" >&2
  exit 1
fi

pairing_token="$(sed -n 's/^BOOKFORGE_CONTROLLER_PAIRING_TOKEN=//p' "$CONTROLLER_ENV")"
if [[ ! "$pairing_token" =~ ^[A-Za-z0-9_-]{32,256}$ ]]; then
  printf 'The controller pairing token is missing or malformed.\n' >&2
  exit 1
fi

device_host="${BOOKFORGE_CONTROLLER_HOST:-}"
if [[ -z "$device_host" ]]; then
  device_name="$(hostname -s)"
  if [[ ! "$device_name" =~ ^[A-Za-z0-9-]{1,63}$ ]]; then
    printf 'Unsafe device hostname: %s\n' "$device_name" >&2
    exit 1
  fi
  device_host="${device_name}.local"
fi
if [[ ! "$device_host" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$ \
  || "$device_host" == *..* ]]; then
  printf 'Unsafe controller host: %s\n' "$device_host" >&2
  exit 1
fi

pairing_url="http://${device_host}:8081/pair#${pairing_token}"
printf 'Pair only while the phone and Jetson are on your private Wi-Fi.\n'
printf '%s\n' "$pairing_url"

if command -v qrencode >/dev/null 2>&1; then
  qrencode -t ANSIUTF8 "$pairing_url"
fi
