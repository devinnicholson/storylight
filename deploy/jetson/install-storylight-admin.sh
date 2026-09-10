#!/usr/bin/env bash
# Install the narrow, root-owned Storylight administrator and its exact sudo delegation.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

admin_user="${SUDO_USER:-operator}"

usage() {
  printf 'Usage: sudo install-storylight-admin.sh [--user USER]\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      [[ $# -ge 2 ]] || { usage >&2; exit 64; }
      admin_user="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 64
      ;;
  esac
done

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run this installer with sudo.\n' >&2
  exit 64
fi
if [[ ! "$admin_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || ! id "$admin_user" >/dev/null 2>&1; then
  printf 'Invalid Storylight administrator user: %s\n' "$admin_user" >&2
  exit 65
fi
if ! command -v visudo >/dev/null 2>&1; then
  printf 'visudo is required to validate the restricted delegation.\n' >&2
  exit 69
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ADMIN_SOURCE="$SCRIPT_DIR/storylight-admin"
readonly POWER_SOURCE="$SCRIPT_DIR/run-power-mode-ab.sh"
readonly SWAP_SOURCE="$SCRIPT_DIR/run-tensorrt-build-swap.sh"
readonly ADMIN_TARGET="/usr/local/sbin/storylight-admin"
readonly POWER_TARGET="/usr/local/libexec/storylight/run-power-mode-ab.sh"
readonly SWAP_TARGET="/usr/local/libexec/storylight/run-tensorrt-build-swap.sh"
readonly CONFIG_TARGET="/etc/storylight/admin.conf"
readonly SUDOERS_TARGET="/etc/sudoers.d/storylight-admin-${admin_user}"

for source_file in "$ADMIN_SOURCE" "$POWER_SOURCE" "$SWAP_SOURCE"; do
  if [[ ! -f "$source_file" ]]; then
    printf 'Required source file is missing: %s\n' "$source_file" >&2
    exit 69
  fi
done

install -d -o root -g root -m 0755 /usr/local/libexec/storylight /usr/local/sbin
install -d -o root -g root -m 0755 /etc/storylight /etc/sudoers.d
install -o root -g root -m 0755 "$ADMIN_SOURCE" "$ADMIN_TARGET"
install -o root -g root -m 0755 "$POWER_SOURCE" "$POWER_TARGET"
install -o root -g root -m 0755 "$SWAP_SOURCE" "$SWAP_TARGET"

config_tmp="$(mktemp /etc/storylight/.admin.conf.XXXXXX)"
sudoers_tmp="$(mktemp /etc/sudoers.d/.storylight-admin.XXXXXX)"
cleanup() {
  rm -f "$config_tmp" "$sudoers_tmp"
}
trap cleanup EXIT INT TERM

printf 'STORYLIGHT_ADMIN_USER=%s\n' "$admin_user" >"$config_tmp"
chown root:root "$config_tmp"
chmod 0600 "$config_tmp"

printf '%s ALL=(root) NOPASSWD: %s\n' "$admin_user" "$ADMIN_TARGET" >"$sudoers_tmp"
chown root:root "$sudoers_tmp"
chmod 0440 "$sudoers_tmp"
visudo -cf "$sudoers_tmp" >/dev/null

mv -f "$config_tmp" "$CONFIG_TARGET"
mv -f "$sudoers_tmp" "$SUDOERS_TARGET"
trap - EXIT INT TERM
visudo -cf "$SUDOERS_TARGET" >/dev/null

if [[ "$(stat -c '%U:%G:%a' "$ADMIN_TARGET")" != "root:root:755" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$POWER_TARGET")" != "root:root:755" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$SWAP_TARGET")" != "root:root:755" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_TARGET")" != "root:root:600" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$SUDOERS_TARGET")" != "root:root:440" ]]; then
  printf 'Installed Storylight delegation failed its ownership/mode verification.\n' >&2
  exit 78
fi

printf 'Restricted Storylight administration installed for %s.\n' "$admin_user"
printf 'No password was stored. Test with: sudo -n %s status\n' "$ADMIN_TARGET"
