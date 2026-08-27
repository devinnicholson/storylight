#!/usr/bin/env bash
# Install the narrow, root-owned Bookforge administrator and its exact sudo delegation.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

admin_user="${SUDO_USER:-operator}"

usage() {
  printf 'Usage: sudo install-bookforge-admin.sh [--user USER]\n'
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
  printf 'Invalid Bookforge administrator user: %s\n' "$admin_user" >&2
  exit 65
fi
if ! command -v visudo >/dev/null 2>&1; then
  printf 'visudo is required to validate the restricted delegation.\n' >&2
  exit 69
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ADMIN_SOURCE="$SCRIPT_DIR/bookforge-admin"
readonly POWER_SOURCE="$SCRIPT_DIR/run-power-mode-ab.sh"
readonly ADMIN_TARGET="/usr/local/sbin/bookforge-admin"
readonly POWER_TARGET="/usr/local/libexec/bookforge/run-power-mode-ab.sh"
readonly CONFIG_TARGET="/etc/bookforge/admin.conf"
readonly SUDOERS_TARGET="/etc/sudoers.d/bookforge-admin-${admin_user}"

for source_file in "$ADMIN_SOURCE" "$POWER_SOURCE"; do
  if [[ ! -f "$source_file" ]]; then
    printf 'Required source file is missing: %s\n' "$source_file" >&2
    exit 69
  fi
done

install -d -o root -g root -m 0755 /usr/local/libexec/bookforge /usr/local/sbin
install -d -o root -g root -m 0755 /etc/bookforge /etc/sudoers.d
install -o root -g root -m 0755 "$ADMIN_SOURCE" "$ADMIN_TARGET"
install -o root -g root -m 0755 "$POWER_SOURCE" "$POWER_TARGET"

config_tmp="$(mktemp /etc/bookforge/.admin.conf.XXXXXX)"
sudoers_tmp="$(mktemp /etc/sudoers.d/.bookforge-admin.XXXXXX)"
cleanup() {
  rm -f "$config_tmp" "$sudoers_tmp"
}
trap cleanup EXIT INT TERM

printf 'BOOKFORGE_ADMIN_USER=%s\n' "$admin_user" >"$config_tmp"
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
  || [[ "$(stat -c '%U:%G:%a' "$CONFIG_TARGET")" != "root:root:600" ]] \
  || [[ "$(stat -c '%U:%G:%a' "$SUDOERS_TARGET")" != "root:root:440" ]]; then
  printf 'Installed Bookforge delegation failed its ownership/mode verification.\n' >&2
  exit 78
fi

printf 'Restricted Bookforge administration installed for %s.\n' "$admin_user"
printf 'No password was stored. Test with: sudo -n %s status\n' "$ADMIN_TARGET"
