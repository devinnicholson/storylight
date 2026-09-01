#!/usr/bin/env bash
# Manage one fixed, temporary NVMe swap file for memory-heavy TensorRT builds.

set -euo pipefail
IFS=$'\n\t'
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE

readonly TRUSTED_ROOT="/var/lib/bookforge-trusted"
readonly SWAP_FILE="$TRUSTED_ROOT/tensorrt-build.swap"
readonly SWAP_BYTES=$((8 * 1024 * 1024 * 1024))

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  printf 'Run this fixed swap runner as root.\n' >&2
  exit 64
fi

is_active() {
  swapon --noheadings --raw --show=NAME | grep -Fxq "$SWAP_FILE"
}

verify_file() {
  [[ -f "$SWAP_FILE" && ! -L "$SWAP_FILE" ]] \
    && [[ "$(stat -c '%U:%G:%a' "$SWAP_FILE")" == "root:root:600" ]] \
    && [[ "$(stat -c '%s' "$SWAP_FILE")" -eq "$SWAP_BYTES" ]]
}

ensure_trusted_root() {
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
}

status() {
  if is_active; then
    printf 'active '
    swapon --noheadings --raw --bytes --show=NAME,SIZE,USED,PRIO \
      | awk -v path="$SWAP_FILE" '$1 == path {print $0}'
  elif [[ -e "$SWAP_FILE" ]]; then
    verify_file || { printf 'unsafe\n' >&2; exit 78; }
    printf 'inactive %s\n' "$SWAP_FILE"
  else
    printf 'absent %s\n' "$SWAP_FILE"
  fi
}

case "${1:-}" in
  prepare)
    [[ $# -eq 1 ]] || exit 64
    ensure_trusted_root
    if is_active; then
      verify_file || { printf 'Active TensorRT swap file is unsafe.\n' >&2; exit 78; }
      status
      exit 0
    fi
    if [[ -e "$SWAP_FILE" ]]; then
      verify_file || { printf 'Existing TensorRT swap file is unsafe.\n' >&2; exit 78; }
    else
      readonly partial="${SWAP_FILE}.partial"
      [[ ! -e "$partial" ]] || { printf 'Refusing stale swap partial: %s\n' "$partial" >&2; exit 78; }
      fallocate --length "$SWAP_BYTES" "$partial"
      chown root:root "$partial"
      chmod 0600 "$partial"
      mkswap "$partial" >/dev/null
      mv "$partial" "$SWAP_FILE"
    fi
    swapon --priority -2 "$SWAP_FILE"
    status
    ;;
  cleanup)
    [[ $# -eq 1 ]] || exit 64
    if is_active; then
      swapoff "$SWAP_FILE"
    fi
    if [[ -e "$SWAP_FILE" ]]; then
      verify_file || { printf 'Refusing to remove unsafe TensorRT swap file.\n' >&2; exit 78; }
      rm -- "$SWAP_FILE"
    fi
    status
    ;;
  status)
    [[ $# -eq 1 ]] || exit 64
    status
    ;;
  *)
    printf 'Usage: run-tensorrt-build-swap.sh prepare|cleanup|status\n' >&2
    exit 64
    ;;
esac
