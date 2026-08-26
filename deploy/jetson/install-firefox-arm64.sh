#!/usr/bin/env bash

set -euo pipefail

readonly FIREFOX_VERSION="153.0esr"
readonly FIREFOX_ARCHIVE="firefox-${FIREFOX_VERSION}.tar.xz"
readonly FIREFOX_SHA256="17c523ed1af68e2204760c51bebd6354bb4c9172b5c09804c6679f3d0049a0fa"
readonly FIREFOX_URL="https://ftp.mozilla.org/pub/firefox/releases/${FIREFOX_VERSION}/linux-aarch64/en-US/${FIREFOX_ARCHIVE}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -m)" != "aarch64" ]]; then
  printf 'The Bookforge Firefox installer is pinned for Jetson aarch64, not %s.\n' "$(uname -m)" >&2
  exit 2
fi

install_root="${BOOKFORGE_FIREFOX_ROOT:-${HOME}/.local/opt}"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/bookforge-downloads"
target_dir="${install_root}/firefox-${FIREFOX_VERSION}"
current_link="${install_root}/firefox-bookforge"
archive_path="${cache_root}/${FIREFOX_ARCHIVE}"

mkdir -p "$install_root" "$cache_root"

if [[ ! -f "$archive_path" ]]; then
  curl --fail --location --retry 2 "$FIREFOX_URL" --output "${archive_path}.part"
  mv "${archive_path}.part" "$archive_path"
fi

actual_sha256="$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$FIREFOX_SHA256" ]]; then
  printf 'Firefox archive checksum mismatch: expected %s, got %s.\n' \
    "$FIREFOX_SHA256" "$actual_sha256" >&2
  exit 1
fi

if [[ ! -x "${target_dir}/firefox" ]]; then
  if [[ -e "$target_dir" ]]; then
    printf 'Firefox target exists but is incomplete: %s\n' "$target_dir" >&2
    exit 1
  fi
  staging_dir="$(mktemp -d "${install_root}/.firefox-stage.XXXXXX")"
  tar --extract --xz --file "$archive_path" --directory "$staging_dir"
  mv "${staging_dir}/firefox" "$target_dir"
  rmdir "$staging_dir"
fi

install -D -m 0644 \
  "$SCRIPT_DIR/firefox-policies.json" \
  "$target_dir/distribution/policies.json"

if [[ -e "$current_link" && ! -L "$current_link" ]]; then
  printf 'Refusing to replace non-symlink browser path: %s\n' "$current_link" >&2
  exit 1
fi
ln -sfn "firefox-${FIREFOX_VERSION}" "$current_link"

"${current_link}/firefox" --version
printf 'Verified Bookforge browser: %s\n' "${current_link}/firefox"
printf 'Set BOOKFORGE_BROWSER_BIN=%s in kiosk.env.\n' "${current_link}/firefox"
