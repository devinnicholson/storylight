#!/usr/bin/env bash
# Bookforge setup helper. With no mutation flag, this only runs device diagnostics.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${BOOKFORGE_VENV_PATH:-${REPO_ROOT}/.venv}"
INSTALL_PACKAGES=0
CREATE_VENV=0
INSTALL_APP=0
STRICT_CHECK=0

usage() {
  cat <<'EOF'
Usage: bootstrap.sh [options]

No options: run read-only diagnostics.

Options:
  --strict-check             Make required diagnostic failures return non-zero.
  --install-system-packages Install only Bookforge's Ubuntu utilities with apt.
  --create-venv              Create or reuse BOOKFORGE_VENV_PATH (default: .venv).
  --install-app              Install this checkout into the existing virtualenv.
  -h, --help                 Show this help.

This script never flashes Jetson Linux, installs a JetPack metapackage, changes
power mode, formats storage, or modifies camera/display configuration.
EOF
}

for argument in "$@"; do
  case "$argument" in
    --strict-check) STRICT_CHECK=1 ;;
    --install-system-packages) INSTALL_PACKAGES=1 ;;
    --create-venv) CREATE_VENV=1 ;;
    --install-app) INSTALL_APP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
  esac
done

if ((STRICT_CHECK == 1)); then
  "${SCRIPT_DIR}/check-device.sh" --strict
else
  "${SCRIPT_DIR}/check-device.sh"
fi

if ((INSTALL_PACKAGES == 0 && CREATE_VENV == 0 && INSTALL_APP == 0)); then
  exit 0
fi

if [[ "$(uname -m)" != "aarch64" || ! -r /etc/nv_tegra_release ]]; then
  printf 'Refusing Jetson setup mutations: this does not look like an aarch64 Jetson host.\n' >&2
  exit 1
fi

if ! grep -q 'R39 (release), REVISION: 2.1' /etc/nv_tegra_release; then
  printf 'Refusing setup mutations: JetPack 7.2.1 requires Jetson Linux 39.2.1.\n' >&2
  printf 'Use NVIDIA official installation guidance; this script never flashes a device.\n' >&2
  exit 1
fi

if ((INSTALL_PACKAGES == 1)); then
  printf '\nInstalling general Ubuntu runtime utilities (explicitly requested).\n'
  sudo apt-get update
  sudo apt-get install python3-venv python3-pip ffmpeg v4l-utils alsa-utils curl ca-certificates git
fi

if ((CREATE_VENV == 1)); then
  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    printf 'Reusing virtual environment: %s\n' "$VENV_PATH"
    if ! grep -q '^include-system-site-packages = true$' "${VENV_PATH}/pyvenv.cfg"; then
      printf 'Existing virtual environment cannot see JetPack Python packages.\n' >&2
      printf 'Move it aside and rerun --create-venv; this script will not delete it.\n' >&2
      exit 1
    fi
  else
    printf 'Creating virtual environment: %s\n' "$VENV_PATH"
    python3 -m venv --system-site-packages "$VENV_PATH"
  fi
fi

if ((INSTALL_APP == 1)); then
  if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    printf 'Virtual environment is missing: %s\n' "$VENV_PATH" >&2
    printf 'Run again with --create-venv --install-app.\n' >&2
    exit 1
  fi
  "${VENV_PATH}/bin/python" -m pip install --upgrade pip
  "${VENV_PATH}/bin/python" -m pip install --upgrade "$REPO_ROOT"
fi

printf '\nSetup actions completed. Re-run %s to inspect readiness.\n' "${SCRIPT_DIR}/check-device.sh"
