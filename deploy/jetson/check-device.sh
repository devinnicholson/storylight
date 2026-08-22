#!/usr/bin/env bash
# Read-only Jetson readiness report. No package, power, storage, or device state is changed.

set -uo pipefail

EXPECTED_JETPACK="7.2.1"
EXPECTED_L4T_RELEASE="R39"
EXPECTED_L4T_REVISION="2.1"
EXPECTED_CUDA_PREFIX="13.2"
EXPECTED_TENSORRT_PREFIX="10.16.2"
STRICT=0
FAILURES=0

usage() {
  printf 'Usage: %s [--strict]\n' "$0"
  printf '  --strict  Return non-zero when a required JetPack component is missing or mismatched.\n'
}

for argument in "$@"; do
  case "$argument" in
    --strict) STRICT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
  esac
done

section() { printf '\n== %s ==\n' "$1"; }
pass() { printf '[PASS] %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

section "Host"
printf 'Kernel: %s\n' "$(uname -srmo 2>/dev/null || uname -a)"
printf 'Architecture: %s\n' "$(uname -m)"
if [[ "$(uname -m)" == "aarch64" ]]; then
  pass "64-bit Arm userspace detected"
else
  warn "Expected aarch64 on Jetson; this may be a development host"
fi

section "JetPack and Jetson Linux"
if [[ -r /etc/nv_tegra_release ]]; then
  L4T_LINE="$(head -n 1 /etc/nv_tegra_release)"
  printf 'L4T marker: %s\n' "$L4T_LINE"
  if [[ "$L4T_LINE" == *"${EXPECTED_L4T_RELEASE}"* && "$L4T_LINE" == *"REVISION: ${EXPECTED_L4T_REVISION}"* ]]; then
    pass "Jetson Linux ${EXPECTED_L4T_RELEASE#R}.${EXPECTED_L4T_REVISION} matches JetPack ${EXPECTED_JETPACK}"
  else
    fail "Expected Jetson Linux ${EXPECTED_L4T_RELEASE#R}.${EXPECTED_L4T_REVISION} for JetPack ${EXPECTED_JETPACK}"
  fi
else
  fail "/etc/nv_tegra_release is missing"
fi

if have dpkg-query && JETPACK_VERSION="$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)"; then
  printf 'nvidia-jetpack package: %s\n' "$JETPACK_VERSION"
  if [[ "$JETPACK_VERSION" == "${EXPECTED_JETPACK}"* ]]; then
    pass "JetPack metapackage version matches ${EXPECTED_JETPACK}"
  else
    warn "JetPack metapackage does not begin with ${EXPECTED_JETPACK}; L4T is the authoritative BSP check"
  fi
else
  warn "nvidia-jetpack metapackage is not installed; runtime components may have been installed separately"
fi

section "CUDA"
if have nvcc; then
  NVCC_LINE="$(nvcc --version 2>/dev/null | tail -n 1)"
  printf '%s\n' "$NVCC_LINE"
  if [[ "$NVCC_LINE" == *"release ${EXPECTED_CUDA_PREFIX}"* ]]; then
    pass "CUDA release matches the JetPack 7.2.1 family"
  else
    warn "Expected CUDA ${EXPECTED_CUDA_PREFIX}.x; inspect the reported version"
  fi
elif [[ -r /usr/local/cuda/version.json ]]; then
  printf 'CUDA version file: /usr/local/cuda/version.json\n'
  grep -m 1 'version' /usr/local/cuda/version.json 2>/dev/null || true
  warn "nvcc is not on PATH"
else
  fail "CUDA toolkit was not detected"
fi

section "TensorRT"
if have dpkg-query; then
  TENSORRT_PACKAGES="$(dpkg-query -W -f='${Package} ${Version}\n' 'libnvinfer*' 2>/dev/null || true)"
else
  TENSORRT_PACKAGES=""
fi
if [[ -n "$TENSORRT_PACKAGES" ]]; then
  printf '%s\n' "$TENSORRT_PACKAGES" | head -n 12
  if [[ "$TENSORRT_PACKAGES" == *"${EXPECTED_TENSORRT_PREFIX}"* ]]; then
    pass "TensorRT ${EXPECTED_TENSORRT_PREFIX}.x package detected"
  else
    warn "TensorRT packages found, but not the ${EXPECTED_TENSORRT_PREFIX}.x JetPack 7.2.1 family"
  fi
else
  fail "TensorRT libnvinfer packages were not detected"
fi

section "Docker and NVIDIA container support"
if have docker; then
  printf 'Docker: %s\n' "$(docker --version 2>/dev/null || true)"
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable by the current user"
    RUNTIMES="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    printf 'Docker runtimes: %s\n' "${RUNTIMES:-unknown}"
  else
    warn "Docker is installed, but its daemon is stopped or inaccessible to this user"
  fi
else
  fail "Docker CLI was not detected"
fi
if have nvidia-ctk; then
  printf 'NVIDIA Container Toolkit: %s\n' "$(nvidia-ctk --version 2>/dev/null || true)"
  pass "nvidia-ctk is available"
else
  warn "nvidia-ctk was not detected"
fi

section "Python"
if have python3; then
  PYTHON_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  printf 'Python: %s\n' "$PYTHON_VERSION"
  if python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    pass "Python satisfies Bookforge >=3.11"
  else
    fail "Bookforge requires Python 3.11 or newer"
  fi
else
  fail "python3 was not detected"
fi

section "Power mode"
if have nvpmodel; then
  nvpmodel -q 2>/dev/null || warn "nvpmodel exists but its current mode could not be read"
  pass "nvpmodel is installed; no power mode was changed"
else
  warn "nvpmodel was not detected"
fi

section "NVMe and mounted storage"
if have lsblk; then
  lsblk -d -o NAME,TRAN,SIZE,MODEL 2>/dev/null || true
  printf '\nMounted filesystems:\n'
  lsblk -o NAME,FSTYPE,SIZE,FSAVAIL,MOUNTPOINTS 2>/dev/null || true
else
  warn "lsblk was not detected"
fi
if compgen -G '/dev/nvme*n*' >/dev/null; then
  pass "At least one NVMe namespace was detected"
else
  warn "No /dev/nvme*n* namespace was detected"
fi

section "Camera"
if compgen -G '/dev/video*' >/dev/null; then
  printf 'Video devices: %s\n' "$(printf '%s ' /dev/video*)"
  pass "At least one V4L2 device node exists"
else
  warn "No /dev/video* device was detected"
fi
if have v4l2-ctl; then
  v4l2-ctl --list-devices 2>/dev/null || warn "v4l2-ctl could not enumerate devices"
else
  warn "v4l2-ctl is not installed"
fi

section "Microphone"
if [[ -d /dev/snd ]]; then
  pass "/dev/snd exists"
else
  warn "/dev/snd is missing"
fi
if have arecord; then
  arecord -l 2>/dev/null || warn "arecord could not enumerate capture hardware"
else
  warn "arecord is not installed"
fi

section "Display"
CONNECTED_DISPLAYS=""
for status_path in /sys/class/drm/card*-*/status; do
  [[ -r "$status_path" ]] || continue
  if [[ "$(<"$status_path")" == "connected" ]]; then
    CONNECTED_DISPLAYS+="${status_path%/status} "
  fi
done
if [[ -n "$CONNECTED_DISPLAYS" ]]; then
  printf 'Connected DRM outputs: %s\n' "$CONNECTED_DISPLAYS"
  pass "A connected display output was detected"
else
  warn "No connected display was found through DRM sysfs"
fi
printf 'Session: XDG_SESSION_TYPE=%s DISPLAY=%s WAYLAND_DISPLAY=%s\n' \
  "${XDG_SESSION_TYPE:-unset}" "${DISPLAY:-unset}" "${WAYLAND_DISPLAY:-unset}"
if have chromium; then
  printf 'Chromium: %s\n' "$(command -v chromium)"
elif have chromium-browser; then
  printf 'Chromium: %s\n' "$(command -v chromium-browser)"
else
  warn "Chromium was not detected; the kiosk launcher will remain unavailable"
fi

section "Thermals"
THERMAL_COUNT=0
for zone in /sys/class/thermal/thermal_zone*; do
  [[ -r "$zone/type" && -r "$zone/temp" ]] || continue
  TYPE="$(<"$zone/type")"
  RAW_TEMP="$(<"$zone/temp")"
  if [[ "$RAW_TEMP" =~ ^[0-9]+$ ]]; then
    awk -v label="$TYPE" -v value="$RAW_TEMP" 'BEGIN { printf "%s: %.1f C\n", label, value / 1000 }'
  else
    printf '%s: %s\n' "$TYPE" "$RAW_TEMP"
  fi
  THERMAL_COUNT=$((THERMAL_COUNT + 1))
done
if ((THERMAL_COUNT > 0)); then
  pass "Read ${THERMAL_COUNT} thermal zones without changing clocks or fan policy"
else
  warn "No readable thermal zones were found"
fi

section "Result"
if ((FAILURES == 0)); then
  pass "No required component failures were detected"
else
  warn "${FAILURES} required component check(s) failed"
fi
printf 'Diagnostic only: no device state was changed.\n'

if ((STRICT == 1 && FAILURES > 0)); then
  exit 1
fi
exit 0
