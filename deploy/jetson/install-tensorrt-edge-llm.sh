#!/usr/bin/env bash
# Build a pinned TensorRT Edge-LLM runtime in the invoking user's home.

set -euo pipefail

readonly EDGELLM_VERSION="v0.10.0"
readonly EDGELLM_REVISION="71dd1bae032e70771265917ec74d3ff4cad07a10"
readonly INSTALL_ROOT="${BOOKFORGE_EDGELLM_ROOT:-$HOME/.local/share/bookforge/tensorrt-edgellm-v0.10.0}"
readonly SOURCE_DIR="$INSTALL_ROOT/src"
readonly BUILD_DIR="$INSTALL_ROOT/build"
readonly VENV_DIR="$INSTALL_ROOT/venv"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  printf 'Run this installer as the Bookforge user, not root.\n' >&2
  exit 64
fi

if [[ ! -r /etc/nv_tegra_release ]] || ! grep -q '^# R39 (release), REVISION: 2.1' /etc/nv_tegra_release; then
  printf 'TensorRT Edge-LLM v0.10.0 is pinned here for JetPack 7.2.1 / L4T 39.2.1.\n' >&2
  exit 65
fi

for executable in git python3 gcc g++ /usr/local/cuda/bin/nvcc; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    printf 'Required build tool is missing: %s\n' "$executable" >&2
    exit 69
  fi
done

if ! dpkg-query -W -f='${Status} ${Version}\n' libnvinfer-dev 2>/dev/null \
  | grep -q '^install ok installed 10\.16\.'; then
  printf 'TensorRT 10.16 development headers are required.\n' >&2
  exit 69
fi

mkdir -p "$INSTALL_ROOT"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --depth 1 --branch "$EDGELLM_VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$SOURCE_DIR"
fi

if [[ $(git -C "$SOURCE_DIR" rev-parse HEAD) != "$EDGELLM_REVISION" ]]; then
  printf 'Refusing unpinned TensorRT Edge-LLM source at %s.\n' "$SOURCE_DIR" >&2
  exit 65
fi
git -C "$SOURCE_DIR" submodule update --init --recursive

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --disable-pip-version-check --no-input \
  cmake==4.1.0 \
  ninja==1.13.0 \
  pybind11==3.0.4 \
  fastapi==0.139.2 \
  uvicorn==0.51.0 \
  python-multipart==0.0.32

readonly PYBIND11_CMAKE_DIR="$("$VENV_DIR/bin/python" -m pybind11 --cmakedir)"

export PATH="/usr/local/cuda/bin:$VENV_DIR/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE="$SOURCE_DIR/cmake/aarch64_linux_toolchain.cmake" \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=13.2 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  -DPython_EXECUTABLE="$VENV_DIR/bin/python"

# A single build worker avoids memory pressure on the 8 GB Orin Nano while the
# kiosk and local Gemma runtime remain available.
cmake --build "$BUILD_DIR" \
  --target NvInfer_edgellm_plugin llm_build llm_inference llm_bench _edgellm_runtime -j1

for binary in llm_build llm_inference llm_bench; do
  test -x "$BUILD_DIR/examples/llm/$binary"
done
test -s "$BUILD_DIR/libNvInfer_edgellm_plugin.so"
find "$BUILD_DIR/pybind" -maxdepth 1 -type f -name '*_edgellm_runtime*.so' -print -quit \
  | grep -q .

printf 'TensorRT Edge-LLM %s is ready at %s.\n' "$EDGELLM_VERSION" "$INSTALL_ROOT"
