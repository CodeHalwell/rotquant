#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="17252c769a63c1cb650ce98ae309cf4de0da7778"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMA_DIR="${1:-${REPO_ROOT}/third_party/llama.cpp}"
BUILD_DIR="${2:-${LLAMA_DIR}/build-rotquant}"
PATCH_FILE="${REPO_ROOT}/integrations/llama.cpp/rotquant-native-v1.patch"
BUILD_JOBS="${ROTQUANT_BUILD_JOBS:-8}"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  if [[ -e "${LLAMA_DIR}" ]]; then
    echo "error: ${LLAMA_DIR} exists but is not a git checkout" >&2
    exit 1
  fi
  git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
  git -C "${LLAMA_DIR}" fetch --depth 1 origin "${PINNED_COMMIT}"
  git -C "${LLAMA_DIR}" checkout --detach "${PINNED_COMMIT}"
fi

ACTUAL_COMMIT="$(git -C "${LLAMA_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "error: llama.cpp must be at ${PINNED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  echo "use a new directory, or explicitly check out the pinned commit" >&2
  exit 1
fi

if git -C "${LLAMA_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "RotQuant patch is already applied."
elif git -C "${LLAMA_DIR}" apply --check "${PATCH_FILE}"; then
  git -C "${LLAMA_DIR}" apply "${PATCH_FILE}"
else
  echo "error: RotQuant patch does not apply cleanly" >&2
  exit 1
fi

if [[ -z "${ROTQUANT_LLAMA_METAL:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    ROTQUANT_LLAMA_METAL=ON
  else
    ROTQUANT_LLAMA_METAL=OFF
  fi
fi

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL="${ROTQUANT_LLAMA_METAL}" \
  -DGGML_METAL_EMBED_LIBRARY=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF
if [[ "${ROTQUANT_LLAMA_METAL}" == "ON" ]]; then
  # Build shaders ahead of time. The embedded-source mode recompiles them on
  # every server startup, which adds roughly 40 seconds on Apple Silicon.
  cmake --build "${BUILD_DIR}" --target ggml-metal-lib -j "${BUILD_JOBS}"
fi
cmake --build "${BUILD_DIR}" --target llama-cli llama-server llama-bench -j "${BUILD_JOBS}"

echo
echo "Built native RotQuant llama.cpp:"
echo "  ${BUILD_DIR}/bin/llama-cli"
echo "  ${BUILD_DIR}/bin/llama-server"
echo "  ${BUILD_DIR}/bin/llama-bench"
