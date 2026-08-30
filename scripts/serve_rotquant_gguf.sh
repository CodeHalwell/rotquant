#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen35-4b-rotquant-joint-kv3p25-native.gguf}"
PORT="${2:-8085}"
SERVER="${ROTQUANT_LLAMA_SERVER:-third_party/llama.cpp/build-rotquant/bin/llama-server}"

if [[ ! -x "${SERVER}" ]]; then
  echo "error: server not found: ${SERVER}" >&2
  echo "run scripts/build_rotquant_llama_cpp.sh first" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "error: model not found: ${MODEL}" >&2
  exit 1
fi

exec "${SERVER}" \
  -m "${MODEL}" \
  -ngl "${ROTQUANT_GPU_LAYERS:-99}" \
  --flash-attn on \
  -c "${ROTQUANT_CONTEXT:-4096}" \
  -np 1 \
  -n "${ROTQUANT_MAX_TOKENS:-64}" \
  -b 512 \
  -ub 512 \
  --no-cont-batching \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --reasoning off \
  --offline \
  --host 127.0.0.1 \
  --port "${PORT}"
