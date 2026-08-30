#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-qwen35-4b-rotquant-joint-kv3p25-native.gguf}"
bench_path="${ROTQUANT_LLAMA_BENCH:-third_party/llama.cpp/build-rotquant/bin/llama-bench}"
depths="${ROTQUANT_KV_DEPTHS:-0,512,2048}"
repetitions="${ROTQUANT_KV_REPETITIONS:-3}"

"${bench_path}" \
  -m "${model_path}" \
  -ngl 99 \
  -fa on \
  -ctk f16 \
  -ctv f16 \
  -p 0 \
  -n 16 \
  -d "${depths}" \
  -r "${repetitions}" \
  --progress
