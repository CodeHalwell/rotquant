#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-qwen35-4b-rotquant-native.gguf}"
bench_path="${ROTQUANT_LLAMA_BENCH:-third_party/llama.cpp/build-rotquant/bin/llama-bench}"
depths="${ROTQUANT_KV_DEPTHS:-0,512,2048}"
repetitions="${ROTQUANT_KV_REPETITIONS:-3}"

for cache_type in f16 q4_0; do
  "${bench_path}" \
    -m "${model_path}" \
    -ngl 99 \
    -fa on \
    -ctk "${cache_type}" \
    -ctv "${cache_type}" \
    -p 0 \
    -n 16 \
    -d "${depths}" \
    -r "${repetitions}" \
    --progress
done
