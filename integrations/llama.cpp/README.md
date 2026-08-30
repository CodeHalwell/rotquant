# Native RotQuant GGUF for llama.cpp

This integration preserves the deployed RotQuant representation instead of
dequantizing it and asking llama.cpp to quantize it again. Each quantized
projection is written as two GGUF tensors:

- `*.rqweight`: raw 4-bit Gaussian code indices and one fp16 scale per group
  of 128 values;
- `*.rqrotation`: the exact sign vector and learned 128-wide butterfly angles.

The included llama.cpp patch recognizes those tensors and evaluates them with
a native `ggml_custom` matrix operator. A stock llama.cpp build fails closed
because the ordinary `*.weight` matrices are intentionally absent.

## Current v1 scope

Native v1 supports the text trunk of Qwen3.5, the exact 4-bit Gaussian
codebook, group size 128, and FWHT or trained butterfly rotations with block
size 128. It does not yet export the vision tower or retained LoRA adapters.
The input checkpoint must therefore report `lora_rank: 0` for every packed
projection.

The joint-release extension also supports its exact frozen mixed K/V recipe:
Gaussian 2/3/4-bit codes, group size 64, one fp16 RMS scale per group, and
randomized FWHT-128 independently for keys and values. The eight
full-attention layers use the embedded maps and average 3.25 bits per value
including scale overhead. Persistent cache rows stay packed; each attention
layer is dequantized into transient fp16 graph storage before the existing
flash-attention operator. Cache shifting and non-flash V layouts fail closed.

The native operator has two implementations:

- a portable scalar CPU reference path; and
- an Apple Metal path that rotates each activation once, decodes packed
  Gaussian nibbles in place, and evaluates four output rows per threadgroup.

The packed weights and rotations remain on the GPU; no dense projection is
materialized. On an Apple M5 Max, the Qwen3.5-4B artifact measured about 23
prompt tokens/s for a 128-token prefill and 47 generated tokens/s for a short
greedy request through `llama-server`. Treat those numbers as a functional
baseline rather than a cross-machine promise.

## Build the pinned runtime

From the RotQuant repository root:

```bash
scripts/build_rotquant_llama_cpp.sh
```

This clones llama.cpp at commit
`17252c769a63c1cb650ce98ae309cf4de0da7778`, applies
`rotquant-native-v1.patch`, compiles the RotQuant metallib, and builds
`llama-cli`, `llama-server`, and `llama-bench`. On macOS Metal is enabled
automatically. Override the checkout/build locations with:

```bash
scripts/build_rotquant_llama_cpp.sh /path/to/llama.cpp /path/to/build
```

Set `ROTQUANT_LLAMA_METAL=OFF` for a CPU-only build.

## Export without double quantization

```bash
uv sync --extra eval
uv run python scripts/export_rotquant_gguf.py \
  HallD/qwen35-4b-rotquant-joint \
  qwen35-4b-rotquant-joint-kv3p25-native.gguf \
  --llama-cpp-dir third_party/llama.cpp
```

The exporter downloads a Hub checkpoint when given a model ID. For a local
artifact, pass its directory instead. It stores the tied text embedding once
and omits optional Qwen MTP weights that are not present in the checkpoint.
The joint checkpoint's deployment manifest supplies the frozen K/V map. For a
compatible checkpoint without embedded deployment metadata, pass
`--kv-cache-config configs/qwen35_4b_frozen_mixed_kv_3p25.json`.

For a release artifact, verify all native tensors against the source:

```bash
uv run python scripts/verify_rotquant_gguf.py \
  HallD/qwen35-4b-rotquant-joint \
  qwen35-4b-rotquant-joint-kv3p25-native.gguf \
  --llama-cpp-dir third_party/llama.cpp
```

This compares every code/scale byte and every float32 rotation value. It does
not compare merely reconstructed dense weights.

## Serve an OpenAI-compatible text API

On Apple Silicon, use the constrained wrapper. It selects one 4096-token slot
instead of llama.cpp's automatic four 262k-token slots, disables reasoning, and
caps default output at 64 tokens:

```bash
scripts/serve_rotquant_gguf.sh qwen35-4b-rotquant-joint-kv3p25-native.gguf 8085
```

Set `ROTQUANT_GPU_LAYERS=0` for the all-CPU reference path. First test the
kernel with llama.cpp's raw completion endpoint. This bypasses chat templates,
system prompts, and tool definitions:

```bash
curl -N http://127.0.0.1:8085/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":4,"temperature":0,"stream":true}'
```

Once that works, call the OpenAI-compatible chat endpoint:

```bash
curl -N http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-4b-rotquant",
    "messages": [{"role": "user", "content": "Explain RotQuant briefly."}],
    "temperature": 0.2,
    "max_tokens": 16,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

This endpoint is text-only in v1. Do not pass images even though the source
Transformers checkpoint is multimodal.

### Browser UI latency

The llama.cpp web UI may attach a system message and tool definitions. A
one-word message can consequently become a prompt containing hundreds of
tokens, so the first response remains slower than the raw completion smoke
test. With the Metal kernel, a 329-token prompt should normally complete in
seconds rather than minutes. If the server reports less than one token/s,
confirm that `-ngl 99` is present and that startup logs compile/load both
`kernel_rotquant_rotate_f32` and `kernel_rotquant_mul_mv_f32`. Disable tools
and clear the system prompt when measuring raw model latency.

## Format notes

The GGUF contains `rotquant.format=rotquant-gguf`, `rotquant.version=1`, and
the exact format constants. A frozen-cache artifact additionally contains the
versioned `rotquant.kv_cache.*` contract, including 32-entry K/V bit maps.
llama.cpp may report a misleading model parameter
count or aggregate BPW because it sees opaque byte tensors rather than their
logical 4-bit matrix shapes. The on-disk file size is authoritative.
