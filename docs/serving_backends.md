# RotQuant serving backends

This document distinguishes preservation of the RotQuant representation from
ordinary export followed by a second quantization pass. The latter can still be
useful, but it is a different model and must not be reported as native RotQuant.

## Compatibility matrix

| Backend | Current state | Preserves RotQuant? | Required integration |
|---|---|---:|---|
| Transformers | `load_packed_model()` returns the normal architecture with `QuantLinear` replacements and supports `forward`/`generate`. | Yes | Package `rotquant`; a fused CUDA/Metal op is still needed for fast compressed execution. A future standard loader should use Transformers' [`HfQuantizer`](https://huggingface.co/docs/transformers/en/quantization/concept_guide) extension point. |
| llama.cpp / Metal | Experimental GGUF v1 loader, tied vocabulary, CPU reference, and Metal weight kernels are implemented in the pinned fork. | Yes | Build with `scripts/build_rotquant_llama_cpp.sh`; use the custom GGUF artifact. |
| vLLM | Stock vLLM does not recognize the checkpoint today. vLLM officially supports [out-of-tree quantization plugins](https://docs.vllm.ai/en/stable/features/quantization/). | Not yet | Implement `QuantizationConfig` + `QuantizeMethodBase`, a sharded packed-weight loader, and CUDA/Triton GEMV/GEMM kernels. This is the first production GPU serving target. |
| SGLang | Stock SGLang has its own quantization registry and GGUF kernels, but no RotQuant method or tensor loader. | Not yet | Port the vLLM kernel contract into SGLang's `QuantizationConfig`/linear method and register the method in its quantization table. Its standard GGUF loader cannot infer RotQuant semantics from opaque custom tensors. |
| Unsloth | Useful for source-model loading, LoRA/QAT, and merged export. Its inference/export paths target standard Transformers, bitsandbytes, vLLM, or standard llama.cpp GGUF formats. | Only through RotQuant's Transformers loader | Keep Unsloth on the training side, then run the RotQuant pack/export step. `save_pretrained_gguf(..., q4_k_m)` reconstructs and requantizes, so it does not preserve RotQuant. See the official [inference](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference) and [deployment](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) guidance. |

## Canonical artifact direction

Checkpoint v1 is safe and self-contained, but backend plugins should converge on
one v2 contract:

- `config.json.quantization_config.quant_method = "rotquant"`;
- packed tensors named by original module prefix rather than manifest ordinal;
- exact codebook, group, scale, rotation, mixed-precision, tied-vocabulary, and
  optional adapter metadata in a versioned JSON schema;
- tensors shardable without reconstruction, with every logical shape recorded;
- a conformance vector that compares selected layer outputs and generated-token
  logits against the reference loader.

The backend order is deliberate: make the Transformers artifact canonical,
implement the fused CUDA operation once, adapt that operation to vLLM, then port
the same tensor contract to SGLang. Unsloth remains a producer of training
checkpoints rather than a separate deployment format.

## Acceptance gates

A backend is not considered supported merely because it loads:

1. packed tensor and metadata bytes match the canonical checkpoint;
2. per-layer outputs agree with the reference loader within a declared tolerance;
3. greedy logits and generated tokens pass fixed conformance prompts;
4. resident memory reflects packed storage without an undeclared dense cache;
5. prefill and decode throughput are reported separately at multiple contexts;
6. quantized K/V is written before persistent HBM/DRAM storage and consumed by
   fused attention without a full-cache dequantization buffer during decode.
