# RotQuant serving backends

This document distinguishes preservation of the RotQuant representation from
ordinary export followed by a second quantization pass. The latter can still be
useful, but it is a different model and must not be reported as native RotQuant.

September 9 priority: the stopped W5 public-task run moves native serving ahead
of another paid sweep. The [native-v3 matrix contract](native_runtime_v3.md)
now preserves 1–8-bit codes with affine uint8/FP16 scales in scalar CPU C++, but
does not yet add full-model W5/W6/W8 execution to any engine. The intended next
product path is the RotQuant llama.cpp integration, Metal locally and CUDA before
Colab; vLLM/SGLang remain later adapters. No GPU readiness is inferred from
matrix conformance or the older W4 integration.

## Compatibility matrix

| Backend | Current state | Preserves RotQuant? | Required integration |
|---|---|---:|---|
| Transformers | `load_packed_model()` returns the normal architecture with `QuantLinear` replacements and supports `forward`/`generate`. | Yes | Package `rotquant`; a fused CUDA/Metal op is still needed for fast compressed execution. A future standard loader should use Transformers' [`HfQuantizer`](https://huggingface.co/docs/transformers/en/quantization/concept_guide) extension point. |
| llama.cpp / Metal | Experimental GGUF v1 W4/Gaussian/g128/FP16-scale model path; old 3.25-bpv KV map implemented but its quality claims are withdrawn. **Not the retained W5/scale8 + W6/W8 recipe.** | Supported v1 representation only | Build with `scripts/build_rotquant_llama_cpp.sh`; native-v3 model/accelerator operators remain pending. Keep cache precision separate from weight validation. |
| vLLM | No RotQuant adapter is implemented here. | Not yet | Quantization adapter, sharded packed-weight loader and CUDA/Triton GEMV/GEMM follow the native serving milestone. |
| SGLang | Stock SGLang has its own quantization registry and GGUF kernels, but no RotQuant method or tensor loader. | Not yet | Port the vLLM kernel contract into SGLang's `QuantizationConfig`/linear method and register the method in its quantization table. Its standard GGUF loader cannot infer RotQuant semantics from opaque custom tensors. |
| Unsloth | Useful for source-model loading, LoRA/QAT, and merged export. Its inference/export paths target standard Transformers, bitsandbytes, vLLM, or standard llama.cpp GGUF formats. | Only through RotQuant's Transformers loader | Keep Unsloth on the training side, then run the RotQuant pack/export step. `save_pretrained_gguf(..., q4_k_m)` reconstructs and requantizes, so it does not preserve RotQuant. See the official [inference](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference) and [deployment](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) guidance. |

## Canonical artifact direction

Checkpoint v3 is the current retained model artifact. A future standard backend
integration should preserve its tensors and ownership while exposing:

- `config.json.quantization_config.quant_method = "rotquant"`;
- packed tensors named by original module prefix rather than manifest ordinal;
- exact codebook, group, scale, rotation, mixed-precision, tied-vocabulary, and
  optional adapter metadata in a versioned JSON schema;
- tensors shardable without reconstruction, with every logical shape recorded;
- a conformance vector that compares selected layer outputs and generated-token
  logits against the reference loader.

The earlier vLLM-first ordering is superseded by the native-first milestone
above. The existing checkpoint-v3 artifact remains canonical; the native matrix
format has an independent version. Do not reinterpret either as a new GGUF
contract without implementing and validating its model operators.

The ordered implementation work, architecture tiers, per-bit kernel rollout,
and definition of done are maintained in [`roadmap.md`](roadmap.md). Its Stage
2.0 reliability checks are merge gates for all backend work.

## Acceptance gates

A backend is not considered supported merely because it loads:

1. packed tensor and metadata bytes match the canonical checkpoint;
2. per-layer outputs agree with the reference loader within a declared tolerance;
3. greedy logits and generated tokens pass fixed conformance prompts;
4. resident memory reflects packed storage without an undeclared dense cache;
5. prefill and decode throughput are reported separately at multiple contexts;
6. **if quantized KV is claimed**, it is written before persistent HBM/DRAM storage and consumed by
   fused attention without a full-cache dequantization buffer during decode.
