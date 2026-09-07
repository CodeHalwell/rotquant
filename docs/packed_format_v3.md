# Experimental packed checkpoint v3: shared vocabulary

V3 extends [v2](packed_format_v2.md) when an artifact contains one shared
`PackedVocabulary` referenced by its input embedding and output head. Ordinary
backbone-only checkpoints still write v2. V3 is a Python/Transformers reference
checkpoint, **not** stock GGUF, llama.cpp, vLLM or SGLang compatibility.

`rotquant_config.json.tied_vocabulary` records:

- two distinct module aliases (`embedding`, `head`), resolving through the
  adapter's input/output embedding accessors;
- vocabulary/hidden dimensions, original source-matrix digest, and a
  `VocabularyConfig` (W6/W8 Gaussian, group/block alignment, row chunks, seed);
- FP16 group scales, packed int32 words, one shared FP32 codebook, and chunk
  row counts covering the entire vocabulary;
- the execution dtype and `projection_mode` (`rotated` or `dense_equivalent`).

Only one copy of the vocabulary codes/scales/codebook and rotation is serialized.
The original dense embedding/head parameter is removed from model state. Shared
owner aliases are restored before loading the ordinary safetensors state.
Backbone dense fallback caches are never serialized. All artifact files have
SHA-256 entries; experiment records bind the manifest digest as well.

The original `rotated` mode rotates the input and multiplies dequantized rotated
rows. `dense_equivalent` inversely rotates each row tile, rounds it to the
execution dtype, then projects unrotated hidden states. It matches the screened
dense weight representation while bounding workspace to row tiles; GEMM rounding
may still differ. V3 files without `projection_mode` retain `rotated` behaviour.

Moving/casting a model changes device and execution dtype without recasting
packed scale metadata. Embedding lookup reconstructs requested rows only;
projection has no persistent dense head. Retie/resize APIs remain unsupported
and must fail instead of recreating dense vocabulary weights. These execution
paths transiently dequantize: no fused throughput claim is implied.

On checkpoint load, framework-owned nonpersistent floating buffers are
reconstructed from the model configuration and retain their constructor dtype
when moved to the requested device. In particular, Qwen's RoPE frequency buffers
must stay FP32 even with FP16/BF16 model parameters. They are not checkpoint
weights, and rounding them to FP16 then converting back to FP32 loses information.
This loader correction does not change the artifact format or serialized bytes.
An explicit later whole-model `.half()` by application code can still downcast
ordinary framework buffers; use the loader's `dtype` argument instead.

The [validation workflow](packed_vocabulary_validation_run.md) checks live
ownership, absent dense caches, actual complete bytes, bounded numerical probes,
fresh-process reload and full development quality. Tiny multimodal-Qwen text-path
conformance is CPU-tested; pretrained full-model GPU validation remains a separate
Colab acceptance step. Retained vision tensors are counted, but vision inference
accuracy is not assessed by the text-only run.
