# RotQuant packed checkpoint format v1

This document defines the portable checkpoint representation consumed by the
Python loader and native exporters. The executable authority is
`rotquant.format`; producers and runtimes must pass its validation and the
round-trip tests before claiming v1 compatibility.

## Artifact layout

A complete checkpoint contains:

- `rotquant_config.json`: manifest, architecture metadata and tensor map.
- `rotquant_model.safetensors`: ordinary model state such as embeddings, norms,
  biases, rotations and retained adapters.
- `rotquant_packed.safetensors`: packed codes, scales and codebooks.
- The normal Transformers configuration and tokenizer/processor files needed
  to reconstruct the source architecture.

The manifest is written last. Its presence therefore marks a completed export.
Pickle is never required.

## Generic code bitstream

Packed scalar-code indices use this exact contract:

| Property | v1 value |
|---|---|
| Storage word | signed `int32` carrying an unsigned 32-bit pattern |
| Bit order | least-significant bit first |
| Logical order | row-major flattened tensor order |
| Code widths | 1–16 bits |
| Kernel-targeted profiles | 1–8 bits |
| Trailing padding | unused high bits in the final word |

For logical code `i` with width `b`, its first bit is at absolute position
`i * b`. The starting word is `floor(i * b / 32)` and its offset is
`(i * b) mod 32`. A code may straddle two words. The exact buffer length is
`ceil(num_codes * b / 32)` words. Negative `int32` values are not negative
codes; they preserve unsigned patterns whose highest bit is set.

The format permits 9–16-bit storage for backwards-compatible research
artifacts. The public optimisation API exposes only 1–8-bit profiles because
those are the widths intended to receive dedicated runtime kernels.

## Quantized module records

Every entry in `quantized_modules` identifies its original module path, logical
input/output dimensions, activation rotation and `QuantizedWeight` tensor map.
The primary packed tensor must contain exactly
`in_features * out_features` scalar codes. Optional residual and sketch streams
use the same generic bitstream and declare their own logical shapes. Scales are
stored as fp16 or fp32 tensors.

New producers include a top-level `packing` object copied from
`CURRENT_PACKING.to_manifest()`. Early v1 checkpoints did not include this
object; absence means the same v1 defaults and remains supported. A present but
different packing object fails closed because silently guessing bit order or
word layout would corrupt every weight.

## Architecture metadata

New manifests record the adapter selected during optimisation and the source
`config.model_type`. This metadata supports diagnostics and future specialized
loaders. It is not needed to rebuild a current checkpoint: the manifest's exact
module paths remain authoritative, so legacy and custom-adapter artifacts can
still load when their Python model class is available.

## Compatibility rules

- Readers must reject unknown format versions.
- Readers must reject unsafe artifact paths, duplicate module names, malformed
  shapes, unsupported rotations and inconsistent logical dimensions.
- Readers may ignore unknown metadata fields that do not alter binary
  interpretation.
- Adding a new optional diagnostic field does not require a version bump.
- Changing bit order, word width, tensor order or required reconstruction
  semantics requires a new format version.
- A runtime must compare decoded indices, scales, codebooks and rotations
  exactly against the Python reference before performance results are valid.

Use `scripts/verify_rotquant_gguf.py` for the existing llama.cpp export and
`pytest tests/test_format.py tests/test_pack.py tests/test_checkpoint.py` for
the generic contract.
