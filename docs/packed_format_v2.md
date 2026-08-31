# RotQuant packed checkpoint format v2

Version 2 retains the v1 `int32`, LSB-first generic code bitstream and adds
formats whose reconstruction semantics cannot safely be ignored by a v1
reader. The executable authority is `rotquant.format`; readers support both v1
and v2, while new writers emit v2.

## Additions over v1

- A codebook record declares `kind: scalar | vector` and `dimension`. Scalar
  streams contain `in_features * out_features` codes. A vector stream of width
  `d` contains `in_features * out_features / d` indices, whose packed bit width
  is `log2(number_of_centroids)`.
- Primary and residual scales may be blockwise uint8. Each block stores an fp16
  offset and fp16 step and reconstructs `offset + code * step`. The manifest
  declares `scale_quant_group_size`; missing offset/step metadata fails closed.
- `rotation_id` preserves one shared rotation object across q/k/v or gate/up
  modules after loading.
- `activation_bits` records the signed per-token activation-quantization
  semantics used by `QuantLinear`.
- Identical codebook tensors are stored once and referenced by all applicable
  modules, which is material for a 65,536-entry finite E8P codebook.

All buffers—including second-level scale metadata—are counted at their actual
retained byte size. A runtime must compare decoded values against the Python
reference before making correctness or performance claims.

See [packed format v1](packed_format_v1.md) for the unchanged word-level
bitstream and artifact-layout rules.
