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
- A butterfly rotation may declare `storage_dtype` as `float16`, `bfloat16`, or
  `float32`. Missing values retain the earlier `float32` behavior. Loaders must
  preserve this dtype after any model-wide device/dtype conversion because the
  angle tensor is part of the deployed byte budget.
- `activation_bits` records the signed per-token activation-quantization
  semantics used by `QuantLinear`.
- Identical codebook tensors are stored once and referenced by all applicable
  modules, which is material for a 65,536-entry finite E8P codebook.

All buffers—including second-level scale metadata—are counted at their actual
retained byte size. A runtime must compare decoded values against the Python
reference before making correctness or performance claims.

See [packed format v1](packed_format_v1.md) for the unchanged word-level
bitstream and artifact-layout rules.

## Generation integrity and publication

New writers also record a `generation_id` and `files_sha256` map for the
configuration, tokenizer/processor and both tensor files. These additive fields
do not change the quantization layout. Export results include `manifest_sha256`;
experiment exports additionally embed revision/seed/trial/allocation identity
under `deployment.experiment_identity`.

The writer builds a complete sibling staging directory before publication.
An overwrite first moves the old artifact to `.NAME.rotquant-previous`, then
publishes the new directory. Normal publication errors restore the old directory;
if the process dies between renames, the Python loader resolves the recovery
copy. A surviving recovery copy is never overwritten automatically. This needs
temporary space for two generations and assumes one writer per destination.
It is not a guarantee of power-loss durability on arbitrary filesystems.

`verify_checkpoint` checks file contents, and experiment resume also requires
the manifest digest recorded by that trial. Loaders remain compatible with
hashless v1/v2 artifacts, but those older artifacts cannot satisfy the new
integrity-bound resume gate. Overwrite of an unverified/legacy directory, or a
checkpoint containing additional user files, requires choosing a new export
path. No unrelated directory contents are removed.
