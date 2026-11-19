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

## Optional entropy-coded code streams (format v4)

`save_packed_checkpoint(model, path, code_encoding="rans-rows-v1")` stores code
streams with the row-wise rANS coder of `rotquant.lossless` instead of `int32`
words. The default, `code_encoding=None`, writes the artifact described above,
unchanged.

**This shrinks the file only.** `load_packed_model` decodes every stream back
to the identical fixed-rate `PackedTensor` (same `int32` words, shape, bits and
numel), so the loaded model, its resident memory, its bits/weight accounting
and its runtime are exactly those of a model saved without the option. GGUF and
native exports are unaffected: they are built from the loaded fixed-rate model
and their formats do not change.

### What is coded

Only the primary code stream (`qweight.packed`) of a backbone `QuantLinear`
whose codebook is scalar and whose code width is at most 14 bits (the coder's
probability resolution). It is split into one independent stream per output
row (`rows = out_features`, `columns = in_features`, the row-major order of the
fixed-rate stream), so a loader can decode rows in parallel. These stay
fixed-rate, exactly as in v2/v3:

- residual streams (`residual_packed`) and QJL sketches (`sketch`);
- vector-codebook indices;
- shared-vocabulary chunks (`tied_vocabulary`, v3);
- any primary stream whose coded form (table, row lengths and words) is not
  strictly smaller than its `int32` words. Each coded row costs eight bytes of
  overhead (the flushed 32-bit state and its word count), so rows of a few
  dozen codes stay fixed-rate.

The save result's `code_streams` entry reports how many streams were coded and
why the rest were not (`vector_codebook`, `code_bits_above_limit`,
`no_saving`).

### Versioning and compatibility

An artifact that stores at least one coded stream has `format_version: 4`
(safetensors metadata `rotquant_format: "4"`) and a top-level `code_encoding`
declaration. If no stream qualifies, the writer emits the ordinary v2 (or v3)
artifact, identical to a save without the option. A reader that predates v4
stops at its version check (`unsupported version 4; supported=(1, 2, 3)`)
instead of looking for words that are not in the file. Apart from the coded
streams, v4 is the v2 layout and may carry a v3 `tied_vocabulary`. The `packing`
block is unchanged: it describes the words the decoder reproduces.

The declaration must equal the following exactly; any other name or parameter
fails closed:

```json
"code_encoding": {
  "name": "rans-rows-v1", "version": 1,
  "probability_bits": 14, "state_bits": 32, "state_lower_bound": 65536,
  "renormalization_bits": 16, "row_split": "output_row", "max_code_bits": 14,
  "tensor_dtypes": {"frequencies": "int32", "row_words": "int32", "words": "int16"},
  "decoded_digest": "sha256_int32_le_words"
}
```

A coded `qweight.packed` entry has no `tensor` key:

| Field | Meaning |
|---|---|
| `encoding` | `"rans-rows-v1"`; must match the declaration |
| `shape`, `bits`, `numel` | logical code tensor, as for a fixed-rate stream |
| `rows`, `columns` | `out_features`, `in_features`; `rows * columns == numel` |
| `frequencies` | key of an `int32 [2**bits]` table; entries `>= 0`, sum `2**14` |
| `row_words` | key of an `int32 [rows]` vector: 16-bit words in each row stream |
| `words` | key of an `int16 [word_count]` vector: all row streams, concatenated |
| `word_count` | number of 16-bit words, `sum(row_words)` |
| `decoded_sha256` | SHA-256 of the decoded fixed-rate words as little-endian `int32` bytes |

The tensors are named `layer_NNNNN.packed.rans_frequencies`, `.rans_row_words`
and `.rans_words`. The words are the coder's `uint16` values stored as their
`int16` bit pattern, as fixed-rate words are `uint32` stored as `int32`.
safetensors has a `U16` dtype, but `int16` is readable by every PyTorch and
NumPy version that reads these files.

### Decoding

With `M = 2**14`, the table `f` and `c[s] = f[0] + ... + f[s-1]`, each row
stream is `[x_hi, x_lo, w_0, w_1, ...]` in consumption order:

1. `x = (x_hi << 16) | x_lo`; it must be at least `2**16`.
2. For each of `columns` codes: `slot = x mod M`; the code is the `s` with
   `c[s] <= slot < c[s] + f[s]`; `x = f[s] * (x >> 14) + slot - c[s]`; if
   `x < 2**16`, `x = (x << 16) | w_next`.
3. The row must end with every word consumed and `x == 2**16`.

The decoded codes are then packed with the generic LSB-first `int32` contract.
The loader also checks the dtypes, table length and sum, one word count of at
least two per row, `word_count`, codes below `2**bits`, the exact packed word
count and `decoded_sha256`. A damaged, truncated or padded stream raises
`ValueError` and nothing loads; the artifact's file hashes catch accidental
corruption before decoding starts. Before writing, the encoder decodes every
stream exactly as the loader will and refuses to publish one that does not
reproduce the deployed words.

### Size accounting

`artifact_bytes` is the sum of the actual file sizes. `code_streams` reports,
for the primary scalar streams, the fixed-rate bytes, the bytes actually stored
(the three coded tensors of each coded stream, the words of the others) and the
difference. `rotquant.validation.audit_artifact` counts tensor payload from the
safetensors headers, so it counts the coded bytes. In memory, `packed_bytes`
and `packed_state_bytes` describe the decoded fixed-rate streams, which is what
a loaded model holds.

Measured on real Qwen2.5-0.5B weights (first two decoder layers, FWHT 128,
groups of 128, Gaussian codebook, 29.8M codes; all 14 streams coded, codes and
logits bit-exact after reload). Savings are net of the eight bytes per row,
about 1.4% of the W4 code bytes for these shapes:

| Stored | Code bytes saved, `rms` scales | `mse_search` scales |
|---|---:|---:|
| W3 | 4.18% | 3.87% |
| W4 | 4.66% | 3.31% |
| W5 | 4.47% | |
| W6 | 4.07% | |

The coder is single-threaded NumPy. On a shared four-core CPU, decoding and
repacking ran at about 15M codes per second (67 ns per code). Loading those
two layers went from 3.2 s to 5.2 s (median of three); at that rate the 358M
backbone codes of the whole model would add about 24 s. Saving with encoding
and the pre-write check ran at about 5M codes per second.

### Tools that read words directly

Loading through `load_packed_model` or `rotquant.checkpoint._read_quantized_weight`
(used by `scripts/export_rotquant_gguf.py` and `scripts/verify_rotquant_gguf.py`)
decodes transparently, and so does `rotquant.gguf_v2.checkpoint_matrix`
(`scripts/export_rotquant_gguf_v2.py`), which reads code streams through the
same reader. The exported GGUF always holds fixed-rate words.
