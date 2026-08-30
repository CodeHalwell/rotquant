# Selective KV retrieval

RotQuant can reuse vector-scoring ideas for the key side of attention without
turning the cache into a general vector database. During autoregressive decode,
the current query is the search query, past key vectors are the index, and token
positions identify the corresponding values.

The useful first design is:

1. Rotate the query into the stored key basis once.
2. Score it directly against packed, low-bit key codes.
3. Force a recent window and the configured attention-sink positions into the
   candidate set.
4. Fill the remaining candidate budget from the largest approximate key scores.
5. Gather and dequantize only those value rows, then perform softmax and the
   weighted value reduction over the candidates.

This still scans K in linear time, but K is compact and the scan is suitable for
nibble-LUT/SIMD kernels. It avoids reading and accumulating the complete V cache,
which is the more expensive payload. A later page-level index can first retrieve
key blocks and then score packed keys exactly inside those blocks.

`rotquant.kv_cache.retrieval_rotquant_decode` defines the reference semantics.
It is intentionally decode-only and currently reconstructs keys in Python;
production performance requires a fused packed-key score/top-k/value-gather
kernel. `kv_retrieval_metrics` reports:

- full-precision attention mass covered by the selected positions;
- relative attention-output MSE and cosine similarity;
- selected/value-read fraction;
- exact packed KV bytes.

The retrieval path changes attention semantics whenever fewer than all positions
are selected. It should therefore be enabled only at long contexts and only when
held-out teacher-KL, perplexity and attention-mass gates pass. Prefill should stay
dense. A safe deployment policy also needs a dense fallback for diffuse queries,
for example when the coarse-score margin or estimated selected softmax mass is
too small.

The next runtime experiment should compare three policies at the same K/V format:

- full packed-key scan plus all V rows;
- full packed-key scan plus retrieved V rows;
- page-centroid shortlist, packed-key rerank, then retrieved V rows.

The first comparison isolates the value-bandwidth saving. Only after it wins
should RotQuant take on an incremental ANN/page index per layer and KV head.

## Initial synthetic screen

On deterministic random tensors with 256 cached positions, four heads and head
dimension 128, retrieving 64 positions (25% of V rows) covered about 58.6% of
the dense softmax mass for a diffuse query and had relative output MSE 0.63. For
a deliberately peaked query, retrieving only 16 positions (6.25% of V rows)
covered 99.4% of the mass and had relative output MSE 0.0094.

This confirms both sides of the design: selective V retrieval can be excellent
for concentrated attention, but a fixed small candidate count is unsafe for
diffuse attention. A production path needs a confidence/mass estimator and
dense fallback; the vector index alone is not sufficient.
