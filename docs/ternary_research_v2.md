# Ternary follow-up: cached decoding, complete turns and inference

Use [the v2 Colab notebook](https://colab.research.google.com/drive/1k2YhEqZoSVLsEF8Z_IfaXpBhlduubRfA), generated from `notebooks/qwen35_4b_ternary_research_v2_colab.ipynb`. The completed v1 notebook and its result roots are retained. The v2 notebook defaults to `STAGE = "parity"`; it does not claim new quality or speed measurements.

The historical input is `/content/drive/MyDrive/rotquant/ternary_research_v1/f1561b8f83766a73-s0`. Every retained payload is checked against its original stage SHA256 receipt, recursively through the parent roots. Historical table values must equal the verified preparation/export/measurement receipts. The child writes only to a new v2 directory. It builds the current runtime; old binary timings stay in the historical table.

## Execution order

1. Verify the historical chain and source/model/data pins; build and test the current native runtime.
2. Freeze private, non-thinking chat diagnostics with source teacher continuations. Compare each retained FP16, W4, W5/W6, W6 and ternary checkpoint in full recomputation, cached teacher forcing and chunk16 prefill, along source-teacher and candidate trajectories. Save errors, raw top-1/tie sets, margins and EOS probability/rank at every step. A failure blocks training. Numerical thresholds remain max error 0.125, mean error 0.005, KL 0.0001 and exact maximum-set agreement 0.99.
3. Freeze complete source-teacher assistant turns ending in a true stop token. Reject cap-truncated teacher responses. Hash distinct train, validation and behavioural-selection prompts. Mix 512/1024/2048 contexts and ordinary immutable C4 fitting windows. Prompt/padding positions are excluded from chat KL/CE; real ordinary-text positions remain supervised. Public benchmark labels never enter teacher fitting or selection.
4. Compare H-aware legal ternary initialisation and block QAT with retained selected latent states. Then run whole-model hard ternary QAT, followed by fixed-assignment scale recovery on the exact selected packed parent. Global training uses teacher top-k UNION response/EOS/format IDs and the exact remaining probability mass. A separate dense validation example audits unresolved tail KL at validation checkpoints.
5. Accept a global/recovery checkpoint only when validation improves, completion/format clear their explicit absolute thresholds and do not regress, and repetition clears its threshold and does not regress. The default absolute gates are completion >=0.9, format >=0.9 and repetition <=0.05. Otherwise restore the baseline and report `best_step=0`; this is not a training gain. Latent/optimizer artifacts remain outside deployed checkpoints.
6. Reload and export each candidate, repeat cached chat parity, then measure held-out quality, the original public tasks, complete model bytes, prefill/decode throughput and answer latency. Each model/runtime pair has its own row and dispatch receipt.

## Runtime profiles

The reference profile retains the original arithmetic. Opt-in exact-operand profiles include decode4, tile8, a warp vocabulary head and specialised group128 W2A16. Graph prefill prunes the vocabulary head only on intermediate no-output batches while retaining recurrent and attention state. Compact greedy sampling checks all logits for finite values on device, keeps lowest-index tie semantics and returns a token rather than a vocabulary vector.

Staged GEMM and fused Tensor Core paths are separately identified because they reorder accumulation. W2A8 is a new numerical lane: signed FWHT128 with the declared half-rounding boundaries, per-group Q8 round-to-even activation quantisation, and integer DP4A dots. It has a dedicated grouped PyTorch/NumPy oracle. It must pass its own operator and cached model gates and receive new quality measurements; a successful W2A16 receipt does not validate it. CUDA device counters record the formats and kernels that actually execute, including graph replays.

Only configured profiles are measured. Defaults are reference, decode4 and ternary-g128. Set `EXPERIMENTAL_PROFILES = ["w2a8", "tensorcore-fused"]` before the first run to include the two additional research profiles; each must pass its own gates. The configuration states the exact profiles and objective. Enabling additional profiles changes the study identity and needs a new child root. No new throughput or capability improvement is established until the notebook produces the corresponding receipts.

## Run from a source bundle

Upload the reviewed source ZIP through the notebook's existing upload cell. It verifies all source checksums and protects Colab's installed CUDA Torch. Keep `HISTORICAL_ROOT` on the completed v1 directory and select A100 80 GB/high RAM.

The command-line equivalent is:

```bash
python -B -m scripts.run_qwen35_ternary_research_v2 \
  --config configs/qwen35_4b_ternary_research_v2_cuda.yaml \
  --historical-root /content/drive/MyDrive/rotquant/ternary_research_v1/f1561b8f83766a73-s0 \
  --output-dir /content/drive/MyDrive/rotquant/ternary_research_v2/NEW_RUN \
  --work-dir /content/rotquant-ternary-v2-work \
  --budget-minutes 480 --stage parity
```

Inspect `comparison.json`, `workflow.json`, and the `cache-reference-*` / `cache-native-*` stage reports. Then use `--stage full` with the same source, config, child root and allowance. Resuming does not renew active or wall time. The full matrix can exceed one allowance; completed stages and partial reports remain available. The built-in 27 private prompts are a reproducible pilot. A larger prompts-only JSONL can be pinned before starting a new study; it must not contain public task answers or labels.

## Reading the comparison

`historical_rows` contains the nine original measured rows with original origins/backends. `rows` contains fresh `model@profile` identities. Missing values stay empty; pending, rejected and failed stages are visible. `parity_complete` clears correctness diagnostics only. `complete` requires every requested fresh comparison row to finish. Complete native payload bytes include retained auxiliary or vision files; language-only GGUF bytes and checkpoint bytes are separate.

The notebook displays historical and fresh tables separately, including completion/repetition/format, selected training step, quality, complete bytes and speed. Fixed-token decode throughput is distinct from latency to a completed answer. Bonsai remains its pinned Prism model/runtime, while FP16 and RotQuant use the current gated runtime.
