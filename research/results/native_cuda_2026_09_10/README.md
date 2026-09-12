# First retained Qwen3.5-4B native CUDA pass

**Ready within the reviewed correctness scope; not a serving-speed or task-quality claim.**

User-supplied Colab bundle: `run1-reports-1789072777350631702`, produced from
clean RotQuant revision `06a4379c71074e4606116c835941388d1876599b` on
10 September 2026. NVIDIA A100-SXM4-40GB; Python 3.13.15;
Torch 2.11.0+cu128; Transformers 5.9.0. The retained arm is `b5_v6_s0`:
200 W5/scale8 backbone matrices and one shared W6/scale16 vocabulary matrix,
fixed FWHT and saved GPTQ recipe. This is not uniform W4 or a new quantization.

## Results

| Check | Observed result | Meaning |
|---|---:|---|
| Pipeline stages | 11/11 passed | Includes native build/load, CPU/CUDA operators, conversion/export and retained parity |
| Retained top-1 agreement | 16/16 positions (100%) | Versus saved **quantized** model logits, not full precision |
| Retained generation | 4/4 exact traces | Four 64-token prompts, eight generated tokens each |
| Retained KL | 0.000008667400893 | KL(saved quantized reference ∥ native), not quantization loss to BF16/FP16 |
| Maximum absolute logit error | 0.0322265625 | Below unchanged 0.125 bound |
| Mean absolute logit error | 0.002836523567 | Below unchanged 0.005 bound |
| CPU and CUDA packed operators | 18 cases each passed | W5/scale8 and W6/W8/scale16; maximum recorded error 0.00048828125 on CUDA |
| Synthetic W6 and W8 Qwen graphs | Both passed | 1/4/17/64-token prompts with cached decoding; not a retained W5/W8 run |
| Offline conversion | 6/6 cases passed | CPU/CUDA at 4/17/64 tokens; random model/fake tokenizer |
| Active execution | 32.2359 minutes | 26.9064 build minutes, 5.3295 minutes for all other stages |
| Retained model load | 4.0722 seconds | One observed load, not repeated latency benchmark |
| Throughput | Not measured | `timing=false`; cannot derive tokens/s from stage duration |

The retained report records `cpu_fallback_forbidden=true`; the log says
`offloaded 33/33 layers to GPU`. The reviewed scheduler patch rejects non-view
compute nodes assigned to non-GPU backends when the runner's
`ROTQUANT_REQUIRE_GPU=1` is set. The custom-operation observer counted 16,080
callbacks during the probe run; that is **not** a hardware kernel-launch or
bandwidth counter. Host staging/output buffers do not imply CPU arithmetic.

The build's `gpu_validated=false` and export's `runtime_validated=false` are
immutable **stage-local** statements, written before later conformance checks.
They are not contradictory failures. The later reports bind the successful
checks to the same library, checkpoint, saved-probe and export hashes. Do not
rewrite earlier receipts to say they independently validated execution.

## Size and memory: keep the denominators separate

| Quantity | Recorded value | Scope |
|---|---:|---|
| Native text GGUF | 2,768,298,560 bytes (2.7683 GB) | Text weights, packed metadata and tokenizer |
| Non-text sidecar | 667,061,152 bytes (0.6671 GB) | Preserved auxiliary tensors; no working native vision tower |
| Combined model payload | 3,435,359,712 bytes (3.4354 GB / 3.1994 GiB) | Sum of GGUF and sidecar; excludes the export receipt itself |
| CUDA model buffer | 2,629.59 MiB | Runtime log; not complete process VRAM |
| CUDA KV / recurrent buffers | 8.00 / 50.25 MiB | This reserved context and model state |
| CUDA compute buffer | 252.63 MiB | Reported reservation, not a full allocation trace |
| Sampled peak process VRAM | 3,446 MiB (3.3652 GiB) | 38 nvidia-smi samples, 0.5-second polling |

VRAM was observed with **batch 1, context capacity 256**, four 64-token input
probes, and FP16 cache. The non-text sidecar is not loaded for text inference.
The sample may miss transient peaks; no long-context/batching/minimum-GPU-size
or DRAM-traffic conclusion follows. Disk payload, device buffers, process VRAM
and complete multimodal residency are not interchangeable sizes.

## Integrity review and archive

`original-reports.tar.gz` preserves all **38 original files** byte-for-byte;
its SHA-256 is
`22c29a0e4c1f9a173adca717c85b26a303234a8a088536c7f8b588542bc80715`.
An in-memory archive round-trip matched every original file's length/hash.
The original download folder is untouched. No model weights are in this archive.

[`audit.json`](audit.json) is the output of [`audit.py`](audit.py). It verifies
11 available stage receipt hashes, 25 recorded producer-code hashes against
the producer's Git objects at `06a4379c7107` (not the evolving working tree),
28 recorded patched-file hashes against
the maintained contract, common runtime identity across seven result/load
reports, source/probe/export bindings, recipe bit widths, reported guard
arithmetic, payload-byte reconciliation and elapsed-time totals. The 38
external artifact paths absent from the reports-only bundle are expected
cache/export files, not proof that those files still exist on Colab.

To repeat the review, extract the original reports into a new directory and run:

```bash
python research/results/native_cuda_2026_09_10/audit.py /path/to/extracted/reports
```

The recorded producer commit must be present in local Git history; the audit
does not fetch code or execute historical scripts. Its report is unchanged by
subsequent notebook/workflow improvements.

**Verification limits:** the archive excludes original/native probe tensors,
weights, exported GGUF and native binaries. The raw KL/logit values and traces
cannot be independently recomputed locally, nor can absent binary hashes be
rehash-verified. This is a consistency/provenance review of user-run evidence,
not a second execution of the GPU experiment. No new GPU job was launched.

## Next experiments, in order

1. Add content-hashed persistence/reuse for the validated native runtime and
   lossless export, with build/toolchain/GPU compatibility checks and a fresh
   binding/parity check on restore. Logs alone cannot restore ephemeral binaries.
   Preserve the original checkpoint and all historical receipts. If this Colab
   VM was deleted, expect one rebuild/export before a reusable cache exists.
2. Run a **bounded W5/W6 performance pilot**: batch 1, contexts 128/512/2048,
   warmup separated, prefill and cached decode measured separately, process
   VRAM sampled and abort caps enforced. Existing optional timing supports
   these shapes, but 16 decode tokens and two measured repetitions are a pilot,
   not sufficient for a polished benchmark. Extend repetitions/tokens only if
   the measured pilot is affordable; do not infer speed from the parity stage.
3. Test the retained W5/W8 checkpoint, broader/longer native parity and cache
   behavior, then profile steady-state CPU-to-GPU copies, allocations and DRAM
   traffic. W8 synthetic success does not substitute for saved-model parity.
4. Establish comparable native FP16/BF16 and pinned Unsloth baselines with
   matched frozen token IDs, tokenizer/stop semantics, hardware, context/cache
   settings and complete byte accounting. Then restart a small runtime-bound
   public-task subset with cost/quality gates before the full frontier sweep.

No allocator, recovery, learned-rotation or 27B sweep is promoted by this run.
The prior quality evidence remains separate; this result removes an execution
blocker, not the remaining performance and broader-evaluation gates.
