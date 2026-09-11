# Native GPU performance pilot — 11 September 2026

## Run this next

**Superseded as the next experiment by the [native optimisation study](native_gpu_optimization.md).**
The [original pilot results](../research/results/native_pilot_2026_09_11/README.md)
are archived, including the partial 2048 run. This simpler reference-only
notebook remains usable; `CONTEXTS` now permits targeted reruns.

Open [the new Colab notebook](../notebooks/qwen35_4b_native_gpu_pilot_colab.ipynb)
after its supporting code has been published. Use a fresh A100 40GB session,
leave W5/W6 selected, verify the source path, then **Runtime → Run all**.
The original correctness notebook is unchanged. Local `gws` authentication is
not required: this notebook uses the normal Colab Drive mount.

The [first successful native CUDA run](../research/results/native_cuda_2026_09_10/README.md)
established bounded saved-model parity. It did **not** measure speed, and its
reports-only ZIP cannot restore binaries or exports after the VM is deleted.
Expect one more build before the new persistent cache can save future builds.

| Control | Default |
|---|---|
| Recipe | `b5_v6_s0`: unchanged W5/scale8 backbone, W6 vocabulary |
| Source | `MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f` |
| Private artifact cache | `MyDrive/rotquant/native_artifact_cache/v1` |
| Results | `MyDrive/rotquant/native_gpu_pilot/<commit>/pilot1` |
| Input tokens | 128, 512, 2048; separate process/context per shape |
| Reserved context capacities | 256, 768, 2304 with default decode length |
| Cached steps | 32 per repetition, fixed count even at EOS |
| Repetitions | 1 excluded warmup + 3 measured |
| Execution allowance | 90 cumulative active minutes; 45-minute build cap |
| Pilot cap | Context/repetition-aware; defaults 4 / 6 / 12 minutes, including load/verification |
| Cost stops | Below 2 measured decode tok/s, or sampled process VRAM above 16,384 MiB |

The speed and VRAM limits are adjustable **spending guards**, not scientific
promotion criteria. A slow first measured repetition stops larger contexts;
the partial result and independent parity pass remain available. Never loosen
numerical thresholds to get a timing pass. If a launch stops, run the Results
cell to see partial evidence and the archive path. No cap stops GPU billing:
**disconnect and delete the runtime when finished or blocked**.

## What gets reused

`scripts/native_gpu_cache.py` persists two private, immutable artifact types:

- Runtime: native libraries (including SONAME aliases copied as regular files)
  and original build receipt. The request key covers compiled repository inputs,
  the pinned llama.cpp base/patch, GPU name/compute capability, driver, libc,
  architecture, CUDA toolkit, C++ compiler, CMake, Torch/CUDA and build-affecting
  environment settings. A different identity causes a miss, not guessed ABI
  compatibility. Source-only/document changes outside these inputs do not force
  a new build. Persistent binary reuse currently supports Linux CUDA only.
- Export: full text GGUF and auxiliary sidecar plus export receipt, keyed by
  verified checkpoint manifest and exporter/format source hashes. Neither
  quantization nor rotation learning runs. The exporter and converter Python
  code come from the pinned checkout, never an executable cache payload.

Every file has a byte count and SHA-256. Incomplete copies remain under a staging
name; only verified entries acquire the final key. Restore verifies the complete
inventory, rejects symlinks/traversal/missing/unexpected files, and rehashes local
copies. A corrupt cache is preserved and rejected, not executed or silently
overwritten. Use only a private cache created by this workflow: hashes are
integrity checks, **not authentication against a malicious cache publisher**.
Do not run two sessions against the same result/cache directory simultaneously.

Libraries restore to a fresh local directory. `LD_LIBRARY_PATH` is scoped to
the driver/children so relocated ELF dependencies resolve to the checked files.
The environment and pinned llama.cpp source checkout are recreated/verified on
each new VM; the Python venv, build objects and prior GPU passes are not cached.
The Drive cache needs several GB per export plus native libraries. Old cache
keys/staging directories are never automatically pruned.

A cache hit establishes **build/export integrity only**, not correctness.
Every pilot invocation performs fresh binding, CPU/CUDA operators, synthetic
whole-model/conversion and retained saved-probe parity before timing. Completed
build/export stages may also resume locally when their artifact hashes match.
Changed controls require a new run name; elapsed active time is cumulative on
resume. Timing is deliberately rerun, not borrowed from an earlier VM.

## Measurement contract

`scripts/run_rq3_performance_pilot.py` requires a passed retained report bound to
the actual runtime, export bytes and saved probe hash. Each context loads one
native model, resets state before each repetition and uses repeated frozen token
IDs from the first saved prompt. These are shape fixtures, not a long-context
quality dataset. Fixed-length greedy stepping through EOS is intentional.

Prefill and cached-decode intervals are separate, synchronous wall-clock calls.
The current private bridge includes synchronization, finite-logit validation,
full-vocabulary host copies and Python argmax. Diagnostic checkpoint writes and
logging are outside the intervals. Reported rates therefore describe this
**instrumented bridge's call times**, not pure GPU kernel time or total job wall
time. Decode rate is total measured steps / total measured seconds, not the mean
of per-repetition rates. Warmup is excluded; observed min/max are not confidence
intervals. Three repetitions are a screening pilot, not a publication benchmark.

Rows and in-progress decode timings are persisted after each step. VRAM uses
0.5-second `nvidia-smi` samples for the current PID, including load and execution
of that context; it can miss transient peaks. GPU memory unavailability prevents
a complete CUDA pilot pass. A VRAM limit is checked between model calls, not a
hardware allocation reservation; OOM/timeout remains possible inside a call.
Non-text tensors remain in the counted sidecar, not in text inference.

No native kernel, saved checkpoint, numerical threshold or old result is
changed. There is no FP16/Unsloth speed ratio, task accuracy, native vision,
DRAM-traffic or production-serving claim in this run.

## Validation and next decision

The notebook's schema/syntax and all code cells are tested top-to-bottom under
explicit Colab/network/hardware mocks. Cache tests exercise fresh-directory
restores, immutable entries, SONAME copies, corruption and invalidation. Timing
tests independently check token/time denominators, warmup exclusion, partial
progress and cost stops. The actual timing loop also ran on the existing tiny
two-layer Metal fixture at all three input lengths, with 8 steps and 2 measured
repetitions. This is local mechanics validation, not CUDA/4B speed evidence.

An HTML notebook preview was generated successfully, but the local browser URL
policy blocked opening it. Visual presentation inspection remains pending;
schema, cell syntax and top-to-bottom mocked execution have been checked.
Final local regression: **881 passed, 17 existing arm64 AVX2 skips**; two
existing SWIG warnings. Repository lint and whitespace checks pass.

Fresh Linux/CUDA cache relocation and this complete 4B pilot still require the
Colab execution. No cloud GPU was launched during preparation. To validate:
publish this code, open the notebook on A100, confirm the original Drive files,
then Run all. Share its reports ZIP; keep the private artifact cache on Drive.

If speed or memory is unsuitable, profile bridge copies/validation and packed
operator hot spots before paying for broad tasks. If acceptable, test retained
W5/W8 plus longer-context parity, then matched native FP16/BF16 and pinned
Unsloth baselines. Only then reopen a bounded public-task subset and the full
Unsloth format frontier. No new allocator or 27B run is scheduled here.
