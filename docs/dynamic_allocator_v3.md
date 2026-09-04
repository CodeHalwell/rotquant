# Dynamic allocator v3

## Why this stage exists

Allocator v2 established that faithful, globally informed mixed-precision
allocation is useful: its three-seed mean teacher KL was 0.02587 versus 0.03453
for matched-size random W3/W4 allocation, a 25.1% reduction. It did not close
the gap to uniform W4 or Unsloth UD-Q4_K_XL.

The completed run also exposed three protocol defects:

1. the W4 protection floor was non-binding, so the two selected finalists had
   identical deployed bit assignments;
2. confirmation did not emit the required direct paired finalist-versus-random
   intervals; and
3. the allocator targeted registered model tensors rather than the complete
   serialized artifact, causing the exported finalist to exceed the 1% provider
   byte gate.

The compact evidence record is
[`research/results/qwen35_4b_allocator_v2_2dce3aa43029.json`](../research/results/qwen35_4b_allocator_v2_2dce3aa43029.json).

## New mechanisms

### Exported-artifact byte targeting

`target_artifact_bytes` is now a first-class allocator target. The search uses

```text
internal target = target_artifact_bytes - artifact_overhead_bytes
```

and reports both estimated registered-model bytes and estimated serialized
artifact bytes. The measured seed-0 v2 overhead was 20,441,655 bytes. The v3
Qwen configuration reserves a conservative 20,750,000 bytes and tightens the
search interval from 1% to 0.1%. The actual exported artifact—not the
estimate—remains authoritative for the final 1% comparison gate.

### Exact broad random control

`allocation: random_pareto` uses the same 2/3/4/5/6/8-bit candidates, complete
byte accounting, constraints, and bucketed multiple-choice solver as the
measured allocator. It replaces sensitivity scores with deterministic seeded
random values. This isolates the value of the learned ranking without changing
the candidate palette or accepting an accidental byte miss.

### Pair-exchange refinement

The bounded Pareto solver deliberately retains only a small frontier per byte
bucket. `refinement_passes` performs deterministic single-layer and two-layer
exchanges after that solve. Every accepted exchange:

- stays inside the exact registered byte interval;
- strictly reduces the measured additive objective; and
- records the affected projections, before/after precisions, bytes, and score.

This repairs solver approximation error. It does not claim to model nonlinear
interactions between simultaneously quantized layers; held-out whole-model KL
and trajectory evaluation remain the authority.

### Allocation identity

Every deployed per-layer quantization recipe receives a SHA-256 allocation
fingerprint over projection names and complete quantizer configurations. The
v3 selector admits at most one finalist per fingerprint, preventing a
non-binding constraint from consuming another three-seed confirmation slot.

## Registered Qwen3.5-4B matrix

The generated
[`allocator-v3 Colab`](../notebooks/qwen35_4b_allocator_v3_colab.ipynb)
runs nine seed-0 arms:

1. source FP16;
2. uniform scale8 W4 quality ceiling;
3. exact-byte broad random allocation;
4. unprotected global Pareto allocation;
5. the same allocation with eight pair-exchange passes;
6. top 5% marginal-KL-sensitive projections protected at W6;
7. top 1% protected at W8;
8. top 2.5% protected at W8; and
9. top 5% protected at W8.

All measured arms use FWHT, Gaussian codebooks, 8-bit stored scales,
MSE-search, group size 128, and act-order GPTQ. The expensive 2--8-bit candidate
table is content-addressed and compatible with the allocator-v2 cache because
only allocation policy and byte targeting changed.

Up to three distinct seed-0 finalists advance to seeds 1 and 2. The runner emits
paired prompt/document bootstrap intervals against both the broad random
control and uniform W4 for every finalist and seed. Seed 0 of the random
control and every finalist is exported so the matched-size assumption is
checked against real serialized artifacts, not only the allocator estimate.

## Promotion gates

A v3 recipe is an internal allocator winner only when it:

- completes every registered seed without a halted evaluation;
- meets the artifact-target estimate at every seed;
- has a direct paired comparison against the broad random control at every
  seed;
- improves aggregate KL over random in every seed and has a paired KL interval
  below zero in every seed;
- improves at least one secondary registered metric in every seed; and
- produces a seed-0 exported artifact within 1% of the pinned Unsloth bundle.

Provider competitiveness additionally requires no worse mean KL and top-1
agreement than the Unsloth development anchor. Any public claim still requires
the licensed 300-prompt, 32-token, engine-neutral protocol.

## Running it

Use an A100 40 GB or larger, allow roughly 20 GB of free Drive space for the
worst-case four seed-0 exports, and run the notebook from top to bottom. By
default it reuses the allocator-v2 candidate-score cache and Unsloth result
from Google Drive when present. Progress is streamed to persistent per-phase
logs. The download cell creates a compact archive and intentionally leaves
multi-gigabyte model artifacts and BF16 reference arrays in Drive.
