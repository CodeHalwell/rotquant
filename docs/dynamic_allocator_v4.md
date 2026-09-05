# Dynamic allocator v4: format-aware rate-distortion optimization

## Decision inherited from allocator v3

Allocator v3 established a real but incomplete result on Qwen3.5-4B. At the
same internal byte budget, its global Pareto recipe reduced three-seed mean KL
by 72.8% versus broad random allocation and improved top-1 agreement by 7.75
percentage points. Its seed-0 export was within 0.087% of the pinned Unsloth
artifact. It nevertheless remained 2.62x worse on mean KL and 3.31 points
behind on top-1 agreement.

The winning recipes used only W3, W4, and W5. Pair refinement made no change,
while hard W6 and W8 islands reduced fidelity because the fixed byte budget
forced damaging downgrades elsewhere. The compact evidence is versioned in
[`research/results/qwen35_4b_allocator_v3_c9efd2d56774.json`](../research/results/qwen35_4b_allocator_v3_c9efd2d56774.json).

## What v4 changes

The allocator previously chose only a bit width. V4 can choose a complete,
named `QuantConfig` per projection. A candidate identity includes its name and
all deployed quantizer fields, so same-width formats remain distinct through:

- candidate-score caching and interrupted-run resume;
- exact byte accounting and Pareto selection;
- seeded same-palette random controls;
- pair-exchange refinement;
- allocation fingerprints and exported diagnostics.

`allocation_formats` restricts a full scored palette at allocation time. It is
excluded from the candidate-cache key, allowing causal ablations to reuse one
expensive table rather than requantizing every projection.

The optimization remains a multiple-choice rate-distortion problem. For layer
`l` and deployable format `f`, the measured cost is

```text
D(l,f) = 0.25 * normalized_relative_reconstruction_error(l,f)
       + 1.00 * normalized_marginal_teacher_KL(l,f)
```

subject to one format per layer and a complete-artifact byte interval around
3,584,533,344 bytes. Whole-model held-out KL and trajectories remain the
authority because the additive objective cannot model cross-layer error
interactions.

## Bounded candidate palette

The registered palette contains six points:

| ID | Bits | Codebook | Group size | Purpose |
|---|---:|---|---:|---|
| `gaussian_w3_g128` | 3 | Gaussian | 128 | v3 low-rate anchor |
| `calibrated_w3_g128` | 3 | per-matrix Lloyd | 128 | learned codebook test |
| `gaussian_w3_g64` | 3 | Gaussian | 64 | finer-scale test |
| `gaussian_w4_g128` | 4 | Gaussian | 128 | promoted W4 primitive |
| `calibrated_w4_g128` | 4 | per-matrix Lloyd | 128 | W4 challenger |
| `gaussian_w5_g128` | 5 | Gaussian | 128 | v3 high-rate endpoint |

All candidates use FWHT, real uint8 scale metadata, MSE-search, act-order GPTQ,
and the same calibration manifests. W2, W6, and W8 were removed based on v3's
observed allocation, not assumed to be universally useless. Vector, spherical,
TurboQuant-scale, learned-sign, and forced high-precision variants remain out
because earlier controlled experiments rejected them or failed to establish a
benefit.

## Registered experiment

The generated
[`allocator-v4 Colab`](../notebooks/qwen35_4b_allocator_v4_colab.ipynb)
screens seven seed-0 arms:

1. source FP16;
2. uniform Gaussian W4;
3. a re-run of the v3 bits-only Pareto policy under the corrected solver;
4. exact-byte random allocation over the new format palette;
5. the full format-aware Pareto allocator;
6. a Gaussian-only allocation that isolates group-64 scaling; and
7. a group-128 allocation that isolates calibrated codebooks.

At most two distinct format recipes proceed to seeds 1 and 2. Confirmation
includes uniform W4, bits-only Pareto, and format-random controls. A recipe is
promoted only if it improves KL over bits-only in every seed, has a paired
prompt-bootstrap KL interval below zero in every seed, beats format-random,
wins a secondary metric consistently, and passes the real exported-byte gate.
Every seed must also retain all screening no-regression guards, and paired
intervals require at least 20 samples plus the runner's reliability flag.
The export gate requires a measured seed-0 artifact bound to the same revision,
trial, allocation fingerprint and manifest digest; estimated tensor bytes cannot
pass. Reports explicitly identify exported seeds.
Provider competitiveness separately requires no worse KL and top-1 than the
pinned prompt-matched Unsloth anchor.

## Operational notes

Use an A100 40 GB or larger and run the notebook top to bottom. The preflight
checks all six formats against the deployed patcher on the actual GPU before
any full-model calibration. Building the candidate table is the expensive
section; it checkpoints every eight
projections, prints per-layer progress, emits 30/60-second GPU heartbeats, and
resumes from Drive. Allocation-only ablations reuse that table. The existing
v2/v3 raw-score cache can also satisfy the unchanged bits-only scoring step;
allocations and held-out results are recomputed under the corrected solver.

This remains a development experiment. The 24-prompt and 25-prompt suites can
select the next recipe, but they cannot support a public comparison. A winner
must still enter the registered engine-neutral 300-prompt/32-token evaluation.
