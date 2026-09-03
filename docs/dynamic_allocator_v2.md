# Dynamic allocator v2

## Decision

RotQuant's first weight mixed-precision allocator is retired as a promotion
candidate. On Qwen3.5-4B it optimized an RMS/no-GPTQ reconstruction proxy and
then deployed MSE-search/GPTQ weights. The resulting FWHT recipe had mean
teacher KL 0.1381, worse than the matched random recipe at 0.1155 and far worse
than uniform W4 at 0.01665.

Allocator v2 makes the search faithful, auditable, model-adapter driven, and
fail-closed. Mixed precision remains a static deployment recipe; precision does
not change token by token at runtime.

## Objective

For projection `i` and candidate format `b`, the search measures:

1. Complete persistent bytes `B_i(b)`, including codes, scales, codebooks,
   bias, and rotation state.
2. Activation-weighted summed output distortion

   ```text
   E_i(b) = mean_x ||x R_i^T (W_hat_i(b) - W_i R_i^T)^T||_2^2
   ```

3. Relative output distortion

   ```text
   Erel_i(b) = E_i(b) / mean_x ||x W_i^T||_2^2
   ```

4. Marginal source-teacher KL `K_i(b)`, measured by installing only that
   candidate in the otherwise full-precision model.

The local and KL penalties are each shifted by their per-layer minimum and
divided by their robust median positive penalty. User weights therefore combine
dimensionless quantities rather than raw errors with unrelated scales:

```text
D_i(b) = alpha * normalized(Erel_i(b))
       + beta  * normalized(K_i(b))
```

The allocator solves the multiple-choice rate-distortion problem

```text
minimize    sum_i D_i(b_i)
subject to  B_fixed + sum_i B_i(b_i) within the registered byte interval.
```

This additive objective is still an approximation: errors from multiple layers
interact. Consequently, it proposes recipes; held-out whole-model KL,
perplexity, and trajectory evaluation decides promotion.

## Faithful candidate construction

By default, candidate scoring inherits the deployed quantizer's scale search
and error compensation. For the Qwen3.5 experiment this means:

- the same Gaussian codebook and group size;
- the same 8-bit stored scales and MSE scale search;
- the same GPTQ block size, activation ordering, and scale recomputation;
- the same streamed source Hessian, rotated into the deployed FWHT basis;
- the same bias correction, when enabled; and
- the same persistent-byte accounting used after patching.

`scoring_error_comp: none` and `scoring_scale: rms` remain available only as
explicit negative controls. `candidate_scoring_matches_deployed` records whether
the faithful path was used.

## Solver

`allocation: pareto` uses a bucketed multiple-choice knapsack dynamic program.
It starts from the highest allowed candidate for every projection, tracks exact
byte savings and additive distortion, and bounds memory by indexing states at a
configurable byte granularity. Final feasibility is checked with exact bytes,
not bucket estimates. Tiny models automatically use their exact storage gcd.

The former greedy adjacent-downgrade solver remains for ablation. A seeded
random adjacent-downgrade solver remains the required matched negative control.

Candidate measurements are stored in a content-addressed JSON cache. The
allocator checkpoints the partial table every `score_checkpoint_interval`
new projections (eight by default), validates the candidate widths on reload,
and resumes only the missing projections. Allocation-only changes reuse the
completed table, so trying a new byte target or protection policy does not
repeat MSE-search, GPTQ, or marginal-KL scoring.

## Safety mechanisms

### Adjacent-bit policy

`allocation_min_bits` and `allocation_max_bits` constrain allocation without
changing the candidate screen. A conservative W3/W4 policy and a broad 2--8-bit
policy can therefore reuse one expensive score table.

### Measured protection

`protect_top_fraction` keeps the most sensitive measured projections at or
above `protect_min_bits`. Sensitivity can use marginal KL, relative local error,
or the combined score. This is model-independent; it does not assume that a
particular layer number or projection suffix is always sensitive. Explicit
glob rules remain available for architecture-specific requirements.

### Proxy audit

The diagnostics report:

- local and marginal-KL monotonicity violation rates across adjacent bit widths;
- Spearman rank correlation between their downgrade penalties;
- robust score scales;
- protected projections and their measured sensitivities; and
- every candidate's raw, normalized, eligibility, selection, and byte fields.

`min_proxy_rank_correlation` can abort allocation when the local proxy and
marginal model behavior disagree. It is an audit gate, not a way to force a bad
correlation to look useful.

## Model support

Target discovery now uses RotQuant's model adapter registry rather than walking
only `nn.Linear` instances directly. Adapters own projection discovery,
conversion to an equivalent linear operation, and installation of a temporary
quantized candidate. This keeps the allocator reusable across dense decoders,
MoE families, recurrent/linear-attention hybrids, and multimodal backbones as
their adapters mature.

## Qwen3.5-4B experiment

`configs/qwen35_4b_allocator_v2_cuda.yaml` registers one faithful broad
2/3/4/5/6/8-bit screen under the complete Unsloth UD-Q4_K_XL byte target. The
following allocation policies reuse it:

1. seeded random W3/W4;
2. Pareto W3/W4 using relative local error;
3. Pareto W3/W4 using local error plus marginal KL;
4. broad Pareto using local error;
5. broad Pareto using local error plus marginal KL; and
6. broad Pareto with the top 10% marginal-KL-sensitive projections protected at
   W4 or above.

Uniform W3, uniform W4, and source FP16 remain controls. Seed 0 screens policies.
A candidate must beat the exact-format random control on KL plus at least one
other registered metric without violating any guard. Only selected recipes and
the controls proceed to seeds 1 and 2.

## Promotion boundary

Allocator v2 is promoted only if it:

- lands within the registered complete-model byte tolerance;
- uses faithful candidate scoring;
- beats the matched random allocation across seeds;
- does not regress the held-out diverse suite or 32-token trajectories;
- improves the size-quality Pareto frontier relative to uniform RotQuant; and
- is reproduced through a packed, reloadable artifact in the same inference
  engine used for the provider comparison.

The 24-prompt internal suite is sufficient to steer development, not to publish
an Unsloth superiority claim. That still requires the registered licensed
300-prompt, engine-neutral evaluation.
