# Hybrid-attention quantisation and learned-rotation follow-up

Recorded 9 September 2026. **Research note and proposed experiments, not new
RotQuant results.** Keep the [public-task gate](public_tasks_run_2026-09-09.md)
and one measured serving path ahead of another broad optimisation sweep.

## Main takeaway

Linear-attention projections may tolerate lower precision than conventional
attention projections in some settings, but this is a hypothesis to measure,
not a safe universal precision map. Architecture, tensor role, quantisation
format, calibration, context length and serving implementation all matter.

The practical objective remains the smallest deployable model within an
explicit quality-loss budget. Truly lossless weight compression reconstructs
the original bits. Matching task scores, a small KL divergence, or matching a
few generations does not establish losslessness. A statistically inconclusive
task difference also does not establish equivalence.

## External evidence and its limits

### Gated DeltaNet / NVFP4 preprint

Kozyrev and Maiboroda's [Why Gated DeltaNet Survives 4-Bit Quantization](https://arxiv.org/abs/2609.04098v1)
was submitted on 3 September 2026. In Qwen3.8-27B, the authors quantise all 496
backbone linear projections, including Gated DeltaNet (GDN) and full attention,
to NVFP4 W4A4. They report a five-task mean 0.52 percentage points below BF16,
with differences within their measured uncertainty; perplexity still worsens.

Their mechanism study attributes GDN robustness to small-block scaling,
gate nonlinearities damping perturbations, and delta-rule updates overwriting
old state errors. This is not evidence that every recurrent operator is robust.

Important limits from [the full paper](https://arxiv.org/html/2609.04098v1):

- One model family/size and one format; NVFP4 uses 16-element scale blocks.
- Full-attention projections were also quantised successfully: the work does
  not establish that full-attention quantisation necessarily causes large losses.
- Low-bit projections are not low-bit recurrent-state storage. Their controlled
  recurrence-noise experiment runs in FP32.
- A fused-GEMM scale mismatch had to be repaired before evaluation. Weight and
  KV-cache precision also interacted, motivating separately calibrated KV scales.

### Unsloth's Qwen3.5 GGUF observations

[Unsloth's Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
report sensitivity in attention tensors and especially `ssm_out`, which belongs
to the linear-attention path; aggressively quantising that projection produced
poor size/quality trade-offs in their experiments. The documented tensor sweep
includes Qwen3.5-35B-A3B, not our dense 4B model.

These findings need not contradict the NVFP4 result: they concern different
models, precisions, formats and protocols. Neither establishes our 4B ranking.
Treat architecture labels as useful experimental groupings, not instructions
to protect or compress a tensor without measuring it.

## Mapping this to our current model

The [archived Qwen3.5-4B configuration](../research/results/raw/qwen35_fresh_quality_d4292d6fdec6/replicas/b5_v6_s1/checkpoint/config.json)
contains 32 language-model blocks: 24 GDN/linear-attention and eight full-attention,
with full attention every fourth block. **75% of blocks is not 75% of weight
bytes.** MLPs exist in both block types and must be classified separately from
their attention/mixing projections.

Our [current validated candidates](fresh_quality_results_2026-09-09.md) use:

- Fixed randomized Hadamard rotations, Gaussian scalar codebooks and GPTQ.
- W5 targeted backbone weights, group size 128, and 8-bit backbone scales.
- One shared packed W6 or W8 embedding/output matrix.
- Retained higher-precision tensors, including the excluded small GDN
  `in_proj_a` and `in_proj_b` gate projections.

The complete artifacts are 3,441,544,638 and 3,600,470,206 bytes. Neither the
NVFP4 format nor its activation quantisation is present in these recipes.
Our exclusions are part of the baseline, not proof the gates require FP16.

Keep these four axes separate:

| Axis | What changes | What a passing test does not prove |
|---|---|---|
| Projection weights | Stored matrices producing Q/K/V, gates or outputs | Robustness to lower-precision activations or caches |
| Activations | Per-request intermediate tensors and GEMM inputs | Weight-only gains or native low-bit kernel speed |
| Full-attention KV cache | Stored keys/values for previous tokens | Robustness of recurrent-state quantisation |
| GDN recurrent state | Fixed-size state updated across tokens | Absence of long-context drift from quantised projections |

## Proposed experiment: attention-type sensitivity

### A. Isolate the effects first

Start from the saved W5/W6 recipe. Keep source revision, tokenizer, calibration,
MLP precision, vocabulary precision, rotation policy and high-precision
exclusions fixed. Do not change activation, KV or recurrent-state precision.
Rebuild changed projections from the pinned source weights; do not round the
already quantised W5 checkpoint into W4 and confound double quantisation.

| Arm | Eligible GDN projections | Full-attention projections | MLPs / vocabulary |
|---|---|---|---|
| Control | W5 | W5 | W5 / W6 |
| Lower GDN only | W4 | W5 | W5 / W6 |
| Lower full attention only | W5 | W4 | W5 / W6 |
| Lower both | W4 | W4 | W5 / W6 |

These are **isolation arms, not matched-byte competitors**. Record the parameter
and actual packed-byte totals for each family. Differences in tensor count or
size must not masquerade as intrinsic sensitivity. If a family is fragile,
separate its QKV, output and gate roles before assigning a blanket precision.
The currently excluded `a`/`b` gates can have a separate later experiment;
changing them in this first matrix would introduce another factor.

Use a fixed FP16 teacher and identical held-out inputs. Measure document-level
KL distributions, top-1 agreement, free-running trajectories and task outcomes.
Include short and long contexts, retrieval probes, and recurrent-state/output
drift by position; keep teacher and candidate windowing/cache policies identical.
Select on a development split, then confirm with untouched examples and seeds
0/1/2. Seeds over the same prompts are not independent datasets. Freeze margins,
sample counts and stopping rules before running; this note sets none implicitly.

### B. Test a real byte-budget decision

Only after that screen, compare architecture-aware allocation with uniform and
matched-size control recipes. Try protecting full-attention projections only
if their measured improvement justifies the bytes; retain a counter-hypothesis
that GDN output projections may need more protection instead. Account for MLPs,
shared vocabulary, scales, rotations and all retained components in actual
exported files. Match bytes, not the number of upgraded/downgraded layers.

Evaluate the assembled model: local error per saved byte is a ranking aid, not
an additive prediction of whole-model KL. Earlier forced W6/W8 islands worsened
our fixed-budget results when paying for them damaged other projections.

## Where learned rotations could interact

This remains a separate, controlled follow-up, not an extra switch in the first
attention-type screen. A different precision map changes the optimisation
problem, so the earlier W4 learned-rotation failures do not settle the W5/W6 case.
Equally, a stronger baseline may leave less error to recover.

[SpinQuant](https://arxiv.org/html/2405.16406v4#S4.SS3.SSS2)
provides precedent for combining learned rotations with GPTQ. Its main
compatibility ablation targets activation error during rotation learning and
then applies GPTQ to weights; this is not direct evidence for our weight-only
Gaussian/GPTQ recipe or a requirement to differentiate through GPTQ.

Our [layerwise trainer](../rotquant/train_rotation.py) currently removes GPTQ
from its training proxy, while the Hessian-based selection gate evaluates the
final GPTQ quantizer. Measure this mismatch rather than assuming more training
steps fix it. Before spending GPU time, audit and regression-test the rotation
storage, orthogonality, deployed-quantizer and reload issues identified in the
[project review](project_deep_dive_2026-09-08.md); this note does not claim those
issues have all been repaired.

A later fixed-precision comparison can isolate fixed rotations, learned backbone
rotations only, learned vocabulary rotations only, and joint learning. Preserve
the shared vocabulary owner and numerical rounding contract. Compare combined
and individual effects on one preregistered metric with paired uncertainty;
beating each individual arm alone is not proof of a super-additive interaction.
Report training budgets and extra artifact/runtime costs for every arm.

The useful compression target is **W4 approaching current W5 quality**, followed
by a controlled W3 test if warranted, not merely reducing W5 KL at any cost.
Learned rotations must either be safely absorbed where architecture permits or
have a bounded, measured execution cost. Do not assume they commute through
GDN gates/nonlinearities, or introduce repeated large rotation reads contrary
to the [memory-bandwidth requirement](roadmap.md#september-9-serving-requirement-minimize-memory-traffic).

## Decision and status

Retain the current checkpoints and frozen experiment results. Complete the
public-task gate and establish one measured packed serving path. Then revisit
the bounded family-sensitivity and rotation experiments with preregistered
quality/byte/runtime gates. No new quantisation run, kernel, notebook, release
claim or automatic precision-map change is made by this note.
