# Qwen3.5-4B dynamic mixed-precision runbook

The generated Colab notebook is
`notebooks/qwen35_4b_dynamic_mixed_precision_colab.ipynb`. Run it from top to
bottom on an A100 40 GB or larger after setting `REPO_REF` to the exact merged
commit. Its default settings run the complete experiment; no stage toggle needs
to be changed for the first run.

## What the run does

1. Replicate promoted W4, learned butterfly signs with fp32 angles, and learned
   signs with fp16 angles at seeds 0/1/2.
2. Screen seven seed-0 controls/candidates: source FP16, uniform W3, uniform
   scale8 W4, random mixed FWHT, and sensitivity-allocated mixed precision with
   no rotation, FWHT, or learned signs.
3. Select at most two mixed candidates using pre-registered quality guards.
   Exact post-pack bytes must be within 1% of the 3,584,533,344-byte Unsloth
   complete bundle, and a candidate must show at least one material signal.
4. Confirm the uniform control, random mixed control, and selected candidates at
   seeds 0/1/2. Seed-0 finalists are exported as reloadable packed artifacts.
5. Rerun the pinned Unsloth UD-Q4_K_XL comparator on the same 24 × 512-token C4
   prompts and refuse cross-system comparison if token hashes differ.
6. Validate completeness and package the compact Drive result directory for
   download.

Every child process uses unbuffered output. The notebook prints runner progress,
30-second notebook heartbeats, 60-second experiment heartbeats, GPU utilisation,
VRAM, and power, while duplicating output to phase-specific files under
`logs/`. C4 tokenized inputs are cached under the Drive result root, so a fresh
Colab process does not need to rescan the streaming dataset. Within one stage
process, the random-mixed and greedy FWHT arms also reuse the same
content-addressed candidate score table instead of repeating all 1,200
layer/bit measurements.

## Resumption

Leave `FORCE_RERUN = False`. Reopening the notebook at the same commit and result
root causes completed arm/seed pairs to resume from their JSON records. An arm
requested for export is rerun only if its packed artifact is incomplete. Set
`FORCE_RERUN = True` only when intentionally replacing records produced by the
same code/config fingerprint.

The durable checkpoints are:

- `phase_summaries/sign_replication.json`;
- `phase_summaries/dynamic_screen.json`;
- `dynamic_finalists.json`;
- `phase_summaries/dynamic_confirm.json`;
- `unsloth_kl/unsloth_ud_q4_kl.json`;
- `rotquant_vs_unsloth.json`; and
- `dynamic_experiment_validation.json`.

## How to interpret it

The random mixed arm is the critical allocation control: mixed precision has
not helped algorithmically unless the sensitivity-guided recipe beats it at the
same bytes. The uniform scale8 W4 arm remains the deployment control. Seed-0
selection is only a screening decision; promotion requires consistent paired
evidence across seeds 1/2.

The 24-prompt C4 evaluation contains more than 10,000 scored next-token
distributions. The additional 25 prompts cover agentic instructions, code,
maths, multilingual inputs, and long documents, and include 32-token greedy
trajectory comparison. They are authored development diagnostics, not the
registered licensed 300-prompt suite. A successful run can select the next
RotQuant recipe, but cannot by itself support a public Dynamic 3.0 parity claim.

## Completed outcome

Revision `3dbae035f0d8bb603d57bb193cbfa0887e331528` completed this
runbook. Learned signs showed small, inconsistent KL/PPL gains but regressed
the trajectory guard and increased bytes, so they are shelved. More
importantly, all three sensitivity-allocated recipes failed selection: the
FWHT recipe's mean KL was 0.1381 versus 0.1155 for the matched random recipe
and 0.01665 for uniform scale8 W4. No mixed finalist advanced.

Do not rerun this allocator as a promotion experiment. Its useful role is now
historical negative-control evidence. The successor is
`notebooks/qwen35_4b_allocator_v2_colab.ipynb`, documented in
`docs/dynamic_allocator_v2.md`; it scores the actual deployed quantizer and
uses a constrained Pareto allocator with proxy audits and resumable candidate
checkpoints.

## Local structural validation

The A100 work cannot be reproduced on a CPU laptop, but the complete plan and
notebook are checked with:

```bash
uv run python scripts/build_qwen35_dynamic_mixed_notebook.py
uv run python scripts/run_qwen35_next_stage.py \
  --output-dir /tmp/rotquant-dynamic-plan --stage signs --seed 0 --dry-run
uv run python scripts/run_qwen35_next_stage.py \
  --output-dir /tmp/rotquant-dynamic-plan --stage dynamic --seed 0 --dry-run
uv run pytest -q
```
