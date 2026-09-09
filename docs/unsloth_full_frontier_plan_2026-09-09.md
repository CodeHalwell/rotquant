# Full Unsloth size–quality comparison

Status: comparison scope and release metadata recorded on September 9, 2026.
No new GPU evaluation has run. The current notebooks/runners still compare
against the pinned `UD-Q4_K_XL` artifact; the multi-artifact runner described
below is **pending implementation**. Do not alter the active public-task run.

## Coverage

The user wants every published quant size, not only Dynamic 4-bit. For each
target model, benchmark all current language-model quant variants at a frozen
repository revision, including standard and Dynamic releases and every suffix
within a nominal bit family. Keep weak/dominated results visible. This means
current released variants, not an unbounded comparison of every historical
replacement, rename or preview revision.

Metadata-only inspection of the two public repositories gives:

| Model | Dynamic variants | Standard variants | Total quants | Language-model file range, decimal GB |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 9 | 12 | 21 | 1.520–5.952 |
| Qwen3.8-27B | 21 | 3 | 24 | 6.192–31.458 |

Sources: [pinned 4B release](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523)
and [pinned 27B release](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502).
The [machine-readable inventory](../research/unsloth_gguf_inventory_2026-09-09.json)
records every file, exact bytes, repository revision and Hub-reported SHA-256.
These hashes still need verification against downloaded bytes. An `UD` filename
does not by itself establish that the artifact uses Dynamic 3.0.

The 4B release has no IQ1 variant in this snapshot. Do not equate coverage with
one hypothetical file at each integer from 1 to 8. BF16 is a reference, not a
quant competitor; its shards form one logical model. Projectors, MTP sidecars
and importance matrices are not additional language-model quant variants.
Refresh discovery explicitly before registering a future run, retaining this
snapshot and old results rather than silently replacing their identities.

## Fair accounting and controls

The user-supplied Qwen3.8-27B chart plots top-1 agreement against quant size
**with MTP removed**. It is a useful presentation target, not a substitute for
raw benchmark records or a compatible evaluation protocol. Reproduce that
accounting as a separately labelled view after auditing tensor bytes, alongside
the actual complete-artifact view. Add RotQuant's measured points; movement
toward the upper-left means less storage for at least as much agreement. Do not
overlay our 4B scores on the 27B curve or infer task accuracy from top-1 agreement.

- Plot exact language-model bytes **and** complete artifact bytes under one
  declared auxiliary policy. For the 4B comparison, retain the existing fixed
  F16 projector policy (672,423,616 bytes) so new points remain comparable with
  the current complete-artifact records. Report text-only runtime memory
  separately; an included projector need not be resident in text-only inference.
- Audit 27B tensor contents before freezing its policy: embedded versus separate
  MTP, vision components and retained high-precision tensors cannot be inferred
  from filenames. Disable speculative/MTP decoding for the common greedy
  quality protocol. Report actual delivered bytes; any decoder-only accounting
  that subtracts unused tensors must be separately labelled, not substituted for
  physical file size. Any rewritten artifact gets its own checksum and identity.
- Pin source and tokenizer provenance, engine/build, prompts, template, token
  IDs, stopping rules, thinking mode, activation/cache settings and evaluation
  precision. Check tokenizer alignment and the full-precision GGUF bridge before
  accepting cross-engine KL. Unknown source provenance blocks a clean
  quantization-only claim, even if a descriptive artifact comparison is possible.
- Preserve both common-teacher fidelity and within-engine full-precision drift
  controls. Reuse a teacher only when the complete protocol/cache key matches.
  Reuse old candidate records only when their artifacts and evaluation protocol
  match too; matching a filename is insufficient.
- A same-size pair requires <=1% byte mismatch. Also report smaller-and-better
  Pareto dominance and the smallest **measured** artifact meeting a predefined
  fidelity/task threshold. No interpolated quality claims or assumed 99%
  agreement at a size that has not been tested.

## Execution plan

1. **Finish the current public-task gate.** Preserve its existing pins, outputs
   and conclusions. Do not restart it to insert additional competitors midway.
2. **Build a manifest-driven 4B runner and resumable Colab.** Replace the
   single-provider constant at the new runner boundary with logical artifact
   records containing all required shards, bytes and checksums. Preflight each
   variant, then run all 21 against one frozen quality protocol and the saved
   RotQuant W5/W6 and W5/W8 candidates. No need to requantize those checkpoints.
3. **Measure all variants, not just finalists.** The common suite should include
   held-out teacher KL, top-1 agreement, perplexity, free-running agreement/
   divergence, and the same bounded public math/code-understanding/instruction
   tests for every variant. Freeze sample counts, token budgets, domain weights,
   timeouts and uncertainty estimates before scoring. Longer agentic and
   long-context follow-ups may prioritize the frontier, but label that subset
   explicitly and retain the complete common-suite table.
4. **Publish the whole curve and coverage ledger.** For each artifact show
   quality, task scores, complete bytes, peak VRAM and measured prefill/decode
   performance where supported. Every inventory entry must be completed,
   failed, unsupported, resource-blocked or pending, with a reason. Low quality
   is a result, not grounds for omission. No claim of full coverage while rows
   remain unmeasured; no claim of faster RotQuant from the current Python path.
5. **Transfer the validated machinery to 27B.** Freeze the optimizer and a fresh
   model-specific protocol, run a small smoke/anchor check, then cover all 24
   provider variants. Fit/offload regimes must be explicit: a large Q8 or BF16
   reference may not fit the intended deployment GPU. Quality collection with
   offload is separate from fully resident runtime comparison. The 27B run is
   a later, separately budgeted experiment, not an immediate GPU launch.

## Cost and observability requirements

The language-model quant files alone total about 60.20 GB for 4B and 414.17 GB
for 27B, before references/projectors. Download and evaluate one logical model
at a time, with an explicit user-controlled cache budget; do not require keeping
the whole release on Colab disk or loading competing models simultaneously.
Any cache eviction must be limited to verified runner-owned downloads, not
saved RotQuant artifacts or user data. Preserve durable scores and provenance.

Persist common teacher outputs once using bounded/sharded storage and matching
fingerprints. Released quants are fixed artifacts: three identical deterministic
reruns are not three quantization seeds. Use paired example-level uncertainty;
evaluate the existing independent RotQuant recipe seeds as such.

Print artifact index/count and name, phase, prompt/token progress, elapsed time,
heartbeat, log path and completion status. Resume at verified completed units,
save after each bounded unit and cap pathological generations. Persist raw
outputs, per-domain summaries, failures and a final all-variant table/plot.
The final question is: **which recipe gives the best measured quality at each
actual storage and runtime-memory budget?** Beating one Q4 checkpoint is one
point on that curve, not proof of beating Unsloth's entire release.
