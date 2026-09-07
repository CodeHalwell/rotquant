# Vocabulary-budget screen: results and decision

Reviewed 7 September 2026. The Colab screen completed 6 September at 22:48 UTC,
using implementation `ce6c8ec861a2597ee5d6df0466cac23ee339859b`, seed 0, A100 40 GB,
FP16 source execution and the pinned Qwen3.5-4B checkpoint.

The [original result archive](../research/results/raw/qwen35_vocabulary_budget_ce6c8ec861a2/)
contains all nine result records and companions, source/runtime metadata, resource
ledgers, preflight and saved selection. Original bytes are retained; the
`evidence_index.json` lists their SHA-256 digests. Logs, weights, logits and
caches are not archived. The saved assessment reproduces exactly after JSON
normalization; both W5 vocabulary candidates pass its quality guards.

| Backbone / vocabulary | Primary KL ↓ | Top-1 match | 32-token positional match | Projected decimal GB |
|---|---:|---:|---:|---:|
| FP16 / FP16 | ≈0 | 100% | 100% | 9.099 |
| FP16 / W8 | 0.000067 | 99.41% | 94.00% | 8.474 |
| FP16 / W6 | 0.000831 | 97.96% | 89.38% | 8.315 |
| W4 / FP16 | 0.016497 | 93.35% | 33.63% | 3.781 |
| W4 / W8 | 0.016552 | 93.35% | 33.63% | 3.155 |
| W4 / W6 | 0.017301 | 93.00% | 34.13% | 2.996 |
| W5 / FP16 | 0.003969 | 96.63% | 80.63% | 4.226 |
| W5 / W8 | 0.004037 | 96.60% | 79.50% | 3.601 |
| W5 / W6 | 0.004741 | 96.05% | 81.63% | 3.442 |

Primary KL uses 24 C4 prompts / 12,264 positions. Diverse KL covers 25 authored
snippets / 3,183 positions, with five snippets each for coding, agentic, maths,
multilingual and short long-document-style content. Trajectories use 25 separate
prompts ×32 greedy tokens. These are reused development suites, not task-success
scores or Unsloth's Divergence-300.

Compared with W4/FP16 vocabulary, primary KL falls **75.5% / 71.3%** with W5/W8
and W5/W6. Diverse KL falls **76.7% / 74.5%**, with improvement in all five small
domains. Exact 32-token sequences improve from 2/25 to 12/25 and 15/25.
The 2.125-point W6-over-W8 trajectory difference is inconclusive: an exploratory
paired-prompt bootstrap (4,000 draws, seed 20260907) gives approximately
[-6.38, +11.75] percentage points. Do not conclude W6 is intrinsically better.

The saved screen's paired-prompt KL differences against W4/FP16 have 95%
intervals [-0.014480, -0.010767] for W5/W8 and [-0.013791, -0.010058] for W5/W6.
They quantify within-seed prompt uncertainty, not replication across seeds.
Vocabulary/backbone interaction intervals include zero on this small primary suite.

All result checksums, aggregate/token-denominator reconciliations, prompt/window
pairing and component ledgers were checked. Source-teacher vocabulary identities
agree and self-KL is effectively zero (a tiny negative is roundoff). Calibration
and evaluation exact-hash disjointness passes; semantic overlap is not ruled out.
PPL dataset revisions remain unpinned, although evaluated window hashes agree.

## Budget interpretation and Unsloth boundary

The shared vocabulary has 635,699,200 parameters. W6's projected packed payload
is 486,710,016 bytes versus 1,271,398,400 FP16 bytes. The savings pay for W5
throughout the targeted backbone instead of forcing many W3/W4 downgrades.
This supports vocabulary-aware mixed precision; it does not validate the old
allocator or show every important-layer heuristic is correct.

W5/W6 projects 3,441,587,632 total bytes, about 143 MB below the historical
3,584,533,344-byte Unsloth bundle. W5/W8 projects 3,600,513,200 bytes, 0.446%
above that bundle. Neither is a measured artifact. The screen executes a dense
vocabulary prototype with dense backbone caches, so these are not runtime-memory
or speed claims.

The [historical Unsloth record](../research/results/raw/qwen35_4b_allocator_v4_8ba3b751bb82/unsloth_kl/unsloth_ud_q4_kl.json)
has KL 0.0118883 / top-1 94.09%; all 24 input hashes match this screen. The new
prototype KL values are numerically 60–66% lower, but **this is not a provider
win**: Unsloth used a BF16-GGUF teacher in llama.cpp; these results use FP16
Transformers. There was no new Unsloth inference in the supplied bundle; its
provider ledger is a header-only audit. Common input hashes do not prove engine
or reference parity.

## Decision

Advance **W5/W6 first**, retaining W5/W8 as the quality comparator, to the
[packed export/reload validation](packed_vocabulary_validation_run.md). Verify
actual files, shared ownership and full quality retention before another search.
Then freeze fresh validation inputs and replicate with appropriate controls;
rerun the provider comparison under explicit engine/reference and byte boundaries.
No new LoRA, learned-rotation or mixed-allocator sweep is warranted before that gate.
