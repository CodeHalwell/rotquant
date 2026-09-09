# Completed fresh-quality evidence: Qwen3.5-4B

Producer: `d4292d6fdec688cdb896d6d2cfea81de1c310235`, CUDA A100-SXM4-40GB.
[Original compact records](../research/results/raw/qwen35_fresh_quality_d4292d6fdec6/evidence_index.json)
are preserved byte-for-byte. This note supersedes the pending-run status of the
[September 8 plan](fresh_quality_run_2026-09-08.md), not its frozen protocol.

## Fidelity and size

The comparison uses identical frozen HF input IDs and a common FP16 teacher.
C4 comprises 24 new documents, 512 tokens each, 12,264 prediction positions.
These checkpoints use **5-bit backbone weights**; W6/W8 names describe their
shared tied vocabulary precision. They are not uniform four-bit checkpoints.
Sizes are decimal GB and include retained multimodal components; vision is not
evaluated. Full measured bytes: W6 3,441,544,638; W8 3,600,470,206;
Unsloth 3,584,533,344 (including its F16 projector sidecar).

| Model/recipe | GB | C4 teacher KL | Source argmax agreement | Authored tasks correct /96 |
|---|---:|---:|---:|---:|
| HF source FP16 | — | 0 | 100% | 77 |
| BF16 GGUF bridge | — | 0.00002712 | 99.576% | 77 |
| Unsloth UD-Q4_K_XL | 3.585 | 0.013379 | 94.039% | 74 |
| W5/W6 seed 0 | 3.442 | 0.005939 | 96.021% | 71 |
| W5/W6 seed 1 | 3.442 | 0.005770 | 95.907% | 80 |
| W5/W6 seed 2 | 3.442 | 0.005633 | 95.947% | 73 |
| W5/W8 seed 0 | 3.600 | 0.005210 | 96.331% | 69 |
| W5/W8 seed 1 | 3.600 | 0.004959 | 96.323% | 81 |
| W5/W8 seed 2 | 3.600 | 0.004901 | 96.510% | 73 |

W6 reduces C4 KL by 55.6–57.9%, at 3.989% fewer bytes. W8 reduces it by
61.1–63.4%, at 0.445% more bytes. All six improve KL on every one of the 24
documents; paired document-bootstrap provider contrasts exclude zero. W8 has
lower C4 KL than W6 for each matched seed. The seeds share calibration selection
and test documents: they are rotation/quantization RNG replications, not three
independent datasets. Do not inflate sample size by pooling them.

The GGUF bridge's 96 task continuations match the HF source exactly. Provider
same-engine BF16 C4 KL is 0.013397, close to its common-FP16 result. Four Hindi
native-tokenization mismatches remain recorded; all mapping and byte-roundtrip
gates passed under `frozen-hf`. This does not prove native text-tokenizer parity.
KL differences are not additive; do not subtract bridge KL from candidate KL.

## What the task suite does and does not show

The 96 tasks are authored diagnostics, not independent public coding/agent
benchmarks. Outcomes vary by seed; lower C4 KL does not consistently improve
their aggregate correctness. Do not select seed 1 because it happened to win.

- Every invalid-JSON case contains the right object inside Markdown fences.
  This is a real strict-format failure, not evidence of wrong arithmetic.
- All six failed lookup cases in every arm select the correct `search` tool,
  but use `"invoice INV-13"`-style queries instead of exactly `"INV-13"`.
  The original prompt did not clearly prohibit that prefix. The oracle is
  overstrict; the uniform 18/24 tool score is not 25% failed tool selection.
- The conditional-tool family only used values 13–48, never testing the >50
  branch. Its coverage is inadequate for a tool-use claim.
- All arms achieve 24/24 on multilingual arithmetic: a ceiling on one narrow
  family, not broad multilingual capability.

The original scores/oracles stay frozen. Any repaired authored suite needs a
new protocol, explicit schema/exact-query instructions and balanced branches.
The next experiment instead adds independent public task sources. Actual tool
interaction and sandboxed coding remain separate release requirements.

## Evidence audit and limits

The archiver checks 1,125 SHA-256 pairs, 1,080 prompt identities across nine
collections, rescoring of 864 strict task outputs, reconciliation of 45 domain
rows and 126 paired contrasts. All six reload probe reports passed (16 positions,
four short generations, zero error); residency evidence shows one shared
vocabulary owner and no persistent dense backbone fallback.

```bash
.venv/bin/python scripts/archive_fresh_quality.py \
  --input-dir /path/to/qwen35_fresh_quality_d4292d6fdec6 \
  --output-dir research/results/raw/qwen35_fresh_quality_d4292d6fdec6
```

The compact bundle lacks checkpoint/probe tensors and full-logit NPZ references.
We can verify their recorded provenance but cannot independently rehash those
absent tensors, recompute raw KL, or replay CUDA from this archive. Original
Drive checkpoints must remain. Logs are not copied into Git. Tokenizer/config
JSON evidence is retained, including identical copies under replica paths.

## Decision

Keep W5/W6 as the under-budget candidate, W5/W8 as the fidelity alternative.
No new recovery/allocator search, no automatic overall promotion. Run the
[public-task release gate](public_tasks_run_2026-09-09.md), then implement one
measured serving path. This is encouraging fidelity evidence against one pinned
Qwen3.5 artifact—not proof of Unsloth Dynamic-v3.0 superiority, native GGUF
interoperability, task dominance or optimized runtime performance.
