# Project review — 2026-09-09

Reviewed at `85e1254` (origin/main, 9 September 2026), the day the
fresh-quality evidence was archived and the public-task gate was prepared.
Fresh `uv sync --locked --extra dev --extra eval` environment (torch
2.12.0+cu130, transformers 5.9.0, Python 3.11, no GPU). This note is
deliberately shorter than the [8 September deep dive](project_deep_dive_2026-09-08.md):
that review's defect lists still stand and are not repeated here. This one
records what the project has actually established, what changed in the day
since, and the order of the next steps. Nothing on Google Drive was
inspected; every statement about a Colab run is a statement about what the
repository records.

## 1. Verdict

RotQuant now has one result worth defending and no runtime for it beyond the
Python reference path.

- The fresh-quality run turned the vocabulary-budget hypothesis into a
  measured, three-seed, common-teacher result: a 5-bit backbone with a packed
  6- or 8-bit tied vocabulary beats the pinned Unsloth `UD-Q4_K_XL` on 24
  fresh C4 documents by 56–63 % teacher KL, at 4 % fewer or 0.4 % more bytes.
  It improves every document in every seed. That is the first fair
  matched-size comparison the project has produced.
- The same run showed that KL is not task accuracy: the 96 authored tasks
  moved with the seed rather than the recipe; the tool-lookup oracle is
  overstrict, the conditional family never exercises its positive branch, and
  the JSON failures are strict-format failures rather than wrong answers. The
  public-task gate (GSM8K, CRUXEval-O, IFEval) is prepared and is the right
  next run.
- The recipe that wins runs only on the tiled Python reference path. The
  native-v2 C++ runtime handles 1–8-bit blocks, but only per matrix and only
  with fp16 scales; the GGUF exporter and the llama.cpp patch are W4 with
  fp16 scales. None of them can consume the W5 artifacts' 8-bit scales or
  packed vocabulary, and nothing on that side has changed since 31 August.
  Until it does there is no memory, throughput or same-engine evidence, and
  every provider comparison stays cross-engine.
- The engineering baseline is sound at this revision: Python CI is green on
  `main` for Python 3.10–3.13, lint is clean, the native runtime builds
  warning-free and passes conformance, and the full suite passes locally
  (§5).
- The twelve library defects listed on 8 September are all still open. The
  reuse freeze that queued them has served its purpose and can be lifted.

## 2. What the project has established

| Claim | Evidence | Status |
|---|---|---|
| The weight quantiser (block FWHT, Gaussian codebook, MSE scale search, act-order GPTQ, 8-bit scales) is correct on the deployed path | Two code audits; exact byte identities; fresh-process reload probes with zero error on six artifacts | Established |
| GPTQ improves W4 in every seed at zero inference bits | Three-seed ladder, 1 September | Claimed; raw records only on Drive |
| The fp16 tied vocabulary, not the quantiser, explained the Unsloth gap | GGUF header decomposition, 6 September | Established |
| W5 backbone + W6/W8 packed vocabulary beats `UD-Q4_K_XL` at matched bytes on fresh inputs, common FP16 teacher, three seeds | `research/results/raw/qwen35_fresh_quality_d4292d6fdec6` | Established on C4 KL; task outcome open |
| Allocation at an fp16-vocabulary budget, learned signs, block recovery, LoRA-QAT, vector codebooks | Allocators v1–v4, sign replication, recovery arms | Not supported at the budgets tried; LoRA-QAT underpowered rather than disproved, vector codebooks research-only |
| Any KV-cache quality number | Simulator defect, 1 September | All withdrawn; nothing re-measured |
| The winning recipe can be served outside Python | Native-v2 lacks an 8-bit scale layout, the packed vocabulary and model execution; GGUF/llama.cpp are W4 only | Not yet |

The fresh-quality numbers, for reference (24 documents, 12,264 positions,
common FP16 teacher):

| Arm | Bytes vs Unsloth | C4 KL, seeds 0 / 1 / 2 | Authored tasks correct of 96, seeds 0 / 1 / 2 |
|---|---:|---|---|
| Unsloth `UD-Q4_K_XL` | 0 | 0.01338 | 74 |
| W5/W6 | −3.99 % | 0.00594 / 0.00577 / 0.00563 | 71 / 80 / 73 |
| W5/W8 | +0.45 % | 0.00521 / 0.00496 / 0.00490 | 69 / 81 / 73 |

Source FP16 scores 77 of 96 on the authored tasks, and the BF16-GGUF bridge
reproduces all 96 source continuations, so the engine bridge is clean on
that suite. The seeds share calibration rows and test documents; they are
quantisation-RNG replications, not independent datasets.

Beyond the numbers, three things have gone right:

- **The review loop works.** The Codex, Copilot and Claude reviews found the
  KV simulator state sharing, the subnormal 8-bit scale step, the
  shared-site rotation gate, the fp32 rotary-buffer downcast and the missing
  generation config, each before a wrong number was published. The cost is
  documentation volume (§3).
- **Provenance is unusually good.** Every archived run pins code and model
  revision, hashes every input and record, and fails closed on mismatch.
  Withdrawn results are marked withdrawn; projections are labelled
  projections; negative results are recorded with the same care as positive
  ones.
- **The engineering foundation is real.** Pickle-free checkpoint v1–v3 with
  transactional export, a dependency-free C++17 runtime with scalar, NEON and
  AVX2 kernels behind a C ABI, sanitizer CI, a llama.cpp patch with a Metal
  path, and Colab runners that resume from checksummed records.

## 3. What has not moved, or is at risk

Ranked by effect on the next decision.

1. **Runtime.** The provider artifact is a GGUF served by llama.cpp; ours
   needs Transformers, this library and a per-layer dequantising reference
   path. No resident-memory or tokens-per-second figure exists for the W5
   recipes, and 3.44/3.60 GB are file sizes, not peak VRAM. The 9 September
   bandwidth requirements in the roadmap are correct, but they are a
   specification, not work done.
2. **Evaluation breadth.** 24 documents and 96 authored tasks, selected by
   the same small suites that chose every earlier recipe. The public-task
   runner is implemented and CPU-tested but has never run on a GPU;
   `lm-eval` is installed and unused; one model; one calibration corpus.
3. **Library defects L1–L12** (deep dive §4.2) are unchanged. Checked here:
   L1 (`rotquant/rotate.py:371` still takes trig in the storage dtype), L4
   (`rotquant/quantize.py:958` still falls back to H = I with a warning), L5
   (neither exporter reads `scale_bits_main`) and L6 (`rotquant/rotate.py:255`
   still silently shrinks a non-dividing block). None affects the archived W5
   artifacts. The reuse test still diffs `rotquant/` against `733bb3d`, but
   the producer run it protected is complete and archived.
4. **Two identity couplings to plan around.** The public-task notebook checks
   out `main` and binds its receipts to HEAD plus a hash of every `.py` under
   `rotquant/` and `scripts/`, so a merge to `main` mid-run forces a new
   output root unless the printed SHA is pinned. The fresh-eval runner treats
   a new JSON under `research/results/raw` as a protocol change. Merge first,
   pin, then run.
5. **Withdrawn numbers still shipped.** `paper/generated/audit_report.json`
   still reports the withdrawn 24.0 %/10.6 % cache reductions and the
   0.80/0.84 matched ratios; `configs/publication_qwen35_joint_cuda.yaml`
   still carries the 3.25-bpv map; the paper keeps the joint weight-plus-KV
   framing.
6. **Documentation volume.** 33 files under `docs/` (564 KB), five review
   documents in ten days, and a README of about 690 lines that still states a
   Llama-2-7B/13B confirmation bar no result has ever met (`README.md:665`,
   the sentence beginning "A finding is confirmed when"). The "four answers
   to what is next" problem from 8 September is fixed: README, roadmap and
   the research README now agree. `CHANGELOG.md` had no entry for the
   9 September work until this review added one, and still has duplicated
   `### Added` and `### Changed` headings.
7. **Release engineering.** `__version__` is 0.1.0 with no tag, so provenance
   cannot separate pre-fix from post-fix runs; no PyPI release, no wheel
   build; the `baselines` extra is pinned to pre-Transformers-5 packages and
   untested.
8. **Scope pressure.** Recorded on 9 September as pending: the 21-variant
   Unsloth 4B sweep, the 24-variant 27B sweep, attention-type sensitivity
   arms, learned-rotation interaction arms, fused bandwidth-minimising
   kernels, and genuine agent benchmarks. Each is defensible on its own.
   Together they are months of one person's time, and none of them is the
   runtime.
9. **Repository hygiene.** The fresh-quality archive adds 91 MB to the
   working tree, including four identical 1.2-million-line `tokenizer.json`
   copies. The pack is 7.5 MiB, so this costs diff noise rather than clone
   time, but the archiver should store one copy and a hash.

## 4. Next steps, in order

Ordering principle: finish the one prepared experiment, then make the
winning recipe runnable, and let that runtime carry every later comparison.
Research branches wait until they can be measured on it.

### A. Run the public-task gate (this week, one A100 session)

1. Merge whatever must land first, then pin the printed SHA in the
   public-task notebook and do not push to `main` until the run completes.
2. Smoke scope, then the standard scope: 128 examples per benchmark, all
   nine arms. Archive the compact evidence the same day.
3. Apply the pre-registered reading: per-benchmark accuracy with intervals
   and paired flips, no overall winner, no promotion on a lucky seed. Check
   whether the BF16 bridge moves task outcomes before attributing anything
   to quantisation.
4. Write the decision rule down before the run: if neither W5 recipe shows a
   material regression against source and Unsloth on any of the three
   benchmarks, freeze W5/W6 as the release candidate and W5/W8 as the
   fidelity alternative, and stop quality search.

### B. Lift the freeze and clear the queue (days, no GPU)

1. Retire or re-review the `733bb3d` reuse-compatibility test now that the
   producer run is archived; the public-task runner binds to its own
   receipts.
2. Land L1–L12 in one PR with the tests listed on 8 September: L1, L2, L5
   and L6 first.
3. Tag `v0.1.0` at the 31 August commit and bump `__version__` to
   `0.2.0.dev0`.
4. Remove the withdrawn cache numbers from the audit script, the publication
   config and the serving docs; either park the KV track explicitly or
   schedule its re-measurement on the fixed simulator.
5. CHANGELOG entry for 9 September, fix the duplicated headings, and cut the
   README to a front page plus one status file that every other document
   points at.

### C. One measured serving path for the retained recipe (weeks)

This is the largest gap and the deliverable that makes the project usable,
and legible, outside the repository. A downloadable GGUF with one
same-engine benchmark table is worth more than any further review document.

1. Add an 8-bit scale layout, the shared packed tied vocabulary and
   model-level execution to native-v2 (its 1–8-bit blocks already exist), and
   extend the GGUF exporter and patch beyond W4; until the scale layout
   exists, refuse `scale_bits_main != 16` on export instead of silently
   re-rounding (L5).
2. Update the llama.cpp patch (CPU and Metal) and make the patch workflow
   compile and run a conformance prompt, not just `git apply --check`.
3. Export the W5/W6 artifact as a GGUF and run the same-engine comparison
   against `UD-Q4_K_XL` inside llama.cpp with a common teacher. That removes
   the engine confound for good.
4. Measure resident memory and prefill/decode throughput on named hardware
   then, and not before.
5. A Triton or CUDA W5 GEMV and the vLLM plugin follow once a GPU CI
   mechanism exists; that decision has been pending since 31 August.

### D. Research branches, only after C

- The 4B all-variant Unsloth sweep is the most valuable: it turns one point
  into a curve using runners that already exist. Do it once the GGUF path
  exists so RotQuant's points come from the same engine.
- Attention-type sensitivity and learned-rotation interactions are lower
  priority; the useful target is W4 approaching current W5 quality.
- 27B waits for layer streaming and a measured cost forecast.

### What not to do next

Another allocator generation; a LoRA or distillation sweep; the 27B model;
a vLLM plugin before GPU CI; another review document before the public-task
result exists.

## 5. Verification record

- Environment: `uv sync --locked --extra dev --extra eval` (torch
  2.12.0+cu130, transformers 5.9.0, Python 3.11.15; 5.2 GB), no GPU,
  full-history clone.
- `uvx ruff@0.16.5 check .` (the locked version): clean. ruff 0.15.8 reports
  102 E402 findings on the `sys.path` bootstraps in `scripts/`; that is a
  ruff version difference, not a defect.
- `pytest tests/ -q` with `ROTQUANT_REQUIRE_GIT_HISTORY=1`: 729 passed,
  17 skipped (NEON-only native cases on x86), 138 s; the reviewed-producer
  reuse guard ran rather than skipped.
- Native: Release shared build with `-DROTQUANT_NATIVE_WARNINGS_AS_ERRORS=ON`,
  both conformance suites pass, `--capabilities` reports scalar and AVX2.
- `configs/smoke_cpu.yaml` end to end (tiny random Llama, CPU, WikiText-2
  through the proxy): completes and writes its result JSON; the deliberately
  meaningless PPL of 32140.2637 matches the 29 August entry in the
  experiment log exactly.
- GitHub: Python CI on `main` green at `85e1254` and `5a98b99`, red only at
  `a4ac776`; no open issues or pull requests; no tags.
- Not done: no model download beyond the tiny fixture, no Colab, no Drive,
  no GPU, no Metal build, no llama.cpp compile.
