# Project deep dive — 2026-09-08

Reviewed at `a4ac776` (origin/main), locally in a fresh `uv sync --locked
--extra dev --extra eval` environment (torch 2.12.0+cu130, transformers 5.9.0,
Python 3.12; no GPU). This review read every document under `docs/`, the
paper, the archived records, the CI history, and the library, scripts,
native runtime and notebooks. Five focused code reviews were run in parallel
and their most severe findings were re-verified by hand before being listed
here. Nothing on Google Drive was inspected; every statement about a Colab
run is a statement about what the repository records.

## 1. Verdict

The project is in better scientific shape than its own documentation
suggests, and in worse engineering shape than its CI badge suggests.

- The weight-only quantiser (block-FWHT, Gaussian codebook, MSE scale
  search, act-order GPTQ, 8-bit scale storage) is numerically sound and
  well-tested. Two independent code audits and this review found no defect
  on the deployed W4/W5 path.
- The decisive result of the last fortnight is not an allocator. It is the
  byte-budget correction: the fp16 tied vocabulary cost 1.27 GB, so every
  "matched-size" allocator run compared a 3.6-bpw backbone against a
  5.3-bpw one. Compressing the vocabulary to W6/W8 and lifting the backbone
  to W5 gives KL 0.0047/0.0040 at 3.44/3.60 GB on the development suite,
  versus 0.0165 for the old W4 recipe. Both artifacts export, reload in a
  fresh process with zero probe error, and pass the development guards.
- That result is seed 0, on the same 24+25 development prompts that selected
  every earlier recipe, scored against a Transformers fp16 teacher. The
  Unsloth numbers it is being compared with come from a llama.cpp BF16
  teacher. The "60% lower KL than Unsloth" figure is therefore not yet a
  result. The prepared fresh-quality run is exactly the experiment that
  turns it into one, and it is the right next experiment.
- Python CI on `main` has been red since the last commit. The next Colab run
  is documented as "publish this code and run it"; the code was published
  with failing tests. Fixed on this branch.
- The winning recipe cannot be served by anything except the tiled Python
  reference path. Native-v2, the llama.cpp patch and the GGUF exporter are
  W4 with fp16 scales only. The runtime story has not moved since 31 August
  while the quality story moved a long way. This is now the biggest gap.
- Every KV-cache quality number is withdrawn; nothing has been re-measured
  since the simulator was fixed on 1 September. The paper draft, the
  publication audit script and one config still carry the withdrawn numbers.

## 2. Where the project stands, on the evidence in the repository

| Question | Status | Evidence in repo |
|---|---|---|
| Is the W4 quantiser correct? | Established | Two code audits (`results_review_2026-09-06.md` §6, this review); exact byte identities reproduce; 695 tests |
| Does GPTQ help on Qwen3.5-4B? | Claimed (three seeds), **not archived** | `experiment_log.md` "three-seed W4 promotion" cites Drive only |
| Does allocation intelligence close the provider gap at an fp16-vocabulary budget? | Refuted | v1 lost to random; v2 −25 % vs random; v3 −73 % vs random but 1.85× uniform W4 and 2.6× Unsloth; v4 format palette gained nothing (compact/raw records) |
| Does vocabulary compression unlock the budget? | Established at seed 0, development suite | Nine-arm screen (raw archived), packed export and revalidation (raw archived) |
| Do learned rotations, learned signs, block recovery, LoRA-QAT help at calibration-scale budgets? | Refuted at the budgets tried | experiment log; some arms confounded by since-fixed bugs (scale encoder, shared-site gate, fp16 butterfly storage, see §4.2) |
| Is any KV-cache result valid? | All withdrawn; one provisional | `scientific_validity_review_2026-09-01.md`; 2-bit E8P four-prompt run predates ageing fix and endpoint check |
| Is RotQuant competitive with UD-Q4_K_XL? | Unmeasured in a comparable sense | Cross-engine, cross-teacher anchors only; fresh-quality run prepared but not completed |
| Can the winning recipe be deployed? | No | GGUF/native are W4 + fp16 scales; no CUDA kernel; Python path dequantises per layer |
| Does the project meet its own confirmation bar? | No, and the bar is obsolete | README demands Llama-2-7B/13B and a zero-shot bundle that has never been run (`lm_eval: not-installed` in every archived environment) |

Two positive things worth saying plainly. The claim boundaries in the docs
are unusually honest: withdrawn results are marked withdrawn, projections are
labelled projections, and every Colab run records revision, model revision,
hashes and byte ledgers. And the adversarial review loop (Codex, Copilot,
Claude) has caught real defects that would have invalidated results. The cost
is that the documentation has become the largest and least consistent
artefact in the repository (see §4.6).

## 3. What this review changed on the branch

All verified locally (`ruff` clean, full suite green, native Release and
ASan/UBSan builds and conformance suites passing) and the branch's Python CI
run is green.

1. **Python CI red on `main` since `a4ac776`, all four Python versions.**
   `tests/test_fresh_quality_reuse.py::test_reuse_compatibility_keeps_reviewed_hf_scoring_and_core_code_unchanged`
   runs `git show 733bb3d…:scripts/run_qwen35_fresh_eval.py`; `actions/checkout`
   defaults to a shallow clone, so the reviewed producer commit is absent and
   `git` exits 128. The tests job now checks out full history and sets
   `ROTQUANT_REQUIRE_GIT_HISTORY=1` so the guard fails, never skips, in CI. On
   a shallow developer clone the test skips with instructions instead of
   crashing inside `subprocess`. The runtime reuse path never calls `git show`,
   so Colab was not affected.
2. **`scripts/build_qwen35_fresh_eval_notebook.py` could not be run as a
   script** (`ModuleNotFoundError: scripts`): it is the only builder without
   the repository-root `sys.path` bootstrap, and its output path was relative
   to the working directory. Fixed; the regenerated notebook is byte-identical
   to the committed one.
3. **Native C ABI undefined behaviour.** Out-of-range `rq_native_v2_kernel`
   and `rq_native_v2_status` integers from a foreign caller were UB at the
   parameter load (UBSan: "load of value 99, which is not a valid value").
   A sentinel enumerator makes every int32 a defined value that the switch
   default rejects; the same call now returns `INVALID_ARGUMENT` with no
   sanitizer report. `native/` is outside the reuse freeze and has its own CI.
4. **Fresh-quality run hygiene.** `summarize()` now reports token-weighted
   `source_nll`, `candidate_nll` and `nll_delta` as the runbook promises; the
   Colab llama.cpp CUDA build timeout is 3600 s instead of 1800 s against a
   recorded ~29-minute build; the runbook gains an "operational notes" section
   (§4.3). None of this touches reviewed scoring code or any reusable record.
5. **Documentation.** README: CI runs Python 3.10–3.13, not 3.10–3.12; `uv sync`
   does not install a CPU torch wheel on Linux (the lock resolves to
   `torch 2.12.0+cu130`, a 5.3 GB environment); the stale "next run" paragraph
   now points at the fresh-quality Colab and this document. CHANGELOG entry
   and a roadmap pointer added. The remaining README inconsistencies are
   listed in §4.6 rather than patched piecemeal, because the README needs
   restructuring rather than patching.

Not changed, deliberately: anything under `rotquant/`, `scripts/run_experiment.py`,
`scripts/run_unsloth_qwen35_4b_kl.py` or `scripts/run_qwen35_packed_validation.py`.
The reuse-compatibility test asserts that `git diff 733bb3d -- rotquant …` is
empty, because the recovery Colab run adopts per-prompt results produced by
that revision. Until that run completes (or `REUSABLE_SOURCE` is re-reviewed),
**the library is frozen by design**, and every library fix in §4.2 has to
queue behind it. Merge this branch before starting the Colab session, then
pin the printed SHA: the reuse receipt binds the consumer to the exact HEAD,
so any commit to `main` mid-run forces a new output root.

## 4. What has been missed

Severity is judged by effect on the next decision, not by how hard the bug
is. "Confirmed" means reproduced here (numerically or by executing the path);
"by reading" means the code path was traced end to end but not executed.

### 4.1 Blocking or distorting the next GPU run

- **Red CI on the published run revision** (fixed, §3).
- **Reuse receipt binds to HEAD** (`source_identity()` = HEAD SHA + hash of
  every `.py` under `rotquant/` and `scripts/`). The notebook sets
  `REPO_REF = "main"`. Merge first, then pin. Also `_code_revision()` returns
  the string `"unknown"` when git fails (`scripts/run_qwen35_next_stage.py:604`),
  so two runs without git would compare as identical code: a fail-open identity.
- **Silent CPU fallback in the experiment runner.** `resolve_device_dtype()`
  (`scripts/run_experiment.py:187-200`) downgrades `device: cuda` to CPU/fp32
  with a warning when CUDA is absent, contrary to the fail-closed rule. The
  notebooks run a CUDA preflight first, which is the only thing standing
  between a detached Colab GPU and a CPU result labelled by its config.
- **Development suites are small and now spent.** Primary KL: 24 C4 prompts.
  Diverse: 25 prompts, 5 per domain, 180–789 characters each. Authored tasks:
  96, with a single multilingual-arithmetic family (so no interval for that
  domain). The registered 300-prompt competitive protocol has never been
  built or run; `docs/competitive_data.md` lists five unstarted prerequisites.
  Nothing here is a task benchmark.

### 4.2 Library defects, queued behind the reuse freeze (core review)

None of these affects the current W5/W6 and W5/W8 artifacts, which store
`torch_dtype: float16` and parameter-free FWHT rotations (checked in the
archived `rotquant_config.json`). They do affect the shelved butterfly and
learned-sign arms, the public API, and any future export of a `scale_bits: 8`
artifact. Apply them, with the listed tests, in one PR as soon as the
recovery run has finished and `REUSABLE_SOURCE` can be re-reviewed.

| # | Where | Defect | Status |
|---|---|---|---|
| L1 | `rotquant/rotate.py:365-377` | `ButterflyRotation._trig()` computes cos/sin in the storage dtype; fp16 angles give a non-orthogonal transform (max abs RᵀR − I = 3.3e-3, uniform gain 1.0033) and the checkpoint gate scores the fp32 rotation before `commit_rotation_storage` runs, so the deployed arm is not the one the gate accepted. The shelved "fp16-angle learned-sign" result carries this confound. One-line fix: trig from `theta.float()`. | confirmed |
| L2 | `rotquant/checkpoint.py:771-787` | `model.to(dtype)` rounds fp32 butterfly angles to the model dtype before the loop casts them "back"; fresh-process logits differ by 2.4e-4 from in-process for an fp16 model. `LearnedRotation.theta` and `DenseOrthogonal.R` are never restored. Capture before `.to()`, as the RoPE buffers already are. | confirmed |
| L3 | `rotquant/checkpoint.py:313-317` | `torch_dtype` is inferred from the first floating tensor in `state_dict()`; for a v3 artifact with packed vocabulary, no biases and butterfly rotations that is the fp32 `theta`, so an fp16 model reloads as fp32 and the vocabulary execution dtype is silently overridden. Record the dtype explicitly. | confirmed |
| L4 | `rotquant/quantize.py:958-963`, `calibrate.py:181-188`, `linear.py:402-403` | GPTQ without a Hessian warns and falls back to plain rounding; `collect_hessians` silently omits layers never invoked; `refresh_quantization()` never has a Hessian, so block training re-packs every GPTQ layer as RTN. Raise, and record in `stats_out`. | confirmed |
| L5 | `rotquant/native.py:238-280`, `rotquant/gguf.py:87-105, 231-285` | Neither exporter checks `scale_bits_main`; 8-bit double-quantised scales are decoded to fp32 and re-rounded to fp16 on export, so scale8 artifacts are not bit-exact with `dequantize()` and are charged 16 bits per scale in the native manifest. `verify_rotquant_gguf.py` compares against the same lossy conversion and passes. Raise on `scale_bits_main != 16` until a native 8-bit-scale layout exists. | confirmed |
| L6 | `rotquant/rotate.py:255-257` | A `block` that does not divide `dim` is silently replaced by the largest power-of-two divisor (odd dims become a pure sign flip). Raise. | confirmed |
| L7 | `rotquant/linear.py:114-149`, `patch.py:437-438` | In-process `.half()`/`.to(dtype)` after patching recasts rotation parameters (scales are protected, rotations are not); the MPS fallback staging path does this to itself. | confirmed |
| L8 | `rotquant/quantize.py:137-146` | `dataclasses.replace(cfg, codebook=...)` keeps the old `mse_search_lo/hi`, so a uniform codebook derived from a Gaussian base clips at 1.5σ (the E2 "Gaussian beats uniform" confound, now reachable from `dynamic.py` palettes). Derive bounds lazily. | confirmed |
| L9 | `scripts/run_experiment.py:723-724`, `linear.py:449-474` | Rotation buffers (`signs`, `DenseOrthogonal.R`) are outside every byte counter and a shared butterfly `theta` is counted once per sibling, so `effective_bits_per_weight` is wrong for dense and shared-rotation runs. | by reading |
| L10 | `calibrate.py:50-54`, `quantize.py:117` | Double damping by default on the public API path (`finalize` ridge + GPTQ `percdamp`); `optimize_model` cannot pass `hessian_damp_frac`. | by reading |
| L11 | `checkpoint.py:739-744` | LoRA storage dtype not preserved across reload; adapter bytes halve on a bf16 reload. Numerically neutral. | confirmed |
| L12 | `quantize.py:400-410` | No finiteness check on fp16 scale storage (group RMS above 65504 stores `inf`). | confirmed |

Smaller: `format.py:250-253` validates residual/sketch specs only structurally;
`vocabulary.py:194-202` re-implements the LSB-first bitstream decoder instead
of calling `pack.unpack_indices`; `PatchConfig.share_rotations=False` while
`RotQuantConfig.share_rotations=True`; `RandomizedHadamard(dtype=...)` is
ignored; `_qjl_residual` uses a device-local generator so codes differ CPU vs
CUDA; `scripts/export_rotquant_gguf.py:24-28` imports `_`-prefixed names from
`rotquant.checkpoint` against the CONTRIBUTING rule.

Test gaps behind these: no reload without an explicit `dtype=`; no fp16
in-process model anywhere in `test_checkpoint.py`; no fp16/bf16 orthogonality
test; no "GPTQ without Hessian raises" test; no non-dividing block test; no
LoRA save/load round trip; the shared-rotation reload branch
(`checkpoint.py:718-721`) is never executed by a test.

### 4.3 Fresh-quality pipeline (the next GPU run)

The runner is unusually well fail-closed: EOS resolution, the token axis,
vocabulary padding, reference hashing, resume and reuse provenance were each
checked against the pinned Hub files and the pinned llama-cpp-python sources
and either match or refuse to proceed. Seeds 1/2 vary exactly what they claim
(quantisation RNG on fixed calibration rows); the fresh C4 slice is disjoint
from every archived selection; the bridge isolates engine and precision
effects; memory fits an A100-40GB with margin. Nothing found would make the
run silently produce wrong KL, top-1 or trajectory numbers. The risks are
operational, and the identity discipline that makes reuse trustworthy is
also what makes the run brittle:

- **Reuse binds the runtime.** The manifest identity includes torch, CUDA and
  Python versions and the GPU marketing name (`run_qwen35_vocabulary_budget.py:91-102`);
  the archived seed-0 session was `torch 2.11.0+cu128`, Python 3.13.15,
  A100-SXM4-40GB. A different SKU or a torch bump makes `reuse` refuse and the
  only fallback is rerunning source and seed 0. Correct behaviour, but the
  plan did not say so. Now recorded in the runbook.
- **The archive is part of the frozen protocol** (`run_qwen35_fresh_eval.py:159-175`):
  committing any JSON under `research/results/raw` mid-run makes every
  remaining phase refuse. The project's own archiving habit invites exactly
  this. Now recorded in the runbook.
- **The llama.cpp build timeout had a one-minute margin** (1800 s against a
  measured ~29 minutes). Raised to 3600 s on this branch.
- **The `condition` tool-selection family never exercises its positive
  branch** (`rotquant/eval/fresh_tasks.py:105-114`: `a = 13 + 7 i`, `i ≤ 5`,
  so `a > 50` is never true); its six variants share the `none` answer with
  the `missing` family, so a none-biased student scores half of tool
  selection for free. The task hash is frozen in the manifest, so the fix
  waits until after this run. Caveat recorded in the runbook.
- **Task-domain intervals resample four families** (one for multilingual):
  at most 35 distinct resamples, so `ci95` on those contrasts is descriptive
  only. The plan disclaimed only the single-family case. Caveat recorded;
  setting `ci95` to null below a minimum family count is a library change
  that waits for the freeze to lift.
- **`summarize` omitted NLL** although the plan reports it. Added on this
  branch (token-weighted, like KL).
- Seed-1/2 recipe identity is checked only after the multi-hour preparation
  (`run_qwen35_fresh_eval.py:734-735`); the current YAML matches the archived
  seed-0 identity, so it passes today. A dry-run pre-check would be cheap.
- Reuse re-reads and re-hashes the ~15 GB reference set several times per
  session (about 100 GB of Drive reads for a full run): hours of wall time,
  no correctness effect.
- Pinned llama-cpp-python `detokenize` uses a 32-byte buffer; an unusually
  long token piece would fail the round-trip gate closed, not silently.
- Multi-step GGUF generation reads correctly from the pinned `eval` and
  `reset`, but the test double hard-codes one step, so an off-by-one would
  not be caught.


### 4.4 Allocator, KV simulator, block training, statistics

No finding invalidates the recorded v2–v4 or post-fix cache numbers. The
pattern is the same as in §4.2: fail-open paths that the project's own
history says are dangerous.

- **`_pareto_selection` in `target_bpw` mode does not optimise the objective**
  (`dynamic.py:1208-1218`, `:1603-1604`). With `tolerance_bytes = 0` and no
  exactly reachable size, the fallback picks the byte-closest state, then
  score; 149 of 300 random instances came back suboptimal, worst 2.9× the
  under-budget optimum, with `target_reached: True`. This is the dataclass
  default mode (`target_bpw=3.625`), and `require_target_match` is forbidden
  there, so it cannot fail closed. The registered v2–v4 configs use byte
  targets with a tolerance and are unaffected. Confirmed.
- **The KV simulator cannot prove decode rows went through it**
  (`eval/kv_cache.py:395, 510-511`). Only `cache.update` is patched; a
  Transformers path that calls `layer.update` directly or returns a new cache
  object stores decode rows unquantised, and the 8-bit endpoint check still
  passes (measured: 2-bit KL 0.0021 on the bypass vs 0.0026 faithful). Count
  hook invocations and assert the returned cache is the patched object.
  Confirmed mechanism; latent in production.
- **Non-tiered 8-bit-scale cache writes are packed per chunk**
  (`eval/kv_cache.py:378-388`): with `recent_window == 0` a 5-token chunk and
  five single writes give different caches at `scale_bits=8`. Registered
  configs use tiers (the invariant path). Decode-row scale metadata is also
  never charged to `packed_kv_bytes`. Confirmed.
- **Missing activations silently switch the allocator's local metric**
  (`dynamic.py:599-604, 786`) to identity-covariance error with no flag, while
  missing Hessians raise. Confirmed.
- **Streamed block training reports zero input drift and trains on
  `source_block(deployed_input)`** rather than the source model's own output
  (`block_train.py:1313`), unlike the in-memory propagated path it claims to
  match. Any recovery-study comparison would be misled. Confirmed.
- **Endpoint check has trivial-pass modes** (`eval/kv_cache.py:281-283, 856`):
  layers without K/V attributes are skipped and `kv_layers` is never compared
  with the model's full-attention count; `endpoint_max_kl=0.01` is ~25× the
  8-bit KL actually observed (4e-4); an all-fp16 tier setting passes.
- **Candidate-score cache key omits dtype, device and code revision**
  (`dynamic.py:375`, `run_experiment.py:1355-1369`); invalidation is a manual
  schema bump. History shows the one numerics change so far did bump it.
- **Solver optimality is unquantified**: the bucketed DP keeps four states per
  256 KiB bucket and the exact MILP runs only when nothing is feasible. The
  same `scipy.optimize.milp` call could solve the 252×6 multiple-choice
  knapsack exactly as an audit, which would settle whether v3's "refinement
  changed nothing" means near-optimal or too-local.
- **Three bootstrap implementations with different RNGs** (`eval/statistics.py`
  torch; `eval/competitive_run.py` and `scripts/run_qwen35_next_stage.py`
  numpy); token-level intervals in `logit_fidelity` and `kv_cache` remain the
  known too-narrow ones, and the registered KV protocol scores 256 tokens.
- Smaller: `teacher_logit_kl` averages per-batch means; protection
  sensitivity picks an arbitrary same-width pair when formats share a width;
  teacher logits stored fp16 (source-arm KL floor ~1e-5); fp16 tiers store
  rotated rows with no finiteness check; `_fake_quant`/`train_kv_rotations`
  are dead and drifted from the evaluator; `_clone_cache` cannot see tensors
  inside arbitrary objects (the next Transformers refactor could re-open the
  1 September bug class silently).

### 4.5 Native runtime, GGUF, llama.cpp, CI, packaging

Built and tested here: Release and ASan/UBSan builds are warning-free under
`-Werror`; both conformance suites pass; an independent fuzz over 88
shape/width cases (bits 1–8, group sizes 1–128 including partial groups,
fp16 subnormal/zero/max scales) found scalar and AVX2 `dequantize` byte-exact
against the Python reference and `matmul` within 1.3e-5 relative. The
hard-coded codebooks, sign tables and butterfly ordering in the llama.cpp
patch match the Python source of truth exactly. The runtime is sound.

- **C ABI enum undefined behaviour** (fixed on this branch, §3):
  out-of-range kernel/status integers from a foreign caller were UB at the
  load; UBSan reported it for the value 99. Now a defined, rejected value.
- **The llama.cpp patch CI never compiles anything** (`llama-cpp-patch.yml`
  runs `git apply --check` only). The patch applies at the pin (`17252c76`,
  29 August) and already fails to apply at upstream HEAD in four files; it
  depends on private converter internals. The Metal path is compiled only on
  the maintainer's Mac. Add a CPU build on ubuntu and a Metal build on
  macos-14 to that workflow.
- **No Windows/MSVC job**; the MSVC branches (`/W4 /WX`, `/arch:AVX2`,
  `__cpuidex`) have never been compiled, and `std::vector` across a
  `__declspec(dllexport)` boundary will warn under `/WX`.
- **CI downloads the CUDA torch stack for lint and all four test jobs**;
  fifteen `nvidia-*` wheels plus triton. Use `uvx ruff` for lint and a CPU
  index for Linux CI (`[[tool.uv.index]]` with `explicit = true` and a
  `tool.uv.sources` entry for torch on `sys_platform == 'linux'`).
- **NEON is never compared against the Python oracle in CI** (only the C++
  self-conformance on macos-14); `native-runtime.yml` ignores changes to
  `rotquant/native_ffi.py` and `tests/test_native_cpp.py`; `setup-uv@v5` is
  unpinned; universal macOS builds would apply `-mavx2` to the arm64 slice.
- **`scripts/inspect_gguf_types.py`** lacks the pinned ggml types 40–42
  (`NVFP4`, `Q1_0`, `Q2_0`, so an artifact using them aborts the whole
  report) and prices RotQuant `.rqweight` tensors at 8 bpw.
- KV-cache min-scale clamp differs between the patch (smallest normal fp16)
  and the Python reference (smallest subnormal); negligible in practice,
  but the "exact vs Python" gate does not hold for near-zero groups.
- Packaging is correct (wheel contents, `py.typed`, no library import of
  `scripts`/`baselines`), but there is no wheel/sdist build in CI, no tag,
  and `tests/` plus two scripts import `scripts.*` as an implicit package.
- **The `baselines` extra is almost certainly non-functional.** The lock pins
  `gptqmodel 4.2.5`, `autoawq 0.2.9`, `aqlm 1.1.7`, all predating
  transformers 5; an unlocked resolution pulls transformers up to 5.16.1, the
  release that broke the KV simulator. No test or notebook exercises
  `baselines/run_baseline.py`, and the paper notes they were never run on
  Qwen3.5. The README's "working GPTQ/AWQ/AQLM wrappers" is unverified.

### 4.6 Documentation and claims

The documentation is large (28 files under `docs/`, a 666-line README) and
has drifted in four ways. The specific instances below were each checked.

- **Withdrawn numbers still asserted or shipped.** The withdrawn 3.25-bpv
  K/V map is hard-coded in `configs/publication_qwen35_joint_cuda.yaml`;
  `scripts/audit_publication.py` still validates the withdrawn 24.0 %/10.6 %
  and matched-seed cache reductions and `paper/generated/audit_report.json`
  reports them `passed: true`; `docs/serving_backends.md` and
  `integrations/llama.cpp/README.md` describe the map as implemented with no
  withdrawal note; `paper/main.tex` keeps the "whole-system weight + KV"
  framing and a headline uniform-W4 result without GPTQ.
- **Four different answers to "what is the next experiment"**: README §1
  (v4 per the performance plan), README §"next run" (packed validation, now
  done), `performance_plan_2026-09-05.md`, and `roadmap.md` (fresh quality
  and seeds 1/2, the correct one). The fresh-quality notebook and runbook are
  not linked from the README at all.
- **Numbers that disagree between documents.** Scale8 W4 complete bytes
  3,759,868,736 vs 3,759,868,416 (and promoted W4 3,787,286,336 vs
  3,787,286,016): traced here to `registered_model_bytes` changing by 320
  bytes between the `8ad3b8e` and `8ba3b75` code eras, an undocumented
  accounting change. Five different "current uniform-W4 KL" values are quoted
  (0.0225, 0.0219, 0.0167, 0.0168, 0.0165) with no canonical one. Native
  decode throughput is 47 tok/s in the llama.cpp README (llama-server) and
  14.7 tok/s in the experiment log and paper (llama-bench), unexplained.
- **Claims without archived evidence.** The three-seed GPTQ promotion (the
  strongest E5 evidence), the OPT ladder, the Algorithm Lab runs, the
  paper's headline PPL figures and exported bytes, and all native throughput
  numbers exist only on Drive or in prose. The seed-0 fresh-quality results
  from the `733bb3d` Colab run are described as completed but recorded
  nowhere. Four compact records carry hashes of raw files that are absent, so
  their per-prompt intervals cannot be recomputed.
- **Version provenance.** `CITATION.cff` and the CHANGELOG describe a 0.1.0
  release dated 31 August; the repository has no tags. Every archived record
  says `rotquant 0.1.0`, so the version cannot distinguish pre- and
  post-fix runs; only `git_sha` can.
- **Stale or orphaned.** `docs/dynamic_allocator_v4.md` (future tense, no
  outcome, linked from nowhere), `docs/turboquant_extensions.md` (orphan),
  three configs referenced by nothing, twenty undocumented scripts including
  the live fresh-eval runner, the README layout block (omits `docs/`,
  `notebooks/`, `native/`, `research/`; says `results/` holds JSONs, which is
  gitignored and empty), the README's Llama-2-7B/13B confirmation bar, the
  README's six-notebook KV/joint/LoRA chain presented as the how-to path
  although its conclusions are withdrawn or retired, `CHANGELOG.md` with no
  entry for the three 8 September commits and duplicated `### Added` /
  `### Changed` headings under `[Unreleased]`, and `docs/experiment_log.md`
  whose "Open experiments" section contains only completed entries and whose
  result summary stops on 30 August.

## 5. Next steps, in order

The ordering principle: one decisive experiment is already prepared, so run
it before touching anything that could change its inputs; then make the
result mean something outside the repository; then build the one runtime
path that lets the winning recipe be served and compared on the provider's
own engine. Everything else waits or is parked explicitly.

### Phase 0 — before any more GPU time (days)

1. Merge this branch so `main` is green at the revision the notebook will
   check out. Pin that SHA in the notebook; do not push to `main` again until
   the run has finished, because the reuse receipt binds to HEAD.
2. Run the fresh-quality notebook as documented: `reuse` from the `733bb3d`
   root, then `bridge`, `unsloth`, seeds 1 and 2, `summary`. Run `--dry-run`
   for every phase first. Confirm the Drive budget (about 140 GB) and that the
   cached llama-cpp-python build is intact before starting the clock.
3. Archive the compact evidence the same day, including the two failed roots
   (`bdf6595`, `733bb3d`), and record the seed-0 numbers that already exist.
4. Apply the decision rule written in `fresh_quality_run_2026-09-08.md`
   without amendment: W6 stays the budget candidate unless W8 wins
   consistently across seeds and domains.

### Phase 1 — interpret the result honestly (days)

- If W5 plus compressed vocabulary beats UD-Q4_K_XL on the common inputs,
  against the common FP16 teacher, at no more than 1 % more bytes, in all
  three seeds: that is the first fair matched-size result the project has
  had. Freeze the recipe (`b5_v6` or `b5_v8`), make it the headline, and
  stop allocator work.
- If it does not: the informative follow-ups are narrow. First, spend the
  ~150 MB of headroom under the W6-vocabulary budget on the projections the
  provider keeps at Q6_K/Q8_0 (`ssm_out`, `down`), as the results review
  proposed, using the existing v3 sensitivity ranking rather than a new
  sweep. Second, test a larger calibration set (128 × 512 tokens gives the
  9,216-wide `down` Hessians about seven samples per dimension). Neither
  needs a new allocator generation.
- Either way, replace the README narrative with the result and its limits.

### Phase 2 — make the result mean something outside the repository (weeks)

- Choose one canonical held-out protocol and build it: either the registered
  300-prompt competitive contract (`docs/competitive_eval.md`, five
  prerequisites untouched since 31 August) or the authored-task suite,
  expanded to several families per domain. Keep it separate from the
  development prompts, which are now spent.
- Add task-level metrics through `lm-eval` (it is in the `eval` extra and has
  never been used): a maths set, an instruction-following set and one code
  set, same prompts for source, the RotQuant artifact and the provider GGUF.
  Perplexity and KL cannot see instruction-following regressions.
- Replicate with an independent calibration corpus. Seeds 1/2 vary only the
  quantisation RNG on fixed calibration documents.
- Archive, or re-run under current code, the three-seed GPTQ promotion. It
  is the strongest weight-only result and exists only on Drive.

### Phase 3 — one runtime path for the winning recipe (weeks)

This is the largest gap. The recipe that wins on quality has no runtime, and
the comparison it is losing or winning is against an artifact that does.

- Extend the native-v2 block format and the GGUF exporter to W5/W6/W8, define
  an 8-bit-scale layout (or refuse scale8 export, L5), and add the shared
  packed tied vocabulary. Then update the llama.cpp patch (CPU and Metal) and
  make the patch workflow compile, not just apply.
- Reasons to do this before a CUDA kernel: the provider artifact is a GGUF
  served by llama.cpp, so a same-engine, same-tokenizer comparison removes
  the teacher/engine confound that currently blocks every provider claim;
  the native runtime, conformance suite, patch and Apple hardware already
  exist; and there is no GPU CI to validate a CUDA kernel. A Triton W5 GEMV
  and the vLLM plugin follow once the GPU CI decision below is made.
- Measure resident memory and decode/prefill throughput on named hardware
  at that point, and not before.

### Phase 4 — KV cache: re-measure or park

Nothing in the cache track has valid quality evidence. Either repeat the
2-bit E8P long-context check on the fixed simulator with the endpoint check
recorded, a single fixed FP16 full-cache teacher, and at least 20 prompts
(roadmap items still unchecked), or park the track explicitly. In both cases
remove the withdrawn 3.25-bpv map from `configs/publication_qwen35_joint_cuda.yaml`,
`scripts/audit_publication.py` and the serving-backend docs now.

### Phase 5 — housekeeping that removes recurring cost (days, interleaved)

- Tag `v0.1.0` at the 31 August commit the citation describes, then bump
  `__version__` to `0.2.0.dev0` so result provenance distinguishes runs made
  after the scale-encoder and rotation-precision fixes.
- Land the §4.2 library queue in one PR, with the listed tests, as soon as
  the reuse freeze lifts. L1, L2, L5 and L6 first.
- Decide GPU CI now: a manually triggered scheduled job (Modal, RunPod or a
  self-hosted runner) that runs the synthetic CUDA preflight and the tiny
  Qwen reload probes is ten minutes of GPU time and would be the first CUDA
  evidence the repository can reproduce without a person in Colab.
- Decide activation quantisation: declare it out of scope for the kernel
  contract until a native A8 GEMM exists. W4A8 cost 6.9 % KL for no speed
  evidence, and the decision has been pending since 31 August.
- Restructure the README into a short front page (what it is, current
  result, next run, install, API, links) and move the run-by-run narrative
  into a history document. Keep one status file as the single source of
  truth for "current result, canonical numbers, next run" and make every
  other document point at it. That is the fix for the four-answers problem.
- CHANGELOG entries for the 8 September commits; fix the duplicated
  headings; document the 320-byte accounting change.
- Either test the `baselines` extra (an import smoke in CI) or mark it
  unmaintained and drop the README claim.
- Re-scope or park the paper. Its audit script must stop passing withdrawn
  numbers either way.
- Make the older notebook builders deterministic (cell ids) so regeneration
  does not produce spurious diffs, or delete builders for completed
  experiments.

### What not to do next

Another allocator generation; a LoRA or distillation recovery sweep; the
27B model; a vLLM plugin before a GPU CI mechanism exists; new model
families; another notebook in the builder inheritance chain. Each of these
is either answered by the data already in hand or blocked on the items above.

## 6. Verification record

- Environment: `uv sync --locked --extra dev --extra eval` (torch 2.12.0+cu130,
  transformers 5.9.0, numpy 2.4.6, scipy 1.17.1, Python 3.12.3), no GPU,
  4 cores, 15 GB RAM.
- `uv run ruff check .`: clean. `pytest tests/ -q`: 695 passed, 17 skipped
  (NEON cases on x86). One failure in a first run was caused by this review
  committing mid-run (the packed-validation identity test hashes HEAD); it
  passes on a stable tree.
- Python CI on this branch (`workflow_dispatch`, run 34281904528): success on
  3.10–3.13 plus lint. Python CI on `main` at `a4ac776`: failure on all four.
- Native: Release (shared, `-Werror`) and Debug ASan/UBSan builds, both
  conformance suites, `--capabilities` (scalar + avx2), the UBSan
  reproduction before the enum fix and its absence after; 98 Python-side
  native tests passed.
- `configs/smoke_cpu.yaml` end to end (tiny random Llama, WikiText-2
  download through the proxy): completes and writes its result JSON.
- `scripts/preflight_packed_validation.py --device cpu` and
  `scripts/run_qwen35_packed_validation.py --dry-run`: pass.
- All 40 configs parse with no dangling file references; all 18 notebooks
  validate as nbformat v4; regenerating the three newest notebooks from their
  builders is byte-identical, and the four older ones differ only in random
  cell ids.
- Dependency check: `uv pip install --dry-run` of the `baselines` extra
  against the locked environment would move transformers to 5.16.1.
- Repository: no secret-shaped strings in tracked files; pack size 3.5 MiB;
  no tags; no open issues or pull requests.
- Not done: no model download beyond the tiny fixtures, no Colab, no Drive,
  no GPU, no Metal build, no llama.cpp compile.
