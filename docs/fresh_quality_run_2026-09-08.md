# Qwen3.5-4B: fresh quality and recipe replication

Use [the new Colab notebook](../notebooks/qwen35_4b_fresh_quality_colab.ipynb).
This follows successful packed revalidation, not another allocator/LoRA sweep.
The notebook is published on `main`. The full pretrained CUDA notebook has not
been executed locally.

## Starting evidence

Original artifacts `8f10ee60fc7f` were revalidated with loader `89d25f3`.
The [compact archive](../research/results/raw/qwen35_packed_revalidation_89d25f3/evidence_index.json)
preserves 37 JSON/checksum files byte-for-byte, including 14 supplied checksum
pairs and the earlier W6 failure. Logs and absent model/probe tensors are omitted.

| Metric | W5 / W6 vocabulary | W5 / W8 vocabulary |
|---|---:|---:|
| Actual complete artifact bytes | 3,441,544,638 | 3,600,470,206 |
| Primary teacher KL | 0.0047409438 | 0.0040373128 |
| Primary source top-1 agreement | 96.0535% | 96.5998% |
| Diverse teacher KL | 0.01071638 | 0.00979986 |
| Diverse trajectory token agreement | 81.625% | 79.500% |
| Reload probe max/mean error and KL | 0 / 0 / 0 | 0 / 0 / 0 |
| Exact short reload generations | 4/4 | 4/4 |

Both pass their saved development-quality guards. Exact equality is limited to
16 sampled positions and four short generations, not all possible inputs. W6
is 143 MB below the 3,584,533,344-byte target. W8 is 15.9 MB above (0.445%), within
the 1% ceiling but not strictly under budget. These are checked run records, not
an independent local inference rerun; weights/probes remain on Drive. W6's small
trajectory lead is uncertain; W8 has lower KL in each of the five tested domains.
Multilingual is the largest observed fidelity gap, based on only five snippets.

## Frozen experiment

1. **Fresh inputs:** 24 C4 documents at eligible skip 32768, 512 tokens each,
   using the pinned dataset revision. Exclude recorded calibration/development
   source rows and exact old token hashes, including all archived C4 manifests.
   Bind the archive digest and freeze before new inference. Skip 16384 was
   already reserved by the older recovery protocol and is not reused here.
2. **Task diagnostics:** 96 authored unit tasks: 24 each for multilingual
   arithmetic, Python code tracing, JSON structure and tool selection. Spanish,
   French, German, Japanese, Arabic and Hindi are represented. Exact typed JSON
   oracles score answers; generated code and proposed tools are never executed.
3. **Common source:** capture the pinned HF model in FP16 once; save full-vocabulary
   logits as FP32 arrays, generated answers, task outcomes and hashes per prompt.
   Each process loads one model; scoring reads one prompt reference at a time.
4. **Saved seed 0:** verify artifact hashes, recipe, packed ownership and small
   saved reload sentinels; evaluate W5/W6 and W5/W8 on new inputs. Do not repeat
   the old full development evaluation or requantize seed 0.
5. **Provider controls:** evaluate pinned BF16 GGUF against common HF FP16, then
   Unsloth UD-Q4_K_XL against both references on identical contexts. Verify the
   installed engine Git provenance, library hashes, token axis and decoded chat
   bytes. Use frozen HF IDs directly; audit native tokenizer splitting separately.
6. **Seeds 1/2:** one W5 backbone preparation per seed, shared by its W6/W8 pair.
   Reuse the existing export preparation and development/prototype evidence,
   then run fresh-process reload probes and the new quality suite. No allocator
   sees the new evaluation inputs when choosing quantization.

Model/tokenizer: `unsloth/Qwen3.5-4B` at
`3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`. GGUF descriptors/hashes remain pinned
in `scripts/run_unsloth_qwen35_4b_kl.py`. This is the existing Qwen3.5 release,
**not evidence of a Qwen3.5 Dynamic-v3.0 release**.

## Measurements and limits

C4 scores all 511 next-token positions/document. Task generation stops at the
source EOS set or 128 tokens, with thinking disabled and greedy decoding. The
stop set combines source generation/model-config EOS IDs with tokenizer EOS,
identically for HF, packed and GGUF arms. The pinned repository has no separate
`generation_config.json`: use `GenerationConfig.from_model_config` to read its
nested `text_config` instead. Its [model EOS](https://huggingface.co/unsloth/Qwen3.5-4B/blob/3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636/config.json)
is 248044 (`<|endoftext|>`), and its pinned tokenizer EOS is 248046 (`<|im_end|>`).
Both stop generation; otherwise normal chat termination could be scored as
truncation. Resolve/log/validate these IDs before artifact scans and C4 capture.
Only confirmed missing Hub files allow fallback; authentication, network,
offline-cache and invalid-config errors remain failures. KL
uses up to 32 positions on the source continuation. Variable EOS lengths mean
this is **not an exact 32-token Divergence-300 reproduction**.

Report teacher KL/top-1, NLL, trajectory agreement, exact trajectory rate, strict
task success, JSON validity, truncation and correct-to-wrong/wrong-to-correct
flips. The source can be wrong: fidelity is not task accuracy. Truncation, extra
keys, wrong JSON types, duplicate keys, Markdown fences and extra prose fail the
strict task oracle. Generated outputs and per-prompt measurements are retained.

These are short **synthetic diagnostics**, not real coding execution, actual
agent completion or an independently sourced public benchmark. Exact row/hash
checks do not establish semantic near-dedup or pretraining cleanliness. Once
used to tune quantization, these inputs become development data.

Variants share authored families. Paired uncertainty resamples family means,
not tokens or duplicated templates. Multilingual arithmetic currently has one
problem family, so its family-bootstrap CI is explicitly unavailable. Expand
languages/task families before language-wide claims. Seeds vary rotation and
quantization RNG on **fixed calibration documents**, not independent calibration
corpora. Keep results per seed; never pool repeated prompts as independent data.
The summary does not automatically promote recipes or declare provider parity.

The HF-vs-BF16-GGUF bridge exposes combined conversion, precision and runtime
effects. KL is not additive: do not subtract bridge KL from provider KL. Common
FP16 comparisons measure the deployed systems; same-engine BF16 helps diagnose
the quantization contribution on those same contexts. Complete artifact bytes
include retained vision/projector state, but vision is not evaluated. W6 is an
under-budget frontier point, not a two-sided 1%-matched pair; W8 is near-size.
Worker timings are diagnostic, not standardized prefill/decode benchmarks.

The pinned upstream [llama.py interface](https://github.com/abetlen/llama-cpp-python/blob/3691546f1c9e0c1bf93323dff02230bd959cf562/llama_cpp/llama.py)
was checked for raw logits and direct context operations. The tokenizer has
248,077 defined IDs whereas the model has padded output slots. Preserve every
logit dimension and verify real token mappings plus the converter's
[`[PAD{id}]` unused slots](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/conversion/base.py).
Do not slice vocabulary rows and renormalize KL to conceal an axis mismatch.

The first Colab native-tokenizer audit found four mismatches out of 96 authored
prompts: the four Hindi additions, each 44 HF tokens versus 36 GGUF-native tokens.
The pinned HF tokenizer regex uses `\p{L}+`, whereas the pinned llama.cpp
[`qwen35` pre-tokenizer](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/llama-vocab.cpp#L360)
groups `\p{L}` and `\p{M}` (combining marks). This explains the observed
letter/mark splitting pattern; it is not evidence of quantization degradation.
A local in-memory regex-only change to the pinned HF tokenizer reproduced
exactly those four length differences and the reported first six differing
GGUF IDs. All 96 original HF prompts decoded to their rendered bytes exactly.
This diagnostic changed no saved inputs, model weights or experiment results.
The original guard blocked the bridge before any prompt scoring.

The notebook now explicitly selects `--gguf-input-policy frozen-hf`. Full
token-ID/row mappings and padding remain hard gates. For every task, current HF
retokenization must reproduce the frozen IDs, and HF decoding plus GGUF decoding
of **both** frozen/native sequences must reproduce the exact rendered UTF-8
bytes. Native segmentation differences are saved in `tokenizer_audit.json`, not
hidden or substituted into inference. Both GGUF generation and KL already feed
the frozen IDs directly. The CLI default remains `strict`, which also requires
native segmentation equality. Neither mode slices vocabulary logits, drops
Hindi prompts, or changes templates. This experiment measures common-input
model fidelity, **not native-text serving/tokenizer parity**. A native-input
deployment comparison remains a separate follow-up.

## Run and resume

```python
RUN_UNSLOTH = True
RUN_REPLICATION = True
REPLICATION_SEEDS = (1, 2)
REQUIRE_FAST_HADAMARD = True
```

Point `ARTIFACT_SOURCE` at the **original** Drive run with tensors:
`/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f`.
The revalidation folder/compact ZIP are not substitutes. Run top to bottom on
A100 40 GB-class hardware with high host RAM. Plan around **140 GB additional
Drive space** for two replication seeds, checkpoints/Hessians/caches, reference
arrays and margin, plus roughly 35 GB local downloads/build space. Two reference
sets alone have an upper bound near 31 GB. These are estimates, not fit/quota
guarantees; original retained artifacts are additional storage.

For a shorter first session set `RUN_REPLICATION=False`, then re-enable on the
same fixed commit/root. Summary expectations come from controls, not successes.
Inspect `complete` and `missing`. Do not change code/dependency versions mid-run.
The notebook prints commit, root, PID, prompt progress and `tail -f` commands.
Completed prompts and reference arrays are hash-checked before reuse. Failure
attempts remain separate records. A writer lock protects the output root; old
artifacts are never overwritten. Browser reconnect does not block stdout;
explicit interruption terminates the subprocess group.

Compact downloads include output text, prompt metrics, manifests, summaries,
logs and small replication evidence—not model/reference/probe binaries. Retain
the latter on Drive for reproducibility and profiling.

### Recover the `bdf65958e248` generation-config failure

The first Colab source phase failed while reading stop IDs, before loading the
FP16 model or writing any per-prompt quality results. Frozen inputs succeeded;
this is a setup failure, not a quantization result. Keep that folder as evidence.

Use the fixed code on `main` and a **new commit-named result root**. Rerun the
checkout/root cell and the freeze cell before the source cell. Do not merely
pull and rerun source against the old `common` command: it still points to the
old root, whose manifest correctly rejects changed code/stopping policy. There
is no need to reinstall unchanged dependencies or requantize the seed-0
checkpoints. Keep `ARTIFACT_SOURCE` pointed at the original `8f10ee60fc7f` folder.
The freeze logs should now print `stop_ids: [248044, 248046]` before C4 capture.

### Recover the `733bb3d2e477` GGUF tokenizer-gate failure

Do not rerun the completed teacher or seed-0 quality evaluations, and do not
rebuild the verified llama.cpp installation. Update the checkout and establish
a **new commit-named** result root. In the updated notebook, set
`REUSE_FRESH_ROOT` to the old `733bb3d2e477/fresh` directory and keep
`GGUF_INPUT_POLICY="frozen-hf"`.

The `reuse` phase verifies the exact reviewed producer revision/source hash,
unchanged runtime, frozen protocol, original preparation evidence, source
reference tensor hashes, and all completed per-prompt HF/packed records. It
preserves the old manifest fingerprint and records a new consumer/reuse receipt.
Only `source_fp16`, `b5_v6_s0`, and `b5_v8_s0` are eligible; no failed or completed
GGUF collection is imported. Original files are only read, not rewritten.
References remain in the original folder, avoiding duplication of large arrays.
Later reads verify the imported record digests and reference tensor digests;
summary rows expose each producer identity. Compact downloads include the
imported small evidence under `reused/` as well as the receipt. Keep both Drive
folders: the original tensors are still required.

Example phase commands (use the notebook variables for absolute paths):

```python
fresh("reuse", "reuse-completed", ["--reuse-root", str(PREVIOUS_FRESH_ROOT)])
fresh("bridge", "gguf-bf16-bridge")
fresh("unsloth", "unsloth-common-and-bf16")
```

Include `--gguf-input-policy frozen-hf` in `common`. Then continue with seeds
1/2 and the summary cells. In a full notebook rerun, the source/seed-0 cells
verify and reuse the adopted completions instead of running inference again.
The consumer code is pinned too; changing it again requires a new reviewed
recovery, not a bypass of manifest identity checks.

Custom diagnostics must import PyTorch **before** llama.cpp. In the supplied
Colab log, importing llama.cpp first exposed `libtorch_cuda.so: undefined symbol:
ncclCommShrink`; the build linked system NCCL. PyTorch-first order successfully
ran the tokenizer diagnostic. The production runner already uses that order.
No package upgrade, engine rebuild or reference regeneration was required for
that diagnostic. Full CUDA execution of the new bridge gate remains to be run
on Colab; local tests cover the audit, guarded reuse and unchanged HF scoring.

## Local validation and next decision

Offline tests cover strict oracles, clustered pairing, full-logit alignment,
a tiny HF model, vocabulary/padding rejection, checksums, resume, missing arms
and archived failure preservation. Notebook schema and Python cells are checked.
To close the execution gap, publish this code and run the notebook on Colab with
the original artifacts. Full pretrained CUDA and llama.cpp inference are not
available on this CPU development host.

Inspect domain/task regressions before deciding whether W6 remains the budget
candidate or W8 earns its extra bytes. Require consistency across seeds. A
larger independently sourced real-task suite and independent calibration
corpora remain prerequisites for competitive release. Packed-runtime profiling
is the next engineering track; LoRA, allocation changes and engine forks stay
conditional on what the new results actually show.
