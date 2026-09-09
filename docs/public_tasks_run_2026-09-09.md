# Qwen3.5-4B public-task release gate

Status update, 9 September: **the user stopped the paid reference-path run**.
FP16 completed in the supplied log; the first W5/W6 arm completed only 30/384
prompts at roughly 0.85 generated tokens/s. There is no complete comparative
task result. Do not launch this nine-arm reference-path sweep again by default.
The next milestone is [native serving](native_runtime_v3.md), followed by model
parity and a short measured speed/memory/cost check before a new run identity.
The original runner/notebook remain unchanged for provenance. This warning is
not an executable interlock in that historical notebook.

Local preparation verified all 2,660 public gold records and exercised the
checker on all 541 IFEval rows. The real pinned Qwen tokenizer froze the standard
384-input subset with all nine artifact identities; maximum prompt lengths were
164 (GSM8K), 198 (CRUXEval-O), and 119 (IFEval) tokens. Positive/negative checker
fixtures and sentence-table checks also pass. These checks neither load the
saved full-model weights nor produce task accuracy measurements.

The historical [public-task Colab](../notebooks/qwen35_4b_public_tasks_colab.ipynb), unlike the old
allocator, vocabulary, packed-validation or fresh-quality notebooks, does not
requantize, train adapters, adjust rotations or change the six saved checkpoints.
Old [results](fresh_quality_results_2026-09-09.md) are neither overwritten nor reused
as public benchmark scores.

## Registered comparison

- Source FP16; pinned Unsloth UD-Q4_K_XL; BF16-GGUF bridge.
- W5/W6 and W5/W8 at seeds 0, 1, 2. Both recipes stay registered regardless of
  seed-0 results. Seeds change quantization/rotation RNG, not calibration corpus.
- Standard scope: deterministic 128-example subset per benchmark, 384 per arm,
  3,456 generations across nine arms. No overall average across benchmarks.
- First run defaults to `SMOKE_RUN=True`: eight per benchmark, five seed-0/control
  arms, a separate root. This only checks plumbing. Then set `SMOKE_RUN=False`.
- `SEED0_ONLY=True` registers a five-arm comparison, explicitly not full seed
  confirmation. `SAMPLE_PER_BENCHMARK=0` selects all 2,660 public examples per
  arm; this is much more expensive. Do not change scope within an existing root.

The standard subset is hash-selected before seeing responses. Selection seed is
20260909, identical for every model. Source row hashes, selected IDs, original
prompts, gold answers, HF chat rendering, token IDs and input hashes are frozen.
Prompts over 2,048 tokens fail preparation; they are not silently truncated or
filtered. There is no semantic or pretraining contamination guarantee. The
benchmarks must not subsequently become recipe calibration/tuning data without
retiring their held-out status.

## Public sources and scoring

| Benchmark | Pinned HF revision | Split / full rows | Measurement |
|---|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f781978c0658806c59bc2815b9866` | main/test, 1,319 | Numeric answer accuracy; terminal `#### NUMBER` |
| [CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) | `b96af0450242eb4da433032b90998f25588a5d0f` | test, 800 | CRUXEval-O typed literal output prediction |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | `966cd89545d6b6acfd7638bc708b98261ca58e84` | train, 541 | Strict/loose prompt and instruction compliance |

GSM8K and CRUXEval are MIT licensed; IFEval is Apache-2.0. Dataset text remains
data, not host instructions. IFEval's published split is named `train`; this run
uses it only for evaluation. CRUXEval measures function-output understanding,
not code-generation pass@1. See the [original CRUXEval project](https://github.com/facebookresearch/cruxeval).

GSM8K accepts a numeric final line (commas/sign/decimal normalization); incidental
numbers in explanations cannot pass. CRUXEval requires one `[ANSWER]literal[/ANSWER]`
pair and type-aware equality with the expected literal. The bounded parser uses
`ast.literal_eval`, never `eval`, `exec`, or a subprocess that executes responses.
No coding agent or live tool calls execute in a Drive-mounted runtime.

IFEval calls the [Google Research checkers](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval)
at `e6890f85757dd84e27ca6df2dd30651dafad28e0`, with four module SHA-256 pins and
the upstream license retained in the cache. NLTK 3.9.2 sentence tables are
plain text pinned to nltk_data `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`;
no pickled resources or implicit download. Checker/language-detection randomness
is fixed independently per example and scoring pass. Missing instructions or
dependencies stop the run instead of silently becoming successes.

All arms use the same source EOS set, neutral greedy decoding and thinking off.
New-token caps: GSM8K 1,024; CRUXEval-O 512; IFEval 2,048. Primary task accuracy
requires both checker success and completed generation. Raw IFEval strict/loose
checker results are retained separately, before this truncation guard. Format
failures, cap hits and unmapped output IDs are explicit. These are custom zero-shot
chat/subset results, not official leaderboard protocols.

## Artifact roots, hardware and progress

Defaults read these **original full** Drive trees:

```text
/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f
  b5_v6_s0/  b5_v8_s0/
/content/drive/MyDrive/rotquant/qwen35_fresh_quality/d4292d6fdec6/replicas
  b5_v6_s1/  b5_v8_s1/  b5_v6_s2/  b5_v8_s2/
```

Each needs its checkpoint, preparation records and saved probe tensors. A compact
download/revalidation folder is insufficient. Preparation hashes, source revision,
recipe configuration and expected bytes are frozen before evaluation; full
artifact hashes and reload/residency probes are checked on consumption. Neither
artifact tree is modified. The full token axis and exact decoded prompt bytes
are checked for both GGUF controls. Native tokenizer differences are audited;
all generation consumes the frozen HF IDs directly. This is controlled input
parity, not native-tokenizer serving equivalence.

Use an A100 40 GB-class GPU and high host RAM. Allow about 35 GB free local space
for source/GGUF/build caches and 1 GB extra Drive headroom for standard text
outputs/logs, in addition to existing artifacts. These are planning estimates,
not measured memory/fit guarantees. The Python packed path may require multiple
sessions. No new Hessians, model exports or full-vocabulary reference-logit
arrays are created. Phase/process and per-prompt progress persists to logs;
60-second heartbeats cover long GPU sections.

Output roots include commit, sample count and seed scope. Pin the printed SHA to
resume after a session loss. Restore the same dependencies/GPU/runtime; changing
the generation runtime intentionally blocks mixed provenance. Binary identity is
recorded for GGUF collections. A rebuilt different binary cannot silently resume
an old collection: preserve that root and start a new one if the identity differs.

Use the printed worker PID and `tail -f` command to check a disconnected session.
Do not start a duplicate worker. Explicit interruption kills the worker process
group; the partially generated prompt restarts, while complete checksummed prompts
resume. Corrupt or mismatched results fail closed. Failures get separate durable
attempt records. Completed-arm resumes do not reload a model.

## Interpretation and next engineering gate

The summary recomputes scores from saved text, checks all registered arms and
refuses to report completion with missing results. It reports per-benchmark
accuracy, Wilson intervals, format/truncation counts, and paired
source-correct → candidate-wrong / wrong → correct flips. Provider and matched-seed
W6/W8 contrasts use paired example-bootstrap intervals, right minus left.
These are descriptive, unadjusted for multiple comparisons and do not pool seeds
as independent prompts. There is no automatic overall winner or promotion.
The 128-example subset is a screen, not a powered non-inferiority study for
one- or two-percentage-point effects. A nonsignificant loss does not establish
equivalence. Identical observed outcomes can give a degenerate paired-bootstrap
interval; that is not proof of equality on unseen tasks. Inconclusive comparisons
need a larger preregistered evaluation, not a claim that no regression exists.

Inspect whether the BF16-GGUF bridge changes task outcomes before attributing
cross-engine differences solely to quantization. Preserve both recipes until
their usefulness at the measured size is assessed. If no material regression is
seen, the next engineering deliverable is one measured packed serving path,
starting with native/GGUF/llama.cpp interoperability and operator profiling.
Actual coding/tool interaction, broader language/model coverage, throughput and
peak-memory tests still precede a competitive product release. No Dynamic-v3.0,
general agent capability, or optimized-runtime claim follows from this run alone.
