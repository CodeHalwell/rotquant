# W4 research comparison

The [Colab notebook](../notebooks/qwen35_4b_w4_research_colab.ipynb) compares the
pinned FP16 Qwen3.5-4B source, a fresh reconstruction of the current W5/W6 recipe,
and W4/W6 candidates. It measures text quality, complete model payload bytes and
native CUDA speed. Results are created by the run; the notebook contains no
precomputed model scores.

## Run it

1. Open the notebook in Colab and select an A100 with at least 40 GB GPU memory
   and high host RAM.
2. Run all cells and upload `rotquant-w4-research-source.zip` when prompted.
   The ZIP contains this implementation, including uncommitted source changes.
   It checks every source file against its manifest. Git mode requires a
   published full commit containing the new implementation.
3. Keep `SEEDS = [0]` and `PUBLIC_SAMPLES = 16` for the initial pilot. The
   default execution allowance is 480 minutes, divided across the chosen seeds.
   Dependencies and kernel setup run before that allowance. Longer studies
   should use separately planned runs and fresh result directories.
4. Read the quality/size, task and speed tables, figures and paired task
   intervals. Download the compact report ZIP. Checkpoints, logits and complete
   receipts remain on Drive. Disconnect and delete the runtime when finished.

Allow substantial Drive space: the run retains the FP16 model, dense teacher
logits, source Hessians, eight quantised checkpoints and eight native exports.
Together these require tens of gigabytes; 150 GB free is a conservative starting
allocation, not a measured final footprint. Runtime caches also consume local
disk. Completed stages resume only when inputs and output hashes match. A
reconnected GPU may need the local native runtime rebuilt. An expired eight-hour
wall deadline does not reset when the notebook reconnects.

To regenerate the source bundle locally without publishing the branch:

```sh
uv run --no-sync python scripts/build_w4_research_bundle.py \
  --output /absolute/path/outside/the/repository/rotquant-w4-research-source.zip
```

The same workflow can run from a prepared CUDA environment:

```sh
uv run --no-sync python scripts/run_qwen35_w4_research.py \
  --config configs/qwen35_4b_w4_research_cuda.yaml \
  --output-dir /absolute/path/to/new-results \
  --work-dir /absolute/path/to/local-build-cache \
  --budget-minutes 480
```

Add `--dry-run` to inspect the protocol without loading a model. The notebook
uses a separate environment while retaining Colab's exact CUDA PyTorch package
and location. It saves setup logs and checks the fast Hadamard kernel before
starting the study.

## Experiment matrix

All quantised arms hold the vocabulary at W6/group128, the FWHT block at 128,
the source checkpoint, calibration tokens, GPTQ settings and scale8 storage
fixed. W4 group sizes and FWHT block size are independent.

| Label | Backbone | Change being measured |
| --- | --- | --- |
| `fp16` | FP16 | Original source, exported through the same native runtime |
| `w5_v6` | W5/group128 | Current W5/W6 recipe, freshly prepared |
| `w4_g128_mse` | W4/group128 | Existing weight-MSE scale search plus GPTQ |
| `w4_g128_h` | W4/group128 | Additional activation-aware scale fitting |
| `w4_g64_h` | W4/group64 | Activation-aware scales with smaller groups |
| `w4_g32_h` | W4/group32 | Activation-aware scales with smaller groups |
| `w4_codes` | Selected initial W4 | Additional discrete-code refinement |
| `w4_recovered` | Same initial W4 | Scale-only teacher recovery |
| `w4_codes_recovered` | Same initial W4 | Code refinement followed by scale-only recovery |

When the selected recipe enables activation-aware scales, scale and code
sweeps alternate; the weight-MSE control retains its original scale policy.
The initial W4 recipe is selected using dense teacher KL on 16 dedicated
selection documents. Code refinement, recovery alone and their combination all
start from that same recipe. The final selection also uses only those selection
documents. Final task scores and test KL never determine the winner. Each arm
is prepared in a new process from the same source; compatible source Hessians
and the unchanged vocabulary are cached. FP16 and W5 serving measurements run
first so a partial experiment retains useful controls.

This implements the first scalar W4 experiment from the scientific review.
Learned channel preconditioning/clipping, fitted shared codebooks, selective
precision and W3/vector coding remain subsequent research ablations.

## Objective and recovery changes

For an input second moment `H = E[x xᵀ]`, reconstruction error for one row is
`D = (w − q) H (w − q)ᵀ`. The quantiser transforms H into the same rotated basis
as the weights. Scale proposals use the full residual projection `(w − q) H`
and exact within-group curvature, followed by a bounded line search. Code
proposals consider neighbouring scalar centroids using the local quadratic
gain. When enabled together, scale and code sweeps alternate.

Every proposal is rounded to the actual storage representation and accepted
only if the full row objective decreases. Scale8 offsets and steps stay fixed;
only representable uint8 scale codes can change. This gives bounded local
improvement of calibration reconstruction, not a global optimum or a guarantee
of improved task accuracy.

Recovery now keeps globally normalised teacher probabilities at the requested
temperature. If `S` is the captured top-k set and `pᵣ = 1 − Σᵢ∈S pᵢ`, its loss is

`T² [Σᵢ∈S pᵢ log(pᵢ/qᵢ) + pᵣ log(pᵣ/qᵣ)]`.

The tail probabilities are computed with masked log-sum-exp for stable
gradients. Identical teacher/student distributions have zero loss and gradient.
The default uses top128, T=2, no supervised CE, fixed rotations and codes, and
at most 200 scale-only steps. Dense held-out KL at T=1 detects distinctions
within the tail that the training bucket cannot represent.

Recovery checkpoints bind the target semantics and temperature. Old recovery
checkpoints are rejected. Scale8 training uses the original storage grid, so a
rejected update restores identical bytes. Resuming an already early-stopped
checkpoint returns the same selected state without another optimisation step.

## Evidence and acceptance

- **Independent inputs:** 128 calibration documents; 64/8/8 recovery
  train/validation/internal-selection documents; 16 outer-selection documents;
  24 final C4 documents; and two separate 2048-token documents. Evaluation
  excludes known historical C4 rows. Source rows and token hashes are checked.
- **Quality:** full-vocabulary next-token KL against frozen source logits,
  token-weighted negative log-likelihood/perplexity, top-1 agreement and
  document-bootstrap 95% intervals. Tasks use pinned GSM8K, CruxEval and IFEval
  inputs/scorers, identical chat templates, greedy decoding and token caps.
- **Task comparisons:** paired accuracy deltas against FP16 and W5/W6 include
  bootstrap and conservative Hoeffding intervals. The one-per-cent task margin
  is recorded, but the pilot never declares non-inferiority or “no loss”.
  Quantisation seeds are reported separately and do not create additional
  independent test examples.
- **Size:** both checkpoint-directory bytes and complete native model payload
  bytes are measured. Native payload includes the GGUF, retained auxiliary
  weights and tokenizer/processor assets; receipts and logs are separate.
  FP16 auxiliary tensors are checked against the source and counted identically.
- **Speed:** FP16/W5/W4 use the same native CUDA bridge, reference kernel mode,
  F16 cache, fixed real prompts (128/512/2048 tokens), fixed 32-token decode
  continuation, three measured repetitions and excluded warmup. This isolates
  runtime speed from model-specific generation lengths. Memory and input hashes
  are recorded. Checkpoints are loaded separately.
- **Gates:** native operators and tiny graphs must pass before source-model
  work; every packed checkpoint must reload with matching outputs; native
  outputs must pass a separate HF parity gate before quality/speed is accepted.
  Source, configuration, runtime and artifact identities bind all stages.

The reference kernel mode measures this common implementation, not each
format's best possible tuned kernel. The model retains its auxiliary tensors,
but this study evaluates text capability only. A smaller W4 artifact is not
promoted merely because calibration error or KL improves.

## Local verification

Focused tests cover the loss identity/gradient, storage rollback and recovery
resume, activation-aware objective improvements, source/token integrity,
selection isolation, stage failures/resume, source ZIP validation, notebook
generation and readout. A tiny Qwen model passes real W4/group32 scale/code
refinement, W6 vocabulary, recovery, packed export and exact-logit reload.

The pinned CPU runtime builds and loads. Its 36 operator cases and eight tiny
hybrid graphs pass. HF-to-packed-to-GGUF CPU parity passes for W5/group128 and
W4/groups128/64/32, and the FP16 export passes the corresponding source parity.
The new CUDA/Metal paths and real 4B quality, bytes and throughput still need the
Colab run; no new GPU model result is claimed here.
