# Ternary QAT study · version 1

The [new Colab notebook](../notebooks/qwen35_4b_ternary_research_colab.ipynb)
compares a pinned Qwen3.5-4B source with fresh W6/W6, W5/W6 and W4/W6 controls,
an open ternary quantization-aware training (QAT) recipe, and the official
Bonsai2-4B model. It measures quality, complete model bytes and runtime speed.
The [W4 notebook](../notebooks/qwen35_4b_w4_research_colab.ipynb) remains separate.
Neither notebook contains precomputed model results.

## Run it

1. Open the ternary notebook in Colab. Select an A100-class GPU with at least
   40 GB memory and high host RAM.
2. Choose **Run all** and upload the supplied source ZIP. Alternatively, paste
   its saved Drive path into `SOURCE_ARCHIVE`. Every extracted source file is
   checked against the bundle manifest, including on resume. Git mode requires
   a published full commit containing the new study.
3. Start with `SEEDS = [0]`, `PUBLIC_SAMPLES = 16` and
   `BUDGET_MINUTES = 480`. The total execution allowance is divided across
   seeds; dependency setup and fast Hadamard installation happen before that allowance.
   The driver's native runtime builds count within the allowance.
   The pilot can stop with partial results when its budget expires.
4. Read the plan before the experiment starts. Set `RUN_EXPERIMENT = False`
   to prepare the environment and inspect the plan without running models.
   Algorithm settings belong in
   [the versioned config](../configs/qwen35_4b_ternary_research_cuda.yaml).
5. Read the status, quality/size, task, paired-difference and speed tables.
   Download the report ZIP. Checkpoints and receipts remain on Drive.
   Disconnect and delete the Colab runtime when finished.

The notebook ports the W4 setup: a separate `uv` environment that retains
Colab's exact CUDA PyTorch version and location, pinned evaluation packages,
the fast Hadamard build and smoke test, and persistent setup logs. Source and
configuration fingerprints determine result directories under
`MyDrive/rotquant/ternary_research_v1`. Changed inputs get new directories;
rerunning unchanged inputs reuses only verified completed stages. Do not run
two sessions against one result directory. An expired wall-clock budget does
not reset when Colab reconnects. A fully completed run reloads its verified
receipts without starting GPU work. Completed quality and task measurements
survive a later timing failure; the affected arm still shows as failed.

Use the existing bundle builder to include current uncommitted implementation
files without publishing them:

```sh
uv run --no-sync python scripts/build_w4_research_bundle.py \
  --output /absolute/path/outside/the/repository/rotquant-ternary-research-source.zip
```

From an already prepared CUDA environment:

```sh
uv run --no-sync python scripts/run_qwen35_ternary_research.py \
  --config configs/qwen35_4b_ternary_research_cuda.yaml \
  --output-dir /absolute/path/to/new-results \
  --work-dir /absolute/path/to/local-build-cache \
  --budget-minutes 480
```

Add `--dry-run` to inspect the protocol without loading models. Retained source
weights, teacher logits, Hessians, QAT optimizer snapshots, checkpoints and
exports require substantial disk space. Start with at least 150 GB free on
Drive and check the available local disk; this is a planning allowance, not a
measured final footprint. Failed or unfinished stages remain visible.

## What is compared

| Label | Arm |
| --- | --- |
| `fp16` | Original source exported through the study runtime |
| `w6_v6` | Fresh W6 backbone and W6 tied vocabulary |
| `w5_v6` | Current W5 backbone/W6 vocabulary recipe, freshly prepared |
| `w4_v6` | W4 backbone/W6 vocabulary baseline |
| `w4_v6_recovered` | W4/W6 with separate teacher recovery |
| `ternary_ptq` | Ternary initialization, before block QAT |
| `ternary_blockqat` | Same ternary representation after block QAT |
| `ternary_recovered` | Block-QAT candidate followed by teacher recovery |
| `bonsai2_4b` | Pinned official Bonsai2-4B, measured on this run's inputs |

The default pilot makes eligible MLP matrices ternary and keeps the remaining
eligible backbone at W4. It prepares that W4 backbone before adapting the
ternary MLPs. Set `ternary.scope: all` for the wider backbone experiment;
every result records its actual `precision_scope`. The tied vocabulary stays
W6 and protected state tensors retain source precision. This creates a larger
storage floor than Bonsai's ternary vocabulary. Keep that difference visible
when comparing complete bytes.

The official Bonsai arm uses its own compatible runtime path, with the same
quality and speed inputs. Its model and runtime identities belong in receipts.
Published Bonsai scores are never substituted for missing measurements.
The initial pins are model revision
`8d6557168168f91ea6697a4a49ed2ef622b53adb` and Prism runtime revision
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`; expected file hashes are recorded in
[`ternary_research_bonsai.py`](../scripts/ternary_research_bonsai.py).
The default `PTQ1_0` language file is counted together with the vendor's
BF16/F32 vision sidecar. RotQuant retains the source's FP16 auxiliary tensors;
the precision difference remains identified in the receipts. This study
evaluates text, not vision quality.

## Open training recipe

Ternary layers combine fixed Hadamard rotations, three weight symbols and
group scales. QAT keeps FP32 latent weights and uses a straight-through
estimator through hard ternary assignments. The forward pass uses scales
rounded to their FP16 storage representation. A fixed half-scale threshold
defines the three symbols; learned thresholds are a future ablation.
The initial native study fixes rotation width and group size at 128. It runs
up to 100 QAT steps per block with latent-weight learning rate `1e-4`, scale
learning rate `1e-3`, and validation every ten steps. The notebook applies
each selected seed to both the global experiment and ternary rotation config.

Block training uses inputs from the already deployed prefix and independent
validation inputs to choose the retained state. Completed blocks and
in-progress latent/optimizer state are saved for resume. A zero-step arm
measures initialization independently of training. Recovery remains a separate
arm so its effect can be compared with block QAT alone.
The block objective is mean squared output error divided by the source output's
mean square, with a `1e-12` denominator floor. Source and student blocks see
the same deployed-prefix inputs. The study uses fixed-length unpadded token
windows; the block trainer does not claim a padding-masked loss.
Block optimization computes gradients in FP32 using compatible Qwen reference
operations. Validation restores source tensor precision and source operations
before selecting the best checkpoint; export preserves protected source tensors.
Recovery allows up to 200 fixed-code, scale-only steps, using temperature-2
teacher KL with a top-128-plus-tail target and supervised next-token CE weight
0.1. Teacher KL weight is 1.0. These settings are explicit in the config.
During FP16 recovery, a model-scoped L2-normalization fallback keeps the original
forward values and computes its analytical backward pass in FP32. Fused FLA
operations remain unchanged. Receipts record the numerical policy, token
presentations, attempt time and peak CUDA allocation when measured on CUDA.
Each comparison row retains these details in `training_metrics`, including
cumulative optimizer token presentations, time for the current attempt and
measured peak CUDA allocation. A resumed attempt's wall time is not the total
cost of every earlier attempt.

The native representation initially stores three symbols in two-bit slots.
That is not a measured 1.58-bit full-model payload. Scales, rotations, higher
precision vocabulary, auxiliary tensors and file metadata also consume bytes.
Dense-trit packing and further vocabulary compression are later experiments.

This version implements block QAT followed by separate scale-only recovery.
The following research arms remain unimplemented: H-aware discrete ternary
fitting, global recovery that changes weight assignments, vocabulary adaptation,
512-wide rotations, learned ternary thresholds and dense-trit storage. The
current two-bit native format can test the training recipe before those changes.

This is an open research implementation, **not a reproduction of PrismML's
private training recipe**. Bonsai2's public reports disclose rotated ternary
weights and runtime formats but not the complete training objective, data or
compute budget. Its [small-family paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-small-whitepaper.pdf)
documents 512-wide rotations for 4B and long thinking-mode evaluation budgets.
That motivates a comparison; it does not establish that a short Colab run will
reach the published quality.

## Read the evidence

- **Quality:** held-out teacher KL, negative log-likelihood/perplexity,
  top-1 agreement, long-context checks and document-bootstrap intervals.
  Calibration, training, validation, selection and final tests are separate.
- **Tasks:** fresh generated answers with task counts and truncations.
  Accuracy is displayed as a percentage; paired deltas and interval bounds
  use fractions. Sixteen examples per benchmark are a pilot, not a
  non-inferiority result. Quantization seeds do not create more independent
  evaluation examples.
- **Size:** plots use `native_payload_bytes`, the complete deployed payload
  including retained auxiliary assets. `language_payload_bytes` and
  checkpoint-directory bytes are separate columns. Language-only Bonsai
  marketing sizes cannot replace measured complete bytes.
- **Speed:** matched real prompts and frozen decode tokens, excluded warmup,
  medians and repeated measurements. All arms use 16-token prefill chunks;
  parity, task generation and timings use the same chunk schedule. This measures
  end-to-end chunked prefill, including bridge synchronization and readback.
  Runtime/backend identity includes `chunk16` and stays visible;
  separate backends receive separate plot panels. Raw speed does not establish
  answer quality or the time needed to solve a task.
- **Status:** pending, measured and failed arms stay distinct. Missing values
  remain empty and are omitted from plots; a measured zero score stays zero.
  Successful measurements from a partial run remain inspectable.

Selection uses its dedicated data, never final task scores or test KL. Native
operator/model parity, source and artifact identities, and completed receipts
are required before interpreting a result. An improvement in reconstruction
loss alone does not establish a smaller model with preserved capability.

## Local checks

```sh
uv run --no-sync python scripts/build_qwen35_ternary_research_notebook.py
uv run --no-sync pytest -q tests/test_ternary_research_notebook.py
```

The focused checks validate the notebook schema, compile every code cell and
its embedded renderer, preserve the W4 setup, exercise seed/config resume
identity, and render synthetic partial reports. They do not train Qwen3.5-4B
or validate Colab CUDA performance. The notebook ships with empty outputs so
synthetic validation images cannot be mistaken for measured model results.
