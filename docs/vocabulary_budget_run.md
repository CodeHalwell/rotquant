# Qwen3.5-4B vocabulary-budget runbook

The seed-0 screen completed successfully on 6 September. See the
[results review](vocabulary_results_2026-09-07.md). For the **next** run use the
[packed-vocabulary validation notebook/runbook](packed_vocabulary_validation_run.md);
the instructions below reproduce the earlier nine-arm screen.

Use [the vocabulary-budget notebook](../notebooks/qwen35_4b_vocabulary_budget_colab.ipynb).
This replaces another allocator-v4 run. The [plan](vocabulary_budget_plan_2026-09-06.md)
explains the hypothesis and the later gates.

## What to run

Use a fresh Colab A100 40 GB-class GPU runtime with high host RAM and Drive
persistence. Run the notebook top to bottom, leaving these defaults:

```python
SEEDS = (0,)
RUN_SCREEN = True
FORCE_RERUN = False
REQUIRE_FAST_HADAMARD = True
RUN_PROVIDER_HEADER_AUDIT = True
```

The code must first be published to the selected `REPO_REF`. The notebook resolves
`main` to an immutable commit and checks that the new runner exists; it will not
silently run an older notebook's experiment. Pin that full commit when resuming.
It keeps Colab's GPU-compatible Torch and uses the previously tested evaluation
dependency pins. A bounded source build handles the optional Hadamard kernel;
failure gives a persistent build log instead of guessing a nonexistent wheel.
HF authentication is optional for these public files and secrets are never logged.

There are nine cells in the experiment matrix:

| Backbone | Vocabulary arms | Purpose |
|---|---|---|
| Source FP16 | FP16, W8, W6 | Isolate vocabulary-only damage |
| Uniform W4, scale8 GPTQ | FP16, W8, W6 | Repaired control and vocabulary interaction |
| Uniform W5, scale8 GPTQ | FP16, W8, W6 | Test spending the saved bytes on the backbone |

Here “uniform” means one backbone bit width, not equally spaced codebook levels.
All compressed vocabulary arms use Gaussian codes, normalized FWHT, group128,
MSE scales stored in FP16, and no GPTQ. The source matrix remains shared by input
embedding and output head. No weights, rotations or adapters are trained.

Source teacher logits/sequences are captured before mutation; source Hessians
are reused across W4/W5 and all vocabulary arms. Each backbone is quantized once,
then its three vocabulary treatments are evaluated. The first run uses the existing
small development suites, not an untouched final benchmark. Results include
primary/diverse KL, top-1 agreement, 32-token trajectory agreement, both perplexities,
per-prompt paired contrasts, scale diagnostics and component byte ledgers.

The pinned provider header audit records actual GGUF tensor types and nominal
payload bytes; **it does not rerun Unsloth quality evaluation**. A fresh matched
comparison comes after the quality screen and real export checks.

## Progress, reconnects and storage

Every subprocess prints its PID and a `tail -f` command. Output goes directly to
`RESULT_ROOT/logs/`, with a notebook heartbeat every 30 seconds and runner GPU
heartbeats every 60 seconds. `RESULT_ROOT/progress.json` records loading,
calibration, teacher capture, vocabulary construction, backbone quantization,
evaluation and completion/failure. Vocabulary chunks print row counts and ETA;
source calibration and backbone patching print layer progress.

A browser disconnect is not a stop command. Reconnect and inspect the printed
PID/log before starting another process. An explicit cell interruption stops its
subprocess group. A new runner refuses to share an output root with another live
runner.

On runtime loss, select the same commit and settings and rerun the setup and screen
cells. Keep `FORCE_RERUN=False`:

- Completed arm JSON files resume only with matching checksums and exact
  source/configuration/runtime/data identities.
- Vocabulary chunks and source Hessians persist in Drive and resume by hash.
- An interrupted backbone patch is rebuilt from completed Hessians. Full teacher
  references are recomputed; there is no claim of exact instruction-level resume.
- A changed revision, GPU/runtime identity or damaged cache fails closed. Use a
  new result directory for a genuinely different run; do not hand-edit hashes.

The runner prints a shape-based resource ledger after model load: source tensor
bytes, Hessian storage and active-group sizes, a host-reference upper bound and
reported filesystem/GPU free space. These are component estimates, not a peak
memory or runtime guarantee; Drive filesystem free space is not its quota API.

Allow substantial Drive space for Hessians, vocabulary chunks and HF downloads
(the latter remain in local Colab cache). Check Drive quota before starting: no
automatic quota purchase, deletion or unvalidated capacity guarantee is made.
The compact download deliberately excludes these large caches and model binaries.
Keep them in Drive if further resume is needed.

## How to interpret the result

The notebook reconstructs compressed vocabulary values into the original dense
tied parameter. This is an inexpensive implementation path for a **quality
screen**, not memory-efficient serving. Its projected artifact byte columns
include an explicitly estimated container allowance; they are not measured file
sizes and cannot satisfy artifact or provider promotion gates.

At most W5/W8 and W5/W6 advance as quality candidates if primary KL improves over
W4/FP16 vocabulary without failing the common quality guards. A seed-0 candidate
is not a promoted recipe. If neither advances, inspect vocabulary-only damage and
scale diagnostics before spending on allocation or recovery.

Bring back the compact ZIP. The next gated work is full-Qwen packed export/reload
parity, exact total bytes, vocabulary-conditioned W4/W5/W6/W8 allocation if useful,
independent confirmation and fresh final manifests/provider evaluation. The
experimental shared packed wrappers and checkpoint-v3 foundation exist, but this
notebook does not claim fused GPU kernels, native-engine support or completed
Qwen packed-artifact validation. Ordinary packed checkpoints remain v2.

## Local validation and execution boundary

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight_vocabulary_budget.py --device cpu
.venv/bin/python scripts/run_qwen35_vocabulary_budget.py --output-dir /tmp/rotquant-vocabulary-plan --dry-run
.venv/bin/python scripts/build_qwen35_vocabulary_budget_notebook.py
```

The offline test runs all nine arms on a tiny model with real backbone patching
and verifies a second run needs no model load. Tests also cover metadata precision,
bounded packing, exception restoration, shared checkpoint reload, tiny Qwen3.5
hybrid generation, corrupt/partial cache handling and notebook schema/code cells.
Full-size Qwen3.5-4B, CUDA numerical parity, GPU headroom and performance have not
been executed on the local Mac. The notebook's synthetic CUDA preflight must pass
before the full screen starts.
