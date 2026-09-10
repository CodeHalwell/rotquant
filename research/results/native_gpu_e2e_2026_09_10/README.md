# Native GPU end-to-end handoff evidence — 10 September 2026

These are development correctness checks, **not retained Qwen3.5-4B results**.
No paid GPU job or pretrained model was run here, no recipe changed, and no
throughput, VRAM minimum or quality improvement is claimed.

- `linux_load.json`: real GCC build and `ctypes.CDLL` in Linux aarch64 with
  Python 3.13, `-Wl,--no-undefined`, and the repaired pinned llama.cpp patch.
  This verifies the missing-symbol fix on Linux, not x86/CUDA conformance.
- `metal-workflow.json`, `metal-summary.json`: exact local driver attempts and
  artifact fingerprints. All eight synthetic-only stages pass. A repeat run
  verifies/reuses six completed stages and rechecks environment and binding
  load. Active time includes both invocations and cached compilation; it is
  not a cold-build or inference benchmark.
- `metal-environment.json`, `metal-build-receipt.json`: actual package versions,
  Apple M5 Max environment and library/source hashes. These pre-commit runs
  truthfully report base `dd87da2` with local edits. The local test used its
  current environment; the Colab managed-venv installation was not run here.
  Package metadata and the CMake executable's reported version are distinct:
  the local PATH selected Homebrew CMake, while the Colab driver prepends its
  managed environment's bin directory.
- `operators-CPU.json`, `operators-MTL0.json`: canonical Torch versus native
  packed operator checks, all 18 cases per backend.
- `whole-model-w6.json`, `whole-model-w8.json`: random two-layer hybrid Qwen
  graphs, 1/4/17/64-token prompts with eight cached steps. Metal uses the
  explicit `GGML_METAL_TENSOR_DISABLE=1` setting, not silent CPU fallback.
- `hf-conversion.json`: offline random Transformers checkpoint through the
  real converter to CPU/Metal, using frozen token IDs and a fake test tokenizer.
  It does not establish real-tokenizer or actual-4B export parity.

## Reproduce local synthetic execution

On a supported Mac with the pinned dependencies and required Xcode tools:

```bash
GGML_METAL_TENSOR_DISABLE=1 .venv/bin/python -u scripts/run_native_gpu_validation.py \
  --output-dir build/rq3-e2e-handoff \
  --work-dir build/rq3-e2e-local \
  --source-root build/no-retained-source \
  --backend Metal --synthetic-only --current-environment \
  --allow-dirty-local-test --jobs 4
```

Repeat the same command to verify resume. The recorded run uses an existing
isolated native source/build cache. New empty cache paths require compilation.
Do not use the local-test switches in the Colab run.

## Remaining execution and presentation gaps

The notebook schema, generated-source identity and code-cell execution order
were checked. Code cells were executed with explicit mocked Colab/GPU/git/driver
operations; these tests cannot establish real Colab success. Managed dependency
commands are covered by a separate mocked test which checks the Torch pin,
virtualenv isolation and PATH restoration.

An HTML preview was rendered to
`build/native_gpu_notebook_review/qwen35_4b_native_gpu_e2e_colab.html`; browser
inspection was denied by the tool's file-URL policy. No visual QA is claimed.
For the real remaining checks, open
`notebooks/qwen35_4b_native_gpu_e2e_colab.ipynb` in Colab, inspect the rendered
instructions, select a fresh CUDA runtime, confirm the original checkpoint
path and select Runtime → Run all. The default retained W6 arm runs with all
numerical gates intact. Stop/delete the GPU runtime after completion or failure.
