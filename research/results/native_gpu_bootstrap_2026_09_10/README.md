# Native notebook bootstrap regression — 10 September 2026

The user's `1507627bca7b/run1` stopped during `venv`'s `ensurepip` step. No
native build, GPU test or retained-model evaluation ran. The pasted traceback
does not contain ensurepip's underlying stderr; a missing/broken ensurepip is
not distinguished by that log. This correction removes the dependency entirely.

## Actual local check

`bootstrap-regression.json` and `summary.json` are unmodified outputs of
`scripts/check_native_gpu_environment.py`. It calls the **same installation
helper as the production driver**, with real subprocesses/network installs;
it does not use `--current-environment` or mock pip.

Environment: disposable Docker Linux aarch64, Python 3.13.15, pip 26.2.1,
Torch 2.12.0+cpu. Base image:
`python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a`.
Only the container's ensurepip directory was renamed to disable it; the host
interpreter and Colab session were not modified.

- Original command: fails in ensurepip and leaves `bin/python`, reproducing
  the observable error (`reproduced-ensurepip-failure.log`, `reproduction.json`).
- Fresh pip-less venv: all pinned dependencies and editable RotQuant install;
  Qwen3.5 class imports and a tiny CPU Torch operation pass.
- Partial venv from the reproduced failure: same install/import checks pass.
- Repeat setup on that repaired venv: same checks pass without clearing it.
- Every case checks target prefix, inherited Torch version/location and base
  distribution inventory. Base inventory and Torch entry-point hash match
  before/after. This is not a bytewise audit of every base-environment file.

The first test pass used an inventory including source-tree egg metadata;
the final recorded check scopes it to base-installed packages, so editable
source metadata cannot cause a false failure on a clean CI checkout. Final
helper/driver/test source hashes are retained in the raw report. No production
numerical code, checkpoint, format or parity threshold changed.

## Reproduce without renting a GPU

From the repository root, with Docker available:

```bash
docker build -t rotquant-bootstrap-ci - < tests/native_gpu_environment.Dockerfile
docker run --rm -v "$PWD:/workspace" rotquant-bootstrap-ci
```

The new CI job uses these commands on Linux x86-64, with the registry's
Python 3.13.15 multi-architecture index pinned in the Dockerfile (different
from the older locally cached image digest recorded above). Local evidence
above is aarch64, not an already-completed x86-64 CI run. The fixture downloads CPU
Torch plus dependencies, never a model or NVIDIA runtime. Use a new output
directory for another local run, preserving old reports.

The local recorded command was:

```bash
docker exec rq3-bootstrap-ensurepip-check python -u scripts/check_native_gpu_environment.py \
  --work-dir /tmp/rq3-bootstrap-test-release \
  --output-dir /workspace/build/rq3-pipless-bootstrap-release
```

## Boundary

This verifies actual setup/imports on Linux/Python 3.13 with missing ensurepip,
not the complete Colab preinstalled-package set, NVIDIA execution, retained
4B parity or model quality. Production CUDA checks remain mandatory and were
not bypassed. Notebook format/alias/ordered-cell mock tests pass; its updated
HTML preview was rendered, but visual inspection remains unavailable under
the file-URL restriction encountered during the previous handoff. Complete
the outstanding notebook/CUDA check by opening the published notebook in a
fresh Colab GPU runtime, retaining the original `SOURCE_ROOT`, and selecting
Runtime → Run all. No paid GPU session was started for this local correction.
