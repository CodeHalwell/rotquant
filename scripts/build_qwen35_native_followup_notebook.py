"""Targeted baseline completion and opt-in decode-kernel experiment."""
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_qwen35_native_optimization_notebook import build_notebook as study_notebook
from scripts.build_qwen35_packed_validation_notebook import md


def build_notebook():
    notebook = study_notebook()
    notebook.cells[0] = md('''
        # RotQuant Qwen3.5-4B — GPU baseline completion and decode optimisation

        ## Goal
        Select **A100 40GB**, check the saved source path, then **Runtime → Run all**.
        Keep the existing **W5/scale8 + W6 vocabulary** weights unchanged.

        The September 11 study confirmed **3.30–3.35× faster prefill** with tiled4,
        but decode remained about **19.8–19.9 tok/s**. BF16 stopped because its
        ordinary input embedding was placed on CPU. This run fixes placement
        without weakening the all-GPU gate. The subsequent Q4_0 preflight
        stopped on CPU/CUDA numerical drift despite matching greedy tokens.
        This revision adds a separately compiled **public llama API caller**
        to distinguish bridge disagreement from shared-backend behaviour.
        It uses the same verified native libraries, not an independent kernel
        implementation. Tiny tests run before downloading the 4B controls.

        First finish the **BF16 and Unsloth UD-Q4** comparison. Then test the
        opt-in **decode4** kernel: one warp per output row, shared codebook,
        hoisted group scales and the same floating-point reduction order.
        It retains tiled4 for prefill. **No decode speedup or CUDA correctness
        is claimed yet.** Exact operator and bounded retained-model numerical
        gates precede timing; their thresholds are unchanged.
        No calibration, requantization, adapters or training are run.
        ''')
    controls = notebook.cells[2]
    controls.source = controls.source.replace('RUN_NAME = "study1"', 'RUN_NAME = "followup2"')
    controls.source = controls.source.replace('CONTEXTS = (128, 512, 2048)', 'CONTEXTS = (128, 512)')
    controls.source = controls.source.replace('RUN_PROFILE = True', 'RUN_PROFILE = False')
    controls.source += '\nCANDIDATE = "decode4"  # "none" runs only fresh reference + BF16/UD-Q4\n'
    controls.source += 'assert CANDIDATE in ("decode4", "none")\n'
    controls.source += 'assert CANDIDATE != "none" or (BASELINES and not RUN_PROFILE)\n'
    for cell in notebook.cells:
        cell.source = cell.source.replace('/content/rotquant-native-study/', '/content/rotquant-native-followup/')
        cell.source = cell.source.replace('MyDrive/rotquant/native_gpu_study', 'MyDrive/rotquant/native_gpu_followup')
    notebook.cells[5] = md('''
        ## Steps and checks
        1. Verify saved evidence; build/restore the hash-checked native library.
        2. Compile the small public-API caller against the verified libraries.
           **Tiny BF16 + Q4_0 preflight:** compare private/public callers on
           CPU and CUDA separately, including eight cached decode steps.
           Same-backend numerical disagreement, CPU fallback and CPU/GPU
           token/greedy-trace disagreement remain hard failures. BF16 also
           retains its CPU/GPU numerical gate. Q4_0 CPU/GPU numerical drift
           keeps the original thresholds and pass flags as a **diagnostic**;
           it cannot establish bridge correctness without the new control.
           Raw logits/traces are saved. No 4B baseline download yet.
        3. Fresh RQ3 operators, W6/W8 tiny-model, conversion and saved-model parity.
        4. Fresh reference timings at **128 and 512 tokens**, three measurements
           plus one excluded warmup, on this runtime/GPU.
        5. Pinned BF16 and UD-Q4: verify files/token-ID maps, replay identical
           decode IDs and save comparisons before trying the experimental kernel.
        6. With `CANDIDATE = "decode4"`: test operator arithmetic, partial row
           groups, permutations, long reduction widths, tiny models and retained
           parity, then collect matched candidate timing. The default kernel is
           unchanged. A failed gate stops execution, never relaxes thresholds.

        Set `CANDIDATE = "none"` for a **baseline-only** follow-up. Fresh reference
        timings still run: old-binary timings cannot establish a matched ratio.
        Profiles and 2,048-token measurements are skipped by default because the
        previous study already answered those questions. You can explicitly add
        2048 to `CONTEXTS` or set `RUN_PROFILE = True` with a candidate and a new
        run name. Controls cannot change inside an existing result directory.

        This preflight repair leaves native kernels and the binary-cache key
        unchanged from **2c4037e76f71**. Its cached CUDA build can be restored
        when hardware/toolchain checks match; only the small caller is compiled.
        A cache miss still needs a cold build (~27 minutes previously).
        Original weights are reused, never trained or requantized.
        Baseline downloads total ~11.34 GB. The private Drive cache is retained.
        Every phase prints a persistent log, cap and heartbeat; every repetition
        prints timing and progress. The **90 active-minute** allowance and phase
        caps stop subprocesses, **not Colab billing**.

        After any stop, run Results and Download; do not add manual repair cells.
        Disconnect/delete the GPU runtime when finished. Never update a running
        checkout. Local validation uses mocks/CPU and cannot establish CUDA
        performance; real execution and rendered outputs must be checked here.
        ''')
    launch = notebook.cells[6]
    launch.source = launch.source.replace('if not RUN_PROFILE:',
        'command.extend(["--study-candidate", CANDIDATE])\nif not RUN_PROFILE:')
    launch.source = launch.source.replace('"native-performance-study"', '"native-followup"')
    notebook.cells[7].source += '''

The conventional preflight summary below separates **required bridge gates**
from cross-backend numerical diagnostics. A Q4_0 diagnostic `False` is not
silently changed to `True`; BF16 still requires those numerical limits.
These tiny shared-library tests do not validate upstream kernels independently.
'''
    notebook.cells[8].source += '''

for path in sorted(RESULT_ROOT.glob("stages/conventional-preflight-*/attempt-*/report.json")):
    report = json.loads(path.read_text())
    print("Conventional preflight:", path.parent.parent.name, path.parent.name,
          {k: report.get(k) for k in ("format", "passed", "gpu_executed", "cross_backend_numerics_role", "error")})
    for device, result in report.get("same_backend", {}).items():
        print("  Required private/public agreement", device, result["passed"], result["metrics"])
    for caller, result in report.get("cross_backend", {}).items():
        print("  CPU/GPU", caller, "original numerical + trace limits passed:", result["passed"], result["metrics"])
'''
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    destination = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_followup_colab.ipynb"
    nbformat.write(notebook, destination)
    print(destination)
