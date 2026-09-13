"""Focused profiling of the validated decode4 kernel, with fresh timing controls."""
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_qwen35_native_optimization_notebook import build_notebook as study_notebook
from scripts.build_qwen35_packed_validation_notebook import md


def build_notebook():
    notebook = study_notebook()
    notebook.cells[0] = md('''
        # RotQuant Qwen3.5-4B — focused decode4 bottleneck profile

        ## Goal
        Select **A100 40GB**, check the saved source path, then **Runtime → Run all**.
        Keep the existing **W5/scale8 + W6 vocabulary** weights unchanged.

        **followup3 passed all 20 gates:** decode4 delivered about **31.9 decode
        tok/s versus 19.9** and retained the **3.3× prefill gain** on A100 40GB.
        See `research/results/native_decode4_2026_09_13/` for that prior evidence;
        these are historical measurements, not outputs from this notebook.

        **Question:** after that improvement, which custom operators still
        dominate prefill and decode? Run fresh reference/decode4 timing pairs,
        then separate CUDA-event profiles at **128 input tokens only**. Preserve
        the current numerical gates; do not promote a new kernel or recipe.

        Existing private build/checkpoint caches are reused when compatible.
        BF16/Unsloth downloads, longer contexts and extra model arms are skipped.
        No calibration, requantization, adapters or training are run.

        ### Key assumptions
        Custom-event profiles attribute rotation, backbone matrix, vocabulary
        head and embedding time. They do **not** measure DRAM bandwidth or
        separately attribute attention/SSM kernels, CPU work or host transfers.
        Profile mode disables CUDA graphs and synchronizes events, so its wall
        rates must never be compared with normal throughput. The earlier
        reference-kernel bottleneck percentages cannot be assumed for decode4.
        ''')
    notebook.cells[1] = md('''
        ## Setup — explicit controls
        Use the original retained **W5/scale8 + W6 vocabulary** checkpoint.
        `SOURCE_ROOT` must include its checkpoint, `prepared.json`,
        `preparation.json` and `packed_probes.safetensors`; reports alone cannot
        run the model. Keep the existing private `CACHE_ROOT` intact.

        The new run name **followup4-profile** isolates these settings from
        followup3. One 128-token prompt shape, 32 cached decode steps, three
        measurements and one excluded warmup per timing/profile process.
        The **60 active-minute** maximum includes setup/build/tests, not idle
        notebook time. It is a spending cap, not an expected duration. A cold
        cache may still need the previously observed ~27-minute build.
        The 2 tok/s and 16 GiB guards are not production-readiness thresholds.
        ''')
    controls = notebook.cells[2]
    controls.source = controls.source.replace('RUN_NAME = "study1"', 'RUN_NAME = "followup4-profile"')
    controls.source = controls.source.replace('CONTEXTS = (128, 512, 2048)', 'CONTEXTS = (128,)')
    controls.source = controls.source.replace('ACTIVE_BUDGET_MINUTES = 90', 'ACTIVE_BUDGET_MINUTES = 60')
    controls.source = controls.source.replace('BASELINES = ("bf16", "ud_q4")', 'BASELINES = ()')
    controls.source += '\nCANDIDATE = "decode4"  # "none" runs only fresh reference + BF16/UD-Q4\n'
    controls.source += 'assert CANDIDATE in ("decode4", "none")\n'
    controls.source += 'assert CANDIDATE != "none" or (BASELINES and not RUN_PROFILE)\n'
    for cell in notebook.cells:
        cell.source = cell.source.replace('/content/rotquant-native-study/', '/content/rotquant-native-followup/')
        cell.source = cell.source.replace('MyDrive/rotquant/native_gpu_study', 'MyDrive/rotquant/native_gpu_followup')
    notebook.cells[5] = md('''
        ## Steps and checks
        1. Verify saved evidence; build/restore the hash-checked native library.
        2. **Early dispatch preflight:** prove one-token decode and four-token
           prefill increment the intended counters, and match reference outputs
           exactly. The report identifies the actual diagnostic library provider.
           Missing/wrong dispatch stops immediately; it is never waived.
        3. Fresh RQ3 operators, W6/W8 tiny-model, conversion and saved-model parity.
        4. Fresh reference timings at **128 tokens**, three measurements
           plus one excluded warmup, on this runtime/GPU.
        5. With `CANDIDATE = "decode4"`: test operator arithmetic, partial row
           groups, permutations, long reduction widths, tiny models and retained
           parity, then collect matched candidate timing. The default kernel is
           unchanged. A failed gate stops execution, never relaxes thresholds.
        6. Separate **reference and decode4 CUDA-event profiles**, three measured
           repetitions plus warmup each, at the same 128-token shape. Results
           stay separate from the uninstrumented timing comparison.

        The completed baseline reports stay historical evidence. They are not
        imported as fresh timings or mixed into this run's speed ratios.
        To explicitly repeat baselines, set `BASELINES = ("bf16", "ud_q4")`.
        That also repeats the conventional public/private preflight first.
        For baseline-only work set those baselines, `CANDIDATE = "none"` and
        `RUN_PROFILE = False`. Fresh reference timings still run.
        `RUN_PROFILE = True` is required for this run's purpose. Explicitly add
        longer contexts only under a new run name. Controls cannot change inside
        an existing result directory. The profile uses the smallest selected
        context, even when more timing contexts are requested.

        This diagnostic repair leaves native kernels and the binary-cache key
        unchanged from **2c4037e76f71**. Its cached CUDA build can be restored
        when hardware/toolchain checks match. No new CUDA compilation is required
        solely for this notebook configuration.
        A cache miss still needs a cold build (~27 minutes previously).
        Original weights are reused, never trained or requantized.
        Baseline downloads (~11.34 GB) are skipped by default. The private Drive
        cache and original results are retained; don't clear them.
        Every phase prints a persistent log, cap and heartbeat; every repetition
        prints timing and progress. The **60 active-minute** allowance and phase
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

BF16/Unsloth timings are intentionally absent with `BASELINES = ()`. See the
archived followup2 results in `research/results/native_followup_2026_09_13/`.
Only fresh matched reference/candidate pairs can produce this run's speed ratios.

The conventional preflight summary below separates **required bridge gates**
from cross-backend numerical diagnostics. A Q4_0 diagnostic `False` is not
silently changed to `True`; BF16 still requires those numerical limits.
These tiny shared-library tests do not validate upstream kernels independently.

Expect two diagnostic groups: `profile-reference-b5_v6_s0-ctx128` and
`profile-decode4-b5_v6_s0-ctx128`. If either is missing or stopped, the profiling
question is incomplete even if earlier timing stages passed. Compare absolute
custom-event milliseconds within each phase first; event sums are not full
model wall time. Zero event milliseconds in **uninstrumented** reports mean
not measured, not zero cost. Return both diagnostic reports and the fresh
throughput pair before deciding whether to optimise backbone or vocabulary.
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
