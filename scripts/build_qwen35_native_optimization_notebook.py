"""An end-to-end performance study built from the tested pilot bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_qwen35_native_gpu_pilot_notebook import build_notebook as pilot_notebook
from scripts.build_qwen35_packed_validation_notebook import code, md


def build_notebook():
    notebook = pilot_notebook()
    notebook.cells[0] = md('''
        # RotQuant Qwen3.5-4B — native GPU optimisation and GGUF comparison

        ## Goal
        Keep the **W5/scale8 + W6-vocabulary model unchanged** while testing the
        runtime. Select **A100 40GB**, then **Runtime → Run all**.

        The previous pilot measured about **36.4 prefill / 19.8 decode tok/s**;
        its 2,048-token stage stopped at an undersized three-minute cap. This
        study uses context/repetition-aware budgets and selectable contexts.

        Compare the original native kernel with an **experimental four-token
        weight-reuse kernel**, then profile custom CUDA operators separately.
        Pinned BF16 and Unsloth UD-Q4 GGUFs use the **same native bridge**, prompt
        IDs, context reservation, FP16 cache, decode replay and timing rules.
        No `llama-cpp-python` build and no new quantization/training are involved.

        **No speedup is known yet.** The original kernel stays the default.
        Correctness gates and stopped/partial-run labels are not weakened.
        ''')
    controls = notebook.cells[2]
    controls.source = controls.source.replace('RUN_NAME = "pilot1"', 'RUN_NAME = "study1"')
    controls.source += '\nBASELINES = ("bf16", "ud_q4")  # () explicitly skips these downloads/runs\nRUN_PROFILE = True\n'
    controls.source += 'assert len(set(BASELINES)) == len(BASELINES) and set(BASELINES) <= {"bf16", "ud_q4"}\n'
    # Reuse bootstrap and checks, but keep this study's checkout/results separate.
    for cell in notebook.cells:
        cell.source = cell.source.replace('/content/rotquant-native-pilot/', '/content/rotquant-native-study/')
        cell.source = cell.source.replace('MyDrive/rotquant/native_gpu_pilot', 'MyDrive/rotquant/native_gpu_study')
    bootstrap = notebook.cells[4]
    bootstrap.source = bootstrap.source.replace('"native_pilot_controls.py"):',
        '"native_pilot_controls.py", "native_performance_study.py", "native_cuda_diagnostics.py", "run_native_gguf_baseline.py"):')
    notebook.cells[5] = md('''
        ## Steps and checks
        1. Verify the original saved model; build/restore the native runtime.
        2. Repeat CPU/CUDA operators, tiny W6/W8 models and conversion gates.
        3. Pass retained-model parity, then time the original kernel.
        4. Check the tiled candidate against canonical arithmetic **and exact
           reference-operator outputs**, including 3/4/5-token tile boundaries.
           Repeat both tiny models and retained-model parity before timing it.
        5. Run separate CUDA-event profiles at the smallest selected context:
           rotation, matrix operations, vocabulary head and embedding lookup.
        6. Download the pinned BF16 / UD-Q4 controls, verify full file hashes and
           their token-ID maps, then run matched fixed-token timing processes.

        The new CUDA code requires **one new build**. The previous runtime cache
        cannot validate changed kernel code. Subsequent compatible builds reuse
        the private cache. Cold build was about 27 minutes on the prior A100.
        Baseline downloads total roughly **11.34 GB**; they stay on local disk.
        Drive cache and original checkpoint directories are never overwritten.

        Default timing caps are **4 / 6 / 12 minutes** for 128/512/2,048 tokens;
        diagnostic caps are doubled, all bounded by the **90 active-minute**
        session allowance. Caps stop child processes, **not Colab billing**.

        To target just 2,048 tokens, set `CONTEXTS = (2048,)` and a new `RUN_NAME`.
        To avoid repeating diagnostics or controls, explicitly set
        `RUN_PROFILE = False` or `BASELINES = ()`. Selected timing measurements
        and correctness gates remain fresh; no old timing is silently promoted.
        After a failure, run the Results cell, download the ZIP and stop the GPU.
        Never update the checkout while the driver is running.
        ''')
    launch = notebook.cells[6]
    launch.source = launch.source.replace('"--performance-pilot"', '"--performance-study"')
    insertion = '''if not RUN_PROFILE:
    command.append("--skip-profile")
if not BASELINES:
    command.append("--no-baselines")
for baseline in BASELINES:
    command.extend(["--baseline", baseline])
'''
    launch.source = launch.source.replace('run_live(command, "native-performance-pilot"', insertion + 'run_live(command, "native-performance-study"')
    notebook.cells[7] = md('''
        ## Results — also run after a stop
        Throughput and diagnostic profiles are shown separately. Warmup is
        excluded; VRAM is sampled process allocation, not an exact transient peak.
        Profiles synchronize CUDA events per custom operation with graphs
        disabled. **Do not compare their wall rates to normal throughput.**
        Non-custom attention/SSM kernels, host work and copy costs are not
        individually attributed by this first profiler; event times do not sum
        to complete model wall time.

        Reference timing generates greedy decode IDs; candidate and conventional
        controls replay those exact IDs, while still performing the same argmax
        and model calls. This isolates shape/token-path differences, not output
        quality. Token-ID maps are checked without retokenizing multilingual text.
        The GGUF controls are **not matched for quality or exact size**. Their
        reported size is text-GGUF bytes; the RotQuant non-text sidecar is not in VRAM.

        `comparison-*.json` contains ratios only for complete matched,
        uninstrumented runs. No automatic kernel or recipe promotion occurs.
        ''')
    notebook.cells[8] = code('''
        from IPython.display import Markdown, display
        from scripts.native_performance_summary import tables
        summary_path = RESULT_ROOT / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print("Workflow:", summary["status"], "| active minutes:", round(summary["active_minutes"], 1))
            for stage in summary["stages"]:
                print(stage["name"], stage["status"], f"{stage.get('elapsed_seconds', 0):.1f}s")
            if summary.get("error"):
                print("Stopped:", summary["error"])
        timing, profiling = tables(RESULT_ROOT)
        display(Markdown("### Uninstrumented throughput\\n\\n" + timing))
        display(Markdown("### Diagnostic profiles — not throughput\\n\\n" + profiling))
        archives = sorted(RESULT_ROOT.parent.glob(RUN_NAME + "-reports-*.zip"))
        if archives:
            REPORTS_ZIP = archives[-1]
            print("Reports ZIP:", REPORTS_ZIP)
        print("DISCONNECT AND DELETE the GPU runtime after downloading your reports.")
        ''')
    notebook.cells[9] = md('''
        ## Download and next steps
        Download the reports ZIP below, then disconnect/delete the GPU runtime.
        Bring back the reports before changing more algorithms or running tasks.
        We need measured bottleneck attribution, unchanged parity and a useful
        uninstrumented improvement before enabling an optimised kernel by default.
        ''')
    notebook.cells.append(code('''
        from google.colab import files
        if "REPORTS_ZIP" in globals():
            files.download(str(REPORTS_ZIP))
        else:
            print("No report archive found. Run the Results cell first.")
        '''))
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    destination = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_optimization_colab.ipynb"
    nbformat.write(notebook, destination)
    print(destination)
