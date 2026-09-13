"""Build the cost-bounded W5 kernel screen and retained-model confirmation notebook."""
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
        # RotQuant Qwen3.5-4B — W5 backbone kernel experiment

        ## Goal
        Select **A100 40GB**, check the saved evidence path, then **Runtime → Run all**.
        Compare three **unvalidated, opt-in** backbone implementations with
        **decode4**, our fastest validated kernel. No speedup is assumed.

        | Candidate | Change relative to decode4 |
        | :-- | :-- |
        | `w5s8` | Specialized W5 / scale8 / metadata-block256 unpacking and scale lookup; four-token prefill tiles |
        | `w5s8-tile8` | Same specialization, eight-token prefill tiles |
        | `w5s8-tile16` | Same specialization, sixteen-token prefill tiles |

        The retained **W5/scale8 + W6 vocabulary checkpoint stays unchanged**.
        Vocabulary, rotation arithmetic and the original reference remain
        available. No quantization, training, adapters, public tasks, Unsloth
        downloads or background GPU provisioning occurs.

        ### Key assumptions
        A resident synthetic rotation-plus-matrix benchmark cheaply selects
        candidates for model testing. It is not a model speedup, bandwidth
        measurement or quality test. Exact operator parity and unchanged
        full-model numerical gates are mandatory before model timings.
        Three synthetic shapes are not a layer-frequency-weighted workload.
        ''')
    notebook.cells[1] = md('''
        ## Setup — explicit spending and experiment controls
        This is a new run, **backbone1**, not a modification of followup4.
        Start with one retained W6 arm and 128-token model timings. W8 and
        512/2048-token confirmation can be selected later under a new run name.

        All three candidates are screened; **at most two** reach full-model
        validation. Use `MAX_FINALISTS = 1` for a smaller run. The driver stops
        before model export if none qualifies. A numerical failure stops the
        workflow immediately and preserves its reports.

        The **90 active-minute cap** includes the new native build, setup and
        tests. It is not an expected duration or a billing limit. New kernels
        require a new compatible binary cache entry; the previous build took
        about 27 minutes on A100. Keep the original checkpoint and private cache.
        ''')
    controls = notebook.cells[2]
    controls.source = controls.source.replace('RUN_NAME = "pilot1"', 'RUN_NAME = "backbone1"')
    controls.source = controls.source.replace('CONTEXTS = (128, 512, 2048)', 'CONTEXTS = (128,)')
    controls.source += '''

KERNEL_CANDIDATES = ("w5s8", "w5s8-tile8", "w5s8-tile16")
MAX_FINALISTS = 2
assert KERNEL_CANDIDATES and len(set(KERNEL_CANDIDATES)) == len(KERNEL_CANDIDATES)
assert set(KERNEL_CANDIDATES) <= {"w5s8", "w5s8-tile8", "w5s8-tile16"}
assert MAX_FINALISTS in (1, 2)
'''
    for cell in notebook.cells:
        cell.source = cell.source.replace('/content/rotquant-native-pilot/', '/content/rotquant-native-backbone/')
        cell.source = cell.source.replace('MyDrive/rotquant/native_gpu_pilot', 'MyDrive/rotquant/native_gpu_backbone')
    notebook.cells[4].source = notebook.cells[4].source.replace('"native_pilot_controls.py"):',
        '"native_pilot_controls.py", "native_kernel_sweep.py", "run_rq3_kernel_screen.py"):')
    notebook.cells[5] = md('''
        ## Steps and checks — run end to end
        1. Verify environment and saved source receipts; build/restore the new
           hash-checked runtime, then test loading it in a fresh process.
        2. **Before loading the model:** exact reference-operator comparisons,
           canonical Torch checks, dispatch counters, tile/row tails, W1–W8 and
           scale-format fallbacks for decode4 and each requested candidate.
        3. Resident synthetic screens use square, expanding and contracting
           shapes at 1 and 128 tokens. Each shape gets **AB then BA** order,
           three measured repetitions of ten graph calls, with warmup excluded.
           Allocations, static uploads and output reads are outside timing;
           rotation, matmul, launch and synchronization are inside. CUDA graphs
           and per-operator event profiling are disabled for these screens.
        4. Shortlist only if every shape/order ratio is at least **0.95×** decode4
           and at least one phase's geometric-mean ratio is **1.05×** or more.
           These declared heuristics are not significance tests. Retain all raw
           timings, including rejected candidates; no automatic promotion.
        5. If there are finalists, run fresh CPU/CUDA, tiny W6/W8 model,
           conversion and retained-model gates, then normal **decode4 timings**.
        6. For each finalist repeat its numerical/model gates before replaying
           the exact same prompt/decode IDs for uninstrumented model timings.

        **Progress:** every phase prints its cap, PID, persistent log and
        heartbeat; screens print shape/order and every pair's measurements;
        model runs print each repetition and every eight decode steps.
        Screen/operator phases have six-minute caps and builds have a
        45-minute cap, bounded by the active allowance. No cost or correctness
        guard is silently relaxed. An interrupted run keeps previous attempts.

        Compatible caches are reused; changed CUDA source hashes cannot reuse
        the old binary. Only hash-checked private caches are trusted. Do not
        clear the cache, alter original evidence or update a running checkout.
        After a stop, run Results and Download. Then **disconnect/delete the
        GPU runtime**: stopping the script does not stop Colab billing.

        Local CPU/mock tests cannot establish CUDA correctness or speed; this
        notebook is the explicit GPU validation handoff.
        ''')
    launch = notebook.cells[6]
    launch.source = launch.source.replace('"--performance-pilot"', '"--kernel-sweep"')
    launch.source = launch.source.replace('run_live(command, "native-performance-pilot"', '''command.extend(["--kernel-finalists", str(MAX_FINALISTS)])
for candidate in KERNEL_CANDIDATES:
    command.extend(["--kernel-candidate", candidate])
run_live(command, "native-backbone-experiment"''')
    notebook.cells[7] = md('''
        ## Results — also run after a stop
        First inspect the synthetic shortlist. **No eligible candidates is a
        valid completed screen**, not proof that the new kernels are faster.
        A stopped/incomplete screen must not be interpreted as a rejection or win.

        The second table contains only model timing reports. It must not mix
        synthetic-screen ratios or diagnostic event timings into serving rates.
        Model timings use normal graph settings, excluded warmup, identical
        fixed-token replay and unchanged parity thresholds. Process memory is
        sampled every 0.5 seconds and may miss transient peaks.

        Whole-model ratios compare against **decode4 in this run**, not the
        slower original kernel or historical Unsloth runs. Retained parity is
        against the saved quantized checkpoint, not teacher/task accuracy.
        ''')
    notebook.cells[8] = code('''
        from IPython.display import Markdown, display
        from scripts.native_performance_summary import tables
        from scripts.native_hashing import digest
        summary_path = RESULT_ROOT / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print("Workflow:", summary["status"], "| active minutes:", round(summary["active_minutes"], 1))
            for stage in summary["stages"]:
                print(stage["name"], stage["status"], f"{stage.get('elapsed_seconds', 0):.1f}s")
            if summary.get("error"):
                print("Stopped:", summary["error"])
        shortlist_path = RESULT_ROOT / "kernel-shortlist.json"
        since_epoch = float("inf")  # No current screen means no current confirmations.
        if shortlist_path.exists():
            shortlist = json.loads(shortlist_path.read_text())
            since_epoch = shortlist["started_utc_epoch"]
            print("Screen completed:", shortlist["completed"], "| outcome:", shortlist.get("outcome", "incomplete"))
            print("Full-model finalists:", shortlist["finalists"])
            rows = ["| Synthetic candidate | Eligible | Worst pair | Decode geomean | Prefill geomean |",
                    "| :-- | :-- | --: | --: | --: |"]
            for item in shortlist["candidates"]:
                ratios = item["geomean_ratios"]
                rows.append(f"| {item['kernel']} | {item['eligible']} | {item['worst_case_ratio']:.3f}× | "
                            f"{ratios['decode']:.3f}× | {ratios['prefill']:.3f}× |")
            display(Markdown("### Synthetic screen — not model throughput\\n\\n" + "\\n".join(rows)))
        timing, _ = tables(RESULT_ROOT, since_epoch=since_epoch)
        display(Markdown("### Uninstrumented model throughput\\n\\n" + timing))
        for path in sorted(RESULT_ROOT.glob("comparison-*.json")):
            record = json.loads(path.read_text())
            if not shortlist_path.exists() or record.get("screen_sha256") != digest(shortlist_path):
                print(path.name, "belongs to a previous screen; excluded from this readout")
                continue
            print(path.name, "completed:", record["completed"], "baseline:", record["baseline_kernel"])
            for pair in record["comparisons"]:
                print("  context:", pair["context"],
                      "prefill ratio:", round(pair["prefill_rate_ratio_to_reference"], 3),
                      "decode ratio:", round(pair["decode_rate_ratio_to_reference"], 3))
        archives = sorted(RESULT_ROOT.parent.glob(RUN_NAME + "-reports-*.zip"))
        if archives:
            REPORTS_ZIP = archives[-1]
            print("Reports ZIP:", REPORTS_ZIP)
        print("DISCONNECT AND DELETE the GPU runtime after downloading reports.")
        ''')
    notebook.cells[9] = md('''
        ## Download and next steps
        Bring back the reports ZIP before enabling a new kernel or buying a
        larger experiment. It contains all attempts and raw timings, not weights
        or executable caches. A model-level win earns longer-context/W8 checks,
        fresh conventional controls and native task-quality evaluation. The
        vocabulary-head optimization remains a separate follow-on change.
        ''')
    notebook.cells.append(code('''
        from google.colab import files
        if "REPORTS_ZIP" in globals():
            files.download(str(REPORTS_ZIP))
        else:
            print("No report archive found; run Results first.")
        '''))
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    destination = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_backbone_colab.ipynb"
    nbformat.write(notebook, destination)
    print(destination)
