#!/usr/bin/env python3
"""Generate the resumable Qwen3.5-4B format-aware allocator Colab."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_qwen35_allocator_v3_notebook as allocator_v3

OUTPUT = Path("notebooks/qwen35_4b_allocator_v4_colab.ipynb")


def _replace_all(source: str) -> str:
    replacements = (
        ("allocator-v3", "allocator-v4"),
        ("allocator_v3", "allocator_v4"),
        ("Allocator-v3", "Allocator-v4"),
        ("ALLOCATOR_V3", "ALLOCATOR_V4"),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    return source


def _cell_after(notebook, heading: str):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip() == heading:
            return notebook.cells[index + 1]
    raise ValueError(f"notebook is missing heading {heading!r}")


def build_notebook():
    notebook = allocator_v3.build_notebook()
    for cell in notebook.cells:
        cell.source = _replace_all(cell.source)
    notebook.cells[0].source = dedent("""
        # RotQuant Qwen3.5-4B format-aware allocator-v4 experiment

        Allocator v3's measured global allocation beat its exact-size broad
        random control, but remained 2.62x worse than the prompt-matched
        Unsloth KL anchor. Its winning recipe used only W3/W4/W5; forced W6/W8
        islands made quality worse at the same size.

        This run therefore holds rotation, GPTQ, scale storage, evaluation, and
        the 3,584,533,344-byte target fixed while allowing the allocator to
        choose among a bounded six-format palette: Gaussian or calibrated
        scalar codebooks at W3/W4, Gaussian group-64 at W3, and Gaussian W5.
        Two allocation-only ablations isolate the value of finer groups from
        learned codebooks while reusing the same expensive candidate table.

        Seed 0 screens seven arms. At most two distinct format-aware recipes
        advance to seeds 1 and 2. Promotion requires direct paired wins over
        both the proven bits-only Pareto baseline and an exact same-palette
        random control. Progress, GPU heartbeats, intermediate score tables,
        summaries, and logs persist to Drive. Use an A100 40 GB or larger.

        Before expensive calibration, a synthetic CUDA preflight checks all
        six formats against the actual patcher and exercises shared-rotation
        inference/generation. Confirmation reapplies every no-regression guard
        on every seed and requires reliable paired intervals. A measured,
        identity-matched seed-0 export is mandatory for promotion; disabling
        exports leaves diagnostics only, never a promoted recipe.
    """).strip()

    settings = _cell_after(notebook, "## 1. Settings")
    settings.source = settings.source.replace(
        'REPO_DIR = Path("/content/rotquant-allocator-v4")',
        'REPO_DIR = Path("/content/rotquant-format-aware")',
    )

    dry_run = _cell_after(notebook, "## 5. Dry-run the registered plan")
    dry_run.source += "\n\n" + dedent("""
        run_live(
            [sys.executable, "-u", str(REPO_DIR / "scripts/preflight_allocator_v4.py"),
             "--device", "cuda", "--output", str(RESULT_ROOT / "runtime_preflight.json")],
            label="runtime-preflight",
        )
        print("Runtime preflight passed. The Qwen experiment may now start.", flush=True)
    """).strip()

    confirmation = _cell_after(
        notebook, "## 8. Phase B — seeds 0/1/2 confirmation and packed export"
    )
    confirmation.source = confirmation.source.replace(
        'tuple(dict.fromkeys(("random_broad_exact", *FINALIST_ARMS)))',
        'tuple(dict.fromkeys(("bits_only_pareto", "format_random_exact", '
        '*FINALIST_ARMS)))',
    )

    validator = _cell_after(
        notebook, "## 12. Validate and display the decision record"
    )
    validator.source = validator.source.replace(
        "if len(rows) != 9:\n"
        '    issues.append(f"expected 9 screen rows, found {len(rows)}")',
        "if len(rows) != 7:\n"
        '    issues.append(f"expected 7 screen rows, found {len(rows)}")',
    )
    validator.source = validator.source.replace(
        'if row["arm"].startswith(("random_", "pareto_")):',
        'if row["arm"].startswith(("bits_", "format_")):',
    )
    validator.source = validator.source.replace(
        'expected_pairs = {\n'
        '        (arm, "random_broad_exact", seed)\n'
        '        for arm in FINALIST_ARMS for seed in CONFIRM_SEEDS\n'
        '    }',
        'expected_pairs = {\n'
        '        (arm, baseline, seed)\n'
        '        for arm in FINALIST_ARMS for seed in CONFIRM_SEEDS\n'
        '        for baseline in ("bits_only_pareto", "format_random_exact")\n'
        '    }',
    )
    validator.source = validator.source.replace(
        'exported_arms = ("random_broad_exact", *FINALIST_ARMS)',
        'exported_arms = ("bits_only_pareto", "format_random_exact", '
        '*FINALIST_ARMS)',
    )
    validator.source = validator.source.replace(
        '"dynamic_estimated_artifact_bytes", "dynamic_counts_by_bits",',
        '"dynamic_estimated_artifact_bytes", "dynamic_counts_by_bits",\n'
        '            "dynamic_counts_by_format",',
    )

    downloader = _cell_after(notebook, "## 13. Download the compact result bundle")
    downloader.source = downloader.source.replace(
        "Large packed artifacts and BF16 references remain in Drive.",
        "Large packed artifacts, candidate caches, and BF16 references remain in Drive.",
    )
    notebook.metadata["colab"]["name"] = OUTPUT.name
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
