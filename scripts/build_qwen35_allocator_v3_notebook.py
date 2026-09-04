#!/usr/bin/env python3
"""Generate the resumable Qwen3.5-4B allocator-v3 Colab experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_qwen35_allocator_v2_notebook as allocator_v2

OUTPUT = Path("notebooks/qwen35_4b_allocator_v3_colab.ipynb")


def _replace_all(source: str) -> str:
    replacements = (
        ("allocator-v2", "allocator-v3"),
        ("allocator_v2", "allocator_v3"),
        ("Allocator-v2", "Allocator-v3"),
        ("ALLOCATOR_V2", "ALLOCATOR_V3"),
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
    notebook = allocator_v2.build_notebook()
    for cell in notebook.cells:
        cell.source = _replace_all(cell.source)
    notebook.cells[0].source = dedent("""
        # RotQuant Qwen3.5-4B allocator-v3 experiment

        This run tests whether deliberately retained high-precision islands
        improve RotQuant at the same complete exported size as Unsloth
        UD-Q4_K_XL. It keeps allocator-v2's faithful MSE-search/GPTQ candidate
        measurements, adds an exact-byte same-palette random control, repairs
        bucketed solutions with deterministic pair exchanges, and compares W6
        and binding W8 protection policies.

        Seed 0 screens nine distinct policies. Allocation fingerprints prevent
        duplicate recipes from consuming confirmation slots. Up to three
        finalists proceed to seeds 1 and 2, with direct paired comparisons
        against both random allocation and uniform W4. The final validator
        fails closed on incomplete rows, missing paired evidence, duplicated
        recipes, mismatched prompts, or exported artifacts outside the 1% gate.
        Use an A100 40 GB or larger.
    """).strip()

    settings = _cell_after(notebook, "## 1. Settings")
    settings.source = settings.source.replace(
        "PERSIST_GGUF_IN_DRIVE = False",
        dedent("""
        PERSIST_GGUF_IN_DRIVE = False
        REUSE_ALLOCATOR_V2_SCORE_CACHE = True
        ALLOCATOR_V2_SCORE_CACHE = Path(
            "/content/drive/MyDrive/rotquant/qwen35_allocator_v2/dynamic_score_cache"
        )
        REUSE_ALLOCATOR_V2_UNSLOTH = True
        ALLOCATOR_V2_UNSLOTH_RESULT = Path(
            "/content/drive/MyDrive/rotquant/qwen35_allocator_v2/"
            "2dce3aa43029/unsloth_kl/unsloth_ud_q4_kl.json"
        )
        """).strip(),
    )

    checkout = _cell_after(notebook, "## 2. GPU, Drive, and immutable checkout")
    checkout.source = checkout.source.replace(
        'DYNAMIC_SCORE_CACHE_ROOT = RESULT_BASE / "dynamic_score_cache"',
        dedent("""
        DYNAMIC_SCORE_CACHE_ROOT = RESULT_BASE / "dynamic_score_cache"
        if (
            USE_GOOGLE_DRIVE
            and REUSE_ALLOCATOR_V2_SCORE_CACHE
            and ALLOCATOR_V2_SCORE_CACHE.exists()
        ):
            DYNAMIC_SCORE_CACHE_ROOT = ALLOCATOR_V2_SCORE_CACHE
            print(f"Reusing allocator-v2 candidate table: {DYNAMIC_SCORE_CACHE_ROOT}")
        """).strip(),
    )

    dry_run = _cell_after(notebook, "## 5. Dry-run the registered plan")
    dry_run.source = dry_run.source.replace(
        '"scripts/compare_qwen35_dynamic_to_unsloth.py",',
        '"scripts/compare_qwen35_dynamic_to_unsloth.py",\n'
        '    "scripts/assess_qwen35_allocator_v3.py",',
    )

    unsloth = _cell_after(notebook, "## 9. Phase C — prompt-matched Unsloth anchor")
    unsloth.source = dedent("""
        def ensure_llama_cpp():
            probe = subprocess.run([
                sys.executable, "-c",
                "import diskcache, llama_cpp; print(llama_cpp.__version__)",
            ], capture_output=True, text=True)
            if probe.returncode == 0:
                print({"llama_cpp_python": probe.stdout.strip(), "status": "resume"})
                return
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "-U",
                "diskcache==5.6.3", "jinja2==3.1.6",
                "typing-extensions==4.15.0",
            ], check=True)
            llama_env = os.environ.copy()
            llama_env.update({
                "CMAKE_ARGS": "-DGGML_CUDA=on",
                "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                "FORCE_CMAKE": "1",
            })
            revision = "3691546f1c9e0c1bf93323dff02230bd959cf562"
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-v", "--no-deps",
                f"git+https://github.com/abetlen/llama-cpp-python.git@{revision}",
            ], check=True, env=llama_env)
            importlib.invalidate_caches()
            subprocess.run([
                sys.executable, "-c", "import diskcache, llama_cpp"
            ], check=True)

        reused_unsloth = False
        if (
            REUSE_ALLOCATOR_V2_UNSLOTH
            and ALLOCATOR_V2_UNSLOTH_RESULT.exists()
        ):
            UNSLOTH_ROOT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                ALLOCATOR_V2_UNSLOTH_RESULT,
                UNSLOTH_ROOT / "unsloth_ud_q4_kl.json",
            )
            reused_unsloth = True
            print(f"Reused prompt-matched Unsloth anchor: {ALLOCATOR_V2_UNSLOTH_RESULT}")

        if RUN_UNSLOTH_KL and not reused_unsloth:
            ensure_llama_cpp()
            gguf_dir = (
                RESULT_BASE / "artifacts/unsloth-qwen35-4b-gguf"
                if PERSIST_GGUF_IN_DRIVE
                else Path("/content/unsloth-qwen35-4b-gguf")
            )
            command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/run_unsloth_qwen35_4b_kl.py"),
                "--output-dir", str(UNSLOTH_ROOT),
                "--artifact-dir", str(gguf_dir),
                "--batches", "24", "--prompt-len", "512", "--skip", "4096",
            ]
            if FORCE_RERUN:
                command.append("--force")
            run_live(command, label="phase-c-unsloth-kl")
        elif not RUN_UNSLOTH_KL and not reused_unsloth:
            print("Unsloth comparison disabled and no reusable result was found.")
    """).strip()

    confirmation = _cell_after(
        notebook, "## 8. Phase B — seeds 0/1/2 confirmation and packed export"
    )
    confirmation.source = confirmation.source.replace(
        "export_arms = FINALIST_ARMS if EXPORT_FINALIST_SEED0 else ()",
        "export_arms = (\n"
        '        tuple(dict.fromkeys(("random_broad_exact", *FINALIST_ARMS)))\n'
        "        if EXPORT_FINALIST_SEED0 else ()\n"
        "    )",
    )

    comparison_heading = "## 10. Same-input, same-byte comparison"
    comparison_code = _cell_after(notebook, comparison_heading)
    comparison_index = notebook.cells.index(comparison_code)
    decision_heading = new_markdown_cell("## 11. Three-seed promotion decision")
    decision_code = new_code_cell(
        dedent("""
        DECISION_PATH = RESULT_ROOT / "allocator_v3_decision.json"
        if RUN_FINAL_COMPARISON and FINALIST_ARMS:
            subprocess.run([
                sys.executable, "-u",
                str(REPO_DIR / "scripts/assess_qwen35_allocator_v3.py"),
                "--summary", str(CONFIRM_SUMMARY),
                "--selection", str(selection_path),
                "--comparison", str(COMPARISON_PATH),
                "--output", str(DECISION_PATH),
            ], cwd=REPO_DIR, check=True)
        else:
            print("Decision skipped because confirmation/comparison is unavailable.")
    """).strip()
    )
    notebook.cells[comparison_index + 1 : comparison_index + 1] = [decision_heading, decision_code]

    # Renumber and replace the inherited validator.
    for cell in notebook.cells:
        if (
            cell.cell_type == "markdown"
            and cell.source == "## 11. Validate and display the decision record"
        ):
            cell.source = "## 12. Validate and display the decision record"
        elif (
            cell.cell_type == "markdown"
            and cell.source == "## 12. Download the compact result bundle"
        ):
            cell.source = "## 13. Download the compact result bundle"

    validator = _cell_after(notebook, "## 12. Validate and display the decision record")
    validator.source = dedent("""
        def load_json(path):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        screen_summary = load_json(SCREEN_SUMMARY)
        confirm_summary = load_json(CONFIRM_SUMMARY)
        comparison = load_json(COMPARISON_PATH)
        decision = load_json(DECISION_PATH)
        issues = []
        rows = (screen_summary or {}).get("rows", [])
        if (screen_summary or {}).get("code_revision") != commit:
            issues.append("screen code revision mismatch")
        if not (screen_summary or {}).get("complete"):
            issues.append("screen summary incomplete")
        if len(rows) != 9:
            issues.append(f"expected 9 screen rows, found {len(rows)}")
        for row in rows:
            if row["arm"].startswith(("random_", "pareto_")):
                if row.get("dynamic_actual_target_match") is not True:
                    issues.append(f"{row['arm']}: exported-byte estimate missed")
                if row.get("dynamic_scoring_matches_deployed") is not True:
                    issues.append(f"{row['arm']}: scoring was not faithful")
                if row.get("dynamic_proxy_rank_correlation") is None:
                    issues.append(f"{row['arm']}: proxy audit missing")
                if not row.get("dynamic_allocation_fingerprint"):
                    issues.append(f"{row['arm']}: allocation fingerprint missing")
                if row.get("dynamic_estimated_artifact_bytes") is None:
                    issues.append(f"{row['arm']}: artifact accounting missing")
            if row.get("logit_fidelity_tokens", 0) < 10_000:
                issues.append(f"{row['arm']}: fewer than 10k KL tokens")

        selected_rows = [
            row for row in rows if row.get("arm") in set(FINALIST_ARMS)
        ]
        fingerprints = [
            row.get("dynamic_allocation_fingerprint") for row in selected_rows
        ]
        if len(fingerprints) != len(set(fingerprints)):
            issues.append("selected finalists contain duplicate allocations")

        if RUN_CONFIRMATION and FINALIST_ARMS:
            confirm_rows = (confirm_summary or {}).get("rows", [])
            if (confirm_summary or {}).get("code_revision") != commit:
                issues.append("confirmation code revision mismatch")
            if tuple((confirm_summary or {}).get("seeds", ())) != tuple(CONFIRM_SEEDS):
                issues.append("confirmation seed set mismatch")
            if not (confirm_summary or {}).get("complete"):
                issues.append("confirmation summary incomplete")
            identities = {
                (row["arm"], int(row["seed"])) for row in confirm_rows
            }
            expected = {
                (arm, seed) for arm in CONFIRMATION_ARMS
                for seed in CONFIRM_SEEDS
            }
            if expected - identities:
                issues.append(f"confirmation missing {sorted(expected - identities)}")
            hash_sets = {
                tuple(row.get("logit_fidelity_input_hashes") or ())
                for row in confirm_rows
            }
            if len(hash_sets) != 1:
                issues.append("confirmation prompt hashes differ")
            paired = {
                (item["candidate_arm"], item["baseline_arm"], int(item["seed"]))
                for item in (confirm_summary or {}).get("paired_comparisons", [])
            }
            expected_pairs = {
                (arm, "random_broad_exact", seed)
                for arm in FINALIST_ARMS for seed in CONFIRM_SEEDS
            }
            if expected_pairs - paired:
                issues.append(
                    "missing direct paired random comparisons: "
                    f"{sorted(expected_pairs - paired)}"
                )
            if EXPORT_FINALIST_SEED0:
                exported_arms = ("random_broad_exact", *FINALIST_ARMS)
                seed0 = {
                    row["arm"]: row for row in confirm_rows
                    if int(row["seed"]) == 0
                }
                for arm in exported_arms:
                    row = seed0.get(arm) or {}
                    actual = row.get("packed_artifact_bytes")
                    target = row.get("dynamic_target_artifact_bytes")
                    if not isinstance(actual, (int, float)):
                        issues.append(f"{arm}: seed-0 packed artifact is missing")
                    elif not isinstance(target, (int, float)):
                        issues.append(f"{arm}: seed-0 artifact target is missing")
                    elif abs(actual / target - 1.0) > 0.01:
                        issues.append(f"{arm}: actual artifact missed the 1% byte gate")

        if comparison:
            if comparison.get("prompt_hashes_match") is not True:
                issues.append("RotQuant/Unsloth prompt hashes differ")
            failed_bytes = [
                row["arm"] for row in comparison.get("comparisons", [])
                if row.get("within_byte_gate") is not True
            ]
            if failed_bytes:
                issues.append(f"exported artifact byte gate failed: {failed_bytes}")
        elif RUN_FINAL_COMPARISON and FINALIST_ARMS:
            issues.append("provider comparison missing")
        if RUN_FINAL_COMPARISON and FINALIST_ARMS and not decision:
            issues.append("three-seed decision record missing")

        validation = {
            "commit": commit,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "ready_for_internal_decision" if not issues else "needs_revision",
            "issues": issues,
            "boundary": (
                "Internal 24-prompt and 25-prompt development suites only. "
                "Public claims require the registered 300-prompt engine-neutral run."
            ),
        }
        validation_path = RESULT_ROOT / "allocator_v3_validation.json"
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps(validation, indent=2))
        assert not issues, "Validation failed; inspect the issue list."

        for label, summary in (
            ("Allocator-v3 screen", screen_summary),
            ("Allocator-v3 confirmation", confirm_summary),
        ):
            frame = pd.DataFrame((summary or {}).get("rows", []))
            if frame.empty:
                continue
            print(label)
            columns = [name for name in (
                "arm", "seed", "mean_teacher_kl", "top1_agreement",
                "ppl_wikitext2", "ppl_c4", "diverse_mean_teacher_kl",
                "diverse_top1_agreement", "diverse_trajectory_token_agreement",
                "dynamic_estimated_artifact_bytes", "dynamic_counts_by_bits",
                "dynamic_refinement_applied", "dynamic_protected_layers",
                "dynamic_score_cache_hit", "dynamic_score_cache_source",
            ) if name in frame]
            display(frame[columns].sort_values(["arm", "seed"]))
        if comparison:
            print("RotQuant minus Unsloth")
            display(pd.DataFrame(comparison["comparisons"]))
        if decision:
            print("Promotion decision")
            display(pd.DataFrame(decision["decisions"]))
        print({"finalists": FINALIST_ARMS, "result_root": str(RESULT_ROOT)})
    """).strip()

    downloader = _cell_after(notebook, "## 13. Download the compact result bundle")
    downloader.source = dedent("""
        if DOWNLOAD_RESULTS:
            import zipfile
            from google.colab import files

            archive = Path("/content") / f"qwen35_allocator_v3_{commit[:12]}.zip"
            excluded_suffixes = {".safetensors", ".gguf", ".npz", ".pt"}
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(RESULT_ROOT.rglob("*")):
                    if not path.is_file() or path.suffix in excluded_suffixes:
                        continue
                    bundle.write(path, path.relative_to(RESULT_ROOT))
            print(f"Compact result bundle: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
            print("Large packed artifacts and BF16 references remain in Drive.")
            files.download(str(archive))
        else:
            print(f"Results remain in {RESULT_ROOT}")
    """).strip()

    notebook.metadata["colab"]["name"] = OUTPUT.name
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
