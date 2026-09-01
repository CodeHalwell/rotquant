# RotQuant paper draft

Build the current arXiv-style draft with:

```bash
uv run python scripts/audit_publication.py
cd paper && tectonic main.tex
```

For the full local release audit, also pass the cached source and RotQuant
snapshot directories plus `--native-gguf`. The command reconciles numerical
aggregates, checkpoint structure, actual bytes, and the native artifact hash,
then regenerates `paper/generated/results.tex`.

Render the immutable model/method command matrix without executing it:

```bash
uv run python scripts/run_publication_suite.py --protocol publication
```

Use `--protocol smoke` for bounded preflight runs. `--execute` is intentionally
required before any model command is launched.

The generated PDF is intentionally ignored by version control. Red `[TODO: ...]`
markers denote missing publication evidence or author metadata rather than prose
that can safely be inferred.

The draft's current claim boundary is deliberate:

- the Qwen3.5-4B result is a development-scale joint weight/KV result;
- CUDA and MPS fallback measurements are quality-only;
- exported checkpoint bytes are measured against a like-for-like pinned source,
  while resident and transient memory remain unmeasured;
- the broader architecture-generic library is presented as future work.
