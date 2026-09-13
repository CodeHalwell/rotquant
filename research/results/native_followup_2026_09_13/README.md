# Native GPU baseline completion — 13 September 2026

Producer `304d3297743abad675d7696f799dc07e9c20bdae`, uploaded bundle
`followup2-reports-1789312959761420396`. `original-reports.tar.gz` preserves the
reports/logs unchanged, not model weights or executable binaries.
Archive SHA-256: `dd7d1dc60030e11d8c19793b3ab254e03285b4939f3fec3c26cf004d9c860ff8`.

Run `node research/results/native_followup_2026_09_13/audit.mjs EXTRACTED_REPORT_DIRECTORY`
to reproduce `audit.json`: receipt hashes, producer sources, per-repetition
arithmetic, matched inputs/settings and saved conventional probe equivalence.
The audit verifies 26 included artifact references and 41 producer-source hashes.
External artifacts absent from the reports bundle are counted, not silently
treated as verified. This is not a CUDA replay or a new model-quality test.

## Same-run measurements

| Prompt tokens | Model/kernel | Prefill tok/s | Decode tok/s | Sampled peak MiB |
|---|---|---:|---:|---:|
|128|RotQuant reference|36.3317|19.9521|3450|
|128|BF16|4926.4477|89.6547|8836|
|128|Unsloth UD-Q4|3296.3572|112.8855|3580|
|512|RotQuant reference|36.4553|19.9651|3972|
|512|BF16|9171.9757|90.7309|9348|
|512|Unsloth UD-Q4|6198.9452|126.6499|4092|

Three measured repetitions plus an excluded warmup, 32 cached decode steps,
same native bridge, GPU/runtime, prompt IDs, decode replay and FP16 cache.
These are synchronous bridge rates, not equal-quality/size or serving-capacity
claims. Memory is process allocation sampled every 0.5 seconds, not a transient
peak bound. RotQuant uses the original reference kernel here, not tiled4.
At 512 tokens UD-Q4 is 6.3436× faster in decode and uses 120 MiB more sampled
memory; substantial runtime work remains despite the compression-quality results.

## Passed and incomplete evidence

22 stages passed; the run stopped after 15.7244 active minutes. The runtime
cache restored in 17.9 seconds. Conventional BF16/Q4_0 private/public callers
agree exactly on each CPU/CUDA backend. Q4_0 cross-backend numerical drift
remains visible as a failed diagnostic; it was not relabeled numerical parity.
RotQuant reference operators, tiny W6/W8 models and retained W5/W6 parity passed.

The decode4 numerical cases printed passes, but its dispatch-counter gate
failed. Candidate whole-model parity and speed were **not established**.
The completed baseline comparisons remain saved despite that final failure.

The repair reproduces a copied-SONAME bug in a CPU-only Linux ELF test:
opening the unversioned CUDA alias can create a second library instance with
zero counters, while inference uses the versioned dependency. Diagnostics now
resolve through the execution-library handle and record the actual provider.
A fresh-counter probe runs before full-model work. This regression confirms
the loader failure mechanism, not candidate CUDA correctness or speed.

Use the updated [follow-up notebook](../../../notebooks/qwen35_4b_native_followup_colab.ipynb)
with `followup3`, no conventional baseline rerun by default, and fresh
reference/candidate timing pairs. Old baseline timings are historical evidence,
not automatically mixed into the new run's comparisons. Native kernels and
cache inputs remain unchanged, allowing compatible build/export reuse.
