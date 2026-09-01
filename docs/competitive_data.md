# Competitive data and run-report pipeline

This pipeline makes the Qwen3.5-4B development results replayable before the
optimizer is frozen for Qwen3.8-27B. It does **not** ship or silently download a
benchmark mixture. Dataset selection, licensing review, and the final immutable
revisions remain explicit release decisions.

## Registered held-out composition

The primary divergence suite contains exactly 300 prompts and generates exactly
32 greedy tokens per prompt. The manifest constructor enforces 60 prompts from
each domain:

| Domain | Candidate primary source | Why it is useful | Adoption constraint |
|---|---|---|---|
| Agentic/tool use | [Berkeley Function Calling Leaderboard](https://github.com/EnlightenedAI/BFCL/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) | Executable and multilingual function-calling cases; Apache-2.0 repository. | Pin a commit/tag and stable case IDs; never use the changing live split by name alone. |
| Code | [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Contamination-aware, continuously updated code problems; MIT repository. | Freeze the benchmark release/time window and exclude every selected problem from calibration. |
| Maths | [OpenAI simple-evals MATH](https://github.com/openai/simple-evals) | Small, inspectable MATH evaluation path with reference scoring; MIT repository. | Freeze the exact 60 row IDs and answer-normalization version; retain upstream provenance. |
| Multilingual/non-Latin | [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | Broad multilingual coverage with explicit language identifiers. | CC BY-SA 4.0, gated, and intended for evaluation rather than training: publish metadata/hashes unless redistribution terms permit more. |
| Long document | [LongBench v2](https://github.com/THUDM/LongBench) | Long-context multiple-choice tasks across several task categories; MIT repository. | Pin the repo/data revision and preserve document/question row IDs and input-length filtering. |

These are the source shortlist, not a claim that the final rows have already
been selected. Every source enters through a declared adapter and a human
licensing check. The same source may also feed a secondary task-outcome suite,
but those rows must remain separate from the 300 divergence prompts.

Secondary release gates should include
[IFEval](https://huggingface.co/datasets/google/IFEval) for verifiable
instruction following and
[Terminal-Bench](https://github.com/harbor-framework/terminal-bench) for real
agentic terminal outcomes. Terminal-Bench is intentionally not forced into the
cheap 300-prompt logit/trajectory loop: its container execution is a separate,
more expensive outcome metric.

## Calibration source shortlist

[calibration-blend](https://huggingface.co/datasets/Calandracas/calibration-blend)
is a useful bootstrap candidate because it already spans chat, code, maths,
multilingual text, instructions, and software-engineering content. It is a
mixed-license aggregate. Its per-record source and license fields must be
preserved; a single repository-level license is not sufficient. It is never
used as the held-out evaluation source.

The first calibration ablation is measured in **post-template tokens**, not
rows: 32K, 128K, 512K, then 1.5M tokens for recipes that survive the cheaper
stages. Keep domain proportions fixed so dataset size is the changed variable.
Each calibration manifest is checked for exact and token 5-gram near-duplicates
using both Jaccard and shorter-sequence containment against the held-out
manifest before a protocol can be built.

## Manifest contract

`rotquant.eval.data_manifest` records:

- immutable dataset revision, split/subset, URL, licenses, and redistribution
  policy for every source;
- stable source row ID, domain, per-row licenses, and normalized metadata;
- exact post-template token IDs, plus a cross-platform SHA-256 identity using
  versioned little-endian unsigned-64 encoding;
- exact tokenizer revision and selected chat-template hash;
- ordered transformations and seed;
- an aggregate content fingerprint that binds item order as well as content.

Mutable revisions such as `main`, `master`, `latest`, and `HEAD` are rejected.
Evaluation manifests reject any composition other than the registered 60 rows
per domain. Duplicate token sequences within a manifest are rejected, and the
protocol builder requires calibration/evaluation tokenizer and template
identity before checking leakage.

For gated or metadata-only sources, keep the replayable manifest in private
experiment storage and publish `--public-summary-output`. The summary contains source
metadata, row identities, token hashes, transformations, and the full-manifest
fingerprint, but no source-derived token IDs. A public summary is auditable but
deliberately cannot be passed off as locally replayable.

### Source config

```yaml
sources:
  - source_id: bfcl-v4
    revision: "<immutable upstream commit>"
    split: test
    subset: simple
    licenses: [Apache-2.0]
    url: https://github.com/EnlightenedAI/BFCL
    redistribution: allowed
transformations:
  - normalize_source_adapter:v1
  - apply_chat_template:add_generation_prompt=true
```

### Prepared JSONL

Each row contains one of `messages`, `text`, or already prepared `token_ids`:

```json
{"item_id":"bfcl-simple-001","domain":"agentic","source_id":"bfcl-v4","source_record_id":"simple_001","licenses":["Apache-2.0"],"messages":[{"role":"user","content":"..."}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"metadata":{"language":"en"}}
```

Build private replayable manifests, check leakage, then bind them to a model:

```bash
python scripts/build_competitive_manifest.py \
  --role calibration --input calibration.jsonl --sources calibration.yaml \
  --tokenizer-id Qwen/Qwen3.5-4B --tokenizer-revision <commit> \
  --output manifests/private/calibration.json \
  --public-summary-output manifests/public/calibration.json

python scripts/build_competitive_manifest.py \
  --role evaluation --input held_out.jsonl --sources held_out.yaml \
  --tokenizer-id Qwen/Qwen3.5-4B --tokenizer-revision <commit> \
  --near-against manifests/private/calibration.json \
  --output manifests/private/held_out.json \
  --public-summary-output manifests/public/held_out.json

python scripts/build_competitive_protocol.py \
  --calibration-manifest manifests/private/calibration.json \
  --evaluation-manifest manifests/private/held_out.json \
  --model-id Qwen/Qwen3.5-4B --model-revision <commit> \
  --output manifests/qwen35-4b-protocol.json
```

Pre-tokenized inputs require `--chat-template-sha256`; this prevents imported
engine outputs from bypassing template identity.

## Engine-neutral run records

Each backend emits a metadata JSON and one observation JSONL row per prompt.
Metadata binds the artifact SHA-256/actual bytes, engine and engine revision,
and protocol fingerprint. An observation carries:

- prompt token identity and domain;
- 32 per-token teacher KL values and top-1 matches;
- the source and candidate 32-token greedy continuations.

Create metadata from the deployed files rather than transcribing sizes. Repeat
`--artifact` for weights, projector, MTP, or any other required module; the
tool sums every file and creates a deterministic bundle identity:

```bash
python scripts/build_competitive_run_metadata.py \
  --protocol manifests/qwen35-4b-protocol.json \
  --name rotquant-w3 --format gguf \
  --engine llama.cpp --engine-revision <commit> \
  --artifact weights.gguf=artifacts/qwen35-4b-w3.gguf \
  --artifact projector.gguf=artifacts/mmproj.gguf \
  --output runs/rotquant-metadata.json
```

Infrastructure failures use a separate structured record with `load`,
`prefill`, `logits`, `generation`, or `scoring` stage. An incomplete run is
reported, but it receives no `ArtifactEvaluation` and cannot enter a quality
comparison. “Completed” means all expected records are present; it is not a
quality pass.

```bash
python scripts/aggregate_competitive_run.py \
  --protocol manifests/qwen35-4b-protocol.json \
  --prompt-manifest manifests/private/held_out.json \
  --metadata runs/rotquant-metadata.json \
  --observations runs/rotquant-observations.jsonl \
  --failures runs/rotquant-failures.jsonl \
  --output runs/rotquant-report.json

python scripts/compare_competitive_runs.py \
  --protocol manifests/qwen35-4b-protocol.json \
  --candidate runs/rotquant-report.json \
  --baseline runs/unsloth-report.json \
  --output runs/rotquant-vs-unsloth.json
```

Comparison requires completed reports, the same prompt identities/protocol,
and deployed sizes within 1% by default. It emits paired prompt deltas,
deterministic bootstrap 95% intervals, domain deltas, and the full per-prompt
audit table. Run and comparison reports are themselves content-addressed, so
post-aggregation edits fail validation. Deltas are always candidate minus baseline; the tool does not
invent a winner or promotion threshold.

### Transformers and RotQuant collector

The Python collector splits source capture from candidate scoring. Source
teacher logits are stored once as object-free FP16 NumPy archives (roughly
3 GB for 300 prompts, 32 tokens, and a 150K-token vocabulary). Candidate runs
then need only one model resident, persist one prompt record at a time, and can
resume safely after a Colab interruption. The candidate must be a local file or
directory: every file named by the run metadata is rehashed before model load,
and the per-prompt resume key includes the resulting artifact metadata.

```bash
python scripts/collect_competitive_transformers.py source \
  --protocol manifests/qwen35-4b-protocol.json \
  --prompt-manifest manifests/private/held_out.json \
  --reference-dir runs/source-references \
  --model-loader multimodal_lm

python scripts/collect_competitive_transformers.py candidate \
  --protocol manifests/qwen35-4b-protocol.json \
  --prompt-manifest manifests/private/held_out.json \
  --reference-dir runs/source-references \
  --candidate-kind rotquant \
  --candidate artifacts/qwen35-4b-rotquant \
  --run-metadata runs/rotquant-metadata.json \
  --fallback \
  --work-dir runs/rotquant-work \
  --observations runs/rotquant-observations.jsonl \
  --failures runs/rotquant-failures.jsonl \
  --model-loader multimodal_lm
```

The source continuation is scored teacher-forced by both models for one full
KL value and top-1 decision per generated token. The candidate also generates
its own 32-token greedy continuation. Explicit all-ones attention masks and
fixed min/max new-token counts avoid PAD/EOS ambiguity. This collector is a
quality reference path, not a packed-throughput benchmark; `--fallback`
materializes RotQuant weights for faster scoring and must not support runtime
memory claims.

## What remains before the 4B competitive run

1. implement and review the five upstream source adapters;
2. select/freeze 60 rows per domain and materialize the licensed private/public
   manifests;
3. add llama.cpp GGUF collectors for the same-size standard and exact-size
   Unsloth artifacts (the source Transformers and RotQuant reference collector
   is implemented);
4. add task-outcome scorers (execution, function arguments, maths, and IFEval)
   alongside—not inside—the divergence metric;
5. run the calibration-size ablation and freeze the winning data/optimizer
   recipe before any Qwen3.8-27B final result is inspected.
