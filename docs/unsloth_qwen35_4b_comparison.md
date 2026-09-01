# Unsloth Qwen3.5-4B comparison note

Research snapshot: 2026-09-01.

## What exists publicly

The current official [Qwen3.5 documentation](https://unsloth.ai/docs/models/qwen3.5.md)
states that its Qwen3.5 GGUF uploads use **Dynamic 2.0**. Dynamic 3.0 is the
newer target described for Qwen3.8, but the exact Qwen3.5-4B artifact available
for the development comparison is therefore a Dynamic-2-era `UD` release, not
a Qwen3.5-4B Dynamic 3.0 result. Keep those names distinct in reports.

The immutable
[`unsloth/Qwen3.5-4B-GGUF` revision](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523)
contains the following files used by the new comparator:

| File | Exact bytes | SHA-256 | Role |
|---|---:|---|---|
| `Qwen3.5-4B-BF16.gguf` | 8,424,393,632 | `9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a` | same-engine teacher |
| `Qwen3.5-4B-UD-Q4_K_XL.gguf` | 2,912,109,728 | `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7` | released text-backbone candidate |
| `mmproj-F16.gguf` | 672,423,616 | `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864` | counted multimodal projector |

The complete released candidate bundle is 3,584,533,344 bytes (3.338 GiB).
RotQuant's current complete W4 accounting is 3,787,286,336 bytes (3.527 GiB),
5.66% larger relative to Unsloth. The nearest released Q4 artifact is useful as
a quality anchor but fails RotQuant's registered <=1% matched-byte comparison
gate. A competitive win/loss cannot be declared from this pair.

Unsloth's public Qwen3.5 benchmark material does not provide a numeric
BF16-to-UD-Q4_K_XL KL result for the 4B model. The repository therefore
measures the released file instead of transcribing a result from a different
model, rate, provider, or custom allocation.

## Exact comparison implemented here

`scripts/run_unsloth_qwen35_4b_kl.py` performs an engine-normalized comparison:

1. recreate the four C4 documents at skip 4096 and their first 512 token IDs,
   matching the completed RotQuant W4A8 logit-fidelity stage;
2. evaluate the pinned BF16 GGUF with the pinned llama.cpp engine and persist
   all next-token distributions as FP16, one prompt at a time;
3. unload BF16, evaluate the exact UD-Q4_K_XL file on the same token IDs, and
   compute full-vocabulary teacher KL, median/p95/max KL, top-1 agreement,
   source/candidate NLL, and paired token bootstrap intervals;
4. verify all downloaded byte counts and SHA-256 values and count the F16
   projector in deployed bytes, while clearly recording that the text-only KL
   path does not execute it;
5. persist source references and candidate prompt records independently so a
   Colab interruption resumes without loading two models together.

The engine is pinned through `llama-cpp-python` revision
`3691546f1c9e0c1bf93323dff02230bd959cf562`, whose llama.cpp submodule is
`4df29be4f4c3673f428170fda944a5b19f743bb8`. This matters because the official
[llama.cpp perplexity documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
warns that its values are implementation-dependent and describes saving the
full source distributions before calculating quantized-model KL. Comparing
BF16 and UD through the same engine removes the largest avoidable engine
confounder. The result still has to match RotQuant's recorded input hashes
before the two KL values are placed in the same development table.

## What this does and does not answer

The next Colab result will answer whether released Unsloth UD-Q4_K_XL changes
the same held-out token distributions more or less than current RotQuant W4 on
this small developmental C4 slice. It also gives top-1 agreement, which directly
answers whether the most likely predicted token changed.

It will not establish Dynamic 3.0 parity, because:

- the released Qwen3.5-4B control is documented as Dynamic 2.0;
- the artifacts differ by 5.66% in complete bytes;
- four 512-token C4 prompts are a drift sentinel, not the frozen 300-prompt
  agentic/code/maths/multilingual/long-document suite;
- no task outcomes or free-running 32-token trajectories are collected by this
  KL-only leg; and
- RotQuant's current Python fallback is not a production runtime comparison.

The final provider comparison must repeat the same engine-normalized method on
the frozen 300-prompt manifest, add greedy trajectory and task outcomes, and
interpolate or build RotQuant/standard-GGUF artifacts within 1% of the released
Unsloth byte count.
