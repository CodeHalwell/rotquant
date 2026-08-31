"""Fixed evaluation protocol for RotQuant experiments.

Submodules (imported lazily by the runner so a CPU-only install works):

* :mod:`~rotquant.eval.perplexity`     -- paired-window PPL with auditable hashes
* :mod:`~rotquant.eval.zeroshot`       -- lm-evaluation-harness bundle
* :mod:`~rotquant.eval.layer_mse`      -- per-layer output drift (E7)
* :mod:`~rotquant.eval.trajectory`     -- multi-token free-running divergence
* :mod:`~rotquant.eval.logit_fidelity` -- teacher-logit KL / agreement
* :mod:`~rotquant.eval.kv_cache`       -- KV-cache quantization boundary
* :mod:`~rotquant.eval.quantization`   -- matched-budget quantizer comparison
* :mod:`~rotquant.eval.throughput`     -- greedy-decode tokens/s + peak VRAM (E8)
* :mod:`~rotquant.eval.competition`    -- exact-size external-artifact contracts
"""
