from types import SimpleNamespace

import numpy as np
import torch


class HFDecodeModel:
    def __init__(self, model, device, *, cached=True):
        self.model, self.device, self.cached = model.eval(), device, cached
        self.cache, self.positions = None, 0

    @torch.no_grad()
    def evaluate(self, ids, *, reset=False, selected=1):
        if reset:
            self.cache, self.positions = None, 0
        if not ids or not 1 <= selected <= len(ids):
            raise ValueError('decode requires tokens and a bounded logits selection')
        values = torch.tensor(list(ids), device=self.device, dtype=torch.long).unsqueeze(0)
        if not self.cached and not reset:
            raise ValueError('uncached model requires the complete history and reset')
        inputs = {'input_ids': values, 'attention_mask': torch.ones(
            (1, self.positions + len(ids)), device=self.device, dtype=torch.long),
            'use_cache': self.cached}
        if self.cache is not None:
            inputs['past_key_values'] = self.cache
        output = self.model(**inputs)
        if self.cached:
            self.cache = getattr(output, 'past_key_values', None)
            if self.cache is None:
                raise ValueError('cached inference returned no cache')
        self.positions += len(ids)
        logits = output.logits[0, -selected:].float().cpu().numpy()
        return logits[0] if selected == 1 else logits


class DecodeTrace:
    def __init__(self, model, prompt, mode, chunk_tokens=16):
        if mode not in {'cached', 'chunked', 'recompute'}:
            raise ValueError('unsupported decode trace mode')
        if not prompt or chunk_tokens < 1:
            raise ValueError('decode trace requires a prompt and positive chunk size')
        self.model, self.history, self.mode = model, list(prompt), mode
        width = chunk_tokens if mode == 'chunked' else len(prompt)
        for start in range(0, len(prompt), width):
            self.logits = model.evaluate(prompt[start:start+width], reset=start == 0)

    def advance(self, token):
        self.history.append(int(token))
        if self.mode == 'recompute':
            self.logits = self.model.evaluate(self.history, reset=True)
        else:
            self.logits = self.model.evaluate([int(token)], reset=False)
        return self.logits


def greedy_trace(model, prompt, stop_ids, steps):
    trace, tokens = DecodeTrace(model, prompt, 'cached'), []
    for step in range(steps):
        token = int(np.asarray(trace.logits).argmax())
        tokens.append(token)
        if token in stop_ids:
            break
        if step + 1 < steps:
            trace.advance(token)
    return SimpleNamespace(tokens=tokens, stopped=bool(tokens and tokens[-1] in stop_ids))
