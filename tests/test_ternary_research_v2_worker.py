import json
from types import SimpleNamespace

import numpy as np

from scripts.native_gpu_workflow import artifacts
from scripts.ternary_inference import InferenceProfile
from scripts.ternary_research_v2_worker import V2Worker


class NativeFixture:
    vocab = 9
    supports_no_logits = True
    supports_compact_greedy = True

    def __init__(self, compact_drift=False):
        self.history, self.requests = [], []
        self.compact_drift = compact_drift

    def evaluate(self, ids, reset=False, selected=1):
        if reset:
            self.history = []
        self.history.extend(map(int, ids))
        self.requests.append(selected)
        result = np.zeros(self.vocab, dtype=np.float32)
        result[sum(self.history) % self.vocab] = 1
        return np.empty((0, self.vocab)) if selected == 0 else result

    def greedy(self, ids, reset=False):
        result = self.evaluate(ids, reset=reset)
        value = int(result.argmax())
        return (value + 1) % self.vocab if self.compact_drift and not reset and len(ids) == 1 else value


def worker():
    result = V2Worker.__new__(V2Worker)
    profile = InferenceProfile(graph_prefill=True, compact_greedy=True)
    result.profile = lambda: profile
    return result


def test_graph_prefill_and_compact_greedy_validate_every_forced_cached_step():
    result = worker()
    raw = NativeFixture()
    case = {'id': 'chat', 'prompt_ids': list(range(40))}
    receipt = result.capability_trace(raw, result.profile().wrap(raw), case, 'teacher', [4, 8, 2])
    assert receipt['passed'] and len(receipt['rows']) == 4
    assert 0 in raw.requests
    broken = NativeFixture(compact_drift=True)
    failed = result.capability_trace(broken, result.profile().wrap(broken), case, 'candidate', [4, 8])
    assert not failed['passed'] and failed['first_failed_step'] == 1
    assert failed['rows'][0]['compact_token_identical']
    assert not failed['rows'][1]['compact_token_identical']


def test_cache_binding_uses_actual_rotquant_checkpoint_manifest(tmp_path):
    result = worker()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    manifest = checkpoint / 'rotquant_config.json'
    manifest.write_text('{}')
    diagnostic = tmp_path / 'diagnostics.json'
    diagnostic.write_text('{}')
    result.args = SimpleNamespace(identity='source', checkpoint=checkpoint, diagnostics=diagnostic, label='ternary')
    binding = result.cache_binding()
    assert binding['checkpoint'] == artifacts([manifest])


def test_bonsai_partial_measurement_context_keeps_prism_backend(monkeypatch, tmp_path):
    from scripts.w4_research_worker import Worker
    result = worker()
    result.args = SimpleNamespace(phase='bonsai', identity='source', label=None, profile='reference')
    result.output, result.data = tmp_path, {'fingerprint': 'frozen-inputs'}
    monkeypatch.setattr(Worker, 'native_results', lambda *args: {'measured': True})
    assert result.native_results(None, None, None, None) == {'measured': True}
    context = json.loads((tmp_path / 'measurement-context.json').read_text())
    assert context['label'] == 'bonsai2_4b' and context['profile'] == 'prism'
    assert context['backend'] == 'CUDA0/Prism/chunk16'
