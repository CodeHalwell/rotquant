import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.cached_decode_models import HFDecodeModel
from scripts.cached_decode_parity import (
    capture_references,
    compare_native,
    stop_metrics,
    validate_cases,
    verify_references,
)


class CachedToy(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
        prior = 0 if past_key_values is None else past_key_values
        totals = input_ids.cumsum(-1) + prior
        logits = -(torch.arange(11)[None, None] - totals[..., None] % 11).float().square()
        assert attention_mask.shape[-1] == (0 if prior == 0 else self.previous) + input_ids.shape[-1]
        self.previous = attention_mask.shape[-1]
        return SimpleNamespace(logits=logits, past_key_values=totals[:, -1:] if use_cache else None)


class DriftModel:
    def __init__(self, cached_fault=False):
        self.ids, self.cached_fault = [], cached_fault

    def evaluate(self, ids, *, reset=False, selected=1):
        self.ids = list(ids) if reset else self.ids + list(ids)
        values = -np.square(np.arange(11) - sum(self.ids) % 11).astype(np.float32)
        if self.cached_fault and not reset:
            values = np.roll(values, 1)
        return values


def cases():
    return [{'id': 'chat', 'prompt_ids': list(range(1, 19)), 'teacher_ids': [2, 3, 10]}]


def test_identical_trajectories_pass_all_cache_and_chunk_modes(tmp_path):
    record = capture_references(lambda mode: DriftModel(), cases(), [10], tmp_path / 'hf', {'a': 1}, 3)
    result = compare_native(lambda mode: DriftModel(), record, {'a': 1}, tmp_path / 'native')
    assert result['passed']
    assert {row['history'] for row in result['rows']} == {'teacher', 'candidate'}
    assert {check['mode'] for row in result['rows'] for check in row['checks']} == {
        'cached', 'chunked', 'recompute'}


def test_cached_drift_fails_and_reports_first_divergence(tmp_path):
    record = capture_references(lambda mode: DriftModel(), cases(), [10], tmp_path / 'hf', {}, 3)
    result = compare_native(lambda mode: DriftModel(True), record, {}, tmp_path / 'native')
    assert not result['passed']
    cached = result['rows'][0]['checks'][0]
    assert cached['first_failed_step'] == cached['first_argmax_difference'] == 1
    assert cached['rows'][1]['actual_logits_path']
    assert 'probability' in cached['rows'][1]['actual']['stop_tokens']['10']
    assert result['rows'][0]['checks'][2]['passed']


def test_tied_argmax_is_recorded_without_relaxing_numerical_gate(tmp_path):
    record = capture_references(lambda mode: DriftModel(), cases(), [10], tmp_path, {}, 3)
    path = record['cases'][0]['histories']['teacher']['reference_paths'][0]
    np.save(path, np.zeros(11))
    with pytest.raises(ValueError, match='logits changed'):
        verify_references(record, {})


@pytest.mark.parametrize('mutation', ['binding', 'prompt', 'teacher', 'case_id', 'cache'])
def test_reference_binding_rejects_tampering(tmp_path, mutation):
    record = capture_references(lambda mode: DriftModel(), cases(), [10], tmp_path, {}, 3)
    changed = copy.deepcopy(record)
    if mutation == 'binding':
        changed['binding'] = {'x': 1}
    elif mutation == 'prompt':
        changed['cases'][0]['prompt_ids'][0] += 1
    elif mutation == 'teacher':
        changed['cases'][0]['histories']['teacher']['continuation_ids'][0] += 1
    elif mutation == 'case_id':
        changed['source_cases'][0]['id'] = '../escape'
    else:
        changed['passed'] = False
    with pytest.raises(ValueError):
        verify_references(changed, {})


def test_hf_adapter_passes_full_attention_history_and_resets():
    adapter = HFDecodeModel(CachedToy(), 'cpu')
    first = adapter.evaluate([1, 2], reset=True)
    second = adapter.evaluate([3])
    full = HFDecodeModel(CachedToy(), 'cpu', cached=False).evaluate([1, 2, 3], reset=True)
    assert first.argmax() == 3 and second.argmax() == 6
    np.testing.assert_array_equal(full, second)
    assert adapter.evaluate([4], reset=True).argmax() == 4


def test_invalid_nonfinite_or_out_of_vocabulary_stops():
    with pytest.raises(ValueError, match='finite'):
        stop_metrics([float('nan'), 0], [1])
    with pytest.raises(ValueError, match='valid vocabulary'):
        stop_metrics([0, 1], [2])
    with pytest.raises(ValueError, match='nonnegative integers'):
        validate_cases([{'id': 'a', 'prompt_ids': [True], 'teacher_ids': [1]}])
