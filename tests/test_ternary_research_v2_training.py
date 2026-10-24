import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import ternary_research_v2_training as training


class Teacher(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        logits = torch.zeros(*input_ids.shape, 64)
        logits[..., 1] = 5
        return SimpleNamespace(logits=logits)


def mixed_fixture(monkeypatch):
    def chat_batches(manifest, split):
        ids = torch.tensor([[5 if split == 'train' else 7, 3, 9, 2]])
        return [{'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                 'response_mask': torch.tensor([[False, False, True, True]])}]
    monkeypatch.setattr(training, 'chat_batches', chat_batches)
    data = {'fingerprint': 'immutable-c4', 'documents': {
        'recovery_train': [{'input_ids': [11, 12, 13, 14]}],
        'recovery_validation': [{'input_ids': [21, 22, 23, 24]}]}}
    config = {'device': 'cpu', 'v2': {
        'chat': {'text_train_documents': 1, 'text_validation_documents': 1},
        'global': {'topk': 2, 'temperature': 1.0}}}
    manifest = {'fingerprint': 'complete-chat', 'required_teacher_ids': [2, 3]}
    return data, config, manifest


def test_mixed_teacher_calls_preserve_chat_masks_text_targets_and_independent_dense_audit(monkeypatch):
    data, config, manifest = mixed_fixture(monkeypatch)
    calls = training.teacher_calls(Teacher(), manifest, config, data)
    assert len(calls['train']) == len(calls['validation']) == 2
    assert calls['train'][0].inputs['response_mask'].tolist() == [[False, False, True, True]]
    assert calls['train'][1].inputs['response_mask'].all()
    for call in calls['train'] + calls['validation']:
        assert call.logits is None and call.topk_valid_mask is not None
        for token in (2, 3):
            assert ((call.topk_indices == token) & call.topk_valid_mask).any(-1).all()
    assert len(calls['audit']) == 1 and calls['audit'][0].logits.shape == (1, 4, 64)
    assert calls['data_id'] == training.fitting_identity(manifest, data, config)
    assert not torch.equal(calls['audit'][0].inputs['input_ids'], calls['train'][0].inputs['input_ids'])


def test_mixed_identity_binds_exact_text_partition_counts_and_rejects_missing_inputs(monkeypatch):
    data, config, manifest = mixed_fixture(monkeypatch)
    before = training.fitting_identity(manifest, data, config)
    changed = copy.deepcopy(config)
    changed['v2']['chat']['text_train_documents'] = 2
    assert before != training.fitting_identity(manifest, data, changed)
    with pytest.raises(ValueError, match='not enough immutable text'):
        training.fitting_batches(manifest, data, changed, 'train')
