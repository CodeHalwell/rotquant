import copy

import pytest

from scripts.ternary_historical import HistoricalStudy
from scripts.w4_research_results import comparison_row


def historical_row():
    measured = {'status': 'complete', 'quality': {'test': {'summary': {'ppl': 14.2}}},
                'tasks': {'rows': [{'id': 'task', 'correct': True}]}, 'speed': {'summary': {'128': {}}},
                'backend': 'CUDA0/reference/chunk16'}
    checkpoint = {'artifact_bytes': 123, 'selection': {'ppl': 13.2}, 'metrics': {'step': 2}}
    exported = {'artifact_bytes': 99, 'language_payload_bytes': 80}
    history = HistoricalStudy.__new__(HistoricalStudy)
    history.records = {name: {'value': value} for name, value in (
        ('measure-w4_v6', measured), ('prepare-w4_v6', checkpoint), ('export-w4_v6', exported))}
    history.origins = {'measure-w4_v6': {'root': '/original', 'identity': 'frozen'}}
    row = comparison_row('w4_v6', checkpoint, exported, measured)
    row.update(backend=measured['backend'], language_payload_bytes=80,
               training_metrics=checkpoint['metrics'], origin={'root': '/original', 'identity': 'frozen'})
    return history, row


@pytest.mark.parametrize('field', ['quality', 'tasks', 'speed', 'checkpoint_bytes',
                                  'native_payload_bytes', 'language_payload_bytes', 'backend',
                                  'training_metrics', 'origin'])
def test_changed_comparison_values_cannot_be_imported_as_verified_measurements(field):
    history, row = historical_row()
    history.validate_comparison_row('w4_v6', row)
    broken = copy.deepcopy(row)
    broken[field] = 'changed'
    with pytest.raises(ValueError, match='historical'):
        history.validate_comparison_row('w4_v6', broken)


def test_historical_save_rejects_every_ancestor_and_descendant(tmp_path):
    history = HistoricalStudy.__new__(HistoricalStudy)
    history.visited = {tmp_path / 'parent' / 'study'}
    for path in (tmp_path / 'parent', tmp_path / 'parent' / 'study', tmp_path / 'parent' / 'study' / 'child'):
        with pytest.raises(ValueError, match='outside'):
            history.save(path)
    assert not list(tmp_path.rglob('historical-import.json'))
