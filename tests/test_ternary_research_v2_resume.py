import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.native_gpu_workflow import artifacts, write_json
from scripts.ternary_research_v2_resume import hydrate, measurement_context, preserve_partial
from scripts.ternary_research_v2_study import V2Study


def study(tmp_path):
    result = V2Study.__new__(V2Study)
    result.root, result.identity = tmp_path, 'exact-source-config-runtime'
    result.args = SimpleNamespace(historical_root=tmp_path.parent / 'historical')
    result.workflow = SimpleNamespace(state={'attempts': []})
    result.data = {'fingerprint': 'frozen'}
    result.teacher, result.bonsai_export = None, {}
    result.historical_rows, result.historical_receipt = [], None
    result.prepared, result.exported, result.measured, result.failures = {}, {}, {}, {}
    result.gates, result.native_parity, result.references = {}, {}, {}
    return result


def passed_worker(result, name, value):
    directory = result.root / 'stages' / name / 'attempt-001'
    write_json(directory / 'report.json', value)
    row = {'name': name, 'directory': str(directory), 'status': 'passed',
           'value': {**value, 'report_path': str(directory / 'report.json')},
           'artifacts': artifacts([directory / 'report.json'])}
    result.workflow.state['attempts'].append(row)
    return row


def test_resume_hydrates_prepared_exported_and_completed_profile_metrics_before_setup_failure(tmp_path):
    result = study(tmp_path)
    passed_worker(result, 'prepare-ternary_global_qat', {'checkpoint': '/saved/checkpoint', 'artifact_bytes': 200})
    passed_worker(result, 'export-ternary_global_qat', {'native': '/saved/native', 'artifact_bytes': 150})
    passed_worker(result, 'measure-ternary_global_qat@decode4',
                  {'status': 'complete', 'quality': {'test': {'summary': {'ppl': 18.2}}}})
    hydrate(result)
    assert result.prepared['ternary_global_qat']['artifact_bytes'] == 200
    assert result.exported['ternary_global_qat']['artifact_bytes'] == 150
    assert result.measured['ternary_global_qat@decode4']['quality']['test']['summary']['ppl'] == 18.2
    result.failures['build-runtime'] = 'new runtime unavailable'
    row = result.fresh_row('ternary_global_qat', 'decode4')
    assert row['quality']['test']['summary']['ppl'] == 18.2
    assert row['native_payload_bytes'] == 150


def test_resume_refuses_changed_receipts_or_reports(tmp_path):
    result = study(tmp_path)
    row = passed_worker(result, 'measure-ternary_global_qat@reference', {'status': 'complete'})
    Path(row['directory'], 'report.json').write_text('{"changed": true}')
    hydrate(result)
    assert result.measured == {}


def partial_files(result, key):
    directory = result.root / 'stages' / ('measure-' + key) / 'attempt-001'
    label, profile = key.split('@')
    write_json(directory / 'measurement-context.json', measurement_context(result, label, profile))
    write_json(directory / 'test' / 'quality.json', {'rows': [{'id': 'q'}], 'summary': {'ppl': 17.5}})
    row = {'name': 'measure-' + key, 'directory': str(directory), 'status': 'interrupted', 'error': 'cap'}
    result.workflow.state['attempts'].append(row)
    return directory, row


def test_partial_measurement_is_bound_hashed_and_reconstructed_after_process_loss(tmp_path):
    result = study(tmp_path)
    key = 'ternary_global_qat@decode4'
    directory, row = partial_files(result, key)
    hydrate(result)
    assert row['partial']['artifacts']
    assert result.measured[key]['quality']['test']['summary']['ppl'] == 17.5
    assert result.measured[key]['status'] == 'failed'
    assert result.row_error('ternary_global_qat', 'decode4') == 'cap'
    assert result.row_error('ternary_global_qat', 'reference') is None
    assert result.fresh_row('ternary_global_qat', 'decode4')['status'] == 'failed'
    assert not (directory / 'speed.json').exists()


def test_partial_wrong_backend_does_not_turn_into_valid_measurement(tmp_path):
    result = study(tmp_path)
    directory, row = partial_files(result, 'bonsai2_4b@prism')
    path = directory / 'measurement-context.json'
    value = json.loads(path.read_text())
    value['backend'] = 'CUDA0/reference/chunk16'
    write_json(path, value)
    preserve_partial(result, directory, 'bonsai2_4b@prism', RuntimeError('failed'), row=row)
    assert 'partial' not in row and not result.measured


@pytest.mark.parametrize('path,value', [
    (('cached_parity', 'max_steps'), True),
    (('chat', 'max_new_tokens'), 9999),
    (('chat', 'context_lengths'), [512.0, 1024, 2048]),
    (('global', 'temperature'), 0),
    (('inference_profiles', 'reference'), {'kernel': 'decode4'}),
    (('inference_profiles', '../bad'), {'kernel': 'reference'}),
    (('refresh_controls',), ['fp16', 'w4_v6', 'w5_v6', 'w6_v6', 'w6_v6']),
])
def test_invalid_dynamic_settings_fail_before_gpu_launch(path, value):
    import yaml

    from scripts.ternary_research_v2_plan import DEFAULT_CONFIG, validate_config
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    target = config['v2']
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError)):
        validate_config(config)


def test_read_only_hydration_runs_after_expiry_and_charges_elapsed_without_renewal(monkeypatch, tmp_path):
    from scripts import ternary_research_v2_resume as resume
    result = study(tmp_path)
    result.workflow.state['active_seconds'] = 480 * 60
    saved = []
    result.workflow.save = lambda: saved.append(result.workflow.state['active_seconds'])
    result.workflow.stage = lambda *args, **kwargs: pytest.fail('expired resume started an execution stage')
    clock = iter((100., 100.1, 102.5))
    monkeypatch.setattr(resume.time, 'monotonic', lambda: next(clock))
    checks = []
    monkeypatch.setattr(resume, 'hydrate', lambda value, verify: checks.append(value.identity))
    resume.hydrate_charged(result)
    assert checks == [result.identity]
    assert result.workflow.state['active_seconds'] == 480 * 60 + 2.5
    assert result.workflow.state['read_only_verification_seconds'] == 2.5
    assert saved == [480 * 60 + 2.5]
