"""Reconstruct verified v2 reports and retain partial measurements on interruption."""
from __future__ import annotations

import time
from pathlib import Path

from scripts.native_gpu_workflow import artifacts
from scripts.ternary_historical import HistoricalStudy
from scripts.ternary_research_continuation import HashVerifier, read
from scripts.ternary_research_plan import LABELS
from scripts.ternary_research_results import partial_measurement


class BoundedVerifier(HashVerifier):
    def __init__(self, seconds):
        super().__init__()
        self.deadline = time.monotonic() + seconds

    def matches(self, path, expected):
        if time.monotonic() > self.deadline:
            raise TimeoutError('read-only saved-result verification time limit exhausted')
        matched = super().matches(path, expected)
        if time.monotonic() > self.deadline:
            raise TimeoutError('read-only saved-result verification time limit exhausted')
        return matched


def hydrate_charged(study, seconds=2700):
    started = time.monotonic()
    try:
        hydrate(study, verify=BoundedVerifier(seconds))
    finally:
        elapsed = time.monotonic() - started
        study.workflow.state['active_seconds'] += elapsed
        study.workflow.state['read_only_verification_seconds'] = (
            study.workflow.state.get('read_only_verification_seconds', 0) + elapsed)
        study.workflow.save()


def load_current_history(root, verify):
    history = HistoricalStudy.__new__(HistoricalStudy)
    history.root, history.verify, history.visited = Path(root).resolve(strict=True), verify, set()
    history.records, history.origins, history.lineage = history.read_generation(history.root)
    history.comparison = read(history.root / 'comparison.json')
    history.validate_complete()
    return history


def verified_record(study, row, verify):
    directory = Path(row['directory'])
    if directory.parent != study.root / 'stages' / row['name'] or not directory.name.startswith('attempt-'):
        return None
    record = row if row['status'] == 'passed' else row.get('partial', {})
    if not record.get('artifacts') or not verify(record['artifacts']):
        return None
    value = record['value']
    if row['status'] == 'passed':
        path = directory / 'report.json'
        expected = {key: val for key, val in value.items() if key != 'report_path'}
        if str(path) not in record['artifacts'] or read(path) != expected:
            return None
    return value


def restore_historical(study, verify):
    path, report = study.root / 'historical-import.json', study.root / 'comparison.json'
    records = [x for x in study.workflow.state['attempts']
               if x['name'] == 'verify-historical-results' and x['status'] == 'passed']
    if not records or not path.exists() or not report.exists():
        return
    row = records[-1]
    if not row.get('artifacts') or not verify(row['artifacts']):
        return
    receipt, previous = read(path), read(report)
    if previous.get('identity') != study.identity or receipt.get('historical_root') != str(study.args.historical_root.resolve()):
        return
    history = HistoricalStudy.__new__(HistoricalStudy)
    history.records, history.origins = receipt['imports'], receipt['origins']
    rows = previous.get('historical_rows', [])
    if {row['label'] for row in rows} != set(LABELS):
        return
    for row in rows:
        history.validate_comparison_row(row['label'], row)
    current = load_current_history(study.args.historical_root, verify)
    if current.receipt() != receipt:
        raise ValueError('historical payloads or lineage changed after the saved import')
    study._verified_history = current
    study.historical_receipt, study.historical_rows = receipt, rows
    restore_history_models(study, history)
    study.bonsai_export = next(row for row in rows if row['label'] == 'bonsai2_4b')


def restore_history_models(study, history):
    study.data, study.teacher = history.value('freeze'), history.value('teacher')
    study.exported['fp16'] = history.value('export-fp16')
    for label in LABELS:
        if label not in ('fp16', 'bonsai2_4b'):
            study.prepared[label] = history.value('prepare-' + label)
            study.exported[label] = history.value('export-' + label)
    assets = history.value('measure-bonsai2_4b')['artifacts']
    study.bonsai_assets = Path(assets['model']).parent
    study.bonsai_artifacts = assets['artifact_files']


def restore_value(study, name, value):
    targets = {'prepare-': study.prepared, 'export-': study.exported,
               'measure-': study.measured, 'cache-native-': study.native_parity,
               'cache-reference-': study.references}
    for prefix, target in targets.items():
        if name.startswith(prefix):
            target[name.removeprefix(prefix)] = value
            if prefix == 'cache-native-':
                study.gates[name.removeprefix(prefix)] = value
            return
    if name.startswith('preflight-'):
        study.gates['preflight@' + name.removeprefix('preflight-')] = value
    if name == 'freeze-complete-teacher-turns':
        study.chat = value
    if name == 'freeze-cached-chat-diagnostics':
        study.diagnostics = value


def hydrate(study, *, verify=None):
    verify = verify or HashVerifier()
    restore_historical(study, verify)
    latest = {row['name']: row for row in study.workflow.state['attempts']}
    for name, row in latest.items():
        if name.startswith('measure-') and row['status'] in ('failed', 'interrupted') and not row.get('partial'):
            recover_interrupted(study, row)
        value = verified_record(study, row, verify)
        if value is not None:
            restore_value(study, name, value)
            if row['status'] != 'passed':
                study.failures[name] = row.get('error', 'measurement interrupted')
    keys = {label + '@reference' for label in LABELS if label != 'bonsai2_4b'}
    if keys <= study.gates.keys() and all(study.gates[key].get('passed') is True for key in keys):
        study.gates['retained-cached-parity'] = {'passed': True}


def measurement_context(study, label, profile):
    return {'protocol': 'ternary-measurement-context-v2', 'identity': study.identity,
            'data_fingerprint': study.data['fingerprint'], 'label': label, 'profile': profile,
            'backend': 'CUDA0/Prism/chunk16' if label == 'bonsai2_4b' else f'CUDA0/{profile}/chunk16'}


def preserve_partial(study, directory, key, error, *, row=None):
    if study.data is None or '@' not in key:
        return
    label, profile = key.rsplit('@', 1)
    try:
        result = partial_measurement(directory, measurement_context(study, label, profile), str(error))
        if result is not None:
            value, files = result
            record = study.workflow.state['attempts'][-1] if row is None else row
            record['partial'] = {'value': value, 'artifacts': artifacts(files)}
            study.measured[key] = value
    except (OSError, ValueError):
        return


def recover_interrupted(study, row):
    directory = Path(row['directory'])
    if directory.parent == study.root / 'stages' / row['name'] and directory.name.startswith('attempt-'):
        preserve_partial(study, directory, row['name'].removeprefix('measure-'),
                         row.get('error', 'measurement process interrupted'), row=row)
