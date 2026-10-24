"""Read-only, recursively verified historical results for a new numerical study."""
from __future__ import annotations

import copy
from pathlib import Path

from scripts.native_gpu_workflow import artifacts, fingerprint, write_json
from scripts.ternary_research_continuation import HashVerifier, read, verify_record
from scripts.ternary_research_plan import LABELS, PROTOCOL
from scripts.ternary_research_results import complete_measurement
from scripts.w4_research_results import comparison_row


class HistoricalStudy:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.verify = HashVerifier()
        self.visited = set()
        self.records, self.origins, self.lineage = self.read_generation(self.root)
        self.comparison = read(self.root / 'comparison.json')
        self.validate_complete()

    def read_generation(self, root):
        if root in self.visited:
            raise ValueError('cyclic historical lineage')
        self.visited.add(root)
        identity, state = read(root / 'identity.json'), read(root / 'workflow.json')
        bound = fingerprint(identity)
        if (identity.get('protocol') != PROTOCOL or state['controls']['identity'] != bound
                or state['status'] != 'passed' and root == self.root
                or any(row['status'] == 'running' for row in state['attempts'])):
            raise ValueError('historical identity/ledger is not terminal and bound')
        if read(root / 'config.json') != identity['config']:
            raise ValueError('historical configuration changed')
        records, origins, lineage = self.inherited(root, state, bound)
        for row in state['attempts']:
            if row['status'] == 'passed' and self.payload_stage(row['name']):
                verify_record(root, row, verify=self.verify)
                records[row['name']] = row
                origins[row['name']] = {'root': str(root), 'identity': bound}
        lineage.append({'root': str(root), 'identity': bound,
                        'files': artifacts([root / 'identity.json', root / 'workflow.json'])})
        return records, origins, lineage

    def inherited(self, root, state, identity):
        path = root / 'continuation.json'
        if not path.exists():
            return {}, {}, []
        receipt = read(path)
        if (receipt.get('child_identity') != identity
                or state['controls'].get('continuation') != fingerprint(receipt)):
            raise ValueError('historical continuation binding changed')
        parent = Path(receipt['parent_root']).resolve(strict=True)
        if parent == root or parent.is_relative_to(root) or root.is_relative_to(parent):
            raise ValueError('historical result roots must be separate')
        self.check_parent_hashes(parent, receipt)
        records, origins, lineage = self.read_generation(parent)
        imported = receipt['imports']
        for name, row in imported.items():
            if self.payload_stage(name) and records.get(name) != row:
                raise ValueError(f'historical imported stage differs from its ancestor: {name}')
        retained = {name: row for name, row in records.items() if name in imported}
        return retained, {name: origins[name] for name in retained}, lineage

    @staticmethod
    def check_parent_hashes(parent, receipt):
        hashes = artifacts([parent / 'identity.json', parent / 'workflow.json'])
        for key in ('identity', 'workflow'):
            if hashes[str(parent / f'{key}.json')]['sha256'] != receipt[f'parent_{key}_sha256']:
                raise ValueError('historical ancestor ledger or identity changed')
        if fingerprint(read(parent / 'identity.json')) != receipt['parent_identity']:
            raise ValueError('historical ancestor identity mismatch')

    @staticmethod
    def payload_stage(name):
        return name in {'freeze', 'teacher'} or name.startswith(('prepare-', 'reload-', 'export-', 'measure-'))

    def value(self, name):
        return copy.deepcopy(self.records[name]['value'])

    def validate_complete(self):
        rows = {row['label']: row for row in self.comparison['rows']}
        if self.comparison.get('complete') is not True or set(rows) != set(LABELS):
            raise ValueError('the requested historical comparison is not fully complete')
        data = self.value('freeze')['fingerprint']
        for label in LABELS:
            value = self.value('measure-' + label)
            if (not complete_measurement(value) or value.get('data_fingerprint') != data
                    or rows[label]['status'] != 'measured' or value['label'] != label):
                raise ValueError(f'historical measurement is incomplete or unbound: {label}')
        for label in LABELS:
            self.validate_comparison_row(label, rows[label])
        self.comparison_hash = artifacts([self.root / 'comparison.json'])

    def validate_comparison_row(self, label, row):
        measurement = self.value('measure-' + label)
        checkpoint = self.value('teacher' if label == 'fp16' else 'prepare-' + label) if label != 'bonsai2_4b' else {}
        if label == 'bonsai2_4b':
            assets = measurement['artifacts']
            exported = {'artifact_bytes': assets['complete_model_payload_bytes'],
                        'language_payload_bytes': assets['language_model_bytes']}
        else:
            exported = self.value('export-' + label)
        expected = comparison_row(label, checkpoint, exported, measurement)
        expected.update(backend=measurement.get('backend'),
                        language_payload_bytes=exported.get('language_payload_bytes'))
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f'historical comparison differs from verified stage values: {label}')
        if row.get('training_metrics') != checkpoint.get('metrics'):
            raise ValueError(f'historical training metrics differ from verified preparation: {label}')
        origin = row.get('origin')
        verified = self.origins['measure-' + label]
        if origin and (not isinstance(origin, dict)
                       or any(origin.get(key) != value for key, value in verified.items())):
            raise ValueError(f'historical comparison origin differs from its verified stage: {label}')

    def receipt(self):
        return {'protocol': 'ternary-historical-import-v1', 'historical_root': str(self.root),
                'lineage': self.lineage, 'comparison': self.comparison_hash,
                'origins': self.origins, 'imports': self.records,
                'policy': 'Read-only historical measurements. New source/runtime results use separate identities.'}

    def save(self, child):
        child = Path(child).resolve()
        if any(child == root or child.is_relative_to(root) or root.is_relative_to(child)
               for root in self.visited):
            raise ValueError('new study must be outside every historical result root')
        receipt, path = self.receipt(), child / 'historical-import.json'
        if path.exists() and read(path) != receipt:
            raise ValueError('historical imports changed; choose a new result root')
        write_json(path, receipt)
        return receipt
