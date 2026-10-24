"""A resumable child study whose correctness gates precede every training stage."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from scripts.native_gpu_workflow import Workflow, fingerprint, write_json
from scripts.run_qwen35_ternary_research import TernaryStudy
from scripts.run_qwen35_vocabulary_budget import runtime_identity
from scripts.ternary_historical import HistoricalStudy
from scripts.ternary_research_continuation import read
from scripts.ternary_research_plan import LABELS
from scripts.ternary_research_results import complete_measurement
from scripts.ternary_research_v2_plan import PROTOCOL, bind_run, candidates, plan
from scripts.w4_research_plan import ROOT
from scripts.w4_research_results import comparison_row


class V2Study(TernaryStudy):
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.root, self.work = args.output_dir.resolve(), args.work_dir.resolve()
        parent = args.historical_root.resolve()
        if parent == self.root or parent.is_relative_to(self.root) or self.root.is_relative_to(parent):
            raise ValueError('v2 needs a separate result root outside its historical parent')
        self.identity = bind_run(self.root, config, runtime_identity(), parent)
        self.config_path = self.root / 'config.json'
        write_json(self.config_path, config)
        self.workflow = Workflow(self.root, ROOT, {'identity': self.identity}, args.budget_minutes,
                                 overnight=args.budget_minutes > 180)
        self.continuation = None
        self.data, self.teacher, self.runtime, self.chat, self.diagnostics = None, None, None, None, None
        self.prepared, self.exported, self.measured, self.failures, self.gates = {}, {}, {}, {}, {}
        self.references, self.native_parity = {}, {}
        self.historical_rows, self.historical_receipt = [], None
        self.bonsai_export = {}
        self.bonsai_artifacts = {}

    @staticmethod
    def stage_files(directory):
        return [p for p in Path(directory).rglob('*') if p.is_file() and not p.is_symlink()
                and 'logs' not in p.parts and p.suffix != '.pyc']

    def command(self, stage, script, *arguments):
        import sys
        stage.command([sys.executable, '-B', '-u', ROOT / 'scripts' / script, *arguments], script)

    @staticmethod
    def runtime_binding(library):
        from scripts.check_rq3_model import runtime_identity as library_identity
        if not Path(library).is_file():
            raise ValueError('current native runtime is missing')
        return library_identity(library)

    def worker_options(self, **options):
        defaults = super().worker_options(**options)
        if 'library' in defaults:
            defaults['runtime_id'] = fingerprint(self.runtime_binding(defaults['library']))
        return defaults

    def worker(self, name, phase, *, minutes=90, **options):
        defaults = self.worker_options(**options)
        self.active_stage = name
        def action(stage):
            return self.worker_action(stage, name, phase, defaults)
        result = self.workflow.stage(name, action, signature={k: str(v) for k, v in defaults.items()}, minutes=minutes)
        self.failures.pop(name, None)
        return result

    def worker_action(self, stage, name, phase, defaults):
        arguments = [str(value) for key, val in defaults.items()
                     for value in ('--' + key.replace('_', '-'), val)]
        try:
            self.command(stage, 'ternary_research_v2_worker.py', phase, '--output', stage.directory, *arguments)
        except BaseException as error:
            if phase in {'native', 'bonsai'}:
                from scripts.ternary_research_v2_resume import preserve_partial
                preserve_partial(self, stage.directory, name.removeprefix('measure-'), error)
            raise
        result = read(stage.directory / 'report.json')
        files = self.stage_files(stage.directory)
        if phase == 'prepare_v2':
            from scripts.ternary_research_v2_training import latent_receipts
            files.extend(Path(receipt['path']) for receipt in latent_receipts(result))
        return {**result, 'report_path': str(stage.directory / 'report.json')}, sorted(set(files))

    def hydrate_verified(self):
        from scripts.ternary_research_v2_resume import hydrate_charged
        hydrate_charged(self)

    def import_history(self):
        def action(stage):
            history = getattr(self, '_verified_history', None) or HistoricalStudy(self.args.historical_root)
            receipt = history.save(self.root)
            self.accept_history(history, receipt)
            write_json(stage.directory / 'report.json', {'passed': True, 'receipt': fingerprint(receipt)})
            return {'passed': True}, [stage.directory / 'report.json', self.root / 'historical-import.json']
        self.workflow.stage('verify-historical-results', action,
                            signature=str(self.args.historical_root.resolve()), minutes=45, reuse=False)

    def accept_history(self, history, receipt):
        original = read(history.root / 'identity.json')['config']
        if any(self.config[key] != original[key] for key in
               ('model', 'model_revision', 'model_loader', 'calib_dataset_revision', 'seed')):
            raise ValueError('v2 source/model/data pins differ from the retained comparison')
        if history.value('freeze')['fingerprint'] != history.value('measure-fp16')['data_fingerprint']:
            raise ValueError('historical controls do not share frozen inputs')
        self.historical_receipt = receipt
        self.data, self.teacher = history.value('freeze'), history.value('teacher')
        self.historical_rows = history.comparison['rows']
        self.exported['fp16'] = history.value('export-fp16')
        for label in LABELS:
            if label not in ('fp16', 'bonsai2_4b'):
                self.prepared[label] = history.value('prepare-' + label)
                self.exported[label] = history.value('export-' + label)
        assets = history.value('measure-bonsai2_4b')['artifacts']
        self.bonsai_assets = Path(assets['model']).parent
        self.bonsai_artifacts = assets['artifact_files']
        self.bonsai_export = next(row for row in self.historical_rows if row['label'] == 'bonsai2_4b')

    def reference(self, label, profile='reference'):
        key = label if profile == 'reference' else label + '@' + profile
        options = {'checkpoint': self.prepared[label]['checkpoint']} if label != 'fp16' else {}
        value = self.worker('cache-reference-' + key, 'cache_reference', label=label, profile=profile,
                            source=self.teacher['source'], diagnostics=self.diagnostics['diagnostics'],
                            minutes=90, **options)
        self.references[key] = value
        return value

    def preflight_profile(self, profile):
        from scripts.ternary_inference import InferenceProfile
        parsed = InferenceProfile.from_mapping(self.config['v2']['inference_profiles'][profile])
        self.active_stage = 'preflight-' + profile
        def action(stage):
            fixture = stage.directory / 'tiny.gguf'
            self.command(stage, 'make_rq3_model_fixture.py', '--llama-dir', self.runtime['source'],
                         '--output', fixture, '--backbone-bits', '2', '--vocabulary-bits', '6')
            self.command(stage, 'check_ternary_inference.py', '--library', self.runtime['library'],
                         '--model', fixture, '--backend', 'CUDA0', '--kernel', parsed.kernel,
                         '--output', stage.directory / 'report.json')
            result = read(stage.directory / 'report.json')
            if (result.get('passed') is not True or not result.get('operators', {}).get('dispatch')
                    or result.get('profile', {}).get('kernel') != parsed.kernel):
                raise ValueError('runtime profile preflight lacks passed device dispatch evidence')
            return result, self.stage_files(stage.directory)
        value = self.workflow.stage('preflight-' + profile, action,
                                    signature={'runtime': self.runtime, 'profile': parsed.identity(),
                                               'binaries': self.runtime_binding(self.runtime['library'])}, minutes=20)
        self.gates['preflight@' + profile] = value
        self.failures.pop('preflight-' + profile, None)
        return value

    def gate(self, label, profile):
        self.preflight_profile(profile)
        arithmetic = self.config['v2']['inference_profiles'][profile].get('arithmetic', 'w2a16')
        key = label if arithmetic == 'w2a16' else label + '@' + profile
        reference = self.references.get(key) or self.reference(label, profile if arithmetic != 'w2a16' else 'reference')
        key = label + '@' + profile
        result = self.worker('cache-native-' + key, 'cache_native',
                             label=label, profile=profile, references=reference['references'],
                             library=self.runtime['library'], native=self.exported[label]['native'], minutes=60)
        self.native_parity[key] = result
        self.gates[key] = result
        self.summarize()
        return result

    def parity_first(self):
        self.diagnostics = self.worker('freeze-cached-chat-diagnostics', 'diagnostics', minutes=30)
        for label in LABELS:
            if label != 'bonsai2_4b':
                self.reference(label)
                self.gate(label, 'reference')
        self.gates['retained-cached-parity'] = {'passed': True}
        self.summarize()

    def train_candidate(self, candidate):
        path = self.root / 'candidates' / f'{candidate.name}.json'
        write_json(path, asdict(candidate))
        options = {}
        if candidate.parent:
            prepared = self.prepared[candidate.parent]
            options = {'parent': prepared['checkpoint'], 'parent_report': prepared['report_path']}
        prepared = self.worker('prepare-' + candidate.name, 'prepare_v2', arm=path,
                               source=self.teacher['source'], teacher_dir=self.teacher['teacher_dir'],
                               chat=self.chat['chat'], calls=self.chat['calls'], minutes=240, **options)
        self.worker('reload-' + candidate.name, 'reload', checkpoint=prepared['checkpoint'],
                     probes=prepared['probes'], minutes=30)
        self.prepared[candidate.name] = prepared
        self.exported[candidate.name] = self.worker('export-' + candidate.name, 'export',
            checkpoint=prepared['checkpoint'], llama_dir=self.runtime['source'], minutes=40)
        self.reference(candidate.name)
        self.gate(candidate.name, 'reference')

    def fresh_measure(self, label, profile):
        key = label + '@' + profile
        try:
            gate = self.gate(label, profile)
            self.measured[key] = self.worker('measure-' + key, 'native', label=label, profile=profile,
                parity=gate['parity'], native=self.exported[label]['native'], library=self.runtime['library'],
                source=self.teacher['source'], teacher_dir=self.teacher['teacher_dir'],
                scorer_dir=self.teacher['scorer_dir'], minutes=120)
        except Exception as error:
            self.failures['measure-' + key] = f'{type(error).__name__}: {error}'
            if profile == 'reference':
                raise
        finally:
            self.summarize()

    def acquire_bonsai(self):
        destination = self.work / self.identity[:12] / 'bonsai-artifacts'
        def action(stage):
            self.command(stage, 'ternary_research_bonsai.py', 'acquire', '--output-dir', destination,
                         '--packing', self.config['bonsai']['packing'], '--report', stage.directory / 'report.json')
            value = read(stage.directory / 'report.json')
            if value['artifact_files'] != self.bonsai_artifacts:
                raise ValueError('fresh Bonsai artifacts differ from the historical pinned model')
            return value, [stage.directory / 'report.json', Path(value['model']), Path(value['vision'])]
        self.active_stage = 'fresh-bonsai-assets'
        result = self.workflow.stage('fresh-bonsai-assets', action,
            signature={'packing': self.config['bonsai'], 'artifacts': self.bonsai_artifacts}, minutes=30)
        self.bonsai_assets = Path(result['model']).parent
        return result

    def fresh_bonsai(self):
        self.acquire_bonsai()
        build_dir = self.work / self.identity[:12] / 'bonsai-build'
        def action(stage):
            self.command(stage, 'ternary_research_bonsai.py', 'build', '--source-dir',
                         self.work / self.identity[:12] / 'bonsai-prism-source', '--build-dir', build_dir,
                         '--backend', 'CUDA', '--jobs', '4', '--report', stage.directory / 'report.json')
            result = read(stage.directory / 'report.json')
            files = [p for p in build_dir.rglob('*') if p.is_file() and ('.so' in p.name or p.suffix == '.dylib')]
            return result, [stage.directory / 'report.json', build_dir / 'build-receipt.json', *files]
        self.active_stage = 'fresh-bonsai-runtime'
        runtime = self.workflow.stage('fresh-bonsai-runtime', action, signature=self.identity, minutes=60)
        self.measured['bonsai2_4b@prism'] = self.worker('measure-bonsai2_4b@prism', 'bonsai',
            assets=self.bonsai_assets, library=runtime['library'], profile='reference',
            reference_gguf=Path(self.exported['fp16']['native']) / 'model.gguf',
            source=self.teacher['source'], teacher_dir=self.teacher['teacher_dir'],
            scorer_dir=self.teacher['scorer_dir'], minutes=120)
        self.summarize()

    def run(self):
        active = 'setup'
        self.hydration_complete = False
        with self.workflow:
            try:
                self.hydrate_verified()
                self.hydration_complete = True
                if self.requested_complete():
                    return
                self.active_stage = 'verify-historical-results'
                self.import_history()
                self.active_stage = 'build-runtime'
                self.build_runtime()
                self.active_stage = 'cached-parity'
                self.parity_first()
                if self.args.stage == 'parity':
                    return
                self.chat = self.worker('freeze-complete-teacher-turns', 'freeze_chat',
                                         diagnostics=self.diagnostics['diagnostics'], minutes=90)
                for candidate in candidates():
                    active = candidate.name
                    self.train_candidate(candidate)
                    for profile in self.config['v2']['inference_profiles']:
                        self.fresh_measure(candidate.name, profile)
                for label in self.config['v2']['refresh_controls']:
                    active = label
                    if label == 'bonsai2_4b':
                        self.fresh_bonsai()
                    else:
                        for profile in self.config['v2']['inference_profiles']:
                            self.fresh_measure(label, profile)
            except BaseException as error:
                self.failures[getattr(self, 'active_stage', active)] = f'{type(error).__name__}: {error}'
                raise
            finally:
                self.finalize_resume()

    def requested_complete(self):
        if (not self.historical_rows
                or self.gates.get('retained-cached-parity', {}).get('passed') is not True):
            return False
        if self.args.stage == 'parity':
            return True
        rows = self.expected_rows()
        if self.failures:
            return False
        return bool(rows) and all(self.completed_row(label, profile) for label, profile in rows)

    def completed_row(self, label, profile):
        key = label + '@' + profile
        value = self.measured.get(key, {})
        if (not complete_measurement(value) or self.data is None
                or value.get('data_fingerprint') != self.data['fingerprint']):
            return False
        if label == 'bonsai2_4b':
            return True
        return (self.native_parity.get(key, {}).get('passed') is True
                and self.gates.get('preflight@' + profile, {}).get('passed') is True)

    def finalize_resume(self):
        previous = self.root / 'comparison.json'
        if not self.hydration_complete and previous.exists():
            write_json(self.root / 'resume-error.json', {
                'identity': self.identity, 'status': 'failed', 'errors': self.failures,
                'preserved_comparison': str(previous)})
            return
        self.summarize()

    def expected_rows(self):
        labels = [candidate.name for candidate in candidates()]
        labels += [x for x in self.config['v2']['refresh_controls'] if x != 'bonsai2_4b']
        result = [(label, profile) for label in labels for profile in self.config['v2']['inference_profiles']]
        if 'bonsai2_4b' in self.config['v2']['refresh_controls']:
            result.append(('bonsai2_4b', 'prism'))
        return result

    def fresh_row(self, label, profile):
        key = label + '@' + profile
        checkpoint = self.teacher if label == 'fp16' else self.prepared.get(label)
        exported = self.exported.get(label, {})
        if label == 'bonsai2_4b':
            exported = {'artifact_bytes': self.bonsai_export.get('native_payload_bytes'),
                        'language_payload_bytes': self.bonsai_export.get('language_payload_bytes')}
        row = comparison_row(key, checkpoint, exported, self.measured.get(key, {}))
        row.update(model=label, profile=profile, phase='fresh', origin={'root': str(self.root), 'identity': self.identity},
                   backend=self.measured.get(key, {}).get('backend'), parity=self.gates.get(key),
                   runtime=self.measured.get(key, {}).get('runtime'),
                   inference=self.measured.get(key, {}).get('inference'),
                   language_payload_bytes=exported.get('language_payload_bytes'),
                   behaviour=(checkpoint or {}).get('behaviour'),
                   training_metrics=(checkpoint or {}).get('metrics'), error=self.row_error(label, profile),
                   precision_scope=(checkpoint or {}).get('precision_scope'))
        if row['error']:
            row['status'] = 'failed'
        elif self.measured.get(key, {}).get('status') == 'failed':
            row.update(status='failed', error=self.measured[key].get('error'))
        elif not complete_measurement(self.measured.get(key, {})):
            row['status'] = 'pending'
        return row

    def row_error(self, label, profile):
        keys = (label + '@' + profile, 'measure-' + label + '@' + profile,
                'cache-native-' + label + '@' + profile, 'preflight-' + profile, 'prepare-' + label, label)
        return next((self.failures[key] for key in keys if key in self.failures), None)

    def summarize(self):
        rows = [self.fresh_row(label, profile) for label, profile in self.expected_rows()]
        result = {'protocol': PROTOCOL, 'identity': self.identity, 'plan': plan(self.config),
                  'stage': self.args.stage,
                  'historical_rows': self.historical_rows, 'rows': rows, 'gates': self.gates,
                  'historical_import': self.historical_receipt, 'errors': self.failures,
                  'error': next(iter(self.failures.values()), None),
                  'parity_complete': self.gates.get('retained-cached-parity', {}).get('passed') is True,
                  'complete': bool(rows) and all(row['status'] == 'measured' for row in rows) and not self.failures,
                  'scope': 'Historical controls retain their original source/runtime/data identities. Fresh rows reuse frozen test inputs with separately gated runtime profiles. New private teacher data never uses public evaluation labels.'}
        write_json(self.root / 'comparison.json', result)
