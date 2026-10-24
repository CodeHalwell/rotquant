import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml

from scripts.ternary_research_v2_plan import DEFAULT_CONFIG, candidates, validate_config
from scripts.ternary_research_v2_study import V2Study


def config():
    return validate_config(yaml.safe_load(DEFAULT_CONFIG.read_text()))


class StudyProbe(V2Study):
    def __init__(self, tmp_path, *, stage='parity', fail=False):
        self.config = config()
        self.args = SimpleNamespace(stage=stage)
        self.root, self.identity = tmp_path, 'new-source'
        self.workflow, self.events, self.fail = nullcontext(), [], fail
        self.prepared, self.exported, self.measured, self.failures, self.gates = {}, {}, {}, {}, {}
        self.historical_rows, self.historical_receipt, self.teacher = [], None, None
        self.bonsai_export = {}

    def hydrate_verified(self):
        return None

    def import_history(self):
        self.events.append('verify-history')

    def build_runtime(self):
        self.events.append('build-current-runtime')

    def parity_first(self):
        self.events.append('cached-parity')
        if self.fail:
            raise RuntimeError('cache divergence')
        self.gates['retained-cached-parity'] = {'passed': True}
        self.diagnostics = {'diagnostics': 'frozen.json'}

    def worker(self, name, phase, **options):
        self.events.append(phase)
        assert self.gates['retained-cached-parity']['passed']
        return {'chat': 'chat.json', 'calls': 'calls.pt'}

    def train_candidate(self, candidate):
        self.events.append('train:' + candidate.name)

    def fresh_measure(self, label, profile):
        self.events.append('measure:' + label + '@' + profile)

    def fresh_bonsai(self):
        self.events.append('measure:bonsai')


def test_parity_is_first_and_default_stage_never_trains(tmp_path):
    probe = StudyProbe(tmp_path)
    probe.run()
    assert probe.events == ['verify-history', 'build-current-runtime', 'cached-parity']
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert report['parity_complete'] and not report['complete']
    assert all(row['status'] == 'pending' and row['quality'] is None for row in report['rows'])


def test_cache_failure_blocks_teacher_capture_training_and_measurement(tmp_path):
    probe = StudyProbe(tmp_path, stage='full', fail=True)
    with pytest.raises(RuntimeError, match='cache divergence'):
        probe.run()
    assert probe.events == ['verify-history', 'build-current-runtime', 'cached-parity']
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert not report['complete'] and not report['parity_complete']
    assert 'cache divergence' in report['errors']['cached-parity']


def test_full_stage_runs_declared_candidates_and_fresh_controls_after_gate(tmp_path):
    probe = StudyProbe(tmp_path, stage='full')
    probe.run()
    assert probe.events[:4] == ['verify-history', 'build-current-runtime', 'cached-parity', 'freeze_chat']
    assert [x for x in probe.events if x.startswith('train:')] == [
        'train:' + candidate.name for candidate in candidates()]
    assert 'measure:fp16@reference' in probe.events
    assert 'measure:w4_v6@decode4' in probe.events
    assert 'measure:w5_v6@ternary-g128' in probe.events
    assert 'measure:w6_v6@reference' in probe.events
    assert 'measure:bonsai' in probe.events


def test_profiles_cannot_claim_unsupported_arithmetic_or_change_identity(tmp_path):
    settings = config()
    changed = copy.deepcopy(settings)
    changed['v2']['inference_profiles']['bad'] = {'kernel': 'reference', 'arithmetic': 'invented'}
    with pytest.raises(ValueError):
        validate_config(changed)
    changed = copy.deepcopy(settings)
    changed['v2']['chat']['context_lengths'] = [512]
    with pytest.raises(ValueError, match='context mixture'):
        validate_config(changed)


def test_profile_preflight_builds_w2_v6_graph_and_requires_device_dispatch(tmp_path):
    probe = StudyProbe(tmp_path)
    library = tmp_path / 'runtime.so'
    library.write_bytes(b'current-runtime')
    probe.runtime = {'source': '/bound/runtime/source', 'library': str(library)}
    probe.calls = []
    stage = SimpleNamespace(directory=tmp_path / 'preflight')
    stage.directory.mkdir()
    def command(stage, script, *arguments):
        probe.calls.append((script, list(map(str, arguments))))
        if script == 'check_ternary_inference.py':
            (stage.directory / 'report.json').write_text(json.dumps({
                'passed': True, 'profile': {'kernel': 'decode4'},
                'operators': {'dispatch': {'w2_g128_s16': {'tiled4': 1}}}}))
    probe.command = command
    class FakeWorkflow:
        def stage(self, name, action, **options):
            assert name == 'preflight-decode4'
            assert options['signature']['runtime'] == probe.runtime
            return action(stage)[0]
    probe.workflow = FakeWorkflow()
    result = probe.preflight_profile('decode4')
    assert result['passed'] and probe.gates['preflight@decode4']['passed']
    assert probe.calls[0][0] == 'make_rq3_model_fixture.py'
    assert probe.calls[0][1][-4:] == ['--backbone-bits', '2', '--vocabulary-bits', '6']
    assert probe.calls[1][0] == 'check_ternary_inference.py'
    assert '--kernel' in probe.calls[1][1]


def test_optional_profile_failure_isolated_and_mandatory_reference_still_stops(tmp_path):
    probe = StudyProbe(tmp_path)
    probe.fresh_measure = lambda label, profile: V2Study.fresh_measure(probe, label, profile)
    def failing_gate(label, profile):
        raise RuntimeError('profile-specific compile failure')
    probe.gate = failing_gate
    probe.native_parity = {}
    probe.fresh_measure('ternary_global_qat', 'decode4')
    assert probe.row_error('ternary_global_qat', 'decode4')
    assert probe.row_error('ternary_global_qat', 'reference') is None
    with pytest.raises(RuntimeError, match='profile-specific'):
        probe.fresh_measure('ternary_global_qat', 'reference')


def test_pre_hydration_expired_cap_never_overwrites_saved_comparison(tmp_path):
    probe = StudyProbe(tmp_path, stage='full')
    previous = {'historical_rows': [{'label': 'fp16', 'measured': 123}],
                'rows': [{'label': 'ternary@decode4', 'quality': {'ppl': 18.2}}]}
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(previous))
    before = path.read_bytes()
    def expired():
        raise TimeoutError('persisted execution allowance exhausted')
    probe.hydrate_verified = expired
    with pytest.raises(TimeoutError, match='exhausted'):
        probe.run()
    assert path.read_bytes() == before
    failure = json.loads((tmp_path / 'resume-error.json').read_text())
    assert failure['preserved_comparison'] == str(path)
    assert probe.events == []


def test_completed_full_resume_verifies_then_returns_without_model_or_build_work(tmp_path):
    probe = StudyProbe(tmp_path, stage='full')
    probe.native_parity = {}
    probe.data = {'fingerprint': 'same-data'}
    def verified():
        probe.events.append('verify-saved-receipts')
        probe.historical_rows = [{'label': 'fp16', 'status': 'measured'}]
        probe.gates['retained-cached-parity'] = {'passed': True}
        for label, profile in probe.expected_rows():
            key = label + '@' + profile
            probe.measured[key] = {
                'status': 'complete', 'data_fingerprint': 'same-data',
                'quality': {part: {'summary': {'ppl': 18.2}} for part in ('test', 'long_test')},
                'tasks': {'rows': [{'id': 'one'}], 'summary': {'task': {'items': 1}}},
                'speed': {'completed': True, 'summary': {'512': {'decode_tokens_per_second': 25.}}}}
            probe.native_parity[key] = {'passed': True}
            probe.gates['preflight@' + profile] = {'passed': True}
    probe.hydrate_verified = verified
    probe.run()
    assert probe.events == ['verify-saved-receipts']
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert report['complete'] and all(row['status'] == 'measured' for row in report['rows'])


def test_v2_rejects_unsupported_precision_scope_and_missing_text_mixture():
    settings = config()
    settings['ternary']['scope'] = 'all'
    with pytest.raises(ValueError, match='precision scope'):
        validate_config(settings)
    settings = config()
    settings['v2']['chat']['text_train_documents'] = 0
    with pytest.raises(ValueError, match='text_train_documents'):
        validate_config(settings)


@pytest.mark.parametrize('changed', [False, True])
def test_fresh_bonsai_reacquires_pinned_weights_outside_the_old_runtime(tmp_path, changed):
    probe = StudyProbe(tmp_path)
    probe.work = tmp_path / 'current-runtime'
    probe.config['bonsai'] = {'packing': 'PTQ1_0'}
    probe.bonsai_assets = tmp_path / 'missing-historical-runtime'
    probe.bonsai_artifacts = {'model.gguf': {'bytes': 10, 'sha256': 'pinned'}}
    def command(stage, script, phase, *args):
        assert script == 'ternary_research_bonsai.py' and phase == 'acquire'
        options = dict(zip(args[::2], args[1::2]))
        destination = options['--output-dir']
        assert destination.is_relative_to(probe.work)
        receipt = {'model': str(destination / 'model.gguf'), 'vision': str(destination / 'vision.gguf'),
                   'artifact_files': {} if changed else probe.bonsai_artifacts}
        options['--report'].write_text(json.dumps(receipt))
    probe.command = command
    def stage(name, action, **kwargs):
        directory = tmp_path / 'acquire-stage'
        directory.mkdir()
        return action(SimpleNamespace(directory=directory))[0]
    probe.workflow = SimpleNamespace(stage=stage)
    if changed:
        with pytest.raises(ValueError, match='historical pinned model'):
            probe.acquire_bonsai()
    else:
        probe.acquire_bonsai()
        assert probe.bonsai_assets.is_relative_to(probe.work)


def test_native_gate_request_changes_when_only_the_cuda_dependency_is_rebuilt(tmp_path):
    library, cuda = tmp_path / 'runtime.so', tmp_path / 'libggml-cuda.so'
    library.write_bytes(b'unchanged-bridge')
    cuda.write_bytes(b'first-cuda-binary')
    probe = StudyProbe(tmp_path)
    probe.data = {'data_path': 'frozen.json'}
    probe.config_path = tmp_path / 'config.json'
    before = probe.worker_options(library=library)['runtime_id']
    cuda.write_bytes(b'rebuilt-cuda-binary')
    after = probe.worker_options(library=library)['runtime_id']
    assert before != after


def test_fresh_measure_rechecks_parity_even_with_a_hydrated_old_gate(tmp_path):
    probe = StudyProbe(tmp_path)
    probe.native_parity = {'ternary_global_qat@reference': {'passed': True, 'parity': 'stale.json'}}
    probe.fresh_measure = lambda label, profile: V2Study.fresh_measure(probe, label, profile)
    def reject_stale(label, profile):
        raise ValueError('rebuilt runtime needs a fresh bound parity gate')
    probe.gate = reject_stale
    with pytest.raises(ValueError, match='fresh bound parity'):
        probe.fresh_measure('ternary_global_qat', 'reference')
