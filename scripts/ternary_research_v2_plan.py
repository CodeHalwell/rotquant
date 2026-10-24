"""Versioned cached-decoding, complete-turn QAT and runtime comparison contract."""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.ternary_research_plan import LABELS
from scripts.ternary_research_plan import validate_config as validate_v1
from scripts.w4_research_plan import ROOT, source_files

PROTOCOL = 'qwen35-ternary-research-v2'
DEFAULT_CONFIG = ROOT / 'configs/qwen35_4b_ternary_research_v2_cuda.yaml'


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    parent: str | None = None


def candidates():
    return [Candidate('ternary_haware_ptq', 'haware'),
            Candidate('ternary_chat_blockqat', 'block'),
            Candidate('ternary_global_qat', 'global', 'ternary_chat_blockqat'),
            Candidate('ternary_chat_recovered', 'recovery', 'ternary_global_qat')]


def bounded_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')


def validate_profiles(profiles, *, require_reference=True):
    from scripts.ternary_inference import InferenceProfile
    if not isinstance(profiles, dict) or require_reference and 'reference' not in profiles:
        raise ValueError('an explicit reference runtime is required')
    for name, profile in profiles.items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name):
            raise ValueError('runtime profile names must be simple identifiers')
        parsed = InferenceProfile.from_mapping(profile)
        if name == 'reference' and parsed != InferenceProfile():
            raise ValueError('reference must preserve the baseline kernel and capabilities')


def validate_chat_settings(chat):
    lengths = chat['context_lengths']
    if any(type(x) is not int for x in lengths) or tuple(lengths) != (512, 1024, 2048):
        raise ValueError('complete-turn data needs the integer 512/1024/2048 context mixture')
    bounded_integer(chat['max_new_tokens'], 'teacher generation cap', 1, 2048)
    for key in ('text_train_documents', 'text_validation_documents'):
        bounded_integer(chat[key], key, 1, 4096)


def validate_global_settings(settings):
    from rotquant.ternary_global_config import TernaryGlobalConfig
    bindings = {'source_id': 'validation', 'data_id': 'validation', 'behaviour_data_id': 'validation',
                'checkpoint_dir': '/validation-only', 'required_ids': (0,)}
    TernaryGlobalConfig(**{**settings, **bindings})


def validate_config(config):
    validate_v1(config)
    study = config['v2']
    bounded_integer(study['cached_parity']['max_steps'], 'cached parity steps', 1, 64)
    validate_profiles(study['inference_profiles'])
    optional = study.get('optional_inference_profiles', {})
    validate_profiles(optional, require_reference=False)
    if any(optional[name] != study['inference_profiles'][name]
           for name in set(optional) & study['inference_profiles'].keys()):
        raise ValueError('enabled and optional inference profile definitions disagree')
    validate_chat_settings(study['chat'])
    validate_fitting_scope(config)
    validate_global_settings(study['global'])
    required = {'fp16', 'w4_v6', 'w5_v6', 'w6_v6', 'bonsai2_4b'}
    controls = study['refresh_controls']
    if not isinstance(controls, list) or len(controls) != len(required) or set(controls) != required:
        raise ValueError('fresh controls must include FP16, W4, W5/W6, W6 and Bonsai exactly once')
    return config


def validate_fitting_scope(config):
    if config['ternary']['scope'] != 'mlp':
        raise ValueError('v2 currently requires the validated MLP/W4 remainder precision scope')
    for split in ('train', 'validation'):
        if config['v2']['chat'][f'text_{split}_documents'] > config['research'][f'recovery_{split}_documents']:
            raise ValueError('text mixture exceeds the retained independent fitting partition')


def bind_run(root, config, runtime, historical):
    from scripts.native_hashing import digest
    sources = source_files()
    sources.update({str(path.relative_to(ROOT)): digest(path)
                    for path in (ROOT / 'integrations').rglob('*.cuh')})
    identity = {'protocol': PROTOCOL, 'config': config, 'source': sources,
                'runtime': runtime, 'historical_root': str(historical)}
    path = Path(root) / 'identity.json'
    if path.exists():
        import json
        if json.loads(path.read_text()) != identity:
            raise ValueError('source/config/runtime changed; choose a new v2 result root')
    write_json(path, identity)
    return fingerprint(identity)


def plan(config):
    return {'protocol': PROTOCOL, 'historical_labels': list(LABELS),
            'cached_gate_labels': [x for x in LABELS if x != 'bonsai2_4b'],
            'candidates': [asdict(x) for x in candidates()],
            'inference_profiles': copy.deepcopy(config['v2']['inference_profiles']),
            'optional_inference_profiles': copy.deepcopy(config['v2'].get('optional_inference_profiles', {})),
            'order': ['verify historical payload hashes', 'build and preflight current runtime',
                      'cached chat parity for retained models', 'freeze complete teacher turns',
                      'H-aware ternary PTQ and block QAT', 'assignment-changing global QAT',
                      'behaviour-gated complete-turn recovery', 'candidate/profile parity',
                      'fresh frozen-input quality, bytes and speed'],
            'result_policy': 'Historical rows stay unchanged; all fresh model/profile rows have new identities.'}
