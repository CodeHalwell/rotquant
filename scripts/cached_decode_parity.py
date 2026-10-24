import hashlib
from pathlib import Path

import numpy as np

from scripts.cached_decode_models import DecodeTrace, greedy_trace
from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.w4_research_worker import parity_row

PROTOCOL = 'ternary-cached-chat-parity-v1'
MODES = ('cached', 'chunked', 'recompute')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_logits(logits):
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('decode parity requires finite full-vocabulary logits')
    return values


def stop_metrics(logits, stop_ids):
    values = checked_logits(logits).astype(np.float64)
    if not stop_ids or min(stop_ids) < 0 or max(stop_ids) >= len(values):
        raise ValueError('stop tokens must be valid vocabulary IDs')
    logz = values.max() + np.log(np.exp(values-values.max()).sum())
    return {'argmax': int(values.argmax()),
            'top2_margin': float(values.max()-np.partition(values, -2)[-2]),
            'stop_tokens': {str(token): {'rank': int(1+(values > values[token]).sum()),
                'probability': float(np.exp(values[token]-logz))} for token in stop_ids}}


def compare_step(actual, reference, stop_ids, step):
    actual, reference = checked_logits(actual), checked_logits(reference)
    comparison = parity_row(actual[None], reference[None])
    return {'step': step, **comparison, 'actual': stop_metrics(actual, stop_ids),
            'reference': stop_metrics(reference, stop_ids)}


def validate_cases(cases, max_steps=64):
    if not cases or len(cases) > 32 or not 1 <= max_steps <= 64:
        raise ValueError('bounded nonempty chat parity cases required')
    names = [case['id'] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError('duplicate cached-decode case IDs')
    for case in cases:
        validate_case(case, max_steps)
    return cases


def validate_case(case, max_steps):
    if Path(case['id']).name != case['id'] or case['id'].startswith('.'):
        raise ValueError('invalid cached-decode case ID')
    for key in ('prompt_ids', 'teacher_ids'):
        ids = case[key]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('chat parity IDs must be nonempty nonnegative integers')
    if len(case['prompt_ids']) > 3072 or len(case['teacher_ids']) > max_steps:
        raise ValueError('chat parity token bound exceeded')


def trace_rows(model, prompt, continuation, mode, stop_ids, reference_paths, directory):
    trace = DecodeTrace(model, prompt, mode)
    rows = []
    for step in range(len(continuation)+1):
        if step:
            trace.advance(continuation[step-1])
        reference = np.load(reference_paths[step], allow_pickle=False)
        row = compare_step(trace.logits, reference, stop_ids, step)
        if not row['passed'] and not any(not item['passed'] for item in rows):
            path = Path(directory) / f'{mode}-first-mismatch.npy'
            np.save(path, checked_logits(trace.logits), allow_pickle=False)
            row['actual_logits_path'] = str(path)
        rows.append(row)
    return {'mode': mode, 'passed': all(row['passed'] for row in rows), 'rows': rows,
            'first_failed_step': next((r['step'] for r in rows if not r['passed']), None),
            'first_argmax_difference': next((r['step'] for r in rows
                if r['actual']['argmax'] != r['reference']['argmax']), None)}


def capture_history(factory, case, continuation, stop_ids, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = factory('recompute')
    trace = DecodeTrace(reference, case['prompt_ids'], 'recompute')
    paths = []
    for step in range(len(continuation)+1):
        if step:
            trace.advance(continuation[step-1])
        path = directory / f'{step:03d}.npy'
        np.save(path, checked_logits(trace.logits), allow_pickle=False)
        paths.append(str(path))
    checks = [trace_rows(factory(mode), case['prompt_ids'], continuation, mode,
                         stop_ids, paths, directory) for mode in ('cached', 'chunked')]
    return {'continuation_ids': continuation, 'reference_paths': paths,
            'reference_sha256': [file_hash(path) for path in paths], 'hf_checks': checks}


def capture_references(factory, cases, stop_ids, directory, binding, max_steps=16):
    cases = validate_cases(cases, max_steps)
    root, records = Path(directory), []
    root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        candidate = greedy_trace(factory('cached'), case['prompt_ids'], set(stop_ids), max_steps)
        histories = {'teacher': case['teacher_ids'], 'candidate': candidate.tokens}
        record = {'id': case['id'], 'prompt_ids': case['prompt_ids'], 'histories': {}}
        for name, continuation in histories.items():
            record['histories'][name] = capture_history(factory, case, continuation, stop_ids,
                                                        root / case['id'] / name)
        records.append(record)
    result = {'protocol': PROTOCOL, 'binding': binding, 'cases_fingerprint': fingerprint(cases),
              'source_cases': cases, 'max_steps': max_steps,
              'stop_ids': list(stop_ids), 'cases': records,
              'passed': all(check['passed'] for case in records for history in case['histories'].values()
                            for check in history['hf_checks'])}
    write_json(root / 'references.json', result)
    return result


def verify_references(record, binding):
    if record.get('protocol') != PROTOCOL or record.get('binding') != binding:
        raise ValueError('cached-decode references do not match checkpoint/data binding')
    if not record.get('passed') or not record.get('cases'):
        raise ValueError('packed PyTorch cached/no-cache parity did not pass')
    cases = validate_cases(record['source_cases'], record['max_steps'])
    if record['cases_fingerprint'] != fingerprint(cases) or len(cases) != len(record['cases']):
        raise ValueError('cached-decode prompt binding changed')
    expected = {case['id']: case for case in cases}
    for case in record['cases']:
        source = expected[case['id']]
        if case['prompt_ids'] != source['prompt_ids'] or set(case['histories']) != {'teacher', 'candidate'}:
            raise ValueError('cached-decode histories changed')
        if case['histories']['teacher']['continuation_ids'] != source['teacher_ids']:
            raise ValueError('cached-decode teacher trajectory changed')
        for history in case['histories'].values():
            verify_history(history)


def verify_history(history):
    paths, hashes = history['reference_paths'], history['reference_sha256']
    if len(paths) != len(hashes) or len(paths) != len(history['continuation_ids'])+1:
        raise ValueError('incomplete cached-decode reference trajectory')
    if any(Path(path).is_symlink() or file_hash(path) != expected for path, expected in zip(paths, hashes)):
        raise ValueError('cached-decode reference logits changed')


def compare_native(factory, references, binding, directory):
    verify_references(references, binding)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in references['cases']:
        for history_name, history in case['histories'].items():
            destination = directory / case['id'] / history_name
            destination.mkdir(parents=True, exist_ok=True)
            checks = [trace_rows(factory(mode), case['prompt_ids'], history['continuation_ids'],
                mode, references['stop_ids'], history['reference_paths'], destination) for mode in MODES]
            rows.append({'id': case['id'], 'history': history_name, 'checks': checks})
    result = {'protocol': PROTOCOL, 'binding': binding, 'cases_fingerprint': references['cases_fingerprint'],
              'passed': all(check['passed'] for row in rows for check in row['checks']), 'rows': rows,
              'boundary': 'Fixed identical histories; raw greedy mismatches retained, exact tie-set numerical conformance is separate from identical generation.'}
    write_json(directory / 'parity.json', result)
    return result
