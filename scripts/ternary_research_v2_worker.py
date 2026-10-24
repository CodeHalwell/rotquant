"""Isolated model processes for the cached-decoding and complete-turn v2 study."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.cached_decode_models import DecodeTrace, HFDecodeModel, greedy_trace
from scripts.cached_decode_parity import capture_references, compare_native
from scripts.native_gpu_workflow import artifacts, fingerprint, write_json
from scripts.ternary_inference import (
    DispatchReceipt,
    InferenceProfile,
    expected_model_dispatch,
    require_dispatch,
)
from scripts.ternary_research_continuation import read
from scripts.ternary_research_worker import TernaryWorker


class V2Worker(TernaryWorker):
    def profile(self):
        return InferenceProfile.from_mapping(self.config['v2']['inference_profiles'][self.args.profile])

    def diagnostics(self):
        from scripts.run_qwen35_fresh_eval import stop_ids_for_source
        from scripts.ternary_chat_data import prompt_identity, template_input_ids
        from scripts.w4_research_model import load_source
        model, tokenizer, _, _ = load_source(self.config)
        prompts = read(ROOT / self.config['v2']['cached_parity']['prompts'])['cases']
        cap = self.config['v2']['cached_parity']['max_steps']
        stops, cases = stop_ids_for_source(tokenizer), []
        for item in prompts:
            ids = template_input_ids(tokenizer, [{'role': 'user', 'content': item['prompt']}])
            teacher = greedy_trace(HFDecodeModel(model, self.config['device']), ids, set(stops), cap)
            cases.append({'id': item['id'], 'prompt_ids': ids, 'teacher_ids': teacher.tokens,
                          'teacher_complete': teacher.stopped, 'prompt_hash': prompt_identity(item)})
        result = {'protocol': 'cached-chat-source-diagnostics-v1', 'cases': cases,
                  'stop_ids': list(stops), 'source_revision': self.config['model_revision'],
                  'template': fingerprint(tokenizer.chat_template), 'max_steps': cap}
        result['fingerprint'] = fingerprint(result)
        path = self.output / 'diagnostics.json'
        write_json(path, result)
        return {'diagnostics': str(path), 'fingerprint': result['fingerprint']}

    def freeze_chat(self):
        from scripts.ternary_chat_data import (
            freeze_chat_dataset,
            load_prompt_jsonl,
            prompt_identity,
        )
        from scripts.ternary_research_v2_training import teacher_calls
        from scripts.w4_research_model import load_source
        model, tokenizer, _, _ = load_source(self.config)
        diagnostics, settings = read(self.args.diagnostics), self.config['v2']['chat']
        prompts = load_prompt_jsonl(settings['prompts']) if settings.get('prompts') else None
        forbidden = [prompt_identity(item) for item in self.data['public_items']]
        forbidden.extend(row['prompt_hash'] for row in diagnostics['cases'])
        manifest = freeze_chat_dataset(
            tokenizer, model, prompts, model_revision=self.config['model_revision'],
            tokenizer_revision=self.config['model_revision'], stop_ids=diagnostics['stop_ids'],
            max_new_tokens=settings['max_new_tokens'], context_lengths=settings['context_lengths'],
            forbidden_prompt_hashes=forbidden,
            forbidden_token_hashes=[fingerprint(x['input_ids']) for x in self.data['public_items']],
            destination=self.output / 'chat.json')
        calls = {**teacher_calls(model, manifest, self.config, self.data), 'chat_fingerprint': manifest['fingerprint']}
        torch.save(calls, self.output / 'teacher-calls.pt')
        return {'chat': str(self.output / 'chat.json'), 'calls': str(self.output / 'teacher-calls.pt'),
                'fingerprint': manifest['fingerprint'], 'token_counts': manifest['token_counts']}

    def cache_binding(self):
        return {'identity': self.args.identity, 'checkpoint': artifacts(
            [self.args.checkpoint / 'rotquant_config.json']) if self.args.checkpoint else artifacts(
                self.args.source.glob('*.safetensors')),
                'diagnostics': artifacts([self.args.diagnostics]), 'model': self.args.label,
                'arithmetic': self.profile().arithmetic}

    def cache_reference(self):
        from rotquant.checkpoint import load_packed_model, verify_checkpoint
        from scripts.w4_research_model import load_source
        if self.args.checkpoint:
            verify_checkpoint(self.args.checkpoint)
            model = load_packed_model(self.args.checkpoint, device=self.config['device'], dtype=torch.float16, fallback=True)
        else:
            model = load_source(self.config)[0]
        diagnostic = read(self.args.diagnostics)
        factory = lambda mode: HFDecodeModel(model, self.config['device'], cached=mode != 'recompute')
        from rotquant.ternary_w2a8 import grouped_w2a8_model
        oracle = grouped_w2a8_model(model) if self.profile().arithmetic == 'w2a8' else nullcontext()
        with oracle:
            record = capture_references(factory, diagnostic['cases'], diagnostic['stop_ids'], self.output,
                                        self.cache_binding(), max_steps=diagnostic['max_steps'])
        if not record['passed']:
            raise ValueError('packed PyTorch cached/full/chunk parity failed; training is blocked')
        return {'references': str(self.output / 'references.json'), 'passed': True,
                'binding': record['binding']}

    def cache_native(self):
        from scripts.check_rq3_model import runtime_identity
        from scripts.rq3_test_runtime import NativeTests
        references, profile = read(self.args.references), self.profile()
        if references.get('binding', {}).get('arithmetic') != profile.arithmetic:
            raise ValueError('cached references use a different numerical arithmetic lane')
        profile.apply()
        os.environ['ROTQUANT_REQUIRE_GPU'] = '1'
        runtime = NativeTests(self.args.library)
        with runtime.model(self.args.native / 'model.gguf', 'CUDA0', 4160) as model:
            result = compare_native(lambda mode: model, references, references['binding'], self.output)
            self.compact_parity(model, references, result)
        result.update(inference=profile.identity(), runtime=artifacts([self.args.library]),
                      runtime_files=runtime_identity(self.args.library),
                      native_artifacts=artifacts(self.args.native.glob('*.gguf')))
        write_json(self.output / 'parity.json', result)
        if not result['passed']:
            raise ValueError('native cached chat parity failed; candidate/profile is blocked')
        return {'passed': True, 'parity': str(self.output / 'parity.json'), 'inference': profile.identity()}

    def compact_parity(self, model, references, result):
        wrapper = self.profile().wrap(model)
        checks = []
        for case in references['cases']:
            for history_name, history in case['histories'].items():
                checks.append(self.capability_trace(model, wrapper, case, history_name,
                                                    history['continuation_ids']))
        result['capability_checks'] = checks
        result['passed'] = result['passed'] and all(row['passed'] for row in checks)

    def capability_trace(self, model, wrapper, case, history_name, continuation):
        raw = DecodeTrace(model, case['prompt_ids'], 'chunked')
        reference = [torch.as_tensor(raw.logits).clone()]
        for token in continuation:
            reference.append(torch.as_tensor(raw.advance(token)).clone())
        selected = DecodeTrace(wrapper, case['prompt_ids'], 'cached')
        rows = []
        for step, expected in enumerate(reference):
            if step:
                selected.advance(continuation[step - 1])
            rows.append({'step': step, 'graph_logits_identical': bool(torch.equal(
                expected, torch.as_tensor(selected.logits)))})
        if self.profile().compact_greedy:
            tokens = self.compact_trace_tokens(wrapper, case['prompt_ids'], continuation)
            for step, (token, expected) in enumerate(zip(tokens, reference)):
                rows[step]['compact_token_identical'] = token == int(expected.argmax())
        passed = [all(value for key, value in row.items() if key != 'step') for row in rows]
        return {'id': case['id'], 'history': history_name, 'passed': all(passed), 'rows': rows,
                'first_failed_step': next((i for i, value in enumerate(passed) if not value), None)}

    @staticmethod
    def compact_trace_tokens(wrapper, prompt, continuation):
        yield wrapper.greedy(prompt, reset=True)
        for token in continuation:
            yield wrapper.greedy([token], reset=False)

    def prepare_v2(self):
        from scripts.ternary_research_v2_plan import Candidate
        from scripts.ternary_research_v2_training import prepare_candidate
        return prepare_candidate(self, Candidate(**read(self.args.arm)), read(self.args.chat))

    def native(self):
        from transformers import AutoTokenizer

        from scripts import public_task_suite
        from scripts.check_rq3_model import runtime_identity
        from scripts.rq3_test_runtime import NativeTests
        from scripts.run_qwen35_fresh_eval import stop_ids_for_source
        from scripts.w4_research_measure import NativeBackend
        profile = self.profile()
        profile.apply()
        os.environ['ROTQUANT_REQUIRE_GPU'] = '1'
        parity = read(self.args.parity)
        if (parity.get('passed') is not True or parity.get('inference') != profile.identity()
                or parity.get('binding', {}).get('model') != self.args.label):
            raise ValueError('fresh measurement requires this profile cached-parity receipt')
        if parity['native_artifacts'] != artifacts(self.args.native.glob('*.gguf')):
            raise ValueError('native payload changed after cached parity')
        if parity.get('runtime') != artifacts([self.args.library]):
            raise ValueError('native runtime changed after cached parity')
        if parity.get('runtime_files') != runtime_identity(self.args.library):
            raise ValueError('native runtime dependencies changed after cached parity')
        tokenizer = AutoTokenizer.from_pretrained(self.args.source)
        scorer = public_task_suite.IFEvalScorer(self.args.scorer_dir)
        dispatch = DispatchReceipt(self.args.library)
        dispatch.snapshot(reset=True)
        runtime = NativeTests(self.args.library)
        with runtime.model(self.args.native / 'model.gguf', 'CUDA0', 4160) as raw:
            model = profile.wrap(raw)
            backend = NativeBackend(model, stop_ids_for_source(tokenizer))
            result = self.native_results(model, backend, tokenizer, scorer)
            if bool(raw.custom_ops) != (self.args.label != 'fp16'):
                raise ValueError('native dispatch does not match the expected model format')
        counters = dispatch.snapshot()
        require_dispatch(counters, expected_model_dispatch(profile, self.args.label))
        return {**result, 'label': self.args.label, 'data_fingerprint': self.data['fingerprint'],
                'backend': f'CUDA0/{self.args.profile}/chunk16', 'inference': profile.identity(),
                'dispatch': counters, 'runtime': runtime_identity(self.args.library),
                'native_artifacts': artifacts(self.args.native.glob('*.gguf')), 'parity': artifacts([self.args.parity])}

    def native_results(self, model, backend, tokenizer, scorer):
        from scripts.w4_research_worker import Worker
        bonsai = self.args.phase == 'bonsai'
        label = 'bonsai2_4b' if bonsai else self.args.label
        profile = 'prism' if bonsai else self.args.profile
        backend_name = 'CUDA0/Prism/chunk16' if bonsai else f'CUDA0/{profile}/chunk16'
        write_json(self.output / 'measurement-context.json', {
            'protocol': 'ternary-measurement-context-v2', 'identity': self.args.identity,
            'data_fingerprint': self.data['fingerprint'], 'label': label,
            'profile': profile, 'backend': backend_name})
        return Worker.native_results(self, model, backend, tokenizer, scorer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('diagnostics', 'freeze_chat', 'cache_reference', 'cache_native',
                                         'prepare_v2', 'reload', 'export', 'native', 'bonsai'))
    for name in ('config', 'root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--identity', required=True)
    for name in ('data', 'arm', 'teacher-dir', 'checkpoint', 'probes', 'llama-dir', 'source', 'quant-export',
                 'library', 'native', 'scorer-dir', 'parent', 'assets', 'reference-gguf', 'diagnostics',
                 'references', 'parity', 'chat', 'calls', 'parent-report'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--label')
    parser.add_argument('--profile', default='reference')
    parser.add_argument('--runtime-id')
    args = parser.parse_args()
    try:
        V2Worker(args).run()
    except Exception as error:
        write_json(args.output / 'report.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
