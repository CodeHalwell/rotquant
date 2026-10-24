"""Complete-turn block QAT, global hard-code QAT and guarded recovery stages."""
from __future__ import annotations

import copy

import torch

from rotquant.checkpoint import load_packed_model, verify_checkpoint
from rotquant.ternary_numerics import stable_qwen_fallbacks
from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.ternary_behaviour import build_behaviour_evaluator, selection_identity
from scripts.ternary_chat_data import chat_batches, validate_chat_manifest
from scripts.ternary_research_model import prepare_mlp_remainder
from scripts.ternary_research_v2_plan import PROTOCOL
from scripts.w4_research_model import export_packed, load_source, quantize_vocabulary_in_place


def fitting_batches(manifest, data, config, split):
    chat = chat_batches(manifest, split)
    count = config['v2']['chat'][f'text_{split}_documents']
    documents = data['documents'][f'recovery_{split}'][:count]
    if len(documents) != count:
        raise ValueError('not enough immutable text fitting documents')
    text = []
    for item in documents:
        ids = torch.tensor([item['input_ids']], dtype=torch.long)
        text.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                     'response_mask': torch.ones_like(ids, dtype=torch.bool)})
    return chat + text


def fitting_identity(manifest, data, config):
    return fingerprint({'chat': manifest['fingerprint'], 'text_data': data['fingerprint'],
                        'text_train_documents': config['v2']['chat']['text_train_documents'],
                        'text_validation_documents': config['v2']['chat']['text_validation_documents'],
                        'protocol': 'complete-chat-and-ordinary-text-mixture-v1'})


def teacher_calls(model, manifest, config, data):
    from rotquant.block_train import collect_teacher_calls
    settings = config['v2']['global']
    result = {}
    for split in ('train', 'validation'):
        batches = fitting_batches(manifest, data, config, split)
        result[split] = collect_teacher_calls(
            model, batches, config['device'], max_batches=len(batches),
            topk=settings['topk'], temperature=settings['temperature'],
            required_ids=manifest['required_teacher_ids'], include_response_ids=True)
    result['audit'] = collect_teacher_calls(model, chat_batches(manifest, 'validation')[:1],
                                            config['device'], max_batches=1, topk=0)
    result['data_id'] = fitting_identity(manifest, data, config)
    return result


def train_blocks(model, candidate, worker, manifest):
    from rotquant.ternary_train import TernaryTrainConfig, train_and_patch_ternary
    cfg = copy.deepcopy(worker.config['ternary'])
    data = {split: [{k: v for k, v in batch.items() if k != 'response_mask'} for batch in
                     fitting_batches(manifest, worker.data, worker.config, split)]
            for split in ('train', 'validation')}
    cfg.update(source_id=worker.args.identity, data_id=fitting_identity(manifest, worker.data, worker.config),
               checkpoint_dir=str(worker.args.root / 'training-state' / candidate.name),
               training_artifact_dir=str(worker.output / 'training-artifacts'),
               retain_latents=True, activation_aware=True,
               train_batches=len(data['train']), validation_batches=len(data['validation']))
    if candidate.method == 'haware':
        cfg['steps'] = 0
    stats = {}
    with stable_qwen_fallbacks(model) as numerics:
        train_and_patch_ternary(model, data['train'], data['validation'], TernaryTrainConfig(**cfg),
                               device=worker.config['device'], stats_out=stats)
    return {**stats, 'numerics': numerics, 'objective': 'deployed-prefix block reconstruction'}


def latent_receipts(prepared):
    metrics = prepared.get('metrics', {})
    if 'ternary_global' in metrics:
        return [metrics['ternary_global']['selected_latents']]
    blocks = metrics.get('ternary_train', {}).get('blocks', {})
    return [row['selected_latents'] for row in blocks.values() if 'selected_latents' in row]


def train_global(model, candidate, worker, manifest):
    from rotquant.ternary_global import TernaryGlobalConfig, train_global_ternary
    from scripts.ternary_research_continuation import read
    parent = read(worker.args.parent_report)
    calls = torch.load(worker.args.calls, map_location='cpu', weights_only=False)
    data_id = fitting_identity(manifest, worker.data, worker.config)
    if calls['chat_fingerprint'] != manifest['fingerprint'] or calls['data_id'] != data_id:
        raise ValueError('teacher targets differ from frozen complete turns')
    settings = {**worker.config['v2']['global'], 'source_id': worker.args.identity,
                'data_id': data_id, 'behaviour_data_id': selection_identity(manifest),
                'checkpoint_dir': str(worker.args.root / 'training-state' / candidate.name),
                'required_ids': tuple(manifest['required_teacher_ids']),
                'learn_assignments': candidate.method != 'recovery'}
    stats = {}
    with stable_qwen_fallbacks(model) as numerics:
        train_global_ternary(model, calls['train'], calls['validation'], TernaryGlobalConfig(**settings),
                             behavioural_evaluator=build_behaviour_evaluator(
                                 manifest, worker.tokenizer, worker.config['device']),
                             training_artifact_receipts=latent_receipts(parent),
                             dense_audit_calls=calls['audit'],
                             device=worker.config['device'], stats_out=stats)
    return {**stats, 'numerics': numerics, 'teacher_objective': 'full-mass top-k union response/EOS/format plus tail'}


def prepare_candidate(worker, candidate, manifest):
    from transformers import AutoProcessor, AutoTokenizer
    validate_chat_manifest(manifest)
    config = worker.config
    if candidate.parent:
        verify_checkpoint(worker.args.parent)
        model = load_packed_model(worker.args.parent, device=config['device'], dtype=torch.float16, fallback=True)
        tokenizer, processor = (AutoTokenizer.from_pretrained(worker.args.source),
                                AutoProcessor.from_pretrained(worker.args.source))
        loader = config['model_loader']
    else:
        model, tokenizer, processor, loader = load_source(config)
    worker.tokenizer = tokenizer
    if candidate.parent:
        metrics = train_global(model, candidate, worker, manifest)
    else:
        remainder = prepare_mlp_remainder(model, tokenizer, config, worker.data, worker.args.root, worker.args.identity)
        metrics = {**train_blocks(model, candidate, worker, manifest), 'remaining_w4': remainder}
        quantize_vocabulary_in_place(model, config, worker.args.root)
    behaviour = build_behaviour_evaluator(manifest, tokenizer, config['device'])(model)
    selection = worker.selection(model, tokenizer, config['device'], worker.output)
    exported = export_packed(model, tokenizer, processor, loader, config, worker.output,
                             worker.args.identity, protocol=PROTOCOL)
    result = {'candidate': candidate.__dict__, 'selection': selection, 'behaviour': behaviour,
              'artifact_bytes': exported['artifact_bytes'], 'export': exported, 'metrics': metrics,
              'checkpoint': str(worker.output / 'checkpoint'), 'probes': str(worker.output / 'probes.npz'),
              'identity': worker.args.identity, 'chat_data': manifest['fingerprint'],
              'precision_scope': 'ternary MLP/group128/FP16 scales; W4 remainder; W6 tied vocabulary'}
    write_json(worker.output / 'prepared.json', result)
    return result
