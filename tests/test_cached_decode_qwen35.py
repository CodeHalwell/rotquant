import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
from rotquant.ternary_train import TernaryTrainConfig, train_and_patch_ternary
from scripts.cached_decode_models import HFDecodeModel
from scripts.cached_decode_parity import capture_references


def test_real_hybrid_qwen_cache_and_chunked_history_after_ternary_reload(tmp_path):
    torch.manual_seed(731)
    torch.set_num_threads(2)
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1, head_dim=128,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128,
        linear_value_head_dim=128, layer_types=['linear_attention', 'full_attention'],
        tie_word_embeddings=True, max_position_embeddings=256,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000000.,
                         'partial_rotary_factor': .5, 'mrope_section': [11, 11, 10]})
    model = Qwen3_5ForCausalLM(config).eval()
    train_and_patch_ternary(model, [{'input_ids': torch.arange(3, 19)[None]}],
        [{'input_ids': torch.arange(23, 39)[None]}],
        TernaryTrainConfig(steps=0, train_batches=1, validation_batches=1), device='cpu')
    save_packed_checkpoint(model, tmp_path / 'checkpoint')
    reloaded = load_packed_model(tmp_path / 'checkpoint', device='cpu', dtype=torch.float32, fallback=True)
    factory = lambda mode: HFDecodeModel(reloaded, 'cpu', cached=mode != 'recompute')
    report = capture_references(factory, [{'id': 'hybrid', 'prompt_ids': list(range(3, 23)),
        'teacher_ids': [7, 8, 9, 2]}], [2], tmp_path / 'parity', {'model': 'random-tiny'}, max_steps=4)
    assert report['passed']
    assert all(check['passed'] for row in report['cases'] for trajectory in row['histories'].values()
               for check in trajectory['hf_checks'])
