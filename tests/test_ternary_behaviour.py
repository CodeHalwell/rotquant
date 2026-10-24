import pytest
import torch
from torch import nn

from scripts.ternary_behaviour import build_behaviour_evaluator, repeated_span, valid_format
from scripts.ternary_chat_data import freeze_chat_dataset
from tests.test_ternary_chat_data import Teacher, Tokenizer, prompts


def test_exact_repetition_detects_long_periodic_regions_but_not_short_or_interrupted_text():
    assert repeated_span([4] * 64) == {"start": 0, "period": 1, "length": 64}
    assert repeated_span([7, 8, 9] * 30)["period"] == 3
    assert repeated_span([1, 2, 3] * 10) is None
    assert repeated_span(list(range(100))) is None


@pytest.mark.parametrize("text,contract,accepted", [
    ('{"items": [1]}', {"kind": "json", "root": "object", "required_keys": ["items"]}, True),
    ('```json\n[1,2]\n```', {"kind": "json", "root": "array"}, True),
    ('{"other": 1}', {"kind": "json", "root": "object", "required_keys": ["items"]}, False),
    ('prose {"items": []}', {"kind": "json"}, False),
    ('[]', {"kind": "json", "root": "object"}, False),
])
def test_format_is_only_the_declared_syntax_contract(text, contract, accepted):
    assert valid_format(text, contract) is accepted


class LoopingCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.seen = []

    def generate(self, input_ids, attention_mask, **settings):
        assert settings["use_cache"] and not settings["do_sample"]
        assert not self.training
        self.seen.append(input_ids[0].tolist())
        return torch.cat((input_ids, torch.full((1, settings["max_new_tokens"]), 10)), -1)


def test_behaviour_evaluates_only_selection_and_reports_truncated_loops_with_stable_identity():
    tokenizer = Tokenizer()
    manifest = freeze_chat_dataset(tokenizer, Teacher(), prompts(), model_revision="a" * 40,
                                    tokenizer_revision="b" * 40, stop_ids=(99,), max_new_tokens=80)
    tokenizer.decode = lambda values, skip_special_tokens: "nonempty response"
    evaluate = build_behaviour_evaluator(manifest, tokenizer, "cpu")
    candidate = LoopingCandidate()
    result = evaluate(candidate)
    assert candidate.training
    assert result["completion_rate"] == 0
    assert result["repetition_rate"] == 1
    assert result["format_rate"] == 1
    assert result["examples"] == 3
    assert result["split"] == "selection"
    expected = [x["input_ids"][:x["prompt_length"]] for x in manifest["records"]["selection"]]
    assert candidate.seen == expected
    assert evaluate(candidate)["data_id"] == result["data_id"]
