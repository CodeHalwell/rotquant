from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ternary_inference import InferenceProfile, require_dispatch
from scripts.w4_research_measure import ChunkedNativeModel, NativeBackend


class NativeFixture:
    supports_no_logits = supports_compact_greedy = True
    vocab = 8

    def __init__(self):
        self.calls = []
        self.next_token = 4

    def evaluate(self, ids, *, reset=False, selected=1):
        self.calls.append((list(ids), reset, selected))
        output = np.zeros((selected, self.vocab), dtype=np.float32)
        if selected:
            output[:, 4] = 1
        return output[0] if selected == 1 else output

    def greedy(self, ids, *, reset=False):
        self.calls.append((list(ids), reset, "compact"))
        token = self.next_token
        self.next_token += 1
        return token


def test_legacy_prefill_stays_compatible():
    native = NativeFixture()
    ChunkedNativeModel(native).evaluate(range(35), reset=True)
    assert [(len(ids), reset, count) for ids, reset, count in native.calls] == [
        (16, True, 1), (16, False, 1), (3, False, 1)]


def test_graph_prefill_requests_no_head_and_retains_selected_positions():
    native = NativeFixture()
    model = ChunkedNativeModel(native, graph_prefill=True)
    output = model.evaluate(range(35), reset=True, selected=4)
    assert output.shape == (4, 8)
    assert [(len(ids), reset, count) for ids, reset, count in native.calls] == [
        (16, True, 0), (16, False, 1), (3, False, 3)]


def test_compact_sampling_preserves_caps_reset_and_explicit_stops():
    native = NativeFixture()
    model = InferenceProfile(graph_prefill=True, compact_greedy=True).wrap(native)
    assert NativeBackend(model, {5}).generate(range(35), 8) == [4, 5]
    assert [(len(ids), reset, count) for ids, reset, count in native.calls] == [
        (16, True, 0), (16, False, 0), (3, False, "compact"), (1, False, "compact")]


def test_requested_capabilities_fail_closed():
    native = SimpleNamespace(supports_no_logits=False, supports_compact_greedy=False)
    with pytest.raises(RuntimeError, match="no-logits"):
        ChunkedNativeModel(native, graph_prefill=True)
    with pytest.raises(RuntimeError, match="compact"):
        ChunkedNativeModel(native, compact_greedy=True)


@pytest.mark.parametrize("values", [{"kernel": "fake"}, {"arithmetic": "w2a8"},
                                    {"version": 1}, {"compact_greedy": "yes"}])
def test_profiles_reject_unimplemented_arithmetic_or_ambiguous_settings(values):
    with pytest.raises(ValueError):
        InferenceProfile.from_mapping(values)


def test_actual_format_specific_dispatch_is_required():
    receipt = {"w2_g128_s16": {"reference": 10, "ternary-g128": 0}}
    with pytest.raises(RuntimeError, match="did not execute"):
        require_dispatch(receipt, {"w2_g128_s16": ["ternary-g128"]})
    receipt["w2_g128_s16"]["ternary-g128"] = 1
    assert require_dispatch(receipt, {"w2_g128_s16": ["ternary-g128"]}) == receipt


def test_compact_speed_replay_calls_native_sampling_not_full_logits(tmp_path):
    from scripts.w4_research_measure import native_speeds
    native = NativeFixture()
    model = InferenceProfile(graph_prefill=True, compact_greedy=True).wrap(native)
    memory = SimpleNamespace(report=lambda: {"sampled_peak_process_vram_mib": 0})
    report = native_speeds(model, {35: {"prompt_ids": list(range(35)), "decode_ids": [1, 2]}},
                           1, tmp_path / "speed.json", memory=memory)
    assert report["compact_sampling"] and report["graph_no_logits_prefill"] and report["completed"]
    assert all(selected in (0, "compact") for _, _, selected in native.calls)
    assert report["rows"][1]["decode_input_ids"] == [1, 2]


def test_generation_reports_finished_answer_latency_and_ttft():
    backend = NativeBackend(InferenceProfile(compact_greedy=True).wrap(NativeFixture()), {5})
    assert backend.generate([1, 2], 5) == [4, 5]
    timing = backend.last_generation_timing
    assert timing["generation_seconds"] >= timing["time_to_first_token_seconds"] > 0
    assert timing["time_to_finished_answer_seconds"] == timing["generation_seconds"]
    assert timing["completed_answers_per_second"] > 0


def test_dispatch_expectations_cover_model_precisions():
    from scripts.ternary_inference import expected_model_dispatch
    profile = InferenceProfile(kernel="head-warp")
    assert expected_model_dispatch(profile, "fp16") == {}
    assert expected_model_dispatch(profile, "w6_v6") == {"w6": ["tile8", "decode4", "head-warp"]}
    assert expected_model_dispatch(profile, "w5_v6")["other"] == ["tile8", "decode4"]
    assert expected_model_dispatch(profile, "ternary_global_qat")["w2_g128_s16"] == ["tile8", "decode4"]
    with pytest.raises(ValueError, match="unknown model"):
        expected_model_dispatch(profile, "unknown")


def test_fused_profiles_require_separate_arithmetic_and_decode_dispatch():
    from scripts.ternary_inference import expected_model_dispatch
    profile = InferenceProfile(kernel="ternary-w2a8", arithmetic="w2a8")
    expected = expected_model_dispatch(profile, "ternary_global_qat")
    assert expected["w2_g128_s16"] == ["ternary-w2a8"]
    assert expected["w4_s8"] == ["tile8", "decode4"]
    tensor = InferenceProfile(kernel="tensorcore-fused")
    assert tensor.identity()["arithmetic_contract"] == "same-half-operands-reordered"
    assert expected_model_dispatch(tensor, "w4_v6")["w4_s8"] == ["tensorcore-fused", "decode4"]
