#include "test_bridge.h"
#include "llama.h"
#include "rotquant-v3.h"
#include "llama-rq3.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct session {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    llama_model_tensor_buft_override input_override[2] = {};
    uint64_t custom_ops = 0;
    int32_t positions = 0;
    llama_sampler * sampler = nullptr;
    bool no_logits = false, compact_supported = false;
    std::string kernel;
    uint64_t output_rows = 0, compact_tokens = 0, no_logits_batches = 0, head_ops = 0;
    ~session() {
        if (context) { llama_free(context); }
        if (model) { llama_model_free(model); }
        if (sampler) { llama_sampler_free(sampler); }
    }
};
struct sampler_state { bool supported = false; };
void sampler_free(llama_sampler * sampler) { delete (sampler_state *) sampler->ctx; }
const char * sampler_name(const llama_sampler *) { return "rq3-finite-greedy-v2"; }
void sampler_apply(llama_sampler *, llama_token_data_array * data) {
    data->selected = 0;
    for (size_t i = 0; i < data->size; ++i) {
        if (!std::isfinite(data->data[i].logit)) { data->selected = -1; return; }
        if (data->data[i].logit > data->data[data->selected].logit) { data->selected = i; }
    }
}
bool sampler_init(llama_sampler * sampler, ggml_backend_buffer_type_t buft, uint32_t) {
    auto * ctx = ggml_init({4096, nullptr, true});
    if (!ctx) { return false; }
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    auto * device = ggml_backend_buft_get_device(buft);
    const bool supported = !device || ggml_backend_dev_supports_op(device, llama_rq3_greedy(ctx, x));
    ggml_free(ctx);
    ((sampler_state *) sampler->ctx)->supported = supported;
    return supported;
}
void sampler_graph(llama_sampler *, ggml_context * ctx, ggml_cgraph *, llama_sampler_data * data) {
    data->sampled = llama_rq3_greedy(ctx, data->logits);
    data->logits = nullptr;
    data->probs = nullptr;
    data->candidates = nullptr;
}
llama_sampler_i sampler_interface = {sampler_name, nullptr, sampler_apply, nullptr, nullptr, sampler_free,
    sampler_init, nullptr, sampler_graph, nullptr, nullptr, nullptr};
void compact_mode(session & s, bool enabled) {
    if (!enabled && s.sampler) {
        if (!llama_set_sampler(s.context, 0, nullptr)) { throw std::runtime_error("cannot detach GPU sampler"); }
        llama_sampler_free(s.sampler); s.sampler = nullptr;
    }
    if (enabled && !s.compact_supported) { throw std::runtime_error("compact backend unsupported; no fallback"); }
    if (enabled && !s.sampler) {
        auto * state = new sampler_state();
        s.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(s.sampler, llama_sampler_init(&sampler_interface, state));
        if (!llama_set_sampler(s.context, 0, s.sampler) || !state->supported) {
            llama_set_sampler(s.context, 0, nullptr);
            llama_sampler_free(s.sampler); s.sampler = nullptr;
            throw std::runtime_error("compact backend sampling unavailable; no fallback");
        }
    }
}
void validate_input(session & s, const int32_t * input, int32_t count, bool reset) {
    if (!input || count < 1) { throw std::runtime_error("invalid token probe"); }
    const char * kernel = getenv("ROTQUANT_RQ3_KERNEL");
    if (s.kernel != (kernel ? kernel : "reference")) { throw std::runtime_error("inference profile changed after model open"); }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(s.model));
    for (int i = 0; i < count; ++i) { if (input[i] < 0 || input[i] >= vocab) { throw std::runtime_error("invalid token ID"); } }
    if (reset) { llama_memory_clear(llama_get_memory(s.context), true); s.positions = 0; }
    if (uint64_t(s.positions) + count > llama_n_ctx(s.context)) { throw std::runtime_error("probe exceeds context"); }
}
bool observe(ggml_tensor * tensor, bool ask, void * data) {
    if (ask && rq3_kind(tensor) && rq3_kind(tensor) != RQ3_GREEDY) {
        ((session *) data)->custom_ops++;
        if (rq3_kind(tensor) == RQ3_HEAD) { ((session *) data)->head_ops++; }
    }
    return false;
}
}

const char * rq3_model_error() { return last_error.c_str(); }
void * rq3_model_open(const char * path, const char * device, uint32_t context) {
    last_error.clear();
    try {
        if (context < 32 || context > 8192) { throw std::runtime_error("bounded preflight context must be 32..8192"); }
        ggml_backend_load_all();
        auto s = std::make_unique<session>();
        const char * kernel = getenv("ROTQUANT_RQ3_KERNEL");
        s->kernel = kernel ? kernel : "reference";
        const bool cpu = strcmp(device, "CPU") == 0;
        s->compact_supported = cpu || strncmp(device, "CUDA", 4) == 0;
        s->devices[0] = cpu ? nullptr : ggml_backend_dev_by_name(device);
        if (!cpu && !s->devices[0]) { throw std::runtime_error("requested GPU unavailable; no fallback"); }
        auto mp = llama_model_default_params();
        mp.devices = s->devices; mp.n_gpu_layers = cpu ? 0 : 999;
        // llama.cpp normally keeps ordinary input embeddings on CPU even when
        // all repeating/output layers are offloaded. Our all-GPU benchmark
        // contract includes GET_ROWS, for conventional GGUFs as well as RQ3.
        // Keep the override alive with the model; never relax the execution gate.
        if (!cpu) {
            auto buft = ggml_backend_dev_buffer_type(s->devices[0]);
            if (!buft) { throw std::runtime_error("selected GPU has no buffer type"); }
            s->input_override[0] = {"^token_embd\\.weight$", buft};
            mp.tensor_buft_overrides = s->input_override;
        }
        mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.check_tensors = true; mp.load_mtp = false;
        s->model = llama_model_load_from_file(path, mp);
        if (!s->model) { throw std::runtime_error("model load failed; inspect persistent log"); }
        char architecture[64] = {};
        llama_model_meta_val_str(s->model, "general.architecture", architecture, sizeof(architecture));
        s->no_logits = strcmp(architecture, "qwen35") == 0;
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = context; cp.n_ubatch = context;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = !cpu; cp.op_offload = !cpu;
        cp.cb_eval = observe; cp.cb_eval_user_data = s.get();
        s->context = llama_init_from_model(s->model, cp);
        if (!s->context) { throw std::runtime_error("context creation failed"); }
        return s.release();
    } catch (const std::exception & e) { last_error = e.what(); return nullptr; }
}
int rq3_model_vocab(void * handle) {
    return llama_vocab_n_tokens(llama_model_get_vocab(((session *) handle)->model));
}
int rq3_model_eval(void * handle, const int32_t * input, int32_t count, bool reset, float * output) {
    return rq3_model_eval_selected(handle, input, count, reset, 1, output);
}
bool rq3_model_is_eog(void * handle, int32_t token) {
    return llama_vocab_is_eog(llama_model_get_vocab(((session *) handle)->model), token);
}
int rq3_model_eval_selected(void * handle, const int32_t * input, int32_t count, bool reset, int32_t selected, float * output) {
    last_error.clear();
    try {
        if (!handle || !input || (selected && !output) || count < 1 || selected < 0 || selected > count || selected > 16) {
            throw std::runtime_error("invalid model probe");
        }
        auto & s = *(session *) handle;
        const int vocab = rq3_model_vocab(handle);
        if (!selected && !s.no_logits) { throw std::runtime_error("graph no-logits requires Qwen35"); }
        compact_mode(s, false);
        validate_input(s, input, count, reset);
        std::vector<llama_token> ids(input, input + count);
        auto batch = llama_batch_get_one(ids.data(), count);
        std::vector<int8_t> outputs(count, 0);
        for (int i = count - selected; i < count; ++i) { outputs[i] = 1; }
        batch.logits = outputs.data();
        if (llama_decode(s.context, batch) != 0) { throw std::runtime_error("model execution/GPU gate failed"); }
        llama_synchronize(s.context);
        for (int pos = 0; pos < selected; ++pos) {
            const float * logits = llama_get_logits_ith(s.context, -selected + pos);
            if (!logits) { throw std::runtime_error("missing model logits"); }
            for (int i = 0; i < vocab; ++i) { if (!std::isfinite(logits[i])) { throw std::runtime_error("non-finite model logits"); } }
            std::memcpy(output + size_t(pos) * vocab, logits, size_t(vocab) * sizeof(float));
        }
        s.positions += count;
        s.output_rows += selected;
        s.no_logits_batches += selected == 0;
        return 0;
    } catch (const std::exception & e) { last_error = e.what(); return 1; }
}
uint64_t rq3_model_custom_ops(void * handle) { return ((session *) handle)->custom_ops; }
void rq3_model_close(void * handle) { delete (session *) handle; }

uint32_t rq3_model_capabilities(void * handle) {
    if (!handle) { return 0; }
    auto & s = *(session *) handle;
    return (s.no_logits ? 1u : 0u) | (s.compact_supported ? 2u : 0u);
}
int rq3_model_greedy(void * handle, const int32_t * input, int32_t count, bool reset, int32_t * token) {
    last_error.clear();
    try {
        if (!handle || !token) { throw std::runtime_error("invalid compact probe"); }
        auto & s = *(session *) handle;
        validate_input(s, input, count, reset);
        compact_mode(s, true);
        std::vector<llama_token> ids(input, input + count);
        auto batch = llama_batch_get_one(ids.data(), count);
        std::vector<int8_t> outputs(count, 0); outputs.back() = 1;
        batch.logits = outputs.data();
        if (llama_decode(s.context, batch)) { throw std::runtime_error("compact model execution failed"); }
        llama_synchronize(s.context);
        *token = llama_get_sampled_token_ith(s.context, -1);
        if (*token < 0 || *token >= rq3_model_vocab(handle)) {
            throw std::runtime_error("device finite-check rejected logits or sampler unavailable");
        }
        s.positions += count;
        s.compact_tokens++;
        return 0;
    } catch (const std::exception & e) { last_error = e.what(); return 1; }
}

int rq3_model_diagnostics(void * handle, uint64_t * stats) {
    if (!handle || !stats) { return 1; }
    auto & s = *(session *) handle;
    stats[0] = s.output_rows; stats[1] = s.compact_tokens;
    stats[2] = s.no_logits_batches; stats[3] = s.head_ops;
    return 0;
}
