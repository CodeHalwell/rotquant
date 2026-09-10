#include "test_bridge.h"
#include "llama.h"
#include "rotquant-v3.h"
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
    uint64_t custom_ops = 0;
    int32_t positions = 0;
    ~session() {
        if (context) { llama_free(context); }
        if (model) { llama_model_free(model); }
    }
};
bool observe(ggml_tensor * tensor, bool ask, void * data) {
    if (ask && rq3_kind(tensor)) { ((session *) data)->custom_ops++; }
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
        const bool cpu = strcmp(device, "CPU") == 0;
        s->devices[0] = cpu ? nullptr : ggml_backend_dev_by_name(device);
        if (!cpu && !s->devices[0]) { throw std::runtime_error("requested GPU unavailable; no fallback"); }
        auto mp = llama_model_default_params();
        mp.devices = s->devices; mp.n_gpu_layers = cpu ? 0 : 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.check_tensors = true; mp.load_mtp = false;
        s->model = llama_model_load_from_file(path, mp);
        if (!s->model) { throw std::runtime_error("model load failed; inspect persistent log"); }
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
        if (!handle || !input || !output || count < 1 || selected < 1 || selected > count || selected > 16) {
            throw std::runtime_error("invalid model probe");
        }
        auto & s = *(session *) handle;
        const int vocab = rq3_model_vocab(handle);
        for (int i = 0; i < count; ++i) { if (input[i] < 0 || input[i] >= vocab) { throw std::runtime_error("invalid token ID"); } }
        if (reset) { llama_memory_clear(llama_get_memory(s.context), true); s.positions = 0; }
        if (uint64_t(s.positions) + count > llama_n_ctx(s.context)) { throw std::runtime_error("probe exceeds context"); }
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
        return 0;
    } catch (const std::exception & e) { last_error = e.what(); return 1; }
}
uint64_t rq3_model_custom_ops(void * handle) { return ((session *) handle)->custom_ops; }
void rq3_model_close(void * handle) { delete (session *) handle; }
