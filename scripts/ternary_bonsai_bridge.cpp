// Thin measurement bridge for the pinned Prism runtime; no model conversion.
#include "llama.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
struct session {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    llama_model_tensor_buft_override overrides[2] = {};
    uint32_t position = 0;
    ~session() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};
void require(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}
}

extern "C" {
const char * bonsai_error() { return error.c_str(); }
const char * bonsai_bridge_version() { return "bonsai-measure-v1"; }
void bonsai_init() { llama_backend_init(); ggml_backend_load_all(); }
void * bonsai_open(const char * path, const char * device, uint32_t capacity) {
    error.clear();
    try {
        require(capacity >= 32 && capacity <= 262144, "context must be 32..262144");
        auto s = std::make_unique<session>();
        const bool cpu = std::strcmp(device, "CPU") == 0;
        s->devices[0] = cpu ? nullptr : ggml_backend_dev_by_name(device);
        require(cpu || s->devices[0], "requested GPU unavailable; no CPU fallback");
        auto mp = llama_model_default_params();
        mp.devices = s->devices;
        mp.n_gpu_layers = cpu ? 0 : 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.check_tensors = true;
        mp.load_mtp = false;
        if (!cpu) {
            auto buft = ggml_backend_dev_buffer_type(s->devices[0]);
            require(buft, "GPU buffer type unavailable");
            s->overrides[0] = {"^token_embd\\.weight$", buft};
            mp.tensor_buft_overrides = s->overrides;
        }
        s->model = llama_model_load_from_file(path, mp);
        require(s->model, "model load failed; inspect native log");
        auto cp = llama_context_default_params();
        cp.n_ctx = capacity;
        cp.n_batch = capacity;
        cp.n_ubatch = capacity;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = !cpu;
        cp.op_offload = !cpu;
        s->context = llama_init_from_model(s->model, cp);
        require(s->context, "context creation failed");
        return s.release();
    } catch (const std::exception & e) { error = e.what(); return nullptr; }
}
int bonsai_vocab(void * handle) {
    return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<session *>(handle)->model));
}
bool bonsai_is_eog(void * handle, int token) {
    return llama_vocab_is_eog(llama_model_get_vocab(static_cast<session *>(handle)->model), token);
}
int bonsai_evaluate(void * handle, const int32_t * input, int count, bool reset,
                    int selected, float * output) {
    error.clear();
    try {
        require(handle && input && output && count > 0 && selected > 0 &&
                selected <= count && selected <= 16, "invalid evaluation arguments");
        auto & s = *static_cast<session *>(handle);
        const int vocab = bonsai_vocab(handle);
        for (int i = 0; i < count; ++i) require(input[i] >= 0 && input[i] < vocab, "invalid token ID");
        if (reset) { llama_memory_clear(llama_get_memory(s.context), true); s.position = 0; }
        require(uint64_t(s.position) + count <= llama_n_ctx(s.context), "evaluation exceeds context");
        std::vector<llama_token> ids(input, input + count);
        auto batch = llama_batch_get_one(ids.data(), count);
        std::vector<int8_t> requested(count, 0);
        for (int i = count - selected; i < count; ++i) requested[i] = 1;
        batch.logits = requested.data();
        require(llama_decode(s.context, batch) == 0, "native decode failed");
        llama_synchronize(s.context);
        for (int row = 0; row < selected; ++row) {
            const float * logits = llama_get_logits_ith(s.context, row - selected);
            require(logits, "missing native logits");
            for (int col = 0; col < vocab; ++col) require(std::isfinite(logits[col]), "non-finite logits");
            std::memcpy(output + size_t(row) * vocab, logits, size_t(vocab) * sizeof(float));
        }
        s.position += count;
        return 0;
    } catch (const std::exception & e) { error = e.what(); return 1; }
}
void bonsai_close(void * handle) { delete static_cast<session *>(handle); }
}
