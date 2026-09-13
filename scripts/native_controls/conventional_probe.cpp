// Independent public-API caller for tiny ordinary GGUFs. No rq3_model_* calls.
// Deliberately shares the tested llama/ggml libraries: this isolates the bridge,
// not defects common to both callers or the upstream quantized kernels.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
bool observe(ggml_tensor *, bool, void *) { return false; }
struct batch_owner {
    llama_batch value = llama_batch_init(64, 0, 1);
    ~batch_owner() { llama_batch_free(value); }
};
void run(const char * path, const char * device, const char * destination) {
    if (std::filesystem::exists(destination)) { throw std::runtime_error("output already exists"); }
    if (std::filesystem::file_size(path) > 16 * 1024 * 1024) { throw std::runtime_error("tiny fixtures only"); }
    const uint32_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1) { throw std::runtime_error("little endian required"); }
    ggml_backend_load_all();
    const bool cpu = std::strcmp(device, "CPU") == 0;
    ggml_backend_dev_t devices[2] = {cpu ? nullptr : ggml_backend_dev_by_name(device), nullptr};
    if (!cpu && !devices[0]) { throw std::runtime_error("requested GPU unavailable"); }
    llama_model_tensor_buft_override overrides[2] = {};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = cpu ? 0 : 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.check_tensors = true;
    mp.load_mtp = false;
    if (!cpu) {
        auto buft = ggml_backend_dev_buffer_type(devices[0]);
        if (!buft) { throw std::runtime_error("no GPU buffer type"); }
        overrides[0] = {"^token_embd\\.weight$", buft};
        mp.tensor_buft_overrides = overrides;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, mp), llama_model_free);
    if (!model) { throw std::runtime_error("public-API model load failed"); }
    char metadata[32];
    if (llama_model_meta_val_str(model.get(), "rotquant.version", metadata, sizeof(metadata)) >= 0) {
        throw std::runtime_error("ordinary GGUF required; RotQuant metadata forbidden");
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    if (vocab != 256 || llama_model_n_layer(model.get()) != 2 || llama_model_n_embd(model.get()) != 256) {
        throw std::runtime_error("requires the bounded two-layer fixture");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.n_ubatch = 256;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = !cpu; cp.op_offload = !cpu;
    cp.cb_eval = observe;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!context) { throw std::runtime_error("public-API context creation failed"); }
    batch_owner batch;
    std::vector<float> captured;
    std::vector<int32_t> traces;
    int position = 0;
    auto evaluate = [&](const std::vector<llama_token> & tokens) {
        if (tokens.empty() || tokens.size() > 64 || position + tokens.size() > 256) {
            throw std::runtime_error("probe exceeds bounded context");
        }
        batch.value.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] < 0 || tokens[i] >= vocab) { throw std::runtime_error("invalid token"); }
            batch.value.token[i] = tokens[i];
            // Explicit positions and sequence ownership exercise a different
            // public-API path from the private bridge's llama_batch_get_one.
            batch.value.pos[i] = position + static_cast<int32_t>(i);
            batch.value.n_seq_id[i] = 1;
            batch.value.seq_id[i][0] = 0;
            batch.value.logits[i] = i + 1 == tokens.size();
        }
        if (llama_decode(context.get(), batch.value) != 0) { throw std::runtime_error("public-API decode failed"); }
        llama_synchronize(context.get());
        const float * values = llama_get_logits_ith(context.get(), -1);
        if (!values) { throw std::runtime_error("missing logits"); }
        std::vector<float> logits(values, values + vocab);
        for (float value : logits) { if (!std::isfinite(value)) { throw std::runtime_error("non-finite logits"); } }
        captured.insert(captured.end(), logits.begin(), logits.end());
        position += static_cast<int>(tokens.size());
        return logits;
    };
    for (int count : {1, 4, 17, 64}) {
        llama_memory_clear(llama_get_memory(context.get()), true);
        position = 0;
        std::vector<llama_token> prompt;
        for (int i = 0; i < count; ++i) { prompt.push_back(3 + i); }
        auto logits = evaluate(prompt);
        for (int step = 0; step < 8; ++step) {
            const auto token = static_cast<llama_token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            traces.push_back(token);
            logits = evaluate({token});
        }
        std::cout << "public API " << device << ": " << count << " prompt tokens + 8 cached steps completed\n" << std::flush;
    }
    std::ofstream output(destination, std::ios::binary | std::ios::out);
    output.write("RQCTRL01", 8);
    const uint32_t shape[4] = {4, 9, 256, 8};
    output.write(reinterpret_cast<const char *>(shape), sizeof(shape));
    output.write(reinterpret_cast<const char *>(captured.data()), captured.size() * sizeof(float));
    output.write(reinterpret_cast<const char *>(traces.data()), traces.size() * sizeof(int32_t));
    if (!output) { throw std::runtime_error("cannot write control probes"); }
}
}
int main(int argc, char ** argv) {
    try {
        if (argc != 4) { throw std::runtime_error("usage: conventional-probe MODEL CPU|CUDA0|MTL0 OUTPUT"); }
        run(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "conventional public-API control failed: " << e.what() << '\n';
        return 1;
    }
}
