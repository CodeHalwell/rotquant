#include "llama-rq3.h"
#include "test_bridge.h"
#include "rotquant-v3.h"
#include "rotquant/native_v3_c.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string error;
extern "C" const char * rq3_test_error() { return error.c_str(); }

extern "C" int rq3_test_eval(const char * device, const uint8_t * data, size_t size,
        int64_t rows, int64_t cols, int64_t tokens, int mode, const void * input,
        const int8_t * signs, const int32_t * row_map, const int32_t * col_map, float * output) {
    error.clear();
    try {
        if (rq_native_v3_validate(data, size, rows, cols) || cols % 128 || !input || !signs || !output ||
            tokens < 1 || tokens > 65535 || mode < 0 || mode > 2) { throw std::runtime_error("invalid operator fixture"); }
        if (data[20] != 128 || data[21] || data[22] || data[23]) { throw std::runtime_error("operator fixture requires g128"); }
        for (int64_t i = 0; i < cols; ++i) {
            if (signs[i] != 1 && signs[i] != -1) { throw std::runtime_error("invalid sign"); }
        }
        for (const auto pair : {std::make_pair(row_map, rows), std::make_pair(col_map, cols)}) {
            if (!pair.first) { continue; }
            std::vector<bool> seen(pair.second, false);
            for (int64_t i = 0; i < pair.second; ++i) {
                const auto value = pair.first[i];
                if (value < 0 || value >= pair.second || seen[value]) { throw std::runtime_error("invalid permutation"); }
                seen[value] = true;
            }
        }
        if (mode && (row_map || col_map)) { throw std::runtime_error("vocabulary cannot use permutations"); }
        if (mode == 2) {
            const auto * ids = (const int32_t *) input;
            for (int64_t i = 0; i < tokens; ++i) { if (ids[i] < 0 || ids[i] >= rows) { throw std::runtime_error("invalid token ID"); } }
        }
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_init_by_name(device, nullptr), ggml_backend_free);
        if (!backend) { throw std::runtime_error("requested backend is unavailable (no fallback)"); }
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        if (!ctx) { throw std::runtime_error("ggml context allocation failed"); }
        ggml_tensor * w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, size);
        ggml_tensor * s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, cols);
        ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), mode == 2 ? GGML_TYPE_I32 : GGML_TYPE_F32, mode == 2 ? tokens : cols, mode == 2 ? 1 : tokens);
        ggml_tensor * rp = row_map ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows) : nullptr;
        ggml_tensor * cp = col_map ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, cols) : nullptr;
        llama_rq3_state state{s, rp, cp, cols, rows, mode == 1};
        ggml_tensor * y = mode == 2 ? llama_rq3_get_rows(ctx.get(), w, x, s) : llama_rq3_mul_mat(ctx.get(), w, x, state);
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, y);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend.get(), ggml_graph_node(graph, i))) {
                throw std::runtime_error("graph op unsupported by requested backend");
            }
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) { throw std::runtime_error("backend buffer allocation failed"); }
        ggml_backend_tensor_set(w, data, 0, size);
        ggml_backend_tensor_set(s, signs, 0, cols);
        ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
        if (rp) { ggml_backend_tensor_set(rp, row_map, 0, rows * 4); }
        if (cp) { ggml_backend_tensor_set(cp, col_map, 0, cols * 4); }
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) { throw std::runtime_error("GPU graph execution failed"); }
        ggml_backend_synchronize(backend.get());
        ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));
        for (int64_t i = 0; i < ggml_nelements(y); ++i) {
            if (!std::isfinite(output[i])) { throw std::runtime_error("non-finite operator output"); }
        }
        return 0;
    } catch (const std::exception & e) { error = e.what(); return 1; }
}
