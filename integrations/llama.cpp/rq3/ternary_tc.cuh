#pragma once
#include <mma.h>

static __global__ void rq3_tensorcore_prefill(rq3_args a, const float * input,
        const uchar * packed, const int * permutation, float * output) {
    const int lane = threadIdx.x;
    const int64_t first_row = int64_t(blockIdx.x) * 16;
    const int64_t first_token = int64_t(blockIdx.y) * 16;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(32) half weights[256], activations[256];
    __shared__ __align__(32) float sums[256];
    __shared__ int invalid;
    if (!lane) { invalid = 0; }
    __syncwarp();
    const rq3_view view = rq3_parse(packed);
    if (view.cols != uint64_t(a.inputs) || view.rows != uint64_t(a.outputs)) {
        for (int index = lane; index < 256; index += 32) {
            const int64_t row = first_row + index / 16, token = first_token + index % 16;
            if (row < a.outputs && token < a.tokens) { output[token * a.outputs + row] = NAN; }
        }
        return;
    }
    const bool ternary = view.bits == 2 && __uint_as_float(rq3_u32(view.book)) == -1.f &&
        __uint_as_float(rq3_u32(view.book + 4)) == 0.f &&
        __uint_as_float(rq3_u32(view.book + 8)) == 1.f &&
        __uint_as_float(rq3_u32(view.book + 12)) == 0.f;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                            nvcuda::wmma::row_major> left;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                            nvcuda::wmma::col_major> right;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accum;
    nvcuda::wmma::fill_fragment(accum, 0.f);
    for (int64_t start = 0; start < a.inputs; start += 16) {
        for (int index = lane; index < 256; index += 32) {
            const int64_t row = first_row + index / 16;
            const int64_t token = first_token + index / 16;
            const int64_t col = start + index % 16;
            const int64_t source = row < a.outputs && permutation ? permutation[row] : row;
            float weight = 0.f, activation = 0.f;
            if (row < a.outputs && col < a.inputs) {
                if (source < 0 || source >= a.outputs) { atomicExch(&invalid, 1); }
                else {
                    weight = rq3_round(rq3_weight(view, source, col));
                    if (ternary) {
                        const uint code = (view.codes[(source * view.cols + col) / 4] >> ((col % 4) * 2)) & 3;
                        if (code == 3) { atomicExch(&invalid, 1); }
                    }
                    if (!isfinite(weight)) { atomicExch(&invalid, 1); }
                }
            }
            if (token < a.tokens && col < a.inputs) {
                activation = rq3_round(input[token * a.inputs + col]);
                if (!isfinite(activation)) { atomicExch(&invalid, 1); }
            }
            weights[index] = __float2half_rn(weight);
            activations[index] = __float2half_rn(activation);
        }
        __syncwarp();
        nvcuda::wmma::load_matrix_sync(left, weights, 16);
        nvcuda::wmma::load_matrix_sync(right, activations, 16);
        nvcuda::wmma::mma_sync(accum, left, right, accum);
        __syncwarp();
    }
    nvcuda::wmma::store_matrix_sync(sums, accum, 16, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < 256; index += 32) {
        const int64_t row = first_row + index / 16;
        const int64_t token = first_token + index % 16;
        if (row < a.outputs && token < a.tokens) {
            output[token * a.outputs + row] = invalid ? NAN : rq3_round(sums[index]);
        }
    }
#else
    for (int index = lane; index < 256; index += 32) {
        const int64_t row = first_row + index / 16, token = first_token + index % 16;
        if (row < a.outputs && token < a.tokens) { output[token * a.outputs + row] = NAN; }
    }
#endif
}

static size_t rq3_launch_tensorcore(ggml_backend_cuda_context & ctx, rq3_args a, ggml_tensor * op) {
    GGML_ASSERT(a.inputs > 0 && a.inputs % 128 == 0 && a.inputs <= INT32_MAX);
    GGML_ASSERT(a.outputs > 0 && a.outputs <= INT32_MAX && a.tokens >= 4 && a.tokens <= 65535);
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 800);
    rq3_tensorcore_prefill<<<dim3((a.outputs + 15) / 16, (a.tokens + 15) / 16), 32, 0, ctx.stream()>>>(a,
        (const float *) op->src[0]->data, (const uchar *) op->src[1]->data,
        a.permutation ? (const int *) op->src[2]->data : nullptr, (float *) op->data);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
