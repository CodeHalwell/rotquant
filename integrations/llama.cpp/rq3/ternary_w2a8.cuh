#pragma once

static constexpr size_t rq3_w2a8_scratch_limit = 256ULL * 1024 * 1024;

static __global__ void rq3_w2a8_rotate_quantize(rq3_args a, const float * input,
        const int8_t * signs, const int * permutation, int8_t * codes, float * scales) {
    const int lane = threadIdx.x;
    const int64_t col = int64_t(blockIdx.x) * 128 + lane;
    const int64_t token = blockIdx.y;
    __shared__ float tile[128], next[128];
    __shared__ int valid;
    if (!lane) { valid = 1; }
    __syncthreads();
    const int64_t source = permutation ? permutation[col] : col;
    const bool legal = source >= 0 && source < a.inputs && (signs[col] == -1 || signs[col] == 1);
    const float value = legal ? rq3_round(input[token * a.inputs + source]) * signs[col] : 0.f;
    if (!legal || !isfinite(value)) { atomicExch(&valid, 0); }
    tile[lane] = isfinite(value) ? value : 0.f;
    rq3_fwht(tile, next, lane);
    const float rotated = rq3_round(tile[lane] * 0.08838834764831845f);
    if (!isfinite(rotated)) { atomicExch(&valid, 0); }
    next[lane] = isfinite(rotated) ? fabsf(rotated) : 0.f;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (lane < stride) { next[lane] = fmaxf(next[lane], next[lane + stride]); }
        __syncthreads();
    }
    const float scale = next[0] > 0.f ? __fdiv_rn(next[0], 127.f) : 1.f;
    const int rounded = __float2int_rn(__fdiv_rn(rotated, scale));
    codes[token * a.inputs + col] = valid ? int8_t(max(-127, min(127, rounded))) : int8_t(0);
    if (!lane) { scales[token * (a.inputs / 128) + blockIdx.x] = valid ? scale : NAN; }
}

__device__ __forceinline__ bool rq3_w2a8_legal(rq3_view v, rq3_args a) {
    return v.bits == 2 && v.sbits == 16 && v.group == 128 &&
        v.cols == uint64_t(a.inputs) && v.rows == uint64_t(a.outputs) &&
        __uint_as_float(rq3_u32(v.book)) == -1.f &&
        __uint_as_float(rq3_u32(v.book + 4)) == 0.f &&
        __uint_as_float(rq3_u32(v.book + 8)) == 1.f &&
        __uint_as_float(rq3_u32(v.book + 12)) == 0.f;
}

static __global__ void rq3_w2a8_dot(rq3_args a, const uchar * packed,
        const int * permutation, const int8_t * activation, const float * activation_scales,
        float * output) {
    const int lane = threadIdx.x;
    const int64_t target = blockIdx.x, token = blockIdx.y;
    const int64_t row = permutation ? permutation[target] : target;
    const rq3_view view = rq3_parse(packed);
    if (row < 0 || row >= a.outputs || !rq3_w2a8_legal(view, a)) {
        if (!lane) { output[token * a.outputs + target] = NAN; }
        return;
    }
    float sum = 0.f;
    for (int64_t group = 0; group < a.inputs / 128; ++group) {
        const uint symbols = view.codes[(row * a.inputs + group * 128) / 4 + lane];
        const int64_t input_offset = token * a.inputs + group * 128 + lane * 4;
        uint weights = 0, values = 0;
        bool valid = true;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int code = (symbols >> (2 * part)) & 3;
            valid = valid && code != 3;
            weights |= uint(uint8_t(int8_t(code - 1))) << (8 * part);
            values |= uint(uint8_t(activation[input_offset + part])) << (8 * part);
        }
        const float weight_scale = rq3_half(view.scales + (row * view.groups + group) * 2);
        const float activation_scale = activation_scales[token * view.groups + group];
        valid = valid && isfinite(weight_scale) && weight_scale > 0.f && isfinite(activation_scale);
        if (__ballot_sync(0xffffffffu, !valid)) {
            if (!lane) { output[token * a.outputs + target] = NAN; }
            return;
        }
        int dot = __dp4a(int(weights), int(values), 0);
        #pragma unroll
        for (int stride = 16; stride; stride /= 2) {
            const int other = __shfl_down_sync(0xffffffffu, dot, stride);
            if (lane < stride) { dot += other; }
        }
        if (!lane) {
            const float scale = __fmul_rn(weight_scale, activation_scale);
            sum = __fadd_rn(sum, __fmul_rn(float(dot), scale));
        }
    }
    if (!lane) { output[token * a.outputs + target] = rq3_round(sum); }
}

static size_t rq3_launch_w2a8(ggml_backend_cuda_context & ctx, rq3_args a, ggml_tensor * op) {
    GGML_ASSERT(a.inputs > 0 && a.inputs % 128 == 0 && a.inputs <= INT32_MAX);
    GGML_ASSERT(a.outputs > 0 && a.outputs <= INT32_MAX && a.tokens > 0 && a.tokens <= 65535);
    const size_t inputs = size_t(a.inputs) * size_t(a.tokens);
    const size_t groups = inputs / 128;
    const size_t scratch = inputs * sizeof(int8_t) + groups * sizeof(float);
    GGML_ASSERT(scratch <= rq3_w2a8_scratch_limit);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I8);
    GGML_ASSERT(ggml_nelements(op->src[2]) >= a.inputs);
    const bool rows_permuted = op->src[3]->type == GGML_TYPE_I32;
    const bool columns_permuted = op->src[4]->type == GGML_TYPE_I32;
    GGML_ASSERT(!rows_permuted || ggml_nelements(op->src[3]) >= a.outputs);
    GGML_ASSERT(!columns_permuted || ggml_nelements(op->src[4]) >= a.inputs);
    ggml_cuda_pool_alloc<int8_t> quantized(ctx.pool(), inputs);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), groups);
    rq3_w2a8_rotate_quantize<<<dim3(a.inputs / 128, a.tokens), 128, 0, ctx.stream()>>>(a,
        (const float *) op->src[0]->data, (const int8_t *) op->src[2]->data,
        columns_permuted ? (const int *) op->src[4]->data : nullptr, quantized.get(), scales.get());
    rq3_w2a8_dot<<<dim3(a.outputs, a.tokens), 32, 0, ctx.stream()>>>(a,
        (const uchar *) op->src[1]->data,
        rows_permuted ? (const int *) op->src[3]->data : nullptr,
        quantized.get(), scales.get(), (float *) op->data);
    CUDA_CHECK(cudaGetLastError());
    return scratch;
}
