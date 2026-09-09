#include "rotquant/native_v3_c.h"
#include "rotquant/native_v2.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef __FAST_MATH__
#error "native-v3 requires IEEE floating point; do not build with fast-math"
#endif

namespace {
thread_local char last_error[256] = {};

size_t add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error("native-v3 size addition overflow");
    }
    return a + b;
}
size_t mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error("native-v3 size multiplication overflow");
    }
    return a * b;
}
size_t ceil_div(size_t a, size_t b) { return a / b + (a % b != 0); }
void require(bool condition, const char *message) {
    if (!condition) { throw std::invalid_argument(message); }
}
uint32_t u32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t u64(const uint8_t *p) { return u32(p) | (uint64_t(u32(p + 4)) << 32); }
size_t dimension(const uint8_t *p) {
    const uint64_t value = u64(p);
    require(value > 0 && value <= uint64_t(std::numeric_limits<int64_t>::max()),
            "dimensions must be positive signed-64-bit integers");
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("dimension exceeds address space");
    }
    return static_cast<size_t>(value);
}
float f32(const uint8_t *p) {
    const uint32_t word = u32(p);
    float result;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "native v3 requires IEEE binary32");
    std::memcpy(&result, &word, sizeof(result));
    return result;
}
float f16(const uint8_t *p) {
    return rotquant::native_v2::fp16_to_fp32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}
void disjoint(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const auto x = reinterpret_cast<uintptr_t>(a);
    const auto y = reinterpret_cast<uintptr_t>(b);
    require((x <= y ? a_bytes <= y - x : b_bytes <= x - y), "buffers overlap");
}
void aligned_float(const void *p) {
    require(reinterpret_cast<uintptr_t>(p) % alignof(float) == 0, "unaligned float buffer");
}

struct Matrix {
    size_t bits, group, rows, cols, sbits, block, groups, scales, words, metadata;
    const uint8_t *codes, *scale_data, *offsets, *steps, *centroids;

    explicit Matrix(const uint8_t *data, size_t size) {
        require(data && size >= 64, "missing native-v3 header");
        require(std::memcmp(data, "RQNATV3\0", 8) == 0 && u32(data + 8) == 3 &&
                u32(data + 12) == 64 && u32(data + 48) == 0 && u32(data + 52) == 0,
                "unsupported native-v3 header, flags or version");
        bits = u32(data + 16); group = u32(data + 20);
        rows = dimension(data + 24); cols = dimension(data + 32);
        sbits = u32(data + 40); block = u32(data + 44);
        require(bits >= 1 && bits <= 8 && group > 0 &&
                ((sbits == 16 && block == 0) || (sbits == 8 && block >= 2)),
                "unsupported native-v3 code/scale layout");
        groups = ceil_div(cols, group);
        scales = mul(rows, groups);
        const size_t code_bits = mul(mul(rows, cols), bits);
        words = ceil_div(code_bits, 32);
        metadata = sbits == 8 ? ceil_div(scales, block) : 0;
        const size_t code_bytes = mul(words, 4);
        const size_t scale_bytes = mul(scales, sbits / 8);
        const size_t metadata_bytes = mul(metadata, 2);
        const size_t payload = add(add(code_bytes, scale_bytes),
                                   add(mul(metadata_bytes, 2), (size_t(1) << bits) * 4));
        require(u64(data + 56) == payload && size == add(64, payload),
                "native-v3 payload length mismatch");
        codes = data + 64;
        scale_data = codes + code_bytes;
        offsets = scale_data + scale_bytes;
        steps = offsets + metadata_bytes;
        centroids = steps + metadata_bytes;
        if (code_bits % 32) {
            require((u32(codes + (words - 1) * 4) >> (code_bits % 32)) == 0,
                    "non-zero unused bits in final code word");
        }
        double max_centroid = 0;
        for (size_t i = 0; i < (size_t(1) << bits); ++i) {
            const float c = f32(centroids + i * 4);
            require(std::isfinite(c), "non-finite centroid");
            if (std::abs(double(c)) > max_centroid) { max_centroid = std::abs(double(c)); }
        }
        for (size_t i = 0; i < metadata; ++i) {
            const float offset = f16(offsets + i * 2), step = f16(steps + i * 2);
            require(std::isfinite(offset) && offset >= 0 && std::isfinite(step) && step >= 0,
                    "non-finite or negative affine metadata");
        }
        for (size_t i = 0; i < scales; ++i) {
            const float s = scale(i);
            require(std::isfinite(s) && s >= 0, "non-finite or negative scale");
            require(double(s) * max_centroid <= std::numeric_limits<float>::max(),
                    "decoded weights overflow float32");
        }
    }

    float scale(size_t index) const {
        if (sbits == 16) { return f16(scale_data + index * 2); }
        // This translation unit is compiled without FP contraction. Keeping
        // the multiply and add separate matches canonical PyTorch scale decode.
        const float product = float(scale_data[index]) * f16(steps + (index / block) * 2);
        return f16(offsets + (index / block) * 2) + product;
    }
    float weight(size_t row, size_t col) const {
        const size_t bit = (row * cols + col) * bits;
        const size_t shift = bit % 32;
        uint32_t code = u32(codes + (bit / 32) * 4) >> shift;
        if (shift + bits > 32) { code |= u32(codes + (bit / 32 + 1) * 4) << (32 - shift); }
        code &= (uint32_t(1) << bits) - 1;
        return f32(centroids + code * 4) * scale(row * groups + col / group);
    }
};

template<class Function> int checked(Function function) noexcept {
    last_error[0] = '\0';
    try { function(); return 0; }
    catch (const std::overflow_error &e) { std::snprintf(last_error, sizeof(last_error), "%s", e.what()); return 2; }
    catch (const std::invalid_argument &e) { std::snprintf(last_error, sizeof(last_error), "%s", e.what()); return 1; }
    catch (...) { std::snprintf(last_error, sizeof(last_error), "unexpected native-v3 error"); return 3; }
}
} // namespace

extern "C" {
uint32_t rq_native_v3_abi_version(void) { return 1; }
uint32_t rq_native_v3_format_version(void) { return 3; }
const char *rq_native_v3_last_error(void) { return last_error; }

int rq_native_v3_dequantize(const uint8_t *data, size_t data_size, float *output,
                          size_t output_size, size_t row_start, size_t row_count) {
    return checked([&] {
        const Matrix m(data, data_size);
        require(row_start < m.rows && row_count > 0 && row_count <= m.rows - row_start,
                "invalid row range");
        require(output && output_size == mul(row_count, m.cols), "invalid output size");
        aligned_float(output);
        disjoint(data, data_size, output, mul(output_size, sizeof(float)));
        for (size_t r = 0; r < row_count; ++r) {
            for (size_t c = 0; c < m.cols; ++c) { output[r * m.cols + c] = m.weight(row_start + r, c); }
        }
    });
}

int rq_native_v3_matmul(const uint8_t *data, size_t data_size, const float *input,
                      size_t input_size, size_t batch, float *output, size_t output_size) {
    return checked([&] {
        const Matrix m(data, data_size);
        require(input && output && batch > 0 && input_size == mul(batch, m.cols) &&
                output_size == mul(batch, m.rows), "invalid matmul shape or buffer");
        aligned_float(input); aligned_float(output);
        const size_t input_bytes = mul(input_size, sizeof(float));
        const size_t output_bytes = mul(output_size, sizeof(float));
        disjoint(data, data_size, output, output_bytes);
        disjoint(data, data_size, input, input_bytes);
        disjoint(input, input_bytes, output, output_bytes);
        for (size_t i = 0; i < input_size; ++i) { require(std::isfinite(input[i]), "non-finite input"); }
        for (size_t b = 0; b < batch; ++b) {
            for (size_t r = 0; r < m.rows; ++r) {
                float sum = 0;
                for (size_t c = 0; c < m.cols; ++c) { sum += input[b * m.cols + c] * m.weight(r, c); }
                output[b * m.rows + r] = sum;
            }
        }
    });
}
}
