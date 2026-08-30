#include "native_v2_internal.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

namespace rotquant::native_v2::detail {
namespace {

template <std::uint32_t Bits>
std::uint32_t read_code_fixed(
        const std::uint8_t * codes,
        std::size_t element) noexcept {
    static_assert(Bits >= kMinBits && Bits <= kMaxBits);
    if constexpr (Bits == 8) {
        return codes[element];
    } else {
        constexpr std::uint32_t mask = (std::uint32_t{1} << Bits) - 1;
        const std::size_t bit_position = element * Bits;
        const std::size_t byte_index = bit_position / 8;
        const std::uint32_t offset =
            static_cast<std::uint32_t>(bit_position % 8);
        std::uint32_t value =
            static_cast<std::uint32_t>(codes[byte_index]) >> offset;
        if (offset + Bits > 8) {
            value |= static_cast<std::uint32_t>(codes[byte_index + 1]) <<
                (8 - offset);
        }
        return value & mask;
    }
}

float read_scale(const std::uint8_t * block) noexcept {
    const std::uint16_t value = static_cast<std::uint16_t>(block[0]) |
        (static_cast<std::uint16_t>(block[1]) << 8);
    return fp16_to_fp32(value);
}

template <std::uint32_t Bits>
__m256 read_codebook8_fixed(
        const std::uint8_t * codes,
        std::size_t element,
        const float * codebook) noexcept {
    constexpr std::uint64_t mask = (std::uint64_t{1} << Bits) - 1;
    const std::size_t byte_index = element * Bits / 8;
    std::uint64_t packed = 0;
    for (std::size_t byte = 0; byte < Bits; ++byte) {
        packed |= static_cast<std::uint64_t>(codes[byte_index + byte]) <<
            (byte * 8);
    }
    const __m256i indices = _mm256_setr_epi32(
        static_cast<int>((packed >> (0 * Bits)) & mask),
        static_cast<int>((packed >> (1 * Bits)) & mask),
        static_cast<int>((packed >> (2 * Bits)) & mask),
        static_cast<int>((packed >> (3 * Bits)) & mask),
        static_cast<int>((packed >> (4 * Bits)) & mask),
        static_cast<int>((packed >> (5 * Bits)) & mask),
        static_cast<int>((packed >> (6 * Bits)) & mask),
        static_cast<int>((packed >> (7 * Bits)) & mask));
    return _mm256_i32gather_ps(codebook, indices, sizeof(float));
}

float horizontal_sum(__m256 value) noexcept {
    const __m128 halves = _mm_add_ps(
        _mm256_castps256_ps128(value),
        _mm256_extractf128_ps(value, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

template <std::uint32_t Bits>
void dequantize_avx2_fixed(
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    const std::size_t row_bytes = layout.row_bytes(in_features);
    const std::size_t groups = layout.groups_for(in_features);
    const std::size_t block_bytes = layout.bytes_per_group();
    for (std::size_t row = 0; row < out_features; ++row) {
        const std::uint8_t * row_data = qdata + row * row_bytes;
        for (std::size_t group = 0; group < groups; ++group) {
            const std::uint8_t * block = row_data + group * block_bytes;
            const std::uint8_t * codes = block + sizeof(std::uint16_t);
            const float scale = read_scale(block);
            const std::size_t start = group * layout.group_size;
            const std::size_t width = std::min(
                static_cast<std::size_t>(layout.group_size),
                in_features - start);
            std::size_t element = 0;
            for (; element + 8 <= width; element += 8) {
                const __m256 centers =
                    read_codebook8_fixed<Bits>(codes, element, codebook);
                _mm256_storeu_ps(
                    output + row * in_features + start + element,
                    _mm256_mul_ps(centers, _mm256_set1_ps(scale)));
            }
            for (; element < width; ++element) {
                output[row * in_features + start + element] =
                    scale * codebook[read_code_fixed<Bits>(codes, element)];
            }
        }
    }
}

template <std::uint32_t Bits>
void matmul_avx2_fixed(
        const float * input,
        std::size_t batch,
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    const std::size_t row_bytes = layout.row_bytes(in_features);
    const std::size_t groups = layout.groups_for(in_features);
    const std::size_t block_bytes = layout.bytes_per_group();
    for (std::size_t item = 0; item < batch; ++item) {
        const float * input_row = input + item * in_features;
        for (std::size_t row = 0; row < out_features; ++row) {
            const std::uint8_t * row_data = qdata + row * row_bytes;
            __m256 vector_sum = _mm256_setzero_ps();
            float scalar_sum = 0.0F;
            for (std::size_t group = 0; group < groups; ++group) {
                const std::uint8_t * block = row_data + group * block_bytes;
                const std::uint8_t * codes = block + sizeof(std::uint16_t);
                const float scale = read_scale(block);
                const std::size_t start = group * layout.group_size;
                const std::size_t width = std::min(
                    static_cast<std::size_t>(layout.group_size),
                    in_features - start);
                std::size_t element = 0;
                for (; element + 8 <= width; element += 8) {
                    const __m256 centers =
                        read_codebook8_fixed<Bits>(codes, element, codebook);
                    const __m256 scaled_centers = _mm256_mul_ps(
                        centers, _mm256_set1_ps(scale));
                    vector_sum = _mm256_add_ps(
                        vector_sum,
                        _mm256_mul_ps(
                            _mm256_loadu_ps(input_row + start + element),
                            scaled_centers));
                }
                for (; element < width; ++element) {
                    scalar_sum += input_row[start + element] * scale *
                        codebook[read_code_fixed<Bits>(codes, element)];
                }
            }
            output[item * out_features + row] =
                horizontal_sum(vector_sum) + scalar_sum;
        }
    }
}

}  // namespace

void dequantize_avx2(
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    switch (layout.bits) {
        case 1: return dequantize_avx2_fixed<1>(
            qdata, codebook, out_features, in_features, layout, output);
        case 2: return dequantize_avx2_fixed<2>(
            qdata, codebook, out_features, in_features, layout, output);
        case 3: return dequantize_avx2_fixed<3>(
            qdata, codebook, out_features, in_features, layout, output);
        case 4: return dequantize_avx2_fixed<4>(
            qdata, codebook, out_features, in_features, layout, output);
        case 5: return dequantize_avx2_fixed<5>(
            qdata, codebook, out_features, in_features, layout, output);
        case 6: return dequantize_avx2_fixed<6>(
            qdata, codebook, out_features, in_features, layout, output);
        case 7: return dequantize_avx2_fixed<7>(
            qdata, codebook, out_features, in_features, layout, output);
        case 8: return dequantize_avx2_fixed<8>(
            qdata, codebook, out_features, in_features, layout, output);
        default:
            throw std::invalid_argument("native v2 AVX2 bits must be in [1, 8]");
    }
}

void matmul_avx2(
        const float * input,
        std::size_t batch,
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    switch (layout.bits) {
        case 1: return matmul_avx2_fixed<1>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 2: return matmul_avx2_fixed<2>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 3: return matmul_avx2_fixed<3>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 4: return matmul_avx2_fixed<4>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 5: return matmul_avx2_fixed<5>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 6: return matmul_avx2_fixed<6>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 7: return matmul_avx2_fixed<7>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        case 8: return matmul_avx2_fixed<8>(
            input, batch, qdata, codebook,
            out_features, in_features, layout, output);
        default:
            throw std::invalid_argument("native v2 AVX2 bits must be in [1, 8]");
    }
}

}  // namespace rotquant::native_v2::detail
