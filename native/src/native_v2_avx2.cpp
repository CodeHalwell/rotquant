#include "native_v2_internal.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <vector>

// AVX2 kernels for native-v2 blocks.
//
// The arithmetic is the reference's up to floating-point reassociation: a
// group's dot product is accumulated in eight lanes and multiplied by the
// group scale once, instead of scaling every centroid first. Three things
// make it faster than the first AVX2 version:
//
//   * codes are split into lanes with variable shifts on one unaligned word
//     load, not eight scalar shift/mask/insert sequences;
//   * codebooks of up to sixteen entries are looked up with register
//     permutes, which cost a fraction of a memory gather;
//   * with more than one input row the packed row is decoded once into a
//     small buffer and every input row then takes a dense dot product with
//     it, so decode work no longer scales with the batch.

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

// Eight consecutive codes occupy exactly `Bits` bytes and start on a byte
// boundary whenever the first element index is a multiple of eight. A full
// eight-byte load is used when it stays inside the tensor; the bytes beyond
// the eight codes are masked off later. x86 is little-endian, so the first
// code sits in the low bits.
template <std::uint32_t Bits>
std::uint64_t load_codes8(
        const std::uint8_t * source,
        const std::uint8_t * last_wide_load) noexcept {
    std::uint64_t packed = 0;
    if (Bits == 8 || source <= last_wide_load) {
        std::memcpy(&packed, source, sizeof(packed));
        return packed;
    }
    for (std::size_t byte = 0; byte < Bits; ++byte) {
        packed |= static_cast<std::uint64_t>(source[byte]) << (byte * 8);
    }
    return packed;
}

template <std::uint32_t Bits>
__m256i split_codes8(std::uint64_t packed) noexcept {
    constexpr int bits = static_cast<int>(Bits);
    if constexpr (Bits == 8) {
        return _mm256_cvtepu8_epi32(
            _mm_cvtsi64_si128(static_cast<long long>(packed)));
    } else if constexpr (Bits <= 4) {
        // All eight codes live in the low 32 bits.
        const __m256i shifts = _mm256_setr_epi32(
            0, bits, 2 * bits, 3 * bits, 4 * bits, 5 * bits, 6 * bits, 7 * bits);
        const __m256i word = _mm256_set1_epi32(
            static_cast<int>(static_cast<std::uint32_t>(packed)));
        return _mm256_and_si256(
            _mm256_srlv_epi32(word, shifts),
            _mm256_set1_epi32((1 << bits) - 1));
    } else {
        const __m256i word = _mm256_set1_epi64x(static_cast<long long>(packed));
        const __m256i mask = _mm256_set1_epi64x((1LL << bits) - 1);
        const __m256i low = _mm256_and_si256(
            _mm256_srlv_epi64(
                word, _mm256_setr_epi64x(0, bits, 2 * bits, 3 * bits)),
            mask);
        const __m256i high = _mm256_and_si256(
            _mm256_srlv_epi64(
                word,
                _mm256_setr_epi64x(4 * bits, 5 * bits, 6 * bits, 7 * bits)),
            mask);
        // 32-bit lanes are now [c0 c4 c1 c5 c2 c6 c3 c7]; restore the order.
        const __m256i mixed = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
        return _mm256_permutevar8x32_epi32(
            mixed, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    }
}

struct Codebook {
    const float * values;
    __m256 low;
    __m256 high;
};

template <std::uint32_t Bits>
Codebook load_codebook(const float * codebook) noexcept {
    Codebook table{codebook, _mm256_setzero_ps(), _mm256_setzero_ps()};
    if constexpr (Bits <= 4) {
        float padded[16] = {};
        std::copy(codebook, codebook + (std::size_t{1} << Bits), padded);
        table.low = _mm256_loadu_ps(padded);
        table.high = _mm256_loadu_ps(padded + 8);
    }
    return table;
}

template <std::uint32_t Bits>
__m256 lookup8(const Codebook & table, __m256i indices) noexcept {
    if constexpr (Bits <= 3) {
        return _mm256_permutevar8x32_ps(table.low, indices);
    } else if constexpr (Bits == 4) {
        // The permute reads only the low three index bits; bit three picks
        // the half. blendv selects on the sign bit, so move bit three there.
        const __m256 low = _mm256_permutevar8x32_ps(table.low, indices);
        const __m256 high = _mm256_permutevar8x32_ps(table.high, indices);
        return _mm256_blendv_ps(
            low, high, _mm256_castsi256_ps(_mm256_slli_epi32(indices, 28)));
    } else {
        return _mm256_i32gather_ps(table.values, indices, sizeof(float));
    }
}

template <std::uint32_t Bits>
__m256 centers8(
        const Codebook & table,
        const std::uint8_t * codes,
        std::size_t element,
        const std::uint8_t * last_wide_load) noexcept {
    return lookup8<Bits>(
        table,
        split_codes8<Bits>(
            load_codes8<Bits>(codes + element * Bits / 8, last_wide_load)));
}

float horizontal_sum(__m256 value) noexcept {
    const __m128 halves = _mm_add_ps(
        _mm256_castps256_ps128(value),
        _mm256_extractf128_ps(value, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

// Write one row of scaled centroids, scale * codebook[code], to `output`.
template <std::uint32_t Bits>
void decode_row(
        const std::uint8_t * row_data,
        const Codebook & table,
        std::size_t in_features,
        const Layout & layout,
        std::size_t groups,
        std::size_t block_bytes,
        const std::uint8_t * last_wide_load,
        float * output) noexcept {
    for (std::size_t group = 0; group < groups; ++group) {
        const std::uint8_t * block = row_data + group * block_bytes;
        const std::uint8_t * codes = block + sizeof(std::uint16_t);
        const float scale = read_scale(block);
        const __m256 scale_lanes = _mm256_set1_ps(scale);
        const std::size_t start = group * layout.group_size;
        const std::size_t width = std::min(
            static_cast<std::size_t>(layout.group_size), in_features - start);
        std::size_t element = 0;
        for (; element + 8 <= width; element += 8) {
            _mm256_storeu_ps(
                output + start + element,
                _mm256_mul_ps(
                    centers8<Bits>(table, codes, element, last_wide_load),
                    scale_lanes));
        }
        for (; element < width; ++element) {
            output[start + element] =
                scale * table.values[read_code_fixed<Bits>(codes, element)];
        }
    }
}

float dense_dot(const float * left, const float * right, std::size_t size) noexcept {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 32 <= size; index += 32) {
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
            _mm256_loadu_ps(left + index), _mm256_loadu_ps(right + index)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
            _mm256_loadu_ps(left + index + 8), _mm256_loadu_ps(right + index + 8)));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
            _mm256_loadu_ps(left + index + 16), _mm256_loadu_ps(right + index + 16)));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
            _mm256_loadu_ps(left + index + 24), _mm256_loadu_ps(right + index + 24)));
    }
    for (; index + 8 <= size; index += 8) {
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
            _mm256_loadu_ps(left + index), _mm256_loadu_ps(right + index)));
    }
    float tail = 0.0F;
    for (; index < size; ++index) {
        tail += left[index] * right[index];
    }
    return horizontal_sum(_mm256_add_ps(
        _mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3))) + tail;
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
    const std::uint8_t * last_wide_load =
        qdata + out_features * row_bytes - sizeof(std::uint64_t);
    const Codebook table = load_codebook<Bits>(codebook);
    for (std::size_t row = 0; row < out_features; ++row) {
        decode_row<Bits>(
            qdata + row * row_bytes, table, in_features, layout, groups,
            block_bytes, last_wide_load, output + row * in_features);
    }
}

// One input row: decode and accumulate in the same pass.
template <std::uint32_t Bits>
float fused_row_dot(
        const float * input_row,
        const std::uint8_t * row_data,
        const Codebook & table,
        std::size_t in_features,
        const Layout & layout,
        std::size_t groups,
        std::size_t block_bytes,
        const std::uint8_t * last_wide_load) noexcept {
    __m256 total = _mm256_setzero_ps();
    float tail = 0.0F;
    for (std::size_t group = 0; group < groups; ++group) {
        const std::uint8_t * block = row_data + group * block_bytes;
        const std::uint8_t * codes = block + sizeof(std::uint16_t);
        const float scale = read_scale(block);
        const std::size_t start = group * layout.group_size;
        const std::size_t width = std::min(
            static_cast<std::size_t>(layout.group_size), in_features - start);
        const float * activation = input_row + start;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        std::size_t element = 0;
        for (; element + 16 <= width; element += 16) {
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                centers8<Bits>(table, codes, element, last_wide_load),
                _mm256_loadu_ps(activation + element)));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                centers8<Bits>(table, codes, element + 8, last_wide_load),
                _mm256_loadu_ps(activation + element + 8)));
        }
        for (; element + 8 <= width; element += 8) {
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                centers8<Bits>(table, codes, element, last_wide_load),
                _mm256_loadu_ps(activation + element)));
        }
        float group_tail = 0.0F;
        for (; element < width; ++element) {
            group_tail += activation[element] *
                table.values[read_code_fixed<Bits>(codes, element)];
        }
        total = _mm256_add_ps(total, _mm256_mul_ps(
            _mm256_set1_ps(scale), _mm256_add_ps(sum0, sum1)));
        tail += scale * group_tail;
    }
    return horizontal_sum(total) + tail;
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
    const std::uint8_t * last_wide_load =
        qdata + out_features * row_bytes - sizeof(std::uint64_t);
    const Codebook table = load_codebook<Bits>(codebook);
    if (batch == 1) {
        for (std::size_t row = 0; row < out_features; ++row) {
            output[row] = fused_row_dot<Bits>(
                input, qdata + row * row_bytes, table, in_features, layout,
                groups, block_bytes, last_wide_load);
        }
        return;
    }
    // Decode each packed row once; it stays in L1 while every input row
    // takes its dot product with it.
    std::vector<float> decoded(in_features);
    for (std::size_t row = 0; row < out_features; ++row) {
        decode_row<Bits>(
            qdata + row * row_bytes, table, in_features, layout, groups,
            block_bytes, last_wide_load, decoded.data());
        for (std::size_t item = 0; item < batch; ++item) {
            output[item * out_features + row] = dense_dot(
                input + item * in_features, decoded.data(), in_features);
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
