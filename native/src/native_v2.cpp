#include "rotquant/native_v2.h"

#include "native_v2_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(ROTQUANT_NATIVE_COMPILED_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ROTQUANT_NATIVE_HAS_NEON 1
#else
#define ROTQUANT_NATIVE_HAS_NEON 0
#endif

namespace rotquant::native_v2 {
namespace {

#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
bool cpu_supports_avx2() noexcept {
#if defined(_MSC_VER)
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 0);
    if (registers[0] < 7) return false;
    __cpuidex(registers, 1, 0);
    constexpr int osxsave_bit = 1 << 27;
    constexpr int avx_bit = 1 << 28;
    if ((registers[2] & (osxsave_bit | avx_bit)) !=
            (osxsave_bit | avx_bit)) return false;
    if ((_xgetbv(0) & 0x6U) != 0x6U) return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
#endif

std::size_t checked_multiply(
        std::size_t left,
        std::size_t right,
        const char * description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
            std::string("native v2 ") + description + " size overflows size_t");
    }
    return left * right;
}

void validate_layout(const Layout & layout) {
    if (layout.bits < kMinBits || layout.bits > kMaxBits) {
        throw std::invalid_argument("native v2 bits must be in [1, 8]");
    }
    if (layout.group_size == 0) {
        throw std::invalid_argument("native v2 group_size must be positive");
    }
}

void validate_matrix(
        const std::uint8_t * qdata,
        std::size_t qdata_size,
        const float * codebook,
        std::size_t codebook_size,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout) {
    validate_layout(layout);
    if (qdata == nullptr || codebook == nullptr) {
        throw std::invalid_argument("native v2 matrix pointers must not be null");
    }
    if (out_features == 0 || in_features == 0) {
        throw std::invalid_argument("native v2 matrix dimensions must be positive");
    }
    if (codebook_size != (std::size_t{1} << layout.bits)) {
        throw std::invalid_argument("native v2 codebook size must equal 2**bits");
    }
    for (std::size_t index = 0; index < codebook_size; ++index) {
        if (!std::isfinite(codebook[index])) {
            throw std::invalid_argument("native v2 codebook values must be finite");
        }
    }
    const std::size_t expected_qdata = checked_multiply(
        out_features, layout.row_bytes(in_features), "qdata");
    if (qdata_size != expected_qdata) {
        throw std::invalid_argument("native v2 qdata size does not match dimensions");
    }
    const std::size_t row_bytes = layout.row_bytes(in_features);
    const std::size_t groups = layout.groups_for(in_features);
    const std::size_t block_bytes = layout.bytes_per_group();
    for (std::size_t row = 0; row < out_features; ++row) {
        for (std::size_t group = 0; group < groups; ++group) {
            const std::uint8_t * block =
                qdata + row * row_bytes + group * block_bytes;
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(block[0]) |
                (static_cast<std::uint16_t>(block[1]) << 8);
            const float scale = fp16_to_fp32(scale_bits);
            if (!std::isfinite(scale) || scale < 0.0F) {
                throw std::invalid_argument(
                    "native v2 scales must be finite and non-negative");
            }
        }
    }
}

std::uint32_t read_code(
        const std::uint8_t * codes,
        std::size_t element,
        std::uint32_t bits) noexcept {
    const std::size_t bit_position = element * bits;
    const std::size_t byte_index = bit_position / 8;
    const std::uint32_t offset = static_cast<std::uint32_t>(bit_position % 8);
    std::uint32_t value = static_cast<std::uint32_t>(codes[byte_index]) >> offset;
    if (offset + bits > 8) {
        value |= static_cast<std::uint32_t>(codes[byte_index + 1]) << (8 - offset);
    }
    return value & ((std::uint32_t{1} << bits) - 1);
}

float read_scale(const std::uint8_t * block) noexcept {
    const std::uint16_t value = static_cast<std::uint16_t>(block[0]) |
        (static_cast<std::uint16_t>(block[1]) << 8);
    return fp16_to_fp32(value);
}

void dequantize_scalar(
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
            for (std::size_t element = 0; element < width; ++element) {
                output[row * in_features + start + element] =
                    scale * codebook[read_code(codes, element, layout.bits)];
            }
        }
    }
}

void matmul_scalar(
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
            float accumulator = 0.0F;
            for (std::size_t group = 0; group < groups; ++group) {
                const std::uint8_t * block = row_data + group * block_bytes;
                const std::uint8_t * codes = block + sizeof(std::uint16_t);
                const float scale = read_scale(block);
                const std::size_t start = group * layout.group_size;
                const std::size_t width = std::min(
                    static_cast<std::size_t>(layout.group_size),
                    in_features - start);
                float group_sum = 0.0F;
                for (std::size_t element = 0; element < width; ++element) {
                    group_sum += input_row[start + element] *
                        codebook[read_code(codes, element, layout.bits)];
                }
                accumulator += scale * group_sum;
            }
            output[item * out_features + row] = accumulator;
        }
    }
}

#if ROTQUANT_NATIVE_HAS_NEON

template <std::uint32_t Bits>
std::uint32_t read_code_fixed(
        const std::uint8_t * codes,
        std::size_t element) noexcept {
    static_assert(Bits >= kMinBits && Bits <= kMaxBits);
    if constexpr (Bits == 8) {
        return codes[element];
    } else if constexpr (Bits == 4) {
        return (codes[element / 2] >> ((element % 2) * 4)) & 0x0FU;
    } else if constexpr (Bits == 2) {
        return (codes[element / 4] >> ((element % 4) * 2)) & 0x03U;
    } else if constexpr (Bits == 1) {
        return (codes[element / 8] >> (element % 8)) & 0x01U;
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

template <std::uint32_t Bits>
float32x4_t read_codebook4_fixed(
        const std::uint8_t * codes,
        std::size_t element,
        const float * codebook) noexcept {
    constexpr std::size_t byte_count = (Bits + 1) / 2;
    constexpr std::uint32_t mask = (std::uint32_t{1} << Bits) - 1;
    const std::size_t bit_position = element * Bits;
    const std::size_t byte_index = bit_position / 8;
    const std::uint32_t bit_offset =
        static_cast<std::uint32_t>(bit_position % 8);
    std::uint32_t packed = 0;
    for (std::size_t byte = 0; byte < byte_count; ++byte) {
        packed |= static_cast<std::uint32_t>(codes[byte_index + byte]) <<
            (byte * 8);
    }
    packed >>= bit_offset;
    alignas(16) float values[4];
    for (std::size_t lane = 0; lane < 4; ++lane) {
        values[lane] = codebook[(packed >> (lane * Bits)) & mask];
    }
    return vld1q_f32(values);
}

float horizontal_sum(float32x4_t value) noexcept {
    const float32x2_t pair = vadd_f32(vget_low_f32(value), vget_high_f32(value));
    return vget_lane_f32(vpadd_f32(pair, pair), 0);
}

template <std::uint32_t Bits>
void dequantize_neon_fixed(
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
            for (; element + 4 <= width; element += 4) {
                const float32x4_t centers =
                    read_codebook4_fixed<Bits>(codes, element, codebook);
                vst1q_f32(
                    output + row * in_features + start + element,
                    vmulq_n_f32(centers, scale));
            }
            for (; element < width; ++element) {
                output[row * in_features + start + element] =
                    scale * codebook[read_code_fixed<Bits>(codes, element)];
            }
        }
    }
}

template <std::uint32_t Bits>
void matmul_neon_fixed(
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
            float32x4_t vector_sum = vdupq_n_f32(0.0F);
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
                for (; element + 4 <= width; element += 4) {
                    const float32x4_t centers =
                        read_codebook4_fixed<Bits>(codes, element, codebook);
                    const float32x4_t scaled_centers = vmulq_n_f32(centers, scale);
                    vector_sum = vmlaq_f32(
                        vector_sum,
                        vld1q_f32(input_row + start + element),
                        scaled_centers);
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

void dequantize_neon(
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    switch (layout.bits) {
        case 1:
            return dequantize_neon_fixed<1>(
                qdata, codebook, out_features, in_features, layout, output);
        case 2:
            return dequantize_neon_fixed<2>(
                qdata, codebook, out_features, in_features, layout, output);
        case 3:
            return dequantize_neon_fixed<3>(
                qdata, codebook, out_features, in_features, layout, output);
        case 4:
            return dequantize_neon_fixed<4>(
                qdata, codebook, out_features, in_features, layout, output);
        case 5:
            return dequantize_neon_fixed<5>(
                qdata, codebook, out_features, in_features, layout, output);
        case 6:
            return dequantize_neon_fixed<6>(
                qdata, codebook, out_features, in_features, layout, output);
        case 7:
            return dequantize_neon_fixed<7>(
                qdata, codebook, out_features, in_features, layout, output);
        case 8:
            return dequantize_neon_fixed<8>(
                qdata, codebook, out_features, in_features, layout, output);
        default:
            throw std::invalid_argument("native v2 NEON bits must be in [1, 8]");
    }
}

void matmul_neon(
        const float * input,
        std::size_t batch,
        const std::uint8_t * qdata,
        const float * codebook,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output) {
    switch (layout.bits) {
        case 1:
            return matmul_neon_fixed<1>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 2:
            return matmul_neon_fixed<2>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 3:
            return matmul_neon_fixed<3>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 4:
            return matmul_neon_fixed<4>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 5:
            return matmul_neon_fixed<5>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 6:
            return matmul_neon_fixed<6>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 7:
            return matmul_neon_fixed<7>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        case 8:
            return matmul_neon_fixed<8>(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
        default:
            throw std::invalid_argument("native v2 NEON bits must be in [1, 8]");
    }
}

#endif

}  // namespace

std::size_t Layout::code_bytes_per_group() const {
    validate_layout(*this);
    const std::size_t code_bits = checked_multiply(
        static_cast<std::size_t>(group_size), bits, "group code");
    return code_bits / 8 + static_cast<std::size_t>(code_bits % 8 != 0);
}

std::size_t Layout::bytes_per_group() const {
    return sizeof(std::uint16_t) + code_bytes_per_group();
}

std::size_t Layout::groups_for(std::size_t in_features) const {
    validate_layout(*this);
    if (in_features == 0) {
        throw std::invalid_argument("native v2 in_features must be positive");
    }
    return 1 + (in_features - 1) / group_size;
}

std::size_t Layout::row_bytes(std::size_t in_features) const {
    return checked_multiply(
        groups_for(in_features), bytes_per_group(), "row");
}

const char * kernel_name(CpuKernel kernel) noexcept {
    switch (kernel) {
        case CpuKernel::automatic: return "auto";
        case CpuKernel::scalar: return "scalar";
        case CpuKernel::neon: return "neon";
        case CpuKernel::avx2: return "avx2";
    }
    return "unknown";
}

std::vector<KernelCapability> available_kernels() {
    std::vector<KernelCapability> kernels = {
        {"portable-scalar", CpuKernel::scalar, kMinBits, kMaxBits, 0},
    };
#if ROTQUANT_NATIVE_HAS_NEON
    kernels.push_back({"arm-neon", CpuKernel::neon, kMinBits, kMaxBits, 0});
#endif
#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
    if (cpu_supports_avx2()) {
        kernels.push_back({"x86-avx2", CpuKernel::avx2, kMinBits, kMaxBits, 0});
    }
#endif
    return kernels;
}

CpuKernel resolve_kernel(CpuKernel requested) {
    if (requested == CpuKernel::automatic) {
#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
        if (cpu_supports_avx2()) return CpuKernel::avx2;
#endif
#if ROTQUANT_NATIVE_HAS_NEON
        return CpuKernel::neon;
#else
        return CpuKernel::scalar;
#endif
    }
    if (requested == CpuKernel::scalar) {
        return requested;
    }
#if ROTQUANT_NATIVE_HAS_NEON
    if (requested == CpuKernel::neon) {
        return requested;
    }
#endif
#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
    if (requested == CpuKernel::avx2 && cpu_supports_avx2()) {
        return requested;
    }
#endif
    throw std::runtime_error(
        std::string("requested RotQuant CPU kernel is not compiled: ") +
        kernel_name(requested));
}

float fp16_to_fp32(std::uint16_t value) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000U) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::int32_t normalized_exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --normalized_exponent;
            }
            mantissa &= 0x03FFU;
            result = sign |
                (static_cast<std::uint32_t>(normalized_exponent + 112) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 0x1FU) {
        result = sign | 0x7F800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float output = 0.0F;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

void dequantize(
        const std::uint8_t * qdata,
        std::size_t qdata_size,
        const float * codebook,
        std::size_t codebook_size,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output,
        std::size_t output_size,
        CpuKernel kernel) {
    validate_matrix(
        qdata, qdata_size, codebook, codebook_size,
        out_features, in_features, layout);
    const std::size_t expected_output = checked_multiply(
        out_features, in_features, "dequantize output");
    if (output == nullptr || output_size != expected_output) {
        throw std::invalid_argument("native v2 dequantize output size is invalid");
    }
    switch (resolve_kernel(kernel)) {
        case CpuKernel::scalar:
            dequantize_scalar(
                qdata, codebook, out_features, in_features, layout, output);
            return;
#if ROTQUANT_NATIVE_HAS_NEON
        case CpuKernel::neon:
            dequantize_neon(
                qdata, codebook, out_features, in_features, layout, output);
            return;
#endif
#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
        case CpuKernel::avx2:
            detail::dequantize_avx2(
                qdata, codebook, out_features, in_features, layout, output);
            return;
#endif
        default:
            throw std::runtime_error("resolved unsupported dequantize kernel");
    }
}

void matmul(
        const float * input,
        std::size_t input_size,
        std::size_t batch,
        const std::uint8_t * qdata,
        std::size_t qdata_size,
        const float * codebook,
        std::size_t codebook_size,
        std::size_t out_features,
        std::size_t in_features,
        const Layout & layout,
        float * output,
        std::size_t output_size,
        CpuKernel kernel) {
    validate_matrix(
        qdata, qdata_size, codebook, codebook_size,
        out_features, in_features, layout);
    if (batch == 0) {
        throw std::invalid_argument("native v2 matmul batch must be positive");
    }
    const std::size_t expected_input = checked_multiply(
        batch, in_features, "matmul input");
    const std::size_t expected_output = checked_multiply(
        batch, out_features, "matmul output");
    if (input == nullptr || input_size != expected_input) {
        throw std::invalid_argument("native v2 matmul input size is invalid");
    }
    if (output == nullptr || output_size != expected_output) {
        throw std::invalid_argument("native v2 matmul output size is invalid");
    }
    switch (resolve_kernel(kernel)) {
        case CpuKernel::scalar:
            matmul_scalar(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
            return;
#if ROTQUANT_NATIVE_HAS_NEON
        case CpuKernel::neon:
            matmul_neon(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
            return;
#endif
#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)
        case CpuKernel::avx2:
            detail::matmul_avx2(
                input, batch, qdata, codebook,
                out_features, in_features, layout, output);
            return;
#endif
        default:
            throw std::runtime_error("resolved unsupported matmul kernel");
    }
}

}  // namespace rotquant::native_v2
