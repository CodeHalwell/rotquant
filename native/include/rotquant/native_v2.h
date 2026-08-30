#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rotquant/native_export.h"

namespace rotquant::native_v2 {

constexpr std::uint32_t kFormatVersion = 2;
constexpr std::uint32_t kMinBits = 1;
constexpr std::uint32_t kMaxBits = 8;

struct ROTQUANT_NATIVE_API Layout {
    std::uint32_t bits = 4;
    std::uint32_t group_size = 128;

    std::size_t code_bytes_per_group() const;
    std::size_t bytes_per_group() const;
    std::size_t groups_for(std::size_t in_features) const;
    std::size_t row_bytes(std::size_t in_features) const;
};

enum class CpuKernel {
    automatic,
    scalar,
    neon,
    avx2,
};

struct KernelCapability {
    std::string name;
    CpuKernel kernel;
    std::uint32_t min_bits;
    std::uint32_t max_bits;
    // Zero means every positive group size is supported.
    std::uint32_t group_size;
};

ROTQUANT_NATIVE_API const char * kernel_name(CpuKernel kernel) noexcept;
ROTQUANT_NATIVE_API std::vector<KernelCapability> available_kernels();
ROTQUANT_NATIVE_API CpuKernel resolve_kernel(CpuKernel requested);

ROTQUANT_NATIVE_API float fp16_to_fp32(std::uint16_t value) noexcept;

ROTQUANT_NATIVE_API void dequantize(
    const std::uint8_t * qdata,
    std::size_t qdata_size,
    const float * codebook,
    std::size_t codebook_size,
    std::size_t out_features,
    std::size_t in_features,
    const Layout & layout,
    float * output,
    std::size_t output_size,
    CpuKernel kernel = CpuKernel::automatic);

ROTQUANT_NATIVE_API void matmul(
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
    CpuKernel kernel = CpuKernel::automatic);

}  // namespace rotquant::native_v2
