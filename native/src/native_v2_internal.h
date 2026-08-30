#pragma once

#include "rotquant/native_v2.h"

namespace rotquant::native_v2::detail {

#if defined(ROTQUANT_NATIVE_COMPILED_AVX2)

void dequantize_avx2(
    const std::uint8_t * qdata,
    const float * codebook,
    std::size_t out_features,
    std::size_t in_features,
    const Layout & layout,
    float * output);

void matmul_avx2(
    const float * input,
    std::size_t batch,
    const std::uint8_t * qdata,
    const float * codebook,
    std::size_t out_features,
    std::size_t in_features,
    const Layout & layout,
    float * output);

#endif

}  // namespace rotquant::native_v2::detail
