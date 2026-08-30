#include "rotquant/native_v2.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace rq = rotquant::native_v2;

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void write_code(
        std::uint8_t * codes,
        std::size_t element,
        std::uint32_t bits,
        std::uint32_t value) {
    const std::size_t bit_position = element * bits;
    const std::size_t byte_index = bit_position / 8;
    const std::uint32_t offset = static_cast<std::uint32_t>(bit_position % 8);
    codes[byte_index] |= static_cast<std::uint8_t>((value << offset) & 0xFFU);
    if (offset + bits > 8) {
        codes[byte_index + 1] |= static_cast<std::uint8_t>(value >> (8 - offset));
    }
}

std::vector<std::uint8_t> make_qdata(
        std::size_t out_features,
        std::size_t in_features,
        const rq::Layout & layout) {
    std::vector<std::uint8_t> qdata(
        out_features * layout.row_bytes(in_features), 0);
    const std::size_t groups = layout.groups_for(in_features);
    for (std::size_t row = 0; row < out_features; ++row) {
        for (std::size_t group = 0; group < groups; ++group) {
            std::uint8_t * block = qdata.data() +
                row * layout.row_bytes(in_features) +
                group * layout.bytes_per_group();
            const std::uint16_t scale = group % 2 == 0 ? 0x3C00U : 0x3800U;
            block[0] = static_cast<std::uint8_t>(scale & 0xFFU);
            block[1] = static_cast<std::uint8_t>(scale >> 8);
            std::uint8_t * codes = block + sizeof(std::uint16_t);
            for (std::size_t element = 0; element < layout.group_size; ++element) {
                const std::size_t column = group * layout.group_size + element;
                const std::uint32_t value = static_cast<std::uint32_t>(
                    (row * in_features + column) % (std::size_t{1} << layout.bits));
                write_code(codes, element, layout.bits, value);
            }
        }
    }
    return qdata;
}

void check_width(std::uint32_t bits, rq::CpuKernel kernel) {
    constexpr std::size_t out_features = 3;
    constexpr std::size_t in_features = 29;
    constexpr std::size_t batch = 2;
    const rq::Layout layout{bits, 13};
    const auto qdata = make_qdata(out_features, in_features, layout);
    std::vector<float> codebook(std::size_t{1} << bits);
    for (std::size_t index = 0; index < codebook.size(); ++index) {
        codebook[index] = static_cast<float>(index) -
            static_cast<float>(codebook.size()) / 2.0F;
    }
    std::vector<float> weight(out_features * in_features);
    rq::dequantize(
        qdata.data(), qdata.size(), codebook.data(), codebook.size(),
        out_features, in_features, layout, weight.data(), weight.size(), kernel);

    for (std::size_t row = 0; row < out_features; ++row) {
        for (std::size_t column = 0; column < in_features; ++column) {
            const std::size_t group = column / layout.group_size;
            const float scale = group % 2 == 0 ? 1.0F : 0.5F;
            const std::size_t code =
                (row * in_features + column) % codebook.size();
            require(
                weight[row * in_features + column] == scale * codebook[code],
                "dequantized weight mismatch");
        }
    }

    std::vector<float> input(batch * in_features);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 7.0F;
    }
    std::vector<float> output(batch * out_features);
    rq::matmul(
        input.data(), input.size(), batch,
        qdata.data(), qdata.size(), codebook.data(), codebook.size(),
        out_features, in_features, layout, output.data(), output.size(), kernel);
    for (std::size_t item = 0; item < batch; ++item) {
        for (std::size_t row = 0; row < out_features; ++row) {
            float expected = 0.0F;
            for (std::size_t column = 0; column < in_features; ++column) {
                expected += input[item * in_features + column] *
                    weight[row * in_features + column];
            }
            const float tolerance = 2e-5F + std::abs(expected) * 2e-6F;
            require(
                std::abs(output[item * out_features + row] - expected) < tolerance,
                "streaming matmul mismatch");
        }
    }
}

}  // namespace

int main() {
    try {
        require(rq::fp16_to_fp32(0x3C00U) == 1.0F, "fp16 one mismatch");
        require(rq::fp16_to_fp32(0x3800U) == 0.5F, "fp16 half mismatch");
        require(rq::fp16_to_fp32(0xBC00U) == -1.0F, "fp16 negative mismatch");
        const auto kernels = rq::available_kernels();
        for (const auto & kernel : kernels) {
            for (std::uint32_t bits = rq::kMinBits; bits <= rq::kMaxBits; ++bits) {
                check_width(bits, kernel.kernel);
            }
        }
        bool scalar_found = false;
        bool avx2_found = false;
        for (const auto & kernel : kernels) {
            if (kernel.kernel == rq::CpuKernel::scalar) scalar_found = true;
            if (kernel.kernel == rq::CpuKernel::avx2) avx2_found = true;
            require(kernel.min_bits == 1, "kernel minimum width mismatch");
            require(kernel.max_bits == 8, "kernel maximum width mismatch");
        }
        require(scalar_found, "scalar kernel missing");
        if (avx2_found) {
            require(
                rq::resolve_kernel(rq::CpuKernel::avx2) == rq::CpuKernel::avx2,
                "reported AVX2 kernel did not resolve");
        } else {
            bool rejected = false;
            try {
                static_cast<void>(rq::resolve_kernel(rq::CpuKernel::avx2));
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            require(rejected, "unavailable AVX2 kernel did not fail closed");
        }
        std::cout << "PASS: native v2 compiled-kernel conformance for bits 1..8\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
