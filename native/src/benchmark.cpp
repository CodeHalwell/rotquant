#include "rotquant/native_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rq = rotquant::native_v2;

namespace {

struct Options {
    std::size_t out_features = 128;
    std::size_t in_features = 512;
    std::size_t group_size = 128;
    std::size_t batch = 4;
    std::size_t iterations = 3;
    rq::CpuKernel kernel = rq::CpuKernel::automatic;
};

rq::CpuKernel parse_kernel(const std::string & value) {
    if (value == "auto") return rq::CpuKernel::automatic;
    if (value == "scalar") return rq::CpuKernel::scalar;
    if (value == "neon") return rq::CpuKernel::neon;
    if (value == "avx2") return rq::CpuKernel::avx2;
    throw std::invalid_argument("unknown kernel: " + value);
}

std::size_t positive_size(const std::string & text, const std::string & key) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(key + " must be a positive integer");
    }
    const unsigned long long parsed = std::stoull(text);
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(key + " must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t checked_product(
        std::size_t left, std::size_t right, const char * description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(description) + " size overflows size_t");
    }
    return left * right;
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("benchmark options require values");
        }
        const std::string text = argv[++index];
        if (key == "--kernel") {
            options.kernel = parse_kernel(text);
            continue;
        }
        const std::size_t value = positive_size(text, key);
        if (key == "--out-features") options.out_features = value;
        else if (key == "--in-features") options.in_features = value;
        else if (key == "--group-size") options.group_size = value;
        else if (key == "--batch") options.batch = value;
        else if (key == "--iterations") options.iterations = value;
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (options.group_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--group-size exceeds uint32 range");
    }
    return options;
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
        checked_product(out_features, layout.row_bytes(in_features), "qdata"), 0);
    const std::size_t groups = layout.groups_for(in_features);
    for (std::size_t row = 0; row < out_features; ++row) {
        for (std::size_t group = 0; group < groups; ++group) {
            std::uint8_t * block = qdata.data() +
                row * layout.row_bytes(in_features) +
                group * layout.bytes_per_group();
            block[0] = 0x00U;
            block[1] = 0x3CU;
            std::uint8_t * codes = block + sizeof(std::uint16_t);
            for (std::size_t element = 0; element < layout.group_size; ++element) {
                const std::uint32_t code = static_cast<std::uint32_t>(
                    (row * 131 + group * 17 + element) %
                    (std::size_t{1} << layout.bits));
                write_code(codes, element, layout.bits, code);
            }
        }
    }
    return qdata;
}

double benchmark_width(const Options & options, std::uint32_t bits) {
    const rq::Layout layout{bits, static_cast<std::uint32_t>(options.group_size)};
    const auto qdata = make_qdata(
        options.out_features, options.in_features, layout);
    std::vector<float> codebook(std::size_t{1} << bits);
    for (std::size_t index = 0; index < codebook.size(); ++index) {
        codebook[index] = -2.5F +
            5.0F * static_cast<float>(index) /
            static_cast<float>(codebook.size() - 1);
    }
    std::vector<float> input(
        checked_product(options.batch, options.in_features, "input"));
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.013F);
    }
    std::vector<float> output(
        checked_product(options.batch, options.out_features, "output"));
    std::vector<float> reference(output.size());
    const rq::CpuKernel kernel = rq::resolve_kernel(options.kernel);
    rq::matmul(
        input.data(), input.size(), options.batch,
        qdata.data(), qdata.size(), codebook.data(), codebook.size(),
        options.out_features, options.in_features, layout,
        reference.data(), reference.size(), rq::CpuKernel::scalar);
    rq::matmul(
        input.data(), input.size(), options.batch,
        qdata.data(), qdata.size(), codebook.data(), codebook.size(),
        options.out_features, options.in_features, layout,
        output.data(), output.size(), kernel);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
        rq::matmul(
            input.data(), input.size(), options.batch,
            qdata.data(), qdata.size(), codebook.data(), codebook.size(),
            options.out_features, options.in_features, layout,
            output.data(), output.size(), kernel);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        static_cast<double>(options.iterations);
    double checksum = 0.0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double max_abs_error = 0.0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        checksum += output[index];
        const double difference =
            static_cast<double>(output[index]) - reference[index];
        squared_error += difference * difference;
        squared_reference +=
            static_cast<double>(reference[index]) * reference[index];
        max_abs_error = std::max(max_abs_error, std::abs(difference));
    }
    const double relative_l2_error = squared_reference == 0.0
        ? std::sqrt(squared_error)
        : std::sqrt(squared_error / squared_reference);
    std::cout << "{\"bits\":" << bits
              << ",\"kernel\":\"" << rq::kernel_name(kernel) << "\""
              << ",\"qdata_bytes\":" << qdata.size()
              << ",\"matmul_ms\":" << elapsed_ms
              << ",\"max_abs_error_vs_scalar\":" << max_abs_error
              << ",\"relative_l2_error_vs_scalar\":" << relative_l2_error
              << ",\"checksum\":" << checksum << '}';
    return checksum;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        Options options = parse_options(argc, argv);
        options.kernel = rq::resolve_kernel(options.kernel);
        std::cout << "{\"schema_version\":1,\"format_version\":"
                  << rq::kFormatVersion
                  << ",\"claim_boundary\":\"compiled CPU kernel benchmark\""
                  << ",\"out_features\":" << options.out_features
                  << ",\"in_features\":" << options.in_features
                  << ",\"group_size\":" << options.group_size
                  << ",\"batch\":" << options.batch
                  << ",\"cases\":[";
        double checksum = 0.0;
        for (std::uint32_t bits = rq::kMinBits; bits <= rq::kMaxBits; ++bits) {
            if (bits != rq::kMinBits) std::cout << ',';
            checksum += benchmark_width(options, bits);
        }
        std::cout << "],\"aggregate_checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "rotquant-native-bench: " << error.what() << '\n';
        return 2;
    }
}
