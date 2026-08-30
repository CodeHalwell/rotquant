#include "rotquant/native_v2.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rq = rotquant::native_v2;

namespace {

template <typename T>
std::vector<T> read_binary(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("unable to open input file: " + path);
    }
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary file size: " + path);
    }
    stream.seekg(0);
    std::vector<T> values(static_cast<std::size_t>(bytes) / sizeof(T));
    if (bytes != 0 && !stream.read(
            reinterpret_cast<char *>(values.data()), bytes)) {
        throw std::runtime_error("unable to read input file: " + path);
    }
    return values;
}

template <typename T>
void write_binary(const std::string & path, const std::vector<T> & values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)))) {
        throw std::runtime_error("unable to write output file: " + path);
    }
}

std::unordered_map<std::string, std::string> parse_options(
        int argc, char ** argv) {
    std::unordered_map<std::string, std::string> options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--capabilities") {
            options.emplace(key, "true");
            continue;
        }
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::invalid_argument("expected --key value arguments");
        }
        options[key] = argv[++index];
    }
    return options;
}

const std::string & required(
        const std::unordered_map<std::string, std::string> & options,
        const std::string & key) {
    const auto iterator = options.find(key);
    if (iterator == options.end()) {
        throw std::invalid_argument("missing required option: " + key);
    }
    return iterator->second;
}

std::size_t positive_size(
        const std::unordered_map<std::string, std::string> & options,
        const std::string & key) {
    const std::string & text = required(options, key);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(key + " must be a positive integer");
    }
    const unsigned long long parsed = std::stoull(text);
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(key + " must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

std::uint32_t positive_u32(
        const std::unordered_map<std::string, std::string> & options,
        const std::string & key) {
    const std::size_t value = positive_size(options, key);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(key + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t checked_product(
        std::size_t left, std::size_t right, const char * description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(description) + " size overflows size_t");
    }
    return left * right;
}

rq::CpuKernel parse_kernel(const std::string & value) {
    if (value == "auto") return rq::CpuKernel::automatic;
    if (value == "scalar") return rq::CpuKernel::scalar;
    if (value == "neon") return rq::CpuKernel::neon;
    if (value == "avx2") return rq::CpuKernel::avx2;
    throw std::invalid_argument("unknown kernel: " + value);
}

void print_capabilities() {
    const auto capabilities = rq::available_kernels();
    std::cout << "{\"format_version\":" << rq::kFormatVersion
              << ",\"kernels\":[";
    for (std::size_t index = 0; index < capabilities.size(); ++index) {
        const auto & capability = capabilities[index];
        if (index != 0) std::cout << ',';
        std::cout << "{\"name\":\"" << capability.name
                  << "\",\"kernel\":\"" << rq::kernel_name(capability.kernel)
                  << "\",\"min_bits\":" << capability.min_bits
                  << ",\"max_bits\":" << capability.max_bits
                  << ",\"group_size\":" << capability.group_size << '}';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.count("--capabilities") != 0) {
            print_capabilities();
            return 0;
        }

        const std::uint32_t bits = positive_u32(options, "--bits");
        const std::uint32_t group_size = positive_u32(options, "--group-size");
        const std::size_t in_features = positive_size(options, "--in-features");
        const std::size_t out_features = positive_size(options, "--out-features");
        const std::size_t batch = positive_size(options, "--batch");
        const rq::Layout layout{
            bits,
            group_size,
        };
        const auto qdata = read_binary<std::uint8_t>(required(options, "--qdata"));
        const auto codebook = read_binary<float>(required(options, "--codebook"));
        const auto input = read_binary<float>(required(options, "--input"));
        std::vector<float> output(
            checked_product(batch, out_features, "output"));
        const auto kernel_iterator = options.find("--kernel");
        const rq::CpuKernel kernel = kernel_iterator == options.end()
            ? rq::CpuKernel::automatic
            : parse_kernel(kernel_iterator->second);
        rq::matmul(
            input.data(), input.size(), batch,
            qdata.data(), qdata.size(),
            codebook.data(), codebook.size(),
            out_features, in_features, layout,
            output.data(), output.size(), kernel);
        write_binary(required(options, "--output"), output);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "rotquant-native-cli: " << error.what() << '\n';
        return 2;
    }
}
