#include "rotquant/native_v2_c.h"

#include "rotquant/native_v2.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace rq = rotquant::native_v2;

namespace {

thread_local std::string last_error;

void set_last_error(const char * message) noexcept {
    try {
        last_error = message == nullptr ? "unknown error" : message;
    } catch (...) {
        last_error.clear();
    }
}

template <typename Function>
rq_native_v2_status guarded_call(Function && function) noexcept {
    try {
        std::forward<Function>(function)();
        last_error.clear();
        return RQ_NATIVE_V2_STATUS_OK;
    } catch (const std::invalid_argument & error) {
        set_last_error(error.what());
        return RQ_NATIVE_V2_STATUS_INVALID_ARGUMENT;
    } catch (const std::overflow_error & error) {
        set_last_error(error.what());
        return RQ_NATIVE_V2_STATUS_OUT_OF_RANGE;
    } catch (const std::out_of_range & error) {
        set_last_error(error.what());
        return RQ_NATIVE_V2_STATUS_OUT_OF_RANGE;
    } catch (const std::runtime_error & error) {
        set_last_error(error.what());
        return RQ_NATIVE_V2_STATUS_KERNEL_UNAVAILABLE;
    } catch (const std::exception & error) {
        set_last_error(error.what());
        return RQ_NATIVE_V2_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_last_error("unknown native v2 error");
        return RQ_NATIVE_V2_STATUS_INTERNAL_ERROR;
    }
}

rq::CpuKernel to_cpp_kernel(rq_native_v2_kernel kernel) {
    switch (kernel) {
        case RQ_NATIVE_V2_KERNEL_AUTO: return rq::CpuKernel::automatic;
        case RQ_NATIVE_V2_KERNEL_SCALAR: return rq::CpuKernel::scalar;
        case RQ_NATIVE_V2_KERNEL_NEON: return rq::CpuKernel::neon;
        case RQ_NATIVE_V2_KERNEL_AVX2: return rq::CpuKernel::avx2;
    }
    throw std::invalid_argument("unknown RotQuant C ABI kernel value");
}

rq_native_v2_kernel from_cpp_kernel(rq::CpuKernel kernel) {
    switch (kernel) {
        case rq::CpuKernel::automatic: return RQ_NATIVE_V2_KERNEL_AUTO;
        case rq::CpuKernel::scalar: return RQ_NATIVE_V2_KERNEL_SCALAR;
        case rq::CpuKernel::neon: return RQ_NATIVE_V2_KERNEL_NEON;
        case rq::CpuKernel::avx2: return RQ_NATIVE_V2_KERNEL_AVX2;
    }
    throw std::invalid_argument("unknown RotQuant C++ kernel value");
}

const char * capability_name(rq::CpuKernel kernel) noexcept {
    switch (kernel) {
        case rq::CpuKernel::scalar: return "portable-scalar";
        case rq::CpuKernel::neon: return "arm-neon";
        case rq::CpuKernel::avx2: return "x86-avx2";
        case rq::CpuKernel::automatic: return "automatic";
    }
    return "unknown";
}

rq::Layout to_cpp_layout(rq_native_v2_layout layout) noexcept {
    return {layout.bits, layout.group_size};
}

}  // namespace

extern "C" {

uint32_t rq_native_v2_abi_version(void) {
    return RQ_NATIVE_V2_ABI_VERSION;
}

uint32_t rq_native_v2_format_version(void) {
    return rq::kFormatVersion;
}

const char * rq_native_v2_last_error(void) {
    return last_error.c_str();
}

const char * rq_native_v2_kernel_name(rq_native_v2_kernel kernel) {
    try {
        return rq::kernel_name(to_cpp_kernel(kernel));
    } catch (...) {
        return "unknown";
    }
}

size_t rq_native_v2_kernel_count(void) {
    try {
        const size_t count = rq::available_kernels().size();
        last_error.clear();
        return count;
    } catch (const std::exception & error) {
        set_last_error(error.what());
        return 0;
    } catch (...) {
        set_last_error("unable to enumerate native v2 kernels");
        return 0;
    }
}

rq_native_v2_status rq_native_v2_kernel_capability_at(
        size_t index,
        rq_native_v2_kernel_capability * output) {
    return guarded_call([&] {
        if (output == nullptr) {
            throw std::invalid_argument("capability output must not be null");
        }
        const auto kernels = rq::available_kernels();
        if (index >= kernels.size()) {
            throw std::out_of_range("kernel capability index is out of range");
        }
        const auto & kernel = kernels[index];
        output->name = capability_name(kernel.kernel);
        output->kernel = from_cpp_kernel(kernel.kernel);
        output->min_bits = kernel.min_bits;
        output->max_bits = kernel.max_bits;
        output->group_size = kernel.group_size;
    });
}

rq_native_v2_status rq_native_v2_resolve_kernel(
        rq_native_v2_kernel requested,
        rq_native_v2_kernel * resolved) {
    return guarded_call([&] {
        if (resolved == nullptr) {
            throw std::invalid_argument("resolved kernel output must not be null");
        }
        *resolved = from_cpp_kernel(rq::resolve_kernel(to_cpp_kernel(requested)));
    });
}

rq_native_v2_status rq_native_v2_dequantize(
        const uint8_t * qdata,
        size_t qdata_size,
        const float * codebook,
        size_t codebook_size,
        size_t out_features,
        size_t in_features,
        rq_native_v2_layout layout,
        float * output,
        size_t output_size,
        rq_native_v2_kernel kernel) {
    return guarded_call([&] {
        rq::dequantize(
            qdata, qdata_size, codebook, codebook_size,
            out_features, in_features, to_cpp_layout(layout),
            output, output_size, to_cpp_kernel(kernel));
    });
}

rq_native_v2_status rq_native_v2_matmul(
        const float * input,
        size_t input_size,
        size_t batch,
        const uint8_t * qdata,
        size_t qdata_size,
        const float * codebook,
        size_t codebook_size,
        size_t out_features,
        size_t in_features,
        rq_native_v2_layout layout,
        float * output,
        size_t output_size,
        rq_native_v2_kernel kernel) {
    return guarded_call([&] {
        rq::matmul(
            input, input_size, batch,
            qdata, qdata_size, codebook, codebook_size,
            out_features, in_features, to_cpp_layout(layout),
            output, output_size, to_cpp_kernel(kernel));
    });
}

}  // extern "C"
