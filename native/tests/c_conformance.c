#include "rotquant/native_v2_c.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int require_condition(bool condition, const char * message) {
    if (condition) return 0;
    fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

int main(void) {
    if (require_condition(
            rq_native_v2_abi_version() == 1U, "C ABI version mismatch")) return 1;
    if (require_condition(
            rq_native_v2_format_version() == 2U, "format version mismatch")) return 1;

    const size_t kernel_count = rq_native_v2_kernel_count();
    if (require_condition(kernel_count >= 1U, "no C ABI kernels reported")) return 1;
    bool scalar_found = false;
    bool avx2_found = false;
    for (size_t index = 0; index < kernel_count; ++index) {
        rq_native_v2_kernel_capability capability;
        if (require_condition(
                rq_native_v2_kernel_capability_at(index, &capability) ==
                    RQ_NATIVE_V2_STATUS_OK,
                "unable to read C ABI capability")) return 1;
        if (capability.kernel == RQ_NATIVE_V2_KERNEL_SCALAR) scalar_found = true;
        if (capability.kernel == RQ_NATIVE_V2_KERNEL_AVX2) avx2_found = true;
        if (require_condition(capability.name != NULL, "capability name is null")) {
            return 1;
        }
    }
    if (require_condition(scalar_found, "scalar C ABI capability missing")) return 1;

    const rq_native_v2_layout layout = {4U, 8U};
    const uint8_t qdata[6] = {0x00U, 0x3CU, 0U, 0U, 0U, 0U};
    float codebook[16] = {0.0F};
    float input[8];
    float output[1] = {0.0F};
    float weight[8] = {0.0F};
    codebook[0] = 2.0F;
    for (size_t index = 0; index < 8U; ++index) input[index] = 1.0F;

    const rq_native_v2_status status = rq_native_v2_matmul(
        input, 8U, 1U,
        qdata, sizeof(qdata), codebook, 16U,
        1U, 8U, layout, output, 1U,
        RQ_NATIVE_V2_KERNEL_SCALAR);
    if (require_condition(status == RQ_NATIVE_V2_STATUS_OK, "C ABI matmul failed")) {
        fprintf(stderr, "detail: %s\n", rq_native_v2_last_error());
        return 1;
    }
    if (require_condition(fabsf(output[0] - 16.0F) < 1e-6F, "C ABI output mismatch")) {
        return 1;
    }
    const rq_native_v2_status dequantize_status = rq_native_v2_dequantize(
        qdata, sizeof(qdata), codebook, 16U,
        1U, 8U, layout, weight, 8U,
        RQ_NATIVE_V2_KERNEL_SCALAR);
    if (require_condition(
            dequantize_status == RQ_NATIVE_V2_STATUS_OK,
            "C ABI dequantize failed")) return 1;
    for (size_t index = 0; index < 8U; ++index) {
        if (require_condition(
                fabsf(weight[index] - 2.0F) < 1e-6F,
                "C ABI dequantized value mismatch")) return 1;
    }

    const rq_native_v2_status invalid = rq_native_v2_matmul(
        input, 8U, 1U,
        qdata, sizeof(qdata) - 1U, codebook, 16U,
        1U, 8U, layout, output, 1U,
        RQ_NATIVE_V2_KERNEL_SCALAR);
    if (require_condition(
            invalid == RQ_NATIVE_V2_STATUS_INVALID_ARGUMENT,
            "malformed C ABI payload was not rejected")) return 1;
    if (require_condition(
            strlen(rq_native_v2_last_error()) > 0U,
            "C ABI error detail is empty")) return 1;

    if (!avx2_found) {
        rq_native_v2_kernel resolved = RQ_NATIVE_V2_KERNEL_AUTO;
        const rq_native_v2_status unavailable = rq_native_v2_resolve_kernel(
            RQ_NATIVE_V2_KERNEL_AVX2, &resolved);
        if (require_condition(
                unavailable == RQ_NATIVE_V2_STATUS_KERNEL_UNAVAILABLE,
                "unavailable C ABI kernel did not fail closed")) return 1;
    }

    printf("PASS: native v2 C ABI conformance\n");
    return 0;
}
