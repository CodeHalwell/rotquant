#ifndef ROTQUANT_NATIVE_V2_C_H
#define ROTQUANT_NATIVE_V2_C_H

#include <stddef.h>
#include <stdint.h>

#include "rotquant/native_export.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RQ_NATIVE_V2_ABI_VERSION 1U
#define RQ_NATIVE_V2_FORMAT_VERSION 2U

typedef enum rq_native_v2_status {
    RQ_NATIVE_V2_STATUS_OK = 0,
    RQ_NATIVE_V2_STATUS_INVALID_ARGUMENT = 1,
    RQ_NATIVE_V2_STATUS_OUT_OF_RANGE = 2,
    RQ_NATIVE_V2_STATUS_KERNEL_UNAVAILABLE = 3,
    RQ_NATIVE_V2_STATUS_INTERNAL_ERROR = 4
} rq_native_v2_status;

typedef enum rq_native_v2_kernel {
    RQ_NATIVE_V2_KERNEL_AUTO = 0,
    RQ_NATIVE_V2_KERNEL_SCALAR = 1,
    RQ_NATIVE_V2_KERNEL_NEON = 2,
    RQ_NATIVE_V2_KERNEL_AVX2 = 3
} rq_native_v2_kernel;

typedef struct rq_native_v2_layout {
    uint32_t bits;
    uint32_t group_size;
} rq_native_v2_layout;

typedef struct rq_native_v2_kernel_capability {
    const char * name;
    rq_native_v2_kernel kernel;
    uint32_t min_bits;
    uint32_t max_bits;
    uint32_t group_size;
} rq_native_v2_kernel_capability;

ROTQUANT_NATIVE_API uint32_t rq_native_v2_abi_version(void);
ROTQUANT_NATIVE_API uint32_t rq_native_v2_format_version(void);

ROTQUANT_NATIVE_API const char * rq_native_v2_last_error(void);
ROTQUANT_NATIVE_API const char * rq_native_v2_kernel_name(
    rq_native_v2_kernel kernel);

ROTQUANT_NATIVE_API size_t rq_native_v2_kernel_count(void);
ROTQUANT_NATIVE_API rq_native_v2_status rq_native_v2_kernel_capability_at(
    size_t index,
    rq_native_v2_kernel_capability * output);
ROTQUANT_NATIVE_API rq_native_v2_status rq_native_v2_resolve_kernel(
    rq_native_v2_kernel requested,
    rq_native_v2_kernel * resolved);

ROTQUANT_NATIVE_API rq_native_v2_status rq_native_v2_dequantize(
    const uint8_t * qdata,
    size_t qdata_size,
    const float * codebook,
    size_t codebook_size,
    size_t out_features,
    size_t in_features,
    rq_native_v2_layout layout,
    float * output,
    size_t output_size,
    rq_native_v2_kernel kernel);

ROTQUANT_NATIVE_API rq_native_v2_status rq_native_v2_matmul(
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
    rq_native_v2_kernel kernel);

#ifdef __cplusplus
}
#endif

#endif
