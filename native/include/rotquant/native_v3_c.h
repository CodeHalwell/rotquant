#ifndef ROTQUANT_NATIVE_V3_C_H
#define ROTQUANT_NATIVE_V3_C_H

#include <stddef.h>
#include <stdint.h>
#include "rotquant/native_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Matrix-only scalar CPU reference. No rotations, FP16 output rounding, GGUF,
 * vocabulary ownership or model execution. Counts below are float elements.
 * Status: 0 success, 1 invalid argument, 2 size overflow, 3 internal error.
 * On validation failure output is unchanged. Buffers must not overlap; input
 * storage must stay valid and immutable for the call. Float buffers must be
 * naturally aligned (packed data need not be). Matmul uses no dense
 * weight allocation. It validates compact storage on every call (not tuned).
 * ABI version and on-disk format version are independently versioned. */
ROTQUANT_NATIVE_API uint32_t rq_native_v3_abi_version(void);
ROTQUANT_NATIVE_API uint32_t rq_native_v3_format_version(void);
ROTQUANT_NATIVE_API const char *rq_native_v3_last_error(void);
ROTQUANT_NATIVE_API int rq_native_v3_dequantize(
    const uint8_t *data, size_t data_size, float *output, size_t output_size,
    size_t row_start, size_t row_count);
ROTQUANT_NATIVE_API int rq_native_v3_matmul(
    const uint8_t *data, size_t data_size, const float *input, size_t input_size,
    size_t batch, float *output, size_t output_size);

#ifdef __cplusplus
}
#endif
#endif
