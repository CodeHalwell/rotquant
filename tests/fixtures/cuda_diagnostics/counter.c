// CPU-only ELF fixture. Simulates process-local counters, never CUDA execution.
#include <stdint.h>
static uint64_t count = 0;
void fixture_increment(void) { ++count; }
uint64_t ggml_cuda_rq3_decode_dispatches(void) { return count; }
int ggml_cuda_rq3_diagnostics(double * ms, uint64_t * calls, uint64_t * tiled) {
    if (!ms || !calls || !tiled) { return 0; }
    for (int i = 0; i < 4; ++i) { ms[i] = 0; calls[i] = count; }
    *tiled = count;
    return 1;
}
