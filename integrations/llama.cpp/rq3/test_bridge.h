#pragma once
#include <stdint.h>
#include <stddef.h>
extern "C" {
const char * rq3_test_error();
int rq3_test_eval(const char *, const uint8_t *, size_t, int64_t, int64_t, int64_t, int,
                  const void *, const int8_t *, const int32_t *, const int32_t *, float *);
const char * rq3_model_error();
void * rq3_model_open(const char *, const char *, uint32_t);
int rq3_model_vocab(void *);
int rq3_model_eval(void *, const int32_t *, int32_t, bool, float *);
int rq3_model_eval_selected(void *, const int32_t *, int32_t, bool, int32_t, float *);
bool rq3_model_is_eog(void *, int32_t);
uint64_t rq3_model_custom_ops(void *);
void rq3_model_close(void *);
}
