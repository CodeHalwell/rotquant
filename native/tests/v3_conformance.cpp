#include "rotquant/native_v3_c.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok) { if (!ok) { throw std::runtime_error("native-v3 conformance failed"); } }
void put32(std::vector<uint8_t> &data, size_t pos, uint32_t word) {
    for (size_t i = 0; i < 4; ++i) { data[pos + i] = uint8_t(word >> (8 * i)); }
}
std::vector<uint8_t> fixture(size_t bits, size_t sbits) {
    // Three 13-value rows, three groups per row; scale blocks cross row boundaries.
    const size_t words = (39 * bits + 31) / 32;
    const size_t metadata = sbits == 8 ? 3 : 0;
    const size_t payload = words * 4 + 9 * sbits / 8 + metadata * 4 + (size_t(1) << bits) * 4;
    std::vector<uint8_t> data(64 + payload, 0);
    std::memcpy(data.data(), "RQNATV3\0", 8);
    put32(data, 8, 3); put32(data, 12, 64); put32(data, 16, uint32_t(bits));
    put32(data, 20, 5); put32(data, 24, 3); put32(data, 32, 13);
    put32(data, 40, uint32_t(sbits)); put32(data, 44, sbits == 8 ? 4 : 0);
    put32(data, 56, uint32_t(payload));
    for (size_t i = 0; i < 39; ++i) {
        const size_t code = i % (size_t(1) << bits);
        for (size_t b = 0; b < bits; ++b) {
            const size_t bit = i * bits + b;
            data[64 + bit / 8] |= uint8_t(((code >> b) & 1) << (bit % 8));
        }
    }
    size_t pos = 64 + words * 4;
    for (size_t i = 0; i < 9; ++i) {
        if (sbits == 8) { data[pos + i] = uint8_t(i); }
        else { data[pos + i * 2 + 1] = 0x3c; } // FP16 one
    }
    pos += 9 * sbits / 8;
    for (size_t i = 0; i < metadata; ++i) {
        data[pos + i * 2 + 1] = 0x3c; // offset one
        data[pos + metadata * 2 + i * 2] = 1; // step = smallest FP16 subnormal
    }
    pos += metadata * 4;
    for (size_t i = 0; i < (size_t(1) << bits); ++i) {
        const float value = float(i) - 2.0f;
        uint32_t word;
        std::memcpy(&word, &value, 4);
        put32(data, pos + i * 4, word);
    }
    return data;
}
}

int main() {
    try {
        check(rq_native_v3_abi_version() == 1 && rq_native_v3_format_version() == 3);
        for (size_t bits = 1; bits <= 8; ++bits) {
            for (size_t sbits : {8, 16}) {
                auto data = fixture(bits, sbits);
                check(rq_native_v3_validate(data.data(), data.size(), 3, 13) == 0);
                check(rq_native_v3_validate(data.data(), data.size(), 4, 13) == 1);
                check(rq_native_v3_validate(data.data(), data.size(), 3, 12) == 1);
                check(rq_native_v3_validate(data.data(), data.size() - 1, 3, 13) == 1);
                float weights[39], x[13] = {}, y[3];
                check(rq_native_v3_dequantize(data.data(), data.size(), weights, 39, 0, 3) == 0);
                for (size_t i = 0; i < 39; ++i) {
                    const float scale = sbits == 8 ? 1.0f + float((i / 13) * 3 + (i % 13) / 5) * 0x1p-24f : 1.0f;
                    check(weights[i] == (float(i % (size_t(1) << bits)) - 2.0f) * scale);
                }
                x[12] = 1;
                check(rq_native_v3_matmul(data.data(), data.size(), x, 13, 1, y, 3) == 0);
                for (size_t r = 0; r < 3; ++r) { check(y[r] == weights[r * 13 + 12]); }
                // Validate before writes; malformed lengths/flags/tails/overflow must fail.
                y[0] = 123;
                check(rq_native_v3_matmul(data.data(), data.size() - 1, x, 13, 1, y, 3) == 1);
                check(y[0] == 123);
                check(rq_native_v3_dequantize(data.data(), data.size(), weights, 13, 3, 1) == 1);
                check(rq_native_v3_matmul(data.data(), data.size(), x, 13, 1, x, 3) == 1);
                auto bad = data; bad[48] = 1;
                check(rq_native_v3_dequantize(bad.data(), bad.size(), weights, 39, 0, 3) == 1);
                bad = data; bad[64 + ((39 * bits + 31) / 32) * 4 - 1] |= 0x80;
                if (39 * bits % 32) {
                    check(rq_native_v3_dequantize(bad.data(), bad.size(), weights, 39, 0, 3) == 1);
                }
                bad = data;
                for (size_t i = 24; i < 40; ++i) { bad[i] = 0xff; }
                bad[31] = 0x7f; bad[39] = 0x7f;
                check(rq_native_v3_dequantize(bad.data(), bad.size(), weights, 39, 0, 3) == 2);
            }
        }
        check(rq_native_v3_dequantize(nullptr, 64, nullptr, 0, 0, 1) == 1);
        check(rq_native_v3_validate(nullptr, 64, 3, 13) == 1);
        std::puts("native-v3: bits 1..8, scale8/16, exact decode and fail-closed checks passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: %s\n", e.what(), rq_native_v3_last_error());
        return 1;
    }
}
