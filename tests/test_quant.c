#include "quant.h"

#include "check.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LONG_ROW = 5632 };

static uint32_t seed = 0x931199b3;

static uint32_t random_bits(void) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

static float random_float(void) {
    return ((int)(random_bits() % 1025) - 512) / 512.0f;
}

static void fill_bytes(void *data, size_t len) {
    uint8_t *bytes = data;
    for (size_t i = 0; i < len; i++) bytes[i] = (uint8_t)random_bits();
}

static void measure(const void *row, gguf_type_t type, const float *x, int n, double bound,
                    double *worst) {
    float weights[LONG_ROW];
    dequantize_row(row, weights, n, type);
    double reference = 0.0;
    for (int i = 0; i < n; i++) reference += (double)x[i] * weights[i];
    double error = fabs(reference - vec_dot(row, x, n, type));
    if (error > *worst) *worst = error;
    if (!(error < bound)) fprintf(stderr, "type=%d n=%d error=%.9g\n", type, n, error);
    CHECK(error < bound);
}

static void known_wire_row(const uint8_t *packed, gguf_type_t type, const float *expected) {
    float decoded[256], x[256], dot = 0.0f;
    dequantize_row(packed, decoded, 256, type);
    for (int i = 0; i < 256; i++) {
        CHECK(decoded[i] == expected[i]);
        x[i] = (float)(i % 7 - 3) * 0.25f;
        dot += expected[i] * x[i];
    }
    CHECK(vec_dot(packed, x, 256, type) == dot);
}

static void known_ggml_blocks(void) {
    /* Wire offsets and expected values are independent of the local structs. */
    uint8_t q2[84] = {[16] = 0xe4, [55] = 3, [81] = 0x3c, [83] = 0x38};
    memset(q2, 0x21, 16);
    q2[2] = 0x43;
    float q2_expected[256];
    for (int i = 0; i < 256; i++) q2_expected[i] = i >= 32 && i < 48 ? -2.0f : -1.0f;
    q2_expected[32] = 1.0f;
    q2_expected[64] = 1.0f;
    q2_expected[96] = 2.0f;
    q2_expected[135] = 2.0f;
    known_wire_row(q2, GGUF_TYPE_Q2_K, q2_expected);

    uint8_t q3[110] = {[32] = 0xe4, [64] = 0xb9, [109] = 0x3c};
    memset(q3, 0xff, 32);
    q3[0] = 0x55;
    memset(q3 + 96, 0x11, 8);
    memset(q3 + 104, 0xaa, 4);
    q3[96] = 0x21;
    q3[104] = 0x96;
    const float q3_expected[256] = {
        [32] = -3, [64] = -30, [96] = -1, [128] = -14, [160] = -2, [192] = 3, [224] = -2,
    };
    known_wire_row(q3, GGUF_TYPE_Q3_K, q3_expected);
}

int main(void) {
    CHECK(fp16_to_fp32(0x3c00) == 1.0f);
    quant_init();
    CHECK(fp16_to_fp32(1) == 0x1p-24f && signbit(fp16_to_fp32(0x8000)));
    CHECK(isinf(fp16_to_fp32(0x7c00)) && isnan(fp16_to_fp32(0x7e00)));
    CHECK(gguf_type_row_size(GGUF_TYPE_Q4_K, 255) == 0);
    known_ggml_blocks();

    unsigned char *samples = malloc(LONG_ROW * sizeof(float));
    CHECK(samples);
    float x[LONG_ROW];
    const gguf_type_t types[] = {
        GGUF_TYPE_F32, GGUF_TYPE_F16, GGUF_TYPE_Q4_0, GGUF_TYPE_Q8_0,
        GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
    };
    double unit_error = 0.0, long_error = 0.0;
    for (size_t type = 0; type < sizeof(types) / sizeof(types[0]); type++) {
        gguf_type_t q = types[type];
        int bs = gguf_type_block_size(q);
        for (int trial = 0; trial < 32; trial++) {
            for (int i = 0; i < LONG_ROW; i++) x[i] = random_float();
            int bytes = gguf_type_quant_size(q);
            fill_bytes(samples, (size_t)LONG_ROW / bs * bytes);
            for (int j = 0; j < LONG_ROW / bs; j++) {
                unsigned char *block = samples + (size_t)j * bytes;
                float f = random_float();
                uint16_t scale = fp32_to_fp16(f * 0.001f);
                if (q == GGUF_TYPE_F32) memcpy(block, &f, 4);
                else if (q == GGUF_TYPE_F16) {
                    uint16_t half = fp32_to_fp16(f);
                    memcpy(block, &half, 2);
                } else if (q == GGUF_TYPE_Q6_K) memcpy(block + 208, &scale, 2);
                else {
                    memcpy(block, &scale, 2);
                    if (q == GGUF_TYPE_Q4_K) memcpy(block + 2, &scale, 2);
                }
            }
            int unit = bs == 1 ? 63 : bs;
            measure(samples, q, x, unit, 1e-4, &unit_error);
            measure(samples, q, x, LONG_ROW, 1e-3, &long_error);
        }
    }

    free(samples);
    uint16_t half[19];
    int8_t raw[19];
    float fp16_acc[19], int8_acc[19], query[19];
    float reference = 0;
    for (int i = 0; i < 19; i++) {
        query[i] = random_float();
        raw[i] = (int8_t)((int)(random_bits() % 255) - 127);
        half[i] = fp32_to_fp16((float)raw[i]);
        reference += raw[i] * query[i];
        fp16_acc[i] = int8_acc[i] = query[i];
    }
    CHECK(fabsf(vec_dot_i8_f32(raw, query, 19) - reference) < 1e-4f);
    vec_accumulate_f16(fp16_acc, half, 19, 0.25f, 0.5f);
    vec_accumulate_i8(int8_acc, raw, 19, 0.25f, 0.5f);
    for (int i = 0; i < 19; i++) {
        float expected = query[i] * 0.25f + raw[i] * 0.5f;
        CHECK(fp16_acc[i] == expected && int8_acc[i] == expected);
    }
    printf("quant %s: 384 dot comparisons plus 2 independent GGML wire vectors; unit max %.9g, long max %.9g\n",
           quant_simd(), unit_error, long_error);
    return 0;
}
