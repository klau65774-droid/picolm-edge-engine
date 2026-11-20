#include "quant.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>

/* ================================================================
 * FP16 <-> FP32 conversion (software, no hardware dependency)
 * ================================================================ */

static float fp16_to_fp32_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; /* +/- zero */
        } else {
            /* subnormal: renormalize */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); /* inf / nan */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static float fp16_to_fp32_lut[65536];
static atomic_int lut_state;

void quant_init(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&lut_state, &expected, 1)) {
        for (unsigned i = 0; i < 65536; i++) {
            fp16_to_fp32_lut[i] = fp16_to_fp32_bits((uint16_t)i);
        }
        atomic_store_explicit(&lut_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&lut_state, memory_order_acquire) != 2) {}
    }
}

float fp16_to_fp32(uint16_t h) {
    if (atomic_load_explicit(&lut_state, memory_order_acquire) == 2) {
        return fp16_to_fp32_lut[h];
    }
    return fp16_to_fp32_bits(h);
}

const char *quant_simd(void) {
#if defined(PICOLM_AVX2)
    return "AVX2+FMA";
#elif defined(PICOLM_NEON)
    return "NEON";
#elif defined(PICOLM_SSE2)
    return "SSE2";
#else
    return "scalar";
#endif
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int      exp  = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) {
        return (uint16_t)sign; /* zero or f32 subnormal -> fp16 zero */
    }
    if (((bits >> 23) & 0xFF) == 0xFF) {
        /* inf / nan */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); /* overflow -> inf */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* too small -> zero */
        /* subnormal fp16 */
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        /* round to nearest */
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }

    /* round to nearest even */
    mant += 0x00001000; /* bit 12 */
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* ---- Q4_K helpers ---- */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

/* ================================================================
 * Dequantization kernels (scalar — used for embedding lookup etc.)
 * ================================================================ */

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t *q = b->qs;
        float *y = dst + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

            for (int l = 0; l < 32; l++) {
                y[l]      = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            }
            y  += 64;
            q  += 32;
            is += 2;
        }
    }
}

/* Within each 128-weight half, one byte covers positions 32 apart. */
static int unpack_k_quants2(const uint8_t *q, int position) {
    int lane = position % 32;
    int shift = 2 * (position / 32 % 4);
    return (q[32 * (position / 128) + lane] >> shift) & 3;
}

void dequantize_row_q3_K(const void *src, float *dst, int n) {
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32(b->d);

        for (int group = 0; group < 16; group++) {
            int low = (b->scales[group % 8] >> (4 * (group / 8))) & 15;
            int high = (b->scales[8 + group % 4] >> (2 * (group / 4))) & 3;
            float scale = d * (float)((low | (high << 4)) - 32);
            for (int pos = group * 16; pos < (group + 1) * 16; pos++) {
                int q = unpack_k_quants2(b->qs, pos);
                if (!(b->hmask[pos % 32] & (1u << (pos / 32)))) q -= 4;
                dst[i * 256 + pos] = scale * (float)q;
            }
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q2_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);

        for (int group = 0; group < 16; group++) {
            float scale = d * (float)(b->scales[group] & 15);
            float min = dmin * (float)(b->scales[group] >> 4);
            for (int pos = group * 16; pos < (group + 1) * 16; pos++) {
                dst[i * 256 + pos] = scale * (float)unpack_k_quants2(b->qs, pos) - min;
            }
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        float *y = dst + i * 256;

        for (int chunk = 0; chunk < 256; chunk += 128) {
            int is = chunk / 16;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                int is_l = is + (l / 16);
                y[l]      = d * (float)sc[is_l + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is_l + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is_l + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is_l + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
        }
    }
}

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_0(const void *src, float *dst, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            uint8_t nibble;
            if (j < 16) {
                nibble = blocks[i].qs[j] & 0xF;
            } else {
                nibble = blocks[i].qs[j - 16] >> 4;
            }
            dst[i * 32 + j] = d * ((float)nibble - 8.0f);
        }
    }
}

void dequantize_row_f16(const void *src, float *dst, int n) {
    const uint16_t *fp16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

void dequantize_row_f32(const void *src, float *dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void dequantize_row(const void *src, float *dst, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   dequantize_row_f32(src, dst, n);  break;
        case GGUF_TYPE_F16:   dequantize_row_f16(src, dst, n);  break;
        case GGUF_TYPE_Q4_0:  dequantize_row_q4_0(src, dst, n); break;
        case GGUF_TYPE_Q8_0:  dequantize_row_q8_0(src, dst, n); break;
        case GGUF_TYPE_Q2_K:  dequantize_row_q2_K(src, dst, n); break;
        case GGUF_TYPE_Q3_K:  dequantize_row_q3_K(src, dst, n); break;
        case GGUF_TYPE_Q4_K:  dequantize_row_q4_K(src, dst, n); break;
        case GGUF_TYPE_Q6_K:  dequantize_row_q6_K(src, dst, n); break;
        default:
            fprintf(stderr, "dequantize_row: unsupported type %d\n", type);
            exit(1);
    }
}

/* ---- Type info ---- */

int gguf_type_block_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 1;
        case GGUF_TYPE_F16:   return 1;
        case GGUF_TYPE_Q4_0:  return 32;
        case GGUF_TYPE_Q4_1:  return 32;
        case GGUF_TYPE_Q5_0:  return 32;
        case GGUF_TYPE_Q5_1:  return 32;
        case GGUF_TYPE_Q8_0:  return 32;
        case GGUF_TYPE_Q8_1:  return 32;
        case GGUF_TYPE_Q2_K:  return 256;
        case GGUF_TYPE_Q3_K:  return 256;
        case GGUF_TYPE_Q4_K:  return 256;
        case GGUF_TYPE_Q5_K:  return 256;
        case GGUF_TYPE_Q6_K:  return 256;
        default: return 0;
    }
}

int gguf_type_quant_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 4;
        case GGUF_TYPE_F16:   return 2;
        case GGUF_TYPE_Q4_0:  return 18;
        case GGUF_TYPE_Q4_1:  return 20;
        case GGUF_TYPE_Q5_0:  return 22;
        case GGUF_TYPE_Q5_1:  return 24;
        case GGUF_TYPE_Q8_0:  return 34;
        case GGUF_TYPE_Q8_1:  return 36;
        case GGUF_TYPE_Q2_K:  return 84;
        case GGUF_TYPE_Q3_K:  return 110;
        case GGUF_TYPE_Q4_K:  return 144;
        case GGUF_TYPE_Q5_K:  return 176;
        case GGUF_TYPE_Q6_K:  return 210;
        default: return 0;
    }
}

size_t gguf_type_row_size(gguf_type_t type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0 || n <= 0 || n % bs != 0 ||
        (size_t)(n / bs) > SIZE_MAX / (size_t)qs) return 0;
    return (size_t)(n / bs) * qs;
}

int gguf_type_supported(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:
        case GGUF_TYPE_F16:
        case GGUF_TYPE_Q4_0:
        case GGUF_TYPE_Q8_0:
        case GGUF_TYPE_Q2_K:
        case GGUF_TYPE_Q3_K:
        case GGUF_TYPE_Q4_K:
        case GGUF_TYPE_Q6_K:
            return 1;
        default:
            return 0;
    }
}

/* Fused kernels unpack eight weights into registers at a time. */
#if defined(PICOLM_AVX2)
typedef __m128i bytes8_t;
typedef __m256 lanes_t;

static inline bytes8_t load_bytes8(const void *src) {
    return _mm_loadl_epi64((const __m128i *)src);
}

static inline bytes8_t unpack_bits(bytes8_t q, int shift, int mask) {
    return _mm_and_si128(_mm_srli_epi16(q, shift), _mm_set1_epi8((char)mask));
}

static inline bytes8_t combine_q6(bytes8_t lo, bytes8_t hi) {
    return _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
}

static inline lanes_t floats_from_bytes(bytes8_t q) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
}

static inline lanes_t load_f16(const uint16_t *p) {
#if defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
#else
    return _mm256_setr_ps(fp16_to_fp32(p[0]), fp16_to_fp32(p[1]),
                         fp16_to_fp32(p[2]), fp16_to_fp32(p[3]),
                         fp16_to_fp32(p[4]), fp16_to_fp32(p[5]),
                         fp16_to_fp32(p[6]), fp16_to_fp32(p[7]));
#endif
}

static inline lanes_t zero_lanes(void) { return _mm256_setzero_ps(); }

static inline float sum_lanes(lanes_t v) {
    return hsum_sse(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)));
}

static inline lanes_t dot8(lanes_t acc, bytes8_t q, const float *x, float scale, float bias) {
    lanes_t weights = _mm256_fmadd_ps(floats_from_bytes(q), _mm256_set1_ps(scale),
                                     _mm256_set1_ps(bias));
    return _mm256_fmadd_ps(weights, _mm256_loadu_ps(x), acc);
}

#elif defined(PICOLM_NEON)
typedef uint8x8_t bytes8_t;
typedef float32x4_t lanes_t;

static inline bytes8_t load_bytes8(const void *src) { return vld1_u8(src); }

static inline bytes8_t unpack_bits(bytes8_t q, int shift, int mask) {
    return vand_u8(vshl_u8(q, vdup_n_s8((int8_t)-shift)), vdup_n_u8((uint8_t)mask));
}

static inline bytes8_t combine_q6(bytes8_t lo, bytes8_t hi) {
    return vorr_u8(lo, vshl_n_u8(hi, 4));
}

static inline lanes_t zero_lanes(void) { return vdupq_n_f32(0.0f); }
static inline float sum_lanes(lanes_t v) { return vaddvq_f32_compat(v); }

static inline lanes_t dot8(lanes_t acc, bytes8_t q, const float *x, float scale, float bias) {
    int16x8_t w = vmovl_s8(vreinterpret_s8_u8(q));
    lanes_t low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
    lanes_t high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
    lanes_t b = vdupq_n_f32(bias);
    acc = vmlaq_f32(acc, vmlaq_n_f32(b, low, scale), vld1q_f32(x));
    return vmlaq_f32(acc, vmlaq_n_f32(b, high, scale), vld1q_f32(x + 4));
}

static inline lanes_t load_f16(const uint16_t *p) {
    lanes_t values = {fp16_to_fp32(p[0]), fp16_to_fp32(p[1]),
                      fp16_to_fp32(p[2]), fp16_to_fp32(p[3])};
    return values;
}

static inline lanes_t load_i8x4(const int8_t *p) {
    lanes_t values = {(float)p[0], (float)p[1], (float)p[2], (float)p[3]};
    return values;
}
#endif

float vec_dot_f32_f32(const void *src, const float *x, int n) {
    const float *w = src;
    int i = 0;
    float sum = 0.0f;
#if defined(PICOLM_AVX2)
    lanes_t a = zero_lanes(), b = zero_lanes();
    for (; i + 15 < n; i += 16) {
        a = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a);
        b = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), b);
    }
    sum = sum_lanes(_mm256_add_ps(a, b));
#elif defined(PICOLM_NEON)
    lanes_t a = zero_lanes(), b = zero_lanes();
    for (; i + 7 < n; i += 8) {
        a = vmlaq_f32(a, vld1q_f32(w + i), vld1q_f32(x + i));
        b = vmlaq_f32(b, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    }
    sum = sum_lanes(vaddq_f32(a, b));
#elif defined(PICOLM_SSE2)
    __m128 a = _mm_setzero_ps(), b = _mm_setzero_ps();
    for (; i + 7 < n; i += 8) {
        a = _mm_add_ps(a, _mm_mul_ps(_mm_loadu_ps(w + i), _mm_loadu_ps(x + i)));
        b = _mm_add_ps(b, _mm_mul_ps(_mm_loadu_ps(w + i + 4), _mm_loadu_ps(x + i + 4)));
    }
    sum = hsum_sse(_mm_add_ps(a, b));
#endif
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;
}

float vec_dot_f16_f32(const void *src, const float *x, int n) {
    const uint16_t *w = src;
    int i = 0;
    float sum = 0.0f;
#if defined(PICOLM_AVX2)
    lanes_t acc = zero_lanes();
    for (; i + 7 < n; i += 8) {
        acc = _mm256_fmadd_ps(load_f16(w + i), _mm256_loadu_ps(x + i), acc);
    }
    sum = sum_lanes(acc);
#elif defined(PICOLM_NEON)
    lanes_t acc = zero_lanes();
    for (; i + 3 < n; i += 4) acc = vmlaq_f32(acc, load_f16(w + i), vld1q_f32(x + i));
    sum = sum_lanes(acc);
#endif
    for (; i < n; i++) sum += fp16_to_fp32(w[i]) * x[i];
    return sum;
}

float vec_dot_i8_f32(const int8_t *src, const float *x, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    lanes_t acc = zero_lanes();
    for (; i + 7 < n; i += 8) acc = dot8(acc, load_bytes8(src + i), x + i, 1.0f, 0.0f);
    sum = sum_lanes(acc);
#endif
    for (; i < n; i++) sum += src[i] * x[i];
    return sum;
}

void vec_accumulate_f16(float *acc, const uint16_t *src, int n, float keep, float mix) {
    int i = 0;
#if defined(PICOLM_AVX2)
    lanes_t a = _mm256_set1_ps(keep), b = _mm256_set1_ps(mix);
    for (; i + 7 < n; i += 8) {
        lanes_t prev = _mm256_mul_ps(_mm256_loadu_ps(acc + i), a);
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(load_f16(src + i), b, prev));
    }
#elif defined(PICOLM_NEON)
    for (; i + 3 < n; i += 4) {
        lanes_t prev = vmulq_n_f32(vld1q_f32(acc + i), keep);
        vst1q_f32(acc + i, vmlaq_n_f32(prev, load_f16(src + i), mix));
    }
#endif
    for (; i < n; i++) acc[i] = acc[i] * keep + fp16_to_fp32(src[i]) * mix;
}

void vec_accumulate_i8(float *acc, const int8_t *src, int n, float keep, float mix) {
    int i = 0;
#if defined(PICOLM_AVX2)
    lanes_t a = _mm256_set1_ps(keep), b = _mm256_set1_ps(mix);
    for (; i + 7 < n; i += 8) {
        lanes_t prev = _mm256_mul_ps(_mm256_loadu_ps(acc + i), a);
        lanes_t v = floats_from_bytes(load_bytes8(src + i));
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(v, b, prev));
    }
#elif defined(PICOLM_NEON)
    for (; i + 3 < n; i += 4) {
        lanes_t prev = vmulq_n_f32(vld1q_f32(acc + i), keep);
        vst1q_f32(acc + i, vmlaq_n_f32(prev, load_i8x4(src + i), mix));
    }
#endif
    for (; i < n; i++) acc[i] = acc[i] * keep + src[i] * mix;
}

float vec_dot_q4_0_f32(const void *src, const float *x, int n) {
    const block_q4_0 *blocks = src;
    float sum = 0.0f;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    lanes_t low = zero_lanes(), high = zero_lanes();
#endif
    for (int i = 0; i < n / 32; i++, x += 32) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *q = blocks[i].qs;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
        for (int j = 0; j < 16; j += 8) {
            bytes8_t packed = load_bytes8(q + j);
            low = dot8(low, unpack_bits(packed, 0, 15), x + j, d, -8.0f * d);
            high = dot8(high, unpack_bits(packed, 4, 15), x + j + 16, d, -8.0f * d);
        }
#else
        for (int j = 0; j < 16; j++) {
            sum += d * ((q[j] & 15) - 8) * x[j];
            sum += d * ((q[j] >> 4) - 8) * x[j + 16];
        }
#endif
    }
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    sum = sum_lanes(low) + sum_lanes(high);
#endif
    return sum;
}

float vec_dot_q8_0_f32(const void *src, const float *x, int n) {
    const block_q8_0 *blocks = src;
    float sum = 0.0f;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    lanes_t a = zero_lanes(), b = zero_lanes();
#endif
    for (int i = 0; i < n / 32; i++, x += 32) {
        float d = fp16_to_fp32(blocks[i].d);
        const int8_t *q = blocks[i].qs;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
        a = dot8(a, load_bytes8(q), x, d, 0.0f);
        b = dot8(b, load_bytes8(q + 8), x + 8, d, 0.0f);
        a = dot8(a, load_bytes8(q + 16), x + 16, d, 0.0f);
        b = dot8(b, load_bytes8(q + 24), x + 24, d, 0.0f);
#else
        for (int j = 0; j < 32; j++) sum += (d * q[j]) * x[j];
#endif
    }
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    sum = sum_lanes(a) + sum_lanes(b);
#endif
    return sum;
}

float vec_dot_q4_K_f32(const void *src, const float *x, int n) {
    const block_q4_K *blocks = src;
    float sum = 0.0f;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    lanes_t low = zero_lanes(), high = zero_lanes();
#endif
    for (int i = 0; i < n / 256; i++) {
        const block_q4_K *b = blocks + i;
        float d = fp16_to_fp32(b->d), dmin = fp16_to_fp32(b->dmin);
        for (int group = 0; group < 8; group += 2, x += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(group, b->scales, &s0, &m0);
            get_scale_min_k4(group + 1, b->scales, &s1, &m1);
            const uint8_t *q = b->qs + group * 16;
            float d0 = d * s0, d1 = d * s1, min0 = dmin * m0, min1 = dmin * m1;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
            for (int j = 0; j < 32; j += 8) {
                bytes8_t packed = load_bytes8(q + j);
                low = dot8(low, unpack_bits(packed, 0, 15), x + j, d0, -min0);
                high = dot8(high, unpack_bits(packed, 4, 15), x + j + 32, d1, -min1);
            }
#else
            for (int j = 0; j < 32; j++) {
                sum += (d0 * (q[j] & 15) - min0) * x[j];
                sum += (d1 * (q[j] >> 4) - min1) * x[j + 32];
            }
#endif
        }
    }
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    sum = sum_lanes(low) + sum_lanes(high);
#endif
    return sum;
}

float vec_dot_q6_K_f32(const void *src, const float *x, int n) {
    const block_q6_K *blocks = src;
    float sum = 0.0f;
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    lanes_t lo = zero_lanes(), hi = zero_lanes();
#endif
    for (int i = 0; i < n / 256; i++) {
        const block_q6_K *b = blocks + i;
        float d = fp16_to_fp32(b->d);
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *qh = b->qh + chunk * 32;
            for (int group = 0; group < 4; group++, x += 32) {
                const uint8_t *ql = b->ql + chunk * 64 + (group & 1) * 32;
                int shift = group >= 2 ? 4 : 0;
                const int8_t *sc = b->scales + chunk * 8 + group * 2;
                float d0 = d * sc[0], d1 = d * sc[1];
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
                for (int j = 0; j < 16; j += 8) {
                    bytes8_t l0 = unpack_bits(load_bytes8(ql + j), shift, 15);
                    bytes8_t h0 = unpack_bits(load_bytes8(qh + j), 2 * group, 3);
                    bytes8_t l1 = unpack_bits(load_bytes8(ql + j + 16), shift, 15);
                    bytes8_t h1 = unpack_bits(load_bytes8(qh + j + 16), 2 * group, 3);
                    lo = dot8(lo, combine_q6(l0, h0), x + j, d0, -32.0f * d0);
                    hi = dot8(hi, combine_q6(l1, h1), x + j + 16, d1, -32.0f * d1);
                }
#else
                for (int j = 0; j < 32; j++) {
                    int q = ((ql[j] >> shift) & 15) | (((qh[j] >> (2 * group)) & 3) << 4);
                    sum += (j < 16 ? d0 : d1) * (q - 32) * x[j];
                }
#endif
            }
        }
    }
#if defined(PICOLM_AVX2) || defined(PICOLM_NEON)
    sum = sum_lanes(lo) + sum_lanes(hi);
#endif
    return sum;
}

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32: return vec_dot_f32_f32(src, x, n);
        case GGUF_TYPE_F16: return vec_dot_f16_f32(src, x, n);
        case GGUF_TYPE_Q4_0: return vec_dot_q4_0_f32(src, x, n);
        case GGUF_TYPE_Q8_0: return vec_dot_q8_0_f32(src, x, n);
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q6_K: return vec_dot_q6_K_f32(src, x, n);
        default: {
            /* Keep the legacy Q2_K/Q3_K path allocation-free, one block at a time. */
            float tmp[256], sum = 0.0f;
            size_t bytes = gguf_type_row_size(type, 256);
            const uint8_t *row = src;
            for (int i = 0; i < n / 256; i++, row += bytes, x += 256) {
                dequantize_row(row, tmp, 256, type);
                sum += vec_dot_f32_f32(tmp, x, 256);
            }
            return sum;
        }
    }
}
