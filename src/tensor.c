#include "tensor.h"
#include "threadpool.h"
#include <math.h>

typedef struct {
    float       *out;
    const float *x;
    const char  *W;
    size_t       row_bytes;
    int          n;        /* input dimension */
    gguf_type_t  qtype;
} matmul_task_t;

static void matmul_rows(void *arg, int first, int end) {
    const matmul_task_t *t = arg;
    for (int i = first; i < end; i++) {
        t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                            t->x, t->n, t->qtype);
    }
}

void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype) {
    matmul_task_t task = {out, x, W, gguf_type_row_size(qtype, n), n, qtype};
    if (d < 4) matmul_rows(&task, 0, d);
    else threadpool_run(d, matmul_rows, &task);
}

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;

#ifdef PICOLM_NEON
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif

    ss = 1.0f / sqrtf(ss / (float)size + eps);

#ifdef PICOLM_NEON
    float32x4_t scale = vdupq_n_f32(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_SSE2)
    __m128 scale = _mm_set1_ps(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 w = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;

#ifdef PICOLM_NEON
    float32x4_t inv_v = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_SSE2)
    __m128 inv_v = _mm_set1_ps(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#else
    for (int i = 0; i < size; i++) x[i] *= inv;
#endif
}

/* Rotary position encoding using pre-computed cos/sin tables */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos) {
    int half = head_dim / 2;

    /* Apply RoPE to all query heads */
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
#ifdef PICOLM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            /* Load pairs: (q0,q1), (q2,q3), ... as interleaved */
            float32x4x2_t qv = vld2q_f32(qh + i * 2);
            float32x4_t cv = vld1q_f32(cos_pos + i);
            float32x4_t sv = vld1q_f32(sin_pos + i);
            /* q_even = q0*cos - q1*sin, q_odd = q0*sin + q1*cos */
            float32x4_t new_even = vmlsq_f32(vmulq_f32(qv.val[0], cv), qv.val[1], sv);
            float32x4_t new_odd  = vmlaq_f32(vmulq_f32(qv.val[0], sv), qv.val[1], cv);
            float32x4x2_t result = {{ new_even, new_odd }};
            vst2q_f32(qh + i * 2, result);
        }
        for (; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#else
        for (int i = 0; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#endif
    }

    /* Apply RoPE to all KV heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
            kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
        }
    }
}

void silu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void elemwise_mul(float *out, const float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
#endif
}

void vec_add(float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(a + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#else
    for (int i = 0; i < size; i++) a[i] += b[i];
#endif
}
