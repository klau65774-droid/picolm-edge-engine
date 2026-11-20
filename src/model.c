#include "model.h"
#include "fileio.h"
#include "gguf.h"
#include "tensor.h"
#include "threadpool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int mmap_file(model_t *m, const char *path) {
#ifdef _WIN32
    wchar_t *wide = utf8_to_wide(path);
    HANDLE file = wide ? CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ, NULL,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)
                       : CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(wide);
    if (file == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
        (uint64_t)size.QuadPart > PTRDIFF_MAX) {
        CloseHandle(file);
        return -1;
    }
    HANDLE mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
    void *addr = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : NULL;
    if (!addr) {
        if (mapping) CloseHandle(mapping);
        CloseHandle(file);
        return -1;
    }
    m->mmap_size = (size_t)size.QuadPart;
    m->file_handle = file;
    m->map_handle = mapping;
#else
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        (uintmax_t)st.st_size > PTRDIFF_MAX) {
        close(fd);
        return -1;
    }
    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return -1;
    }
    m->mmap_size = (size_t)st.st_size;
    m->fd = fd;
#endif
    m->mmap_addr = addr;
    return 0;
}

static void munmap_file(model_t *m) {
    if (!m->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(m->mmap_addr);
    CloseHandle(m->map_handle);
    CloseHandle(m->file_handle);
#else
    munmap(m->mmap_addr, m->mmap_size);
    close(m->fd);
#endif
    m->mmap_addr = NULL;
}

static int mul_size(size_t a, size_t b, size_t *size) {
    if (b && a > SIZE_MAX / b) return 0;
    *size = a * b;
    return 1;
}

static int add_size(size_t a, size_t b, size_t *size) {
    if (a > SIZE_MAX - b) return 0;
    *size = a + b;
    return 1;
}

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int half = c->head_dim / 2;
    for (int i = 0; i < half; i++) {
        s->rope_cos[i] = powf(c->rope_freq_base, (float)(2 * i) / (float)c->head_dim);
    }
    for (int pos = 1; pos < c->max_seq_len; pos++) {
        size_t row = (size_t)pos * half;
        for (int i = 0; i < half; i++) {
            /* Use the original division to preserve RoPE rounding. */
            float theta = (float)pos / s->rope_cos[i];
            s->rope_cos[row + i] = cosf(theta);
            s->rope_sin[row + i] = sinf(theta);
        }
    }
    for (int i = 0; i < half; i++) s->rope_cos[i] = 1.0f;
}

static int allocate_run_state(model_t *m) {
    /* Bound eager allocations even on systems where malloc overcommits memory. */
    const size_t max_runtime = (size_t)1 << 30;
    const model_config_t *c = &m->config;
    run_state_t *s = &m->state;
    size_t dim = (size_t)c->n_embd, half = (size_t)c->head_dim / 2;
    size_t activations, ffn, rope, norms, floats, float_bytes;
    size_t slots, data_bytes, scale_bytes = 0, plane_bytes, kv_bytes, total;
    size_t elem_bytes = s->kv_quant == KV_INT8 ? 1 : sizeof(uint16_t);

    if (!mul_size(4, dim, &activations) || !mul_size(2, (size_t)c->n_ffn, &ffn) ||
        !mul_size((size_t)c->max_seq_len, half, &rope) ||
        !mul_size(dim, (size_t)(2 * c->n_layers + 1), &norms) ||
        !add_size(activations, ffn, &floats) ||
        !add_size(floats, (size_t)c->vocab_size, &floats) ||
        !add_size(floats, rope, &floats) || !add_size(floats, rope, &floats) ||
        !add_size(floats, norms, &floats) || !mul_size(floats, sizeof(float), &float_bytes) ||
        !mul_size((size_t)c->n_layers, (size_t)c->max_seq_len, &slots) ||
        !mul_size(slots, (size_t)c->n_kv_heads, &slots) ||
        !mul_size(slots, (size_t)c->head_dim, &data_bytes) ||
        !mul_size(data_bytes, elem_bytes, &data_bytes) ||
        (s->kv_quant == KV_INT8 && !mul_size(slots, sizeof(uint16_t), &scale_bytes)) ||
        !add_size(data_bytes, scale_bytes, &plane_bytes) || !mul_size(plane_bytes, 2, &kv_bytes) ||
        !add_size(float_bytes, kv_bytes, &total) || total > max_runtime) {
        fprintf(stderr, "Model runtime allocation exceeds 1024 MiB. Try a smaller -c context.\n");
        return -1;
    }
    s->mem_block = calloc(floats, sizeof(float));
    s->kv_block = calloc(1, kv_bytes);
    if (!s->mem_block || !s->kv_block) {
        fprintf(stderr, "Could not allocate %zu bytes of runtime state\n", total);
        return -1;
    }
    s->mem_size = total;
    s->kv_size = kv_bytes;
    float *p = s->mem_block;
    s->x = p; p += dim;
    s->xb = p; p += dim;
    s->xb2 = p; p += dim;
    s->q = p; p += dim;
    s->hb = p; p += c->n_ffn;
    s->hb2 = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->rope_cos = p; p += rope;
    s->rope_sin = p; p += rope;
    s->norm_weights = p;

    uint8_t *kv = s->kv_block;
    s->key_cache.data = kv;
    s->val_cache.data = kv + data_bytes;
    if (s->kv_quant == KV_INT8) {
        s->key_cache.scales = (uint16_t *)(kv + 2 * data_bytes);
        s->val_cache.scales = (uint16_t *)(kv + 2 * data_bytes + scale_bytes);
    }

    for (int l = 0; l < c->n_layers; l++) {
        const layer_weights_t *w = m->weights.layers + l;
        s->attn_norm_w[l] = p;
        dequantize_row(w->attn_norm, p, c->n_embd, w->type_attn_norm);
        p += dim;
        s->ffn_norm_w[l] = p;
        dequantize_row(w->ffn_norm, p, c->n_embd, w->type_ffn_norm);
        p += dim;
    }
    s->output_norm_w = p;
    dequantize_row(m->weights.output_norm, p, c->n_embd, m->weights.type_output_norm);
    init_rope_tables(s, c);
    return 0;
}

int model_load_ex(model_t *m, const char *path, int context, kv_quant_t kv_quant) {
    memset(m, 0, sizeof(*m));
    if (kv_quant != KV_FP16 && kv_quant != KV_INT8) return -1;
    if (mmap_file(m, path) != 0) {
        fprintf(stderr, "Cannot map model file: %s\n", path);
        return -1;
    }
    m->state.kv_quant = kv_quant;
    if (gguf_parse(m, context) != 0) {
        model_free(m);
        return -1;
    }
    quant_init();
    if (allocate_run_state(m) != 0) {
        model_free(m);
        return -1;
    }
    const model_config_t *c = &m->config;
    fprintf(stderr, "Model: dim=%d ffn=%d layers=%d heads=%d kv_heads=%d vocab=%d context=%d\n",
            c->n_embd, c->n_ffn, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size,
            c->max_seq_len);
    return 0;
}

int model_load(model_t *m, const char *path, int max_seq_len) {
    return model_load_ex(m, path, max_seq_len, KV_FP16);
}

void model_free(model_t *m) {
    free(m->state.mem_block);
    free(m->state.kv_block);
    memset(&m->state, 0, sizeof(m->state));
    munmap_file(m);
}

static void store_kv(kv_plane_t *plane, const float *x, size_t slot, int heads, int dim) {
    if (!plane->scales) {
        uint16_t *dst = (uint16_t *)plane->data + slot * dim;
        for (int i = 0; i < heads * dim; i++) dst[i] = fp32_to_fp16(x[i]);
        return;
    }
    int8_t *dst = (int8_t *)plane->data + slot * dim;
    for (int h = 0; h < heads; h++, x += dim, dst += dim) {
        float max_abs = 0.0f;
        for (int i = 0; i < dim; i++) {
            if (isfinite(x[i])) max_abs = fmaxf(max_abs, fabsf(x[i]));
        }
        uint16_t bits = fp32_to_fp16(fminf(max_abs / 127.0f, 65504.0f));
        if (!bits && max_abs > 0.0f) bits = 1; /* Smallest FP16 subnormal. */
        plane->scales[slot + h] = bits;
        float inverse = bits ? 1.0f / fp16_to_fp32(bits) : 0.0f;
        for (int i = 0; i < dim; i++) {
            float value = isfinite(x[i]) ? x[i] * inverse : 0.0f;
            dst[i] = (int8_t)roundf(fminf(127.0f, fmaxf(-127.0f, value)));
        }
    }
}

typedef struct {
    const model_config_t *config;
    run_state_t *state;
    size_t layer_start;
    int pos;
    float inv_sqrt_d;
} attention_t;

/* Online softmax keeps only the current max, denominator and weighted value. */
static void attention_heads(void *arg, int first, int end) {
    const attention_t *a = arg;
    const model_config_t *c = a->config;
    run_state_t *s = a->state;
    int d = c->head_dim, group = c->n_heads / c->n_kv_heads;
    int int8 = s->kv_quant == KV_INT8;
    for (int h = first; h < end; h++) {
        const float *q = s->q + (size_t)h * d;
        float *acc = s->xb + (size_t)h * d;
        memset(acc, 0, (size_t)d * sizeof(*acc));
        float best = -INFINITY, denominator = 0.0f;
        size_t slot = a->layer_start + (size_t)(h / group);
        for (int t = 0; t <= a->pos; t++, slot += c->n_kv_heads) {
            size_t offset = slot * d;
            float score;
            if (int8) {
                const int8_t *k = (const int8_t *)s->key_cache.data + offset;
                score = vec_dot_i8_f32(k, q, d) * fp16_to_fp32(s->key_cache.scales[slot]);
            } else {
                const uint16_t *k = (const uint16_t *)s->key_cache.data + offset;
                score = vec_dot_f16_f32(k, q, d);
            }
            score *= a->inv_sqrt_d;

            float keep = 1.0f, mix;
            if (score > best) {
                keep = expf(best - score);
                mix = 1.0f;
                best = score;
            } else {
                mix = expf(score - best);
            }
            denominator = denominator * keep + mix;
            if (int8) {
                const int8_t *v = (const int8_t *)s->val_cache.data + offset;
                vec_accumulate_i8(acc, v, d, keep, mix * fp16_to_fp32(s->val_cache.scales[slot]));
            } else {
                const uint16_t *v = (const uint16_t *)s->val_cache.data + offset;
                vec_accumulate_f16(acc, v, d, keep, mix);
            }
        }
        float inverse = 1.0f / denominator;
        for (int i = 0; i < d; i++) acc[i] *= inverse;
    }
}

float *model_forward_ex(model_t *m, int token, int pos, int compute_logits) {
    const model_config_t *c = &m->config;
    const model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;
    if (!s->mem_block || token < 0 || token >= c->vocab_size ||
        pos < 0 || pos >= c->max_seq_len) return NULL;

    int dim = c->n_embd, ffn = c->n_ffn;
    int kv_dim = c->n_kv_heads * c->head_dim;
    float inv_sqrt_d = 1.0f / sqrtf((float)c->head_dim);
    size_t rope_offset = (size_t)pos * (c->head_dim / 2);
    size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
    const void *embedding = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
    dequantize_row(embedding, s->x, dim, w->type_token_embd);

    for (int l = 0; l < c->n_layers; l++) {
        const layer_weights_t *lw = w->layers + l;
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_epsilon);
        matmul(s->q, s->xb, lw->attn_q, dim, dim, lw->type_attn_q);
        matmul(s->xb2, s->xb, lw->attn_k, dim, kv_dim, lw->type_attn_k);
        rope(s->q, s->xb2, c->head_dim, c->n_heads, c->n_kv_heads,
             s->rope_cos + rope_offset, s->rope_sin + rope_offset);

        size_t layer_start = (size_t)l * c->max_seq_len * c->n_kv_heads;
        size_t slot = layer_start + (size_t)pos * c->n_kv_heads;
        store_kv(&s->key_cache, s->xb2, slot, c->n_kv_heads, c->head_dim);
        matmul(s->xb2, s->xb, lw->attn_v, dim, kv_dim, lw->type_attn_v);
        store_kv(&s->val_cache, s->xb2, slot, c->n_kv_heads, c->head_dim);

        attention_t attention = {c, s, layer_start, pos, inv_sqrt_d};
        if (pos >= 64 && threadpool_size() > 1) {
            threadpool_run(c->n_heads, attention_heads, &attention);
        } else {
            attention_heads(&attention, 0, c->n_heads);
        }
        matmul(s->xb2, s->xb, lw->attn_output, dim, dim, lw->type_attn_output);
        vec_add(s->x, s->xb2, dim);

        rmsnorm(s->xb, s->x, s->ffn_norm_w[l], dim, c->rms_epsilon);
        matmul(s->hb, s->xb, lw->ffn_gate, dim, ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up, dim, ffn, lw->type_ffn_up);
        silu(s->hb, ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, ffn);
        matmul(s->xb, s->hb, lw->ffn_down, ffn, dim, lw->type_ffn_down);
        vec_add(s->x, s->xb, dim);
    }

    if (!compute_logits) return NULL;
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_epsilon);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);
    return s->logits;
}

float *model_forward(model_t *m, int token, int pos) {
    return model_forward_ex(m, token, pos, 1);
}

