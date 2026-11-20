#include "gguf.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    META_U8, META_I8, META_U16, META_I16, META_U32, META_I32, META_F32,
    META_BOOL, META_STRING, META_ARRAY, META_U64, META_I64, META_F64,
};

typedef struct {
    const uint8_t *data;
    size_t size, pos;
    int error;
} reader_t;

typedef struct {
    const char *str;
    size_t len;
} gguf_str_t;

typedef struct {
    gguf_str_t name;
    uint64_t dims[4];
    uint32_t n_dims, type;
    uint64_t offset;
    size_t bytes;
} tensor_info_t;

static const uint8_t *take(reader_t *r, uint64_t size) {
    if (r->error || size > r->size - r->pos) {
        r->error = 1;
        return NULL;
    }
    const uint8_t *p = r->data + r->pos;
    r->pos += (size_t)size;
    return p;
}

static uint64_t read_uint(reader_t *r, size_t width) {
    const uint8_t *p = take(r, width);
    uint64_t value = 0;
    if (p) {
        for (size_t i = 0; i < width; i++) value |= (uint64_t)p[i] << (8 * i);
    }
    return value;
}

static uint32_t read_u32(reader_t *r) { return (uint32_t)read_uint(r, 4); }
static uint64_t read_u64(reader_t *r) { return read_uint(r, 8); }

static gguf_str_t read_gguf_string(reader_t *r) {
    uint64_t len = read_u64(r);
    const uint8_t *data = take(r, len);
    return (gguf_str_t){(const char *)data, data ? (size_t)len : 0};
}

static int str_eq(gguf_str_t s, const char *literal) {
    size_t n = strlen(literal);
    return s.str && s.len == n && memcmp(s.str, literal, n) == 0;
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case META_U8: case META_I8: case META_BOOL: return 1;
        case META_U16: case META_I16: return 2;
        case META_U32: case META_I32: case META_F32: return 4;
        case META_U64: case META_I64: case META_F64: return 8;
        default: return 0;
    }
}

static void skip_meta_array(reader_t *r, uint32_t type, uint64_t count, unsigned depth) {
    size_t size = scalar_size(type);
    if (size) {
        if (count <= (r->size - r->pos) / size) take(r, count * size);
        else r->error = 1;
    } else if (type == META_STRING && count <= (r->size - r->pos) / 8) {
        for (uint64_t i = 0; i < count && !r->error; i++) read_gguf_string(r);
    } else if (type == META_ARRAY && depth < 16 && count <= (r->size - r->pos) / 12) {
        for (uint64_t i = 0; i < count && !r->error; i++) {
            uint32_t subtype = read_u32(r);
            uint64_t subcount = read_u64(r);
            skip_meta_array(r, subtype, subcount, depth + 1);
        }
    } else {
        r->error = 1;
    }
}

static void skip_meta_value(reader_t *r, uint32_t type) {
    size_t size = scalar_size(type);
    if (size) {
        take(r, size);
    } else if (type == META_STRING) {
        read_gguf_string(r);
    } else if (type == META_ARRAY) {
        uint32_t subtype = read_u32(r);
        uint64_t count = read_u64(r);
        skip_meta_array(r, subtype, count, 0);
    } else {
        r->error = 1;
    }
}

static int read_dimension(reader_t *r, uint32_t type, int allow_nonpositive) {
    int is_signed = type == META_I8 || type == META_I16 ||
                    type == META_I32 || type == META_I64;
    size_t width = scalar_size(type);
    if (!width || type == META_F32 || type == META_F64 || type == META_BOOL) {
        r->error = 1;
        return 0;
    }
    uint64_t value = read_uint(r, width);
    if (is_signed && (value >> (width * 8 - 1))) {
        if (!allow_nonpositive) r->error = 1;
        return 0;
    }
    if (value > INT_MAX) {
        r->error = 1;
        return 0;
    }
    return (int)value;
}

static float read_positive_float(reader_t *r, uint32_t type) {
    float value = 0.0f;
    if (type == META_F32) {
        uint32_t bits = read_u32(r);
        memcpy(&value, &bits, sizeof(value));
    } else if (type == META_F64) {
        uint64_t bits = read_u64(r);
        double wide;
        memcpy(&wide, &bits, sizeof(wide));
        if (wide > 0.0 && wide <= 3.402823466e38) value = (float)wide;
    }
    if (!isfinite(value) || value <= 0.0f) r->error = 1;
    return value;
}

static int read_bool(reader_t *r, uint32_t type) {
    if (type != META_BOOL) {
        r->error = 1;
        return 0;
    }
    uint64_t value = read_uint(r, 1);
    if (value > 1) r->error = 1;
    return value == 1;
}

static void unsupported(reader_t *r, const char *feature) {
    fprintf(stderr, "GGUF: unsupported %s\n", feature);
    r->error = 1;
}

static void read_token_array(reader_t *r, model_t *m, uint32_t type, int scores) {
    if (type != META_ARRAY) {
        r->error = 1;
        return;
    }
    uint32_t subtype = read_u32(r);
    uint64_t count = read_u64(r);
    if (count > INT_MAX || subtype != (scores ? META_F32 : META_STRING)) {
        r->error = 1;
        return;
    }
    size_t start = r->pos;
    skip_meta_array(r, subtype, count, 0);
    if (r->error) return;
    if (scores) {
        if (m->tok_scores_data) r->error = 1;
        m->tok_scores_data = r->data + start;
        m->tok_n_scores = count;
    } else {
        if (m->tok_tokens_data) r->error = 1;
        m->tok_tokens_data = r->data + start;
        m->tok_tokens_size = r->pos - start;
        m->tok_n_tokens = count;
    }
}

static void read_metadata(reader_t *r, model_t *m, uint64_t count, int *rope_dims) {
    model_config_t *cfg = &m->config;
    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->rms_epsilon = 1e-5f;
    cfg->max_seq_len = 2048;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    for (uint64_t i = 0; i < count && !r->error; i++) {
        gguf_str_t key = read_gguf_string(r);
        uint32_t type = read_u32(r);
        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")) {
            cfg->n_embd = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")) {
            cfg->n_ffn = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.attention.head_count")) {
            cfg->n_heads = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.attention.head_count_kv")) {
            cfg->n_kv_heads = read_dimension(r, type, 1);
        } else if (str_eq(key, "llama.block_count")) {
            cfg->n_layers = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.context_length")) {
            cfg->max_seq_len = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.rope.freq_base")) {
            cfg->rope_freq_base = read_positive_float(r, type);
        } else if (str_eq(key, "llama.rope.dimension_count")) {
            *rope_dims = read_dimension(r, type, 0);
            if (!*rope_dims) r->error = 1;
        } else if (str_eq(key, "llama.rope.scaling.type")) {
            if (type != META_STRING || !str_eq(read_gguf_string(r), "none")) {
                unsupported(r, "RoPE scaling type");
            }
        } else if (str_eq(key, "llama.rope.scaling.factor") ||
                   str_eq(key, "llama.rope.scale_linear") || str_eq(key, "llama.rope.scale")) {
            if (read_positive_float(r, type) != 1.0f) unsupported(r, "RoPE scale factor");
        } else if (str_eq(key, "llama.attention.layer_norm_rms_epsilon")) {
            cfg->rms_epsilon = read_positive_float(r, type);
        } else if (str_eq(key, "general.alignment")) {
            cfg->alignment = read_dimension(r, type, 0);
        } else if (str_eq(key, "llama.vocab_size")) {
            cfg->vocab_size = read_dimension(r, type, 0);
        } else if (str_eq(key, "general.architecture")) {
            if (type != META_STRING || !str_eq(read_gguf_string(r), "llama")) r->error = 1;
        } else if (str_eq(key, "tokenizer.ggml.model")) {
            if (type != META_STRING || !str_eq(read_gguf_string(r), "llama")) {
                unsupported(r, "tokenizer (expected llama SentencePiece)");
            }
        } else if (str_eq(key, "tokenizer.ggml.add_bos_token")) {
            m->tok_no_bos = !read_bool(r, type);
        } else if (str_eq(key, "tokenizer.ggml.add_space_prefix")) {
            m->tok_no_space_prefix = !read_bool(r, type);
        } else if (str_eq(key, "tokenizer.ggml.remove_extra_whitespaces")) {
            if (read_bool(r, type)) unsupported(r, "tokenizer whitespace normalization");
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            m->tok_bos_id = (uint32_t)read_dimension(r, type, 0);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            m->tok_eos_id = (uint32_t)read_dimension(r, type, 0);
        } else if (str_eq(key, "tokenizer.ggml.unknown_token_id")) {
            m->tok_unk_id = (uint32_t)read_dimension(r, type, 0);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
            read_token_array(r, m, type, 0);
        } else if (str_eq(key, "tokenizer.ggml.scores")) {
            read_token_array(r, m, type, 1);
        } else {
            skip_meta_value(r, type);
        }
    }
}

static int read_tensor(reader_t *r, tensor_info_t *t) {
    t->name = read_gguf_string(r);
    t->n_dims = read_u32(r);
    if (r->error || t->name.len == 0 || t->name.len > 64 ||
        memchr(t->name.str, '\0', t->name.len) || t->n_dims == 0 || t->n_dims > 4) return 0;
    for (uint32_t i = 0; i < t->n_dims; i++) t->dims[i] = read_u64(r);
    t->type = read_u32(r);
    t->offset = read_u64(r);
    if (r->error || t->type > INT_MAX || t->dims[0] > INT_MAX) return 0;
    t->bytes = gguf_type_row_size((gguf_type_t)t->type, (int)t->dims[0]);
    if (!t->bytes) return 0;
    for (uint32_t i = 1; i < t->n_dims; i++) {
        if (!t->dims[i] || t->dims[i] > SIZE_MAX / t->bytes) return 0;
        t->bytes *= (size_t)t->dims[i];
    }
    return 1;
}

static int by_tensor_name(const void *a, const void *b) {
    gguf_str_t x = ((const tensor_info_t *)a)->name;
    gguf_str_t y = ((const tensor_info_t *)b)->name;
    size_t shared = x.len < y.len ? x.len : y.len;
    int cmp = memcmp(x.str, y.str, shared);
    if (cmp) return cmp;
    return (x.len > y.len) - (x.len < y.len);
}

static const tensor_info_t *find_tensor(const tensor_info_t *tensors, size_t count,
                                        const char *name) {
    tensor_info_t key = {.name = {name, strlen(name)}};
    return bsearch(&key, tensors, count, sizeof(*tensors), by_tensor_name);
}

static int bind_tensor(const tensor_info_t *t, const char *name, const uint8_t *base,
                       int columns, int rows, const void **dst, gguf_type_t *type) {
    if (!t || !gguf_type_supported((gguf_type_t)t->type) ||
        t->n_dims != (rows ? 2u : 1u) || t->dims[0] != (uint64_t)columns ||
        (rows && t->dims[1] != (uint64_t)rows)) {
        fprintf(stderr, "GGUF: missing, unsupported or incompatible tensor %s\n", name);
        return 0;
    }
    *dst = base + (size_t)t->offset;
    *type = (gguf_type_t)t->type;
    return 1;
}

static int bind_weights(model_t *m, const tensor_info_t *tensors, size_t count,
                        const uint8_t *base) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    const char *name = "token_embd.weight";
    const tensor_info_t *embedding = find_tensor(tensors, count, name);
    if (!c->vocab_size && embedding && embedding->n_dims == 2 && embedding->dims[1] <= INT_MAX) {
        c->vocab_size = (int)embedding->dims[1];
    }
    if (c->vocab_size <= 0 || !m->tok_n_tokens ||
        m->tok_n_tokens != (uint64_t)c->vocab_size ||
        (m->tok_n_scores && m->tok_n_scores != m->tok_n_tokens) ||
        m->tok_bos_id >= m->tok_n_tokens || m->tok_eos_id >= m->tok_n_tokens ||
        m->tok_unk_id >= m->tok_n_tokens) {
        fprintf(stderr, "GGUF: inconsistent vocabulary or special token IDs\n");
        return 0;
    }
    if (!bind_tensor(embedding, name, base, c->n_embd, c->vocab_size,
                     &w->token_embd, &w->type_token_embd)) return 0;
    name = "output_norm.weight";
    if (!bind_tensor(find_tensor(tensors, count, name), name, base, c->n_embd, 0,
                     &w->output_norm, &w->type_output_norm)) return 0;
    name = "output.weight";
    const tensor_info_t *output = find_tensor(tensors, count, name);
    if (output) {
        if (!bind_tensor(output, name, base, c->n_embd, c->vocab_size,
                         &w->output, &w->type_output)) return 0;
    } else {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

    int dim = c->n_embd, ffn = c->n_ffn, kv_dim = c->n_kv_heads * c->head_dim;
    for (int i = 0; i < c->n_layers; i++) {
        layer_weights_t *l = w->layers + i;
        const struct {
            const char *suffix;
            int cols, rows;
            const void **dst;
            gguf_type_t *type;
        } specs[] = {
            {"attn_norm", dim, 0, &l->attn_norm, &l->type_attn_norm},
            {"attn_q", dim, dim, &l->attn_q, &l->type_attn_q},
            {"attn_k", dim, kv_dim, &l->attn_k, &l->type_attn_k},
            {"attn_v", dim, kv_dim, &l->attn_v, &l->type_attn_v},
            {"attn_output", dim, dim, &l->attn_output, &l->type_attn_output},
            {"ffn_norm", dim, 0, &l->ffn_norm, &l->type_ffn_norm},
            {"ffn_gate", dim, ffn, &l->ffn_gate, &l->type_ffn_gate},
            {"ffn_down", ffn, dim, &l->ffn_down, &l->type_ffn_down},
            {"ffn_up", dim, ffn, &l->ffn_up, &l->type_ffn_up},
        };
        for (size_t j = 0; j < sizeof(specs) / sizeof(specs[0]); j++) {
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s.weight", i, specs[j].suffix);
            const tensor_info_t *t = find_tensor(tensors, count, tensor_name);
            if (!bind_tensor(t, tensor_name, base, specs[j].cols, specs[j].rows,
                             specs[j].dst, specs[j].type)) return 0;
        }
    }
    c->weight_type = w->layers[0].type_attn_q;
    return 1;
}

int gguf_parse(model_t *m, int max_seq_len) {
    reader_t r = {.data = m->mmap_addr, .size = m->mmap_size};
    uint32_t magic = read_u32(&r);
    uint32_t version = read_u32(&r);
    uint64_t count = read_u64(&r), metadata = read_u64(&r);
    if (r.error || magic != GGUF_MAGIC || version < 2 || version > 3 ||
        !count || metadata > (r.size - r.pos) / 13) {
        fprintf(stderr, "GGUF: invalid header or unsupported version\n");
        return -1;
    }
    int rope_dims = 0;
    read_metadata(&r, m, metadata, &rope_dims);
    model_config_t *c = &m->config;
    if (c->n_kv_heads <= 0) c->n_kv_heads = c->n_heads;
    if (r.error || c->n_embd <= 0 || c->n_ffn <= 0 || c->max_seq_len <= 0 ||
        c->n_layers <= 0 || c->n_layers > MAX_LAYERS ||
        c->n_heads <= 0 || c->n_kv_heads <= 0 || c->n_heads % c->n_kv_heads ||
        c->n_embd % c->n_heads || (c->n_embd / c->n_heads) % 2 ||
        c->alignment <= 0 || c->alignment % 8 ||
        count > (r.size - r.pos) / 32 || count > SIZE_MAX / sizeof(tensor_info_t)) {
        fprintf(stderr, "GGUF: invalid metadata, model dimensions or tensor count\n");
        return -1;
    }
    c->head_dim = c->n_embd / c->n_heads;
    if (rope_dims && rope_dims != c->head_dim) {
        fprintf(stderr, "GGUF: unsupported partial-head RoPE\n");
        return -1;
    }
    if (max_seq_len > 0 && max_seq_len < c->max_seq_len) c->max_seq_len = max_seq_len;

    size_t n = (size_t)count;
    tensor_info_t *tensors = malloc(n * sizeof(*tensors));
    if (!tensors) {
        fprintf(stderr, "GGUF: unable to allocate tensor metadata\n");
        return -1;
    }
    const char *error = "invalid tensor descriptor";
    for (size_t i = 0; i < n; i++) {
        if (!read_tensor(&r, tensors + i)) goto fail;
    }

    size_t align = (size_t)c->alignment;
    size_t pad = (align - r.pos % align) % align;
    error = "tensor data extends beyond the file";
    const uint8_t *base = take(&r, pad);
    if (!base) goto fail;
    base = r.data + r.pos;
    size_t remaining = r.size - r.pos;
    for (size_t i = 0; i < n; i++) {
        const tensor_info_t *t = tensors + i;
        if (t->offset % align || t->offset > remaining ||
            t->bytes > remaining - (size_t)t->offset) goto fail;
    }
    qsort(tensors, n, sizeof(*tensors), by_tensor_name);
    error = "duplicate tensor name";
    for (size_t i = 1; i < n; i++) {
        if (!by_tensor_name(tensors + i - 1, tensors + i)) goto fail;
    }
    error = "unsupported RoPE frequency tensor";
    if (find_tensor(tensors, n, "rope_freqs.weight") ||
        find_tensor(tensors, n, "rope_factors_long.weight") ||
        find_tensor(tensors, n, "rope_factors_short.weight")) goto fail;
    int ok = bind_weights(m, tensors, n, base);
    free(tensors);
    return ok ? 0 : -1;

fail:
    fprintf(stderr, "GGUF: %s\n", error);
    free(tensors);
    return -1;
}
