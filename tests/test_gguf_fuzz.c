#include "gguf.h"
#include "model.h"

#include "check.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

enum { MICRO_DIM = 32, MICRO_FFN = 64, MICRO_VOCAB = 8 };

typedef struct {
    uint8_t bytes[32768];
    size_t size, head, layers, epsilon, alignment, token_type, token_count;
    size_t first_name, first_rank, first_dim, first_rows, first_offset, weight_data;
} fixture_t;

typedef struct {
    const char *tokenizer, *scaling, *extra;
    unsigned rope_dims;
    int no_json;
} fixture_options_t;

static void bytes(fixture_t *f, const void *data, size_t n) {
    CHECK(n <= sizeof(f->bytes) - f->size);
    memcpy(f->bytes + f->size, data, n);
    f->size += n;
}

static void u32(fixture_t *f, uint32_t n) { bytes(f, &n, sizeof(n)); }
static void u64(fixture_t *f, uint64_t n) { bytes(f, &n, sizeof(n)); }

static void string(fixture_t *f, const char *s) {
    size_t n = strlen(s);
    u64(f, n);
    bytes(f, s, n);
}

static size_t meta(fixture_t *f, const char *key, uint32_t value) {
    string(f, key);
    u32(f, 4);
    size_t pos = f->size;
    u32(f, value);
    return pos;
}

static void pad(fixture_t *f) {
    while (f->size % 32) {
        uint8_t zero = 0;
        bytes(f, &zero, 1);
    }
}

static void meta_string(fixture_t *f, const char *key, const char *value) {
    string(f, key);
    u32(f, 8);
    string(f, value);
}

static fixture_t micro_gguf(fixture_options_t options) {
    static const struct { const char *name; int cols, rows; gguf_type_t type; } plan[] = {
        {"token_embd.weight", MICRO_DIM, MICRO_VOCAB, GGUF_TYPE_Q4_0},
        {"output_norm.weight", MICRO_DIM, 0, GGUF_TYPE_F32},
        {"blk.0.attn_norm.weight", MICRO_DIM, 0, GGUF_TYPE_F32},
        {"blk.0.attn_q.weight", MICRO_DIM, MICRO_DIM, GGUF_TYPE_Q4_0},
        {"blk.0.attn_k.weight", MICRO_DIM, MICRO_DIM, GGUF_TYPE_Q8_0},
        {"blk.0.attn_v.weight", MICRO_DIM, MICRO_DIM, GGUF_TYPE_Q8_0},
        {"blk.0.attn_output.weight", MICRO_DIM, MICRO_DIM, GGUF_TYPE_Q4_0},
        {"blk.0.ffn_norm.weight", MICRO_DIM, 0, GGUF_TYPE_F32},
        {"blk.0.ffn_gate.weight", MICRO_DIM, MICRO_FFN, GGUF_TYPE_Q4_0},
        {"blk.0.ffn_down.weight", MICRO_FFN, MICRO_DIM, GGUF_TYPE_Q8_0},
        {"blk.0.ffn_up.weight", MICRO_DIM, MICRO_FFN, GGUF_TYPE_Q4_0},
        {"fixture.extra.weight", 256, 0, GGUF_TYPE_Q4_K},
    };
    const size_t tensors = sizeof(plan) / sizeof(plan[0]);
    fixture_t f = {0};
    u32(&f, GGUF_MAGIC);
    u32(&f, 3);
    u64(&f, tensors);
    u64(&f, 11);
    meta(&f, "llama.embedding_length", MICRO_DIM);
    meta(&f, "llama.feed_forward_length", MICRO_FFN);
    f.head = meta(&f, "llama.attention.head_count", 4);
    f.layers = meta(&f, "llama.block_count", 1);
    f.alignment = meta(&f, "general.alignment", 32);
    meta(&f, "llama.context_length", 16);
    string(&f, "llama.attention.layer_norm_rms_epsilon");
    u32(&f, 6);
    f.epsilon = f.size;
    float eps = 2e-5f;
    bytes(&f, &eps, sizeof(eps));
    meta(&f, "llama.rope.dimension_count", options.rope_dims ? options.rope_dims : 8);
    meta_string(&f, "llama.rope.scaling.type", options.scaling ? options.scaling : "none");
    meta_string(&f, "tokenizer.ggml.model", options.tokenizer ? options.tokenizer : "llama");
    string(&f, "tokenizer.ggml.tokens");
    u32(&f, 9);
    f.token_type = f.size;
    u32(&f, 8);
    f.token_count = f.size;
    u64(&f, MICRO_VOCAB);
    const char *const vocab[MICRO_VOCAB] = {"<unk>", "<s>", "</s>",
                                           options.no_json ? "xyz" : "\xE2\x96\x81", "x",
                                           options.no_json ? "ab" : "{}",
                                           options.no_json ? "a" : "{", options.no_json ? "b" : "}"};
    for (size_t i = 0; i < MICRO_VOCAB; i++) string(&f, vocab[i]);

    size_t offsets = 0;
    for (size_t i = 0; i < tensors; i++) {
        if (!i) f.first_name = f.size;
        string(&f, i + 1 == tensors && options.extra ? options.extra : plan[i].name);
        if (!i) f.first_rank = f.size;
        u32(&f, plan[i].rows ? 2 : 1);
        if (!i) f.first_dim = f.size;
        u64(&f, (uint64_t)plan[i].cols);
        if (!i) f.first_rows = f.size;
        if (plan[i].rows) u64(&f, (uint64_t)plan[i].rows);
        u32(&f, (uint32_t)plan[i].type);
        if (!i) f.first_offset = f.size;
        u64(&f, offsets);
        size_t row = gguf_type_row_size(plan[i].type, plan[i].cols);
        size_t size = row * (size_t)(plan[i].rows ? plan[i].rows : 1);
        offsets += (size + 31) & ~(size_t)31;
    }
    pad(&f);
    f.weight_data = f.size;
    for (size_t i = 0; i < tensors; i++) {
        gguf_type_t type = plan[i].type;
        int rows = plan[i].rows ? plan[i].rows : 1;
        int count = plan[i].cols * rows / gguf_type_block_size(type);
        for (int j = 0; j < count; j++) {
            if (type == GGUF_TYPE_F32) {
                float one = 1.0f;
                bytes(&f, &one, sizeof(one));
            } else if (type == GGUF_TYPE_Q4_0) {
                block_q4_0 block = {.d = fp32_to_fp16(0.0078125f)};
                for (int k = 0; k < 16; k++) block.qs[k] = (uint8_t)(i + j * 3 + k * 7);
                bytes(&f, &block, sizeof(block));
            } else if (type == GGUF_TYPE_Q8_0) {
                block_q8_0 block = {.d = fp32_to_fp16(0.0009765625f)};
                for (int k = 0; k < 32; k++) block.qs[k] = (int8_t)((j * 3 + k * 7) % 101 - 50);
                bytes(&f, &block, sizeof(block));
            } else {
                block_q4_K block = {0};
                bytes(&f, &block, sizeof(block));
            }
        }
        if (i + 1 < tensors) pad(&f);
    }
    return f;
}

static void reject_gguf(const fixture_t *f, size_t size) {
    model_t m = {.mmap_addr = (void *)f->bytes, .mmap_size = size};
    CHECK(gguf_parse(&m, 0) == -1);
}

static void gguf_cases(const fixture_t *f) {
    const struct { size_t pos, width; uint64_t value; } changes[] = {
        {0, 4, 0}, {4, 4, 4}, {8, 8, UINT64_MAX}, {16, 8, UINT64_MAX},
        {24, 8, UINT64_MAX}, {24 + 8 + 22, 4, 13},
        {f->token_type, 4, 6}, {f->token_count, 8, UINT64_MAX},
        {f->head, 4, 0}, {f->layers, 4, MAX_LAYERS + 1}, {f->head, 4, 3},
        {f->epsilon, 4, 0xbf800000}, {f->alignment, 4, 4},
        {f->first_name, 8, 65}, {f->first_rank, 4, 5}, {f->first_dim, 8, 0},
        {f->first_offset, 8, 1u << 20}, {f->first_rows, 8, MICRO_VOCAB - 1},
    };
    reject_gguf(f, 16);
    reject_gguf(f, f->size - 1);
    for (size_t i = 0; i < sizeof(changes) / sizeof(changes[0]); i++) {
        fixture_t bad = *f;
        memcpy(bad.bytes + changes[i].pos, &changes[i].value, changes[i].width);
        reject_gguf(&bad, bad.size);
    }
}

static void write_file(const char *path, const void *data, size_t n, const char *mode) {
    FILE *file = fopen(path, mode);
    CHECK(file);
    CHECK(fwrite(data, 1, n, file) == n);
    CHECK(fclose(file) == 0);
}

static size_t read_cache(const char *path, uint8_t *data, size_t capacity) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    size_t n = fread(data, 1, capacity, f);
    CHECK(n < capacity && !ferror(f) && feof(f));
    fclose(f);
    return n;
}

static void check_sha256(void) {
    static const char message[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    static const uint8_t expected[] = {
        0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8, 0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
        0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67, 0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
    };
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, message, 9);
    sha256_update(&hash, message + 9, sizeof(message) - 10);
    uint8_t digest[SHA256_BYTES];
    sha256_finish(&hash, digest);
    CHECK(!memcmp(digest, expected, sizeof(digest)));
}

static void check_legacy_cache(const model_t *m, const char *path, kv_quant_t quant,
                               const uint8_t *cached, size_t size) {
    enum { HEADER_V3 = 8 * sizeof(uint32_t) + 2 * SHA256_BYTES };
    uint8_t legacy[4096];
    uint32_t dim = (uint32_t)(m->config.head_dim * m->config.n_kv_heads);
    uint32_t layers = (uint32_t)m->config.n_layers;
    uint32_t v1[] = {KVCACHE_MAGIC, 2, layers, dim};
    uint32_t v2[] = {KVCACHE_MAGIC, 0, 2, 2, layers, dim, (uint32_t)m->config.n_kv_heads, KV_INT8};
    size_t header_size = quant == KV_FP16 ? sizeof(v1) : sizeof(v2);
    size_t payload = size - HEADER_V3 - SHA256_BYTES;
    memcpy(legacy, quant == KV_FP16 ? v1 : v2, header_size);
    memcpy(legacy + header_size, cached + HEADER_V3, payload);
    write_file(path, legacy, header_size + payload, "wb");
}

static const int saved_prompt[] = {1, 4, 5};

static void check_cache(model_t *m, const char *path, kv_quant_t quant) {
    float *logits = model_forward_ex(m, saved_prompt[0], 0, 0);
    CHECK(!logits);
    logits = model_forward(m, saved_prompt[1], 1);
    CHECK(logits && isfinite(logits[0]));
#ifndef _WIN32
    CHECK(chmod(path, 0600) == 0);
    mode_t old_mask = umask(022);
#endif
    CHECK(kvcache_save(m, path, saved_prompt, 2) == 0);
#ifndef _WIN32
    umask(old_mask);
    struct stat info;
    CHECK(stat(path, &info) == 0 && (info.st_mode & 0777) == 0600);
#endif
    unsigned char *snapshot = malloc(m->state.kv_size);
    CHECK(snapshot);
    memcpy(snapshot, m->state.kv_block, m->state.kv_size);
    memset(m->state.kv_block, 0, m->state.kv_size);
    const int other_prompt[] = {1, 5, 4};
    CHECK(kvcache_load(m, path, other_prompt, 3, 0) == 0);
    CHECK(kvcache_load(m, path, saved_prompt, 1, 0) == 0);
    CHECK(kvcache_load(m, path, saved_prompt, 3, 0) == 2);
    CHECK(!memcmp(snapshot, m->state.kv_block, m->state.kv_size));

    uint8_t cached[4096], bad[4096];
    size_t size = read_cache(path, cached, sizeof(cached));
    uint32_t version;
    memcpy(&version, cached + 8, sizeof(version));
    CHECK(version == KVCACHE_VERSION && !cached[4]);
    const struct { size_t length, offset; uint32_t value; } cases[] = {
        {size, 0, 0}, {10, 0, KVCACHE_MAGIC}, {size - 1, 0, KVCACHE_MAGIC},
        {size + 1, 0, KVCACHE_MAGIC}, {size, 12, 17},
        {size, 16, 2}, {size, 8, 99}, {size, 20, 3},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        memcpy(bad, cached, size);
        bad[size] = 0;
        memcpy(bad + cases[i].offset, &cases[i].value, 4);
        write_file(path, bad, cases[i].length, "wb");
        CHECK(kvcache_load(m, path, saved_prompt, 3, 0) == -1);
        CHECK(!memcmp(snapshot, m->state.kv_block, m->state.kv_size));
    }
    memcpy(bad, cached, size);
    bad[size - 1] ^= 1;
    write_file(path, bad, size, "wb");
    CHECK(kvcache_load(m, path, saved_prompt, 3, 0) == -1);
    for (size_t i = 0; i < m->state.kv_size; i++) CHECK(!((const uint8_t *)m->state.kv_block)[i]);

    check_legacy_cache(m, path, quant, cached, size);
    CHECK(kvcache_load(m, path, saved_prompt, 3, 0) == 0);
    CHECK(kvcache_load(m, path, saved_prompt, 3, 1) == 2);
    CHECK(!memcmp(snapshot, m->state.kv_block, m->state.kv_size));
    write_file(path, cached, size, "wb");
    free(snapshot);
}

int main(int argc, char **argv) {
    fixture_t f = micro_gguf((fixture_options_t){0});
    if (argc == 3 && (!strcmp(argv[1], "--fixture") || !strcmp(argv[1], "--fixture-no-json"))) {
        if (!strcmp(argv[1], "--fixture-no-json")) {
            f = micro_gguf((fixture_options_t){.no_json = 1});
        }
        write_file(argv[2], f.bytes, f.size, "wbx");
        return 0;
    }
    gguf_cases(&f);
    const fixture_options_t unsupported[] = {
        {.tokenizer = "gpt2"}, {.scaling = "linear"}, {.rope_dims = 4},
        {.extra = "rope_freqs.weight"},
    };
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        fixture_t other = micro_gguf(unsupported[i]);
        reject_gguf(&other, other.size);
    }
    check_sha256();
    unsigned long pid;
#ifdef _WIN32
    pid = (unsigned long)GetCurrentProcessId();
#else
    pid = (unsigned long)getpid();
#endif
    char gguf[100], cache[100], alternate[100];
    snprintf(gguf, sizeof(gguf), "tests/picolm_micro_%lu.gguf", pid);
    snprintf(cache, sizeof(cache), "tests/picolm_micro_%lu.kvc", pid);
    snprintf(alternate, sizeof(alternate), "tests/picolm_alternate_%lu.gguf", pid);
    write_file(gguf, f.bytes, f.size, "wbx");
    write_file(cache, f.bytes, 0, "wbx");

    model_t m;
    CHECK(model_load(&m, gguf, 0) == 0);
    CHECK(m.config.n_heads == 4 && m.config.n_kv_heads == 4);
    CHECK(m.config.rms_epsilon == 2e-5f);
    check_cache(&m, cache, KV_FP16);
    size_t fp16 = m.state.kv_size;
    model_free(&m);
    CHECK(model_load_ex(&m, gguf, 0, KV_INT8) == 0);
    CHECK(m.state.kv_size < fp16);
    check_cache(&m, cache, KV_INT8);
    model_free(&m);
    f.bytes[f.weight_data + 2] ^= 1; /* Same metadata and shapes, different valid weights. */
    write_file(alternate, f.bytes, f.size, "wbx");
    CHECK(model_load_ex(&m, alternate, 0, KV_INT8) == 0);
    CHECK(kvcache_load(&m, cache, saved_prompt, 3, 0) == 0);
    model_free(&m);
    CHECK(remove(alternate) == 0);
    CHECK(remove(cache) == 0);
    CHECK(remove(gguf) == 0);
    puts("binary validation: 24 GGUF + 18 KVCP rejected; SHA-256, v3 identities and v1/v2 migration passed");
    return 0;
}
