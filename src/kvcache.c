#include "model.h"
#include "fileio.h"

#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* v1: magic, positions, layers, dim (FP16). v2 uses zero as the second
 * word and adds version, heads and quant. v3 adds the model and prefix
 * digests to the v2 header, and a payload digest at the end of the file. */
typedef struct {
    uint32_t version, positions, layers, dim, heads, quant;
} cache_header_t;

typedef struct {
    size_t packed, stride;
    uint8_t *data;
} cache_plane_t;

static size_t cache_planes(const model_t *m, int positions, cache_plane_t planes[4]) {
    const model_config_t *c = &m->config;
    const run_state_t *s = &m->state;
    int use_int8 = s->kv_quant == KV_INT8;
    size_t row = (size_t)c->n_kv_heads * c->head_dim * (use_int8 ? 1 : sizeof(uint16_t));
    size_t scale_row = (size_t)c->n_kv_heads * sizeof(uint16_t);
    planes[0] = (cache_plane_t){row * positions, row * c->max_seq_len, s->key_cache.data};
    planes[1] = (cache_plane_t){row * positions, row * c->max_seq_len, s->val_cache.data};
    planes[2] = (cache_plane_t){scale_row * positions, scale_row * c->max_seq_len,
                                (uint8_t *)s->key_cache.scales};
    planes[3] = (cache_plane_t){scale_row * positions, scale_row * c->max_seq_len,
                                (uint8_t *)s->val_cache.scales};
    return use_int8 ? 4 : 2;
}

static const uint8_t *model_digest(model_t *m) {
    if (!m->cache_digest_ready) {
        sha256_t hash;
        sha256_init(&hash);
        sha256_update(&hash, m->mmap_addr, m->mmap_size);
        sha256_finish(&hash, m->cache_digest);
        m->cache_digest_ready = 1;
    }
    return m->cache_digest;
}

static void prefix_digest(const int *tokens, int positions, uint8_t digest[SHA256_BYTES]) {
    sha256_t hash;
    sha256_init(&hash);
    for (int i = 0; i < positions; i++) {
        uint32_t id = (uint32_t)tokens[i];
        uint8_t bytes[] = {(uint8_t)id, (uint8_t)(id >> 8), (uint8_t)(id >> 16), (uint8_t)(id >> 24)};
        sha256_update(&hash, bytes, sizeof(bytes));
    }
    sha256_finish(&hash, digest);
}

static FILE *open_private_temp(const char *path) {
#ifdef _WIN32
    return open_utf8(path, "wbx");
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0) return NULL;
    FILE *file = fdopen(fd, "wb");
    if (!file) {
        int error = errno;
        close(fd);
        unlink(path);
        errno = error;
    }
    return file;
#endif
}

static int finish_cache_file(const char *temp, const char *path, int ok) {
#ifdef _WIN32
    wchar_t *wide_temp = utf8_to_wide(temp);
    wchar_t *wide_path = utf8_to_wide(path);
    ok = ok && wide_temp && wide_path && MoveFileExW(wide_temp, wide_path, MOVEFILE_REPLACE_EXISTING);
    if (!ok && wide_temp) DeleteFileW(wide_temp);
    free(wide_temp);
    free(wide_path);
#else
    ok = ok && rename(temp, path) == 0;
    if (!ok) unlink(temp);
#endif
    return ok ? 0 : -1;
}

int kvcache_save(model_t *m, const char *path, const int *tokens, int positions) {
    const model_config_t *c = &m->config;
    if (!m->state.kv_block || !tokens || positions <= 0 || positions > c->max_seq_len) return -1;
    unsigned long pid;
#ifdef _WIN32
    pid = (unsigned long)GetCurrentProcessId();
#else
    pid = (unsigned long)getpid();
#endif
    static atomic_uint temp_id;
    size_t path_len = strlen(path);
    if (path_len > SIZE_MAX - 48) return -1;
    size_t cap = path_len + 48;
    char *temp = malloc(cap);
    if (!temp) return -1;
    FILE *f = NULL;
    for (int attempt = 0; attempt < 16; attempt++) {
        unsigned id = atomic_fetch_add_explicit(&temp_id, 1, memory_order_relaxed);
        snprintf(temp, cap, "%s.%lu.%u.tmp", path, pid, id);
        f = open_private_temp(temp);
        if (f || errno != EEXIST) break;
    }
    if (!f) {
        free(temp);
        return -1;
    }

    uint32_t header[] = {KVCACHE_MAGIC, 0, KVCACHE_VERSION, (uint32_t)positions,
                         (uint32_t)c->n_layers, (uint32_t)(c->head_dim * c->n_kv_heads),
                         (uint32_t)c->n_kv_heads, (uint32_t)m->state.kv_quant};
    uint8_t digest[SHA256_BYTES];
    prefix_digest(tokens, positions, digest);
    int ok = fwrite(header, 1, sizeof(header), f) == sizeof(header) &&
             fwrite(model_digest(m), 1, SHA256_BYTES, f) == SHA256_BYTES &&
             fwrite(digest, 1, sizeof(digest), f) == sizeof(digest);

    sha256_t hash;
    sha256_init(&hash);
    cache_plane_t planes[4];
    size_t count = cache_planes(m, positions, planes);
    for (size_t i = 0; i < count && ok; i++) {
        for (int l = 0; l < c->n_layers && ok; l++) {
            const uint8_t *row = planes[i].data + (size_t)l * planes[i].stride;
            ok = fwrite(row, 1, planes[i].packed, f) == planes[i].packed;
            if (ok) sha256_update(&hash, row, planes[i].packed);
        }
    }
    sha256_finish(&hash, digest);
    ok = ok && fwrite(digest, 1, sizeof(digest), f) == sizeof(digest);
    if (fclose(f) != 0) ok = 0;
    int status = finish_cache_file(temp, path, ok);
    free(temp);
    return status;
}

static int read_cache_header(FILE *f, cache_header_t *info) {
    uint32_t first[4], next[4];
    if (fread(first, sizeof(first[0]), 4, f) != 4 || first[0] != KVCACHE_MAGIC) return 0;
    if (first[1]) {
        *info = (cache_header_t){1, first[1], first[2], first[3], 0, KV_FP16};
    } else {
        if ((first[2] != 2 && first[2] != KVCACHE_VERSION) ||
            fread(next, sizeof(next[0]), 4, f) != 4) return 0;
        *info = (cache_header_t){first[2], first[3], next[0], next[1], next[2], next[3]};
        if (!info->heads || info->dim % info->heads || info->quant > KV_INT8) return 0;
    }
    return info->positions && info->positions <= INT_MAX && info->layers &&
           info->layers <= MAX_LAYERS && info->dim && info->dim <= INT_MAX;
}

/* Seek over the weight vectors to validate just the scale planes. A fixed
 * buffer avoids a second copy of KV and keeps rejected headers/scales read-only. */
static int valid_scales(FILE *f, long end, uint64_t count) {
    uint16_t buffer[2048];
    if (fseek(f, end - (long)(count * sizeof(buffer[0])), SEEK_SET)) return 0;
    while (count) {
        size_t n = count > sizeof(buffer) / sizeof(buffer[0])
                   ? sizeof(buffer) / sizeof(buffer[0]) : (size_t)count;
        if (fread(buffer, sizeof(buffer[0]), n, f) != n) return 0;
        for (size_t i = 0; i < n; i++) {
            if (buffer[i] > 0x7bff) return 0;
        }
        count -= n;
    }
    return 1;
}

int kvcache_load(model_t *m, const char *path, const int *tokens, int count, int trust_legacy) {
    const model_config_t *c = &m->config;
    if (!m->state.kv_block || !tokens || count <= 0) return -1;
    FILE *f = open_utf8(path, "rb");
    if (!f) {
        if (errno == ENOENT) return 0;
        fprintf(stderr, "KVCP: cannot open cache\n");
        return -1;
    }

    int status = -1, copying = 0;
    cache_header_t info;
    uint8_t saved_model[SHA256_BYTES], saved_prefix[SHA256_BYTES], actual[SHA256_BYTES];
    if (!read_cache_header(f, &info)) goto done;
    int verified_format = info.version == KVCACHE_VERSION;
    if (verified_format &&
        (fread(saved_model, 1, sizeof(saved_model), f) != sizeof(saved_model) ||
         fread(saved_prefix, 1, sizeof(saved_prefix), f) != sizeof(saved_prefix))) goto done;

    long data_start = ftell(f);
    if (data_start < 0 || fseek(f, 0, SEEK_END)) goto done;
    long end = ftell(f);
    long trailer = verified_format ? SHA256_BYTES : 0;
    if (end < data_start || end - data_start < trailer) goto done;
    long payload_end = end - trailer;
    uint64_t bytes = (uint64_t)(payload_end - data_start);
    uint64_t scale_count = info.quant == KV_INT8 ? (uint64_t)info.heads * 2 : 0;
    uint64_t row_pair = info.quant == KV_INT8 ? (uint64_t)info.dim * 2 + scale_count * 2
                                            : (uint64_t)info.dim * 4;
    uint64_t layer_positions = (uint64_t)info.positions * info.layers;
    if (bytes % row_pair || bytes / row_pair != layer_positions) goto done;

    if (info.positions > (uint32_t)count || info.positions > (uint32_t)c->max_seq_len ||
        info.layers != (uint32_t)c->n_layers || info.dim != (uint32_t)(c->head_dim * c->n_kv_heads) ||
        (info.heads && info.heads != (uint32_t)c->n_kv_heads) || info.quant != (uint32_t)m->state.kv_quant) {
        status = 0;
        goto done;
    }
    if (verified_format) {
        prefix_digest(tokens, (int)info.positions, actual);
        if (memcmp(actual, saved_prefix, sizeof(actual)) ||
            memcmp(model_digest(m), saved_model, sizeof(saved_model))) {
            status = 0;
            goto done;
        }
    } else if (!trust_legacy) {
        status = 0;
        goto done;
    }

    if ((scale_count && !valid_scales(f, payload_end, scale_count * layer_positions)) ||
        fseek(f, data_start, SEEK_SET)) goto done;

    sha256_t hash;
    sha256_init(&hash);
    cache_plane_t planes[4];
    size_t planes_count = cache_planes(m, (int)info.positions, planes);
    copying = 1;
    for (size_t i = 0; i < planes_count; i++) {
        for (int l = 0; l < c->n_layers; l++) {
            uint8_t *row = planes[i].data + (size_t)l * planes[i].stride;
            if (fread(row, 1, planes[i].packed, f) != planes[i].packed) goto done;
            if (verified_format) sha256_update(&hash, row, planes[i].packed);
        }
    }
    if (verified_format) {
        uint8_t saved[SHA256_BYTES];
        sha256_finish(&hash, actual);
        if (fread(saved, 1, sizeof(saved), f) != sizeof(saved) ||
            memcmp(saved, actual, sizeof(actual))) goto done;
    }
    if (fgetc(f) != EOF || ferror(f)) goto done;
    status = (int)info.positions;

done:
    fclose(f);
    if (status < 0) {
        if (copying) memset(m->state.kv_block, 0, m->state.kv_size);
        fprintf(stderr, "KVCP: invalid or unreadable cache\n");
    }
    return status;
}
