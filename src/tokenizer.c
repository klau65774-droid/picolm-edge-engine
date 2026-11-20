#include "tokenizer.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char space_marker[] = "\xE2\x96\x81";

static uint32_t hash_bytes(uint32_t hash, const char *p, size_t len) {
    for (size_t i = 0; i < len; i++) hash = (hash ^ (uint8_t)p[i]) * UINT32_C(16777619);
    return hash;
}

static size_t vocab_slot(const tokenizer_t *t, const char *first, size_t n,
                         const char *second, size_t m) {
    uint32_t hash = hash_bytes(hash_bytes(UINT32_C(2166136261), first, n), second, m);
    size_t slot = hash & t->hash_mask;
    while (t->hash[slot]) {
        int id = t->hash[slot] - 1;
        if (t->vocab_len[id] == n + m && !memcmp(t->vocab[id], first, n) &&
            (!m || !memcmp(t->vocab[id] + n, second, m))) break;
        slot = (slot + 1) & t->hash_mask;
    }
    return slot;
}

static int vocab_lookup(const tokenizer_t *t, const char *str, size_t len) {
    return t->hash[vocab_slot(t, str, len, NULL, 0)] - 1;
}

static int hex_digit(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int byte_token(const char *text, size_t len) {
    if (len != 6 || memcmp(text, "<0x", 3) || text[5] != '>') return -1;
    int hi = hex_digit((uint8_t)text[3]), lo = hex_digit((uint8_t)text[4]);
    return hi < 0 || lo < 0 ? -1 : 16 * hi + lo;
}

int tokenizer_load(tokenizer_t *t, const model_t *m) {
    memset(t, 0, sizeof(*t));
    int vs = m->config.vocab_size;
    if (vs <= 0 || (size_t)vs > SIZE_MAX / sizeof(tokenizer_piece_t) ||
        (size_t)vs > SIZE_MAX / 4 || m->tok_n_tokens > (uint64_t)vs ||
        m->tok_n_scores > (uint64_t)vs || !m->tok_tokens_data) return -1;
    size_t capacity = 2;
    while (capacity < 2 * (size_t)vs) capacity *= 2;
    if (capacity > SIZE_MAX / sizeof(*t->hash)) return -1;

    t->vocab_size = vs;
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;
    t->unk_id = m->tok_unk_id;
    t->add_bos = !m->tok_no_bos;
    t->add_space_prefix = !m->tok_no_space_prefix;
    t->vocab = calloc((size_t)vs, sizeof(*t->vocab));
    t->vocab_len = calloc((size_t)vs, sizeof(*t->vocab_len));
    t->pieces = calloc((size_t)vs, sizeof(*t->pieces));
    t->scores = calloc((size_t)vs, sizeof(*t->scores));
    t->hash = calloc(capacity, sizeof(*t->hash));
    t->hash_mask = capacity - 1;
    if (!t->vocab || !t->vocab_len || !t->pieces || !t->scores || !t->hash) goto fail;

    const uint8_t *data = m->tok_tokens_data;
    size_t remaining = m->tok_tokens_size;
    for (uint64_t i = 0; i < m->tok_n_tokens; i++) {
        uint64_t len64;
        if (remaining < sizeof(len64)) goto fail;
        memcpy(&len64, data, sizeof(len64));
        data += sizeof(len64);
        remaining -= sizeof(len64);
        if (len64 > remaining || len64 > (SIZE_MAX - 3) / 2) goto fail;
        size_t len = (size_t)len64;
        const char *text = (const char *)data;
        int marker = len >= 3 && !memcmp(text, space_marker, 3);
        int byte = byte_token(text, len);
        size_t extra = marker ? len - 1 : byte >= 0 ? 2 : 0;
        char *raw = malloc(len + 1 + extra);
        if (!raw) goto fail;
        t->vocab[i] = raw;
        t->vocab_len[i] = len;
        memcpy(raw, text, len);
        raw[len] = '\0';
        if (marker) {
            char *decoded = raw + len + 1;
            decoded[0] = ' ';
            memcpy(decoded + 1, raw + 3, len - 2);
            t->pieces[i] = (tokenizer_piece_t){decoded, len - 2, 1};
        } else if (byte >= 0) {
            char *decoded = raw + len + 1;
            decoded[0] = (char)byte;
            decoded[1] = '\0';
            t->pieces[i] = (tokenizer_piece_t){decoded, 1, 0};
        } else {
            t->pieces[i] = (tokenizer_piece_t){raw, len, 0};
        }
        if (len) {
            size_t slot = vocab_slot(t, raw, len, NULL, 0);
            if (!t->hash[slot]) t->hash[slot] = (int)i + 1;
        }
        data += len;
        remaining -= len;
    }
    if (remaining) goto fail;
    if (m->tok_n_scores) {
        if (!m->tok_scores_data) goto fail;
        memcpy(t->scores, m->tok_scores_data, (size_t)m->tok_n_scores * sizeof(*t->scores));
    }
    return 0;

fail:
    fprintf(stderr, "Tokenizer: invalid vocabulary data or allocation failure\n");
    tokenizer_free(t);
    return -1;
}

typedef struct {
    int token, previous, next;
    int slot, merged;
} symbol_t;

typedef struct {
    symbol_t *symbols;
    int *heap;
    int count;
    const tokenizer_t *tokenizer;
} merges_t;

static void push_symbol(symbol_t *symbols, int *count, int token) {
    int i = (*count)++;
    symbols[i] = (symbol_t){token, i - 1, i + 1, -1, -1};
}

static void append_symbol(const tokenizer_t *t, const char *symbol, size_t n,
                          symbol_t *symbols, int *count) {
    int id = vocab_lookup(t, symbol, n);
    if (id >= 0) {
        push_symbol(symbols, count, id);
        return;
    }
    static const char hex[] = "0123456789ABCDEF";
    int ids[4]; /* A UTF-8 symbol has at most four bytes. */
    for (size_t i = 0; i < n; i++) {
        unsigned char value = (unsigned char)symbol[i];
        char fallback[] = {'<', '0', 'x', hex[value >> 4], hex[value & 15], '>'};
        ids[i] = vocab_lookup(t, fallback, sizeof(fallback));
        if (ids[i] < 0) {
            push_symbol(symbols, count, (int)t->unk_id);
            return;
        }
    }
    for (size_t i = 0; i < n; i++) push_symbol(symbols, count, ids[i]);
}

static size_t utf8_width(const char *p, size_t remaining) {
    unsigned char c = (unsigned char)*p;
    size_t n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 :
               c >= 0xf0 && c <= 0xf4 ? 4 : 1;
    if (n > remaining) return 1;
    for (size_t i = 1; i < n; i++) {
        if (((unsigned char)p[i] & 0xc0) != 0x80) return 1;
    }
    return n;
}

static int higher_score(const merges_t *q, int a, int b) {
    const float *scores = q->tokenizer->scores;
    float x = scores[q->symbols[a].merged], y = scores[q->symbols[b].merged];
    /* Retain the leftmost tie-break of the original full scan. */
    return x > y || (x == y && a < b);
}

static void place_merge(merges_t *q, int slot, int symbol) {
    q->heap[slot] = symbol;
    q->symbols[symbol].slot = slot;
}

static void repair_heap(merges_t *q, int slot, int symbol) {
    if (slot > 0 && higher_score(q, symbol, q->heap[(slot - 1) / 2])) {
        do {
            int parent = (slot - 1) / 2;
            place_merge(q, slot, q->heap[parent]);
            slot = parent;
        } while (slot > 0 && higher_score(q, symbol, q->heap[(slot - 1) / 2]));
    } else {
        while (slot < q->count / 2) {
            int child = slot * 2 + 1;
            if (child + 1 < q->count && higher_score(q, q->heap[child + 1], q->heap[child])) child++;
            if (!higher_score(q, q->heap[child], symbol)) break;
            place_merge(q, slot, q->heap[child]);
            slot = child;
        }
    }
    place_merge(q, slot, symbol);
}

static void set_merge(merges_t *q, int left, int merged) {
    symbol_t *s = q->symbols + left;
    int slot = s->slot;
    s->merged = merged;
    if (merged < 0) {
        if (slot < 0) return;
        s->slot = -1;
        int last = q->heap[--q->count];
        if (slot < q->count) repair_heap(q, slot, last);
    } else {
        if (slot < 0) slot = q->count++;
        repair_heap(q, slot, left);
    }
}

static void update_merge(merges_t *q, int left) {
    if (left < 0) return;
    const tokenizer_t *t = q->tokenizer;
    const symbol_t *a = q->symbols + left;
    int merged = -1;
    if (a->next >= 0) {
        int first = a->token, second = q->symbols[a->next].token;
        if (first != (int)t->unk_id && second != (int)t->unk_id) {
            size_t slot = vocab_slot(t, t->vocab[first], t->vocab_len[first],
                                     t->vocab[second], t->vocab_len[second]);
            int id = t->hash[slot] - 1;
            if (id >= 0 && t->scores[id] > -1e30f) merged = id;
        }
    }
    set_merge(q, left, merged);
}

int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    if (max_tokens < 0 || (max_tokens && !tokens) || !t->hash) return -1;
    int emitted = 0;
    if (add_bos && max_tokens > 0) tokens[emitted++] = (int)t->bos_id;
    if (!text || !*text || emitted == max_tokens) return emitted;

    size_t len = strlen(text);
    if (len > (size_t)(INT_MAX - 3) / 3 ||
        len > (SIZE_MAX / sizeof(symbol_t) - 3) / 3) return -1;
    symbol_t *symbols = malloc((len * 3 + 3) * sizeof(*symbols));
    if (!symbols) return -1;

    int count = 0;
    if (t->add_space_prefix) append_symbol(t, space_marker, 3, symbols, &count);
    for (size_t i = 0; i < len;) {
        if (text[i] == ' ') {
            append_symbol(t, space_marker, 3, symbols, &count);
            i++;
        } else {
            size_t n = utf8_width(text + i, len - i);
            append_symbol(t, text + i, n, symbols, &count);
            i += n;
        }
    }
    symbols[count - 1].next = -1;

    int *heap = malloc((size_t)count * sizeof(*heap));
    if (!heap) {
        free(symbols);
        return -1;
    }
    merges_t merges = {symbols, heap, 0, t};
    for (int i = 0; i < count - 1; i++) update_merge(&merges, i);
    while (merges.count) {
        int left = heap[0], right = symbols[left].next;
        set_merge(&merges, right, -1);
        symbols[left].token = symbols[left].merged;
        int next = symbols[right].next;
        symbols[left].next = next;
        if (next >= 0) symbols[next].previous = left;
        update_merge(&merges, left);
        update_merge(&merges, symbols[left].previous);
    }
    for (int i = 0; i >= 0 && emitted < max_tokens; i = symbols[i].next) {
        tokens[emitted++] = symbols[i].token;
    }
    free(heap);
    free(symbols);
    return emitted;
}

tokenizer_piece_t tokenizer_piece(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size || !t->pieces[token].data) {
        return (tokenizer_piece_t){"", 0, 0};
    }
    tokenizer_piece_t piece = t->pieces[token];
    if (piece.leading_space && t->add_space_prefix && prev_token == (int)t->bos_id) {
        piece.data++;
        piece.len--;
    }
    return piece;
}

const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token) {
    return tokenizer_piece(t, prev_token, token).data;
}

void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_len);
    free(t->pieces);
    free(t->scores);
    free(t->hash);
    memset(t, 0, sizeof(*t));
}
