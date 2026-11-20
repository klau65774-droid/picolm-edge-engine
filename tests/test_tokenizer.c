#include "tokenizer.h"

#include "check.h"
#include <stdio.h>
#include <string.h>

static void add_fake_token(unsigned char **next, const char *text, size_t size) {
    uint64_t length = size;
    memcpy(*next, &length, 8);
    memcpy(*next + 8, text, size);
    *next += 8 + size;
}

static void check_bpe_order(void) {
    const char *const words[] = {
        "<unk>", "<s>", "</s>", "a", "b", "aa", "ab", "ba", "bb",
        "aba", "baa", "abba", "abb", "bab", "aaab",
    };
    float scores[] = {0, 0, 0, 0, 0, 3, 3, 3, 3, 5, 1, 6, 4, 4, 2};
    const int vocab = (int)(sizeof(words) / sizeof(words[0]));
    unsigned char memory[512], *end = memory;
    for (int i = 0; i < vocab; i++) add_fake_token(&end, words[i], strlen(words[i]));
    model_t m = {
        .config = {.vocab_size = vocab}, .tok_bos_id = 1, .tok_eos_id = 2,
        .tok_no_bos = 1, .tok_no_space_prefix = 1,
        .tok_tokens_data = memory, .tok_tokens_size = (size_t)(end - memory),
        .tok_n_tokens = (uint64_t)vocab, .tok_scores_data = scores, .tok_n_scores = (uint64_t)vocab,
    };
    tokenizer_t t;
    CHECK(tokenizer_load(&t, &m) == 0 && !t.add_bos && !t.add_space_prefix);
    /* Compare the heap against a simple left-to-right scan, including ties. */
    for (int len = 1; len <= 10; len++) {
        for (unsigned bits = 0; bits < (1u << len); bits++) {
            char text[11] = {0};
            int expected[10], actual[10], n = len;
            for (int i = 0; i < len; i++) {
                int one = (int)((bits >> i) & 1);
                text[i] = (char)('a' + one);
                expected[i] = 3 + one;
            }
            for (;;) {
                int pair = -1, merged = -1;
                float score = -1e30f;
                for (int i = 0; i + 1 < n; i++) {
                    char word[16];
                    snprintf(word, sizeof(word), "%s%s", words[expected[i]], words[expected[i + 1]]);
                    for (int j = 5; j < vocab; j++) {
                        if (scores[j] > score && !strcmp(word, words[j])) {
                            pair = i;
                            merged = j;
                            score = scores[j];
                        }
                    }
                }
                if (pair < 0) break;
                expected[pair] = merged;
                memmove(expected + pair + 1, expected + pair + 2,
                        (size_t)(n - pair - 2) * sizeof(*expected));
                n--;
            }
            CHECK(tokenizer_encode(&t, text, actual, len, 0) == n);
            CHECK(!memcmp(expected, actual, (size_t)n * sizeof(*expected)));
        }
    }
    tokenizer_free(&t);
}

int main(void) {
    unsigned char memory[4096], *end = memory;
    const char *base[] = {"<unk>", "<s>", "</s>", "\xE2\x96\x81", "<0x00>", "<0xFF>", "x", "\xE2\x96\x81x"};
    float scores[19] = {0};
    for (size_t i = 0; i < sizeof(base) / sizeof(base[0]); i++) {
        add_fake_token(&end, base[i], strlen(base[i]));
    }
    char repeated[513];
    memset(repeated, 'a', 512);
    repeated[512] = '\0';
    for (int i = 0; i < 10; i++) {
        add_fake_token(&end, repeated, (size_t)1 << i);
        scores[8 + i] = 10.0f - (float)i;
    }
    char long_token[516];
    memcpy(long_token, "\xE2\x96\x81", 3);
    memcpy(long_token + 3, repeated, sizeof(repeated));
    add_fake_token(&end, long_token, sizeof(long_token) - 1);
    model_t m = {
        .config = {.vocab_size = 19}, .tok_bos_id = 1, .tok_eos_id = 2,
        .tok_tokens_data = memory, .tok_tokens_size = (size_t)(end - memory), .tok_n_tokens = 19,
        .tok_scores_data = scores, .tok_n_scores = 19,
    };
    tokenizer_t t;
    CHECK(tokenizer_load(&t, &m) == 0);

    int ids[4] = {-1, -1, -1, -1};
    CHECK(tokenizer_encode(&t, "x", ids, 4, 1) == 2);
    CHECK(ids[0] == 1 && ids[1] == 7);
    CHECK(!strcmp(tokenizer_decode(&t, ids[0], ids[1]), "x"));
    CHECK(!strcmp(tokenizer_decode(&t, 2, ids[1]), " x"));

    CHECK(tokenizer_encode(&t, repeated, ids, 4, 1) == 2);
    CHECK(ids[0] == 1 && ids[1] == 18);
    tokenizer_piece_t piece = tokenizer_piece(&t, 2, 18);
    CHECK(piece.len == 513 && piece.data[0] == ' ' && !memcmp(piece.data + 1, repeated, 512));
    CHECK(strlen(tokenizer_decode(&t, 1, 18)) == 512);
    CHECK(tokenizer_encode(&t, "\xFF", ids, 4, 0) == 2 && ids[1] == 5);
    CHECK(tokenizer_encode(&t, "\xF0\x9F\x99\x82", ids, 4, 0) == 2 && ids[1] == 0);
    CHECK((unsigned char)*tokenizer_decode(&t, 0, 5) == 0xff);
    piece = tokenizer_piece(&t, 0, 4);
    CHECK(piece.len == 1 && piece.data[0] == '\0');
    ids[1] = -7;
    CHECK(tokenizer_encode(&t, "x", ids, 1, 1) == 1 && ids[1] == -7);
    CHECK(tokenizer_encode(&t, "x", NULL, 0, 1) == 0);
    tokenizer_free(&t);

    m.tok_tokens_size--;
    CHECK(tokenizer_load(&t, &m) < 0);
    tokenizer_free(&t);
    check_bpe_order();
    puts("tokenizer: bounded encode, unknown/byte fallback and 2046 BPE reference comparisons");
    return 0;
}
