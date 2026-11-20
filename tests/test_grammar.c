#include "grammar.h"

#include "check.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static void valid(const char *input) {
    grammar_state_t g;
    grammar_init(&g);
    for (const char *p = input; *p; p++) CHECK(grammar_feed(&g, p, 1));
    CHECK(grammar_can_end(&g));
}

static void invalid(const char *input) {
    grammar_state_t g;
    grammar_init(&g);
    CHECK(!grammar_feed(&g, input, strlen(input)) || !grammar_can_end(&g));
}

static void completion(const char *prefix) {
    grammar_state_t g;
    grammar_init(&g);
    CHECK(grammar_feed(&g, prefix, strlen(prefix)));
    char suffix[JSON_SUFFIX_CAP];
    int size = grammar_finish(&g, suffix);
    CHECK(size >= 0 && grammar_can_end(&g));
    grammar_init(&g);
    CHECK(grammar_feed(&g, prefix, strlen(prefix)));
    CHECK(grammar_feed(&g, suffix, (size_t)size));
    CHECK(grammar_can_end(&g));
}

int main(void) {
    const char *accepted[] = {
        "{}", "[]", "\"text\"", "null", "false", "true", "-0", "0.0", "1E+9",
        " \r\n [0, -1.5e-3, null, {\"x\":[true,false,{}]}] \t",
        "{\"a\\u1234\":\"\\n\\\"\\/\\\\\\b\\f\\r\\t\"}",
        "\"\xE6\xA0\x91\xF0\x9F\x98\x80\"",
    };
    const char *rejected[] = {
        "{,}", "{\"a\" \"b\"}", "{\"a\":1,}", "[1,]", "[,1]", "[}",
        "{a:2}", "{\"a\":}", "[1 2]", "00", "+1", ".4", "-.2", "1.",
        "1.e2", "1e", "1e+-1", "1e2E3", "nullfalse", "NaN", "true false",
        "\"\\u12g4\"", "\"\\x\"", "\"\n\"", "\"\xED\xA0\x80\"",
        "\"\xC0\xAF\"", "\"\xF4\x90\x80\x80\"",
    };
    const char *prefixes[] = {
        "", "  \t", "{\"key\":", "{\"key\":1,", "[true,",
        "\"escaped\\", "\"\\uD", "{\"split\":\"\xF0\x9F", "[1.2e-", "3", "-",
    };
    for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); i++) valid(accepted[i]);
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++) invalid(rejected[i]);
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) completion(prefixes[i]);

    grammar_state_t g;
    grammar_init(&g);
    char deep[JSON_MAX_DEPTH + 1], suffix[JSON_SUFFIX_CAP];
    memset(deep, '[', sizeof(deep));
    CHECK(!grammar_feed(&g, deep, sizeof(deep)));
    CHECK(grammar_feed(&g, deep, JSON_MAX_DEPTH));
    CHECK(grammar_finish(&g, suffix) == JSON_MAX_DEPTH);
    CHECK(grammar_is_complete(&g));

    tokenizer_piece_t pieces[] = {
        {"<s>", 3, 0}, {"</s>", 4, 0}, {"{}", 2, 0},
        {"{,}", 3, 0}, {"true", 4, 0}, {" \0", 2, 0},
    };
    tokenizer_t tok = {.pieces = pieces, .vocab_size = 6, .bos_id = 0, .eos_id = 1};
    float logits[6] = {0};
    grammar_init(&g);
    CHECK(grammar_apply(&g, &tok, 0, logits) == GRAMMAR_READY);
    CHECK(logits[0] == -INFINITY && logits[1] == -INFINITY);
    CHECK(isfinite(logits[2]) && logits[3] == -INFINITY && isfinite(logits[4]));
    CHECK(logits[5] == -INFINITY);
    const float rejected_logits[] = {-INFINITY, -INFINITY, -INFINITY, 0, -INFINITY, -INFINITY};
    memcpy(logits, rejected_logits, sizeof(logits));
    CHECK(grammar_apply(&g, &tok, 0, logits) == GRAMMAR_NO_CANDIDATE);
    CHECK(grammar_apply(&g, &tok, 0, logits) == GRAMMAR_NO_FINITE_LOGITS);
    puts("grammar: 12 valid, 27 invalid, 11 completions, depth and masking outcomes");
    return 0;
}
