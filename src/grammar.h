#ifndef GRAMMAR_H
#define GRAMMAR_H

#include "tokenizer.h"

#define JSON_MAX_DEPTH 128
#define JSON_SUFFIX_CAP (JSON_MAX_DEPTH + 16)

typedef enum {
    EXPECT_VALUE, OBJ_START_OR_END, OBJ_KEY, OBJ_COLON, OBJ_AFTER_VAL,
    ARR_START_OR_END, ARR_AFTER_VAL, IN_STRING, IN_ESCAPE,
    IN_UNICODE, IN_NUMBER, IN_LITERAL, DONE,
} json_phase_t;

typedef struct {
    json_phase_t phase;
    unsigned char stack[JSON_MAX_DEPTH];
    unsigned depth;
    unsigned char key, digits, number, literal, literal_pos;
    unsigned char utf8_left, utf8_min, utf8_max;
} grammar_state_t;

typedef enum {
    GRAMMAR_NO_FINITE_LOGITS = -1,
    GRAMMAR_NO_CANDIDATE = 0,
    GRAMMAR_READY = 1,
} grammar_filter_t;

void grammar_init(grammar_state_t *g);

/* Commit a complete token span only if it is valid at the current position. */
int grammar_feed(grammar_state_t *g, const char *bytes, size_t len);
int grammar_is_complete(const grammar_state_t *g);
int grammar_can_end(const grammar_state_t *g);
grammar_filter_t grammar_apply(const grammar_state_t *g, const tokenizer_t *tok,
                               int previous, float *logits);

/* Append a syntactic ending to a valid prefix, up to JSON_SUFFIX_CAP
 * bytes. Separately counted by the CLI and never fed back to the model. */
int grammar_finish(grammar_state_t *g, char suffix[JSON_SUFFIX_CAP]);

#endif
