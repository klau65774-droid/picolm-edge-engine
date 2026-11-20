#include "grammar.h"

#include <math.h>

enum { OBJECT = 1, ARRAY };
enum { SIGN, ZERO, INTEGER, DOT, FRACTION, EXP, EXP_SIGN, EXP_DIGITS };
static const char *const literals[] = {"null", "true", "false"};

static int space(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static int digit(unsigned char c) { return c >= '0' && c <= '9'; }

static void value_complete(grammar_state_t *g) {
    g->phase = !g->depth ? DONE : g->stack[g->depth - 1] == OBJECT ? OBJ_AFTER_VAL : ARR_AFTER_VAL;
}

static void close_container(grammar_state_t *g) {
    g->depth--;
    value_complete(g);
}

static int begin_value(grammar_state_t *g, unsigned char c) {
    if (space(c)) return 1;
    if (c == '{' || c == '[') {
        if (g->depth == JSON_MAX_DEPTH) return 0;
        g->stack[g->depth++] = c == '{' ? OBJECT : ARRAY;
        g->phase = c == '{' ? OBJ_START_OR_END : ARR_START_OR_END;
    } else if (c == '"') {
        g->key = 0;
        g->phase = IN_STRING;
    } else if (c == 'n' || c == 't' || c == 'f') {
        g->literal = c == 'n' ? 0 : c == 't' ? 1 : 2;
        g->literal_pos = 1;
        g->phase = IN_LITERAL;
    } else if (c == '-' || digit(c)) {
        g->number = c == '-' ? SIGN : c == '0' ? ZERO : INTEGER;
        g->phase = IN_NUMBER;
    } else {
        return 0;
    }
    return 1;
}

static int step_number(grammar_state_t *g, unsigned char c) {
    switch (g->number) {
        case SIGN:
            if (!digit(c)) return 0;
            g->number = c == '0' ? ZERO : INTEGER;
            return 1;
        case INTEGER:
            if (digit(c)) return 1;
            /* fall through */
        case ZERO:
            if (c == '.') {
                g->number = DOT;
                return 1;
            }
            break;
        case DOT:
            if (!digit(c)) return 0;
            g->number = FRACTION;
            return 1;
        case FRACTION:
            if (digit(c)) return 1;
            break;
        case EXP:
            if (c == '+' || c == '-') {
                g->number = EXP_SIGN;
                return 1;
            }
            /* fall through */
        case EXP_SIGN:
            if (!digit(c)) return 0;
            g->number = EXP_DIGITS;
            return 1;
        case EXP_DIGITS:
            if (digit(c)) return 1;
            return -1;
    }
    if (c == 'e' || c == 'E') {
        g->number = EXP;
        return 1;
    }
    return -1;
}

static int step_string(grammar_state_t *g, unsigned char c) {
    if (g->utf8_left) {
        if (c < g->utf8_min || c > g->utf8_max) return 0;
        g->utf8_left--;
        g->utf8_min = 0x80;
        g->utf8_max = 0xbf;
        return 1;
    }
    if (c == '"') {
        if (g->key) g->phase = OBJ_COLON;
        else value_complete(g);
    } else if (c == '\\') {
        g->phase = IN_ESCAPE;
    } else if (c < 0x20) {
        return 0;
    } else if (c >= 0x80) {
        g->utf8_min = 0x80;
        g->utf8_max = 0xbf;
        if (c >= 0xc2 && c <= 0xdf) {
            g->utf8_left = 1;
        } else if (c >= 0xe0 && c <= 0xef) {
            g->utf8_left = 2;
            if (c == 0xe0) g->utf8_min = 0xa0;
            if (c == 0xed) g->utf8_max = 0x9f;
        } else if (c >= 0xf0 && c <= 0xf4) {
            g->utf8_left = 3;
            if (c == 0xf0) g->utf8_min = 0x90;
            if (c == 0xf4) g->utf8_max = 0x8f;
        } else {
            return 0;
        }
    }
    return 1;
}

static int step(grammar_state_t *g, unsigned char c) {
    if (g->phase == IN_NUMBER) {
        int status = step_number(g, c);
        if (status >= 0) return status;
        value_complete(g);  /* The same delimiter also belongs to the parent. */
    }
    switch (g->phase) {
        case EXPECT_VALUE:
            return begin_value(g, c);
        case OBJ_START_OR_END:
            if (c == '}') {
                close_container(g);
                return 1;
            }
            /* fall through */
        case OBJ_KEY:
            if (space(c)) return 1;
            if (c != '"') return 0;
            g->key = 1;
            g->phase = IN_STRING;
            return 1;
        case OBJ_COLON:
            if (space(c)) return 1;
            if (c != ':') return 0;
            g->phase = EXPECT_VALUE;
            return 1;
        case OBJ_AFTER_VAL:
            if (space(c)) return 1;
            if (c == '}') close_container(g);
            else if (c == ',') g->phase = OBJ_KEY;
            else return 0;
            return 1;
        case ARR_START_OR_END:
            if (c == ']') {
                close_container(g);
                return 1;
            }
            return begin_value(g, c);
        case ARR_AFTER_VAL:
            if (space(c)) return 1;
            if (c == ']') close_container(g);
            else if (c == ',') g->phase = EXPECT_VALUE;
            else return 0;
            return 1;
        case IN_STRING:
            return step_string(g, c);
        case IN_ESCAPE:
            if (c == 'u') {
                g->digits = 4;
                g->phase = IN_UNICODE;
                return 1;
            }
            if (c != '"' && c != '\\' && c != '/' && c != 'b' &&
                c != 'f' && c != 'n' && c != 'r' && c != 't') return 0;
            g->phase = IN_STRING;
            return 1;
        case IN_UNICODE:
            if (!digit(c) && !(c >= 'a' && c <= 'f') && !(c >= 'A' && c <= 'F')) return 0;
            if (--g->digits == 0) g->phase = IN_STRING;
            return 1;
        case IN_LITERAL: {
            const char *literal = literals[g->literal];
            if (c != (unsigned char)literal[g->literal_pos]) return 0;
            if (!literal[++g->literal_pos]) value_complete(g);
            return 1;
        }
        case DONE:
            return space(c);
        case IN_NUMBER:
            break;
    }
    return 0;
}

void grammar_init(grammar_state_t *g) {
    *g = (grammar_state_t){.phase = EXPECT_VALUE};
}

int grammar_feed(grammar_state_t *g, const char *bytes, size_t len) {
    grammar_state_t trial = *g;
    for (size_t i = 0; i < len; i++) {
        if (!step(&trial, (unsigned char)bytes[i])) return 0;
    }
    *g = trial;
    return 1;
}

int grammar_is_complete(const grammar_state_t *g) {
    return g->phase == DONE;
}

static int number_complete(const grammar_state_t *g) {
    return g->number == ZERO || g->number == INTEGER ||
           g->number == FRACTION || g->number == EXP_DIGITS;
}

int grammar_can_end(const grammar_state_t *g) {
    return grammar_is_complete(g) || (!g->depth && g->phase == IN_NUMBER && number_complete(g));
}

grammar_filter_t grammar_apply(const grammar_state_t *g, const tokenizer_t *tok,
                               int previous, float *logits) {
    grammar_filter_t result = GRAMMAR_NO_FINITE_LOGITS;
    for (int id = 0; id < tok->vocab_size; id++) {
        if (!isfinite(logits[id])) continue;
        if (result == GRAMMAR_NO_FINITE_LOGITS) result = GRAMMAR_NO_CANDIDATE;
        if (id == (int)tok->eos_id) {
            if (grammar_can_end(g)) result = GRAMMAR_READY;
            else logits[id] = -INFINITY;
            continue;
        }
        tokenizer_piece_t piece = tokenizer_piece(tok, previous, id);
        if (id == (int)tok->bos_id || !piece.len) {
            logits[id] = -INFINITY;
            continue;
        }
        grammar_state_t trial = *g;
        for (size_t i = 0; i < piece.len; i++) {
            if (!step(&trial, (unsigned char)piece.data[i])) {
                logits[id] = -INFINITY;
                break;
            }
        }
        if (isfinite(logits[id])) result = GRAMMAR_READY;
    }
    return result;
}

int grammar_finish(grammar_state_t *g, char suffix[JSON_SUFFIX_CAP]) {
    int len = 0;
    while (!grammar_can_end(g)) {
        unsigned char c;
        switch (g->phase) {
            case EXPECT_VALUE: c = g->depth ? 'n' : '{'; break;
            case OBJ_START_OR_END: case OBJ_AFTER_VAL: c = '}'; break;
            case ARR_START_OR_END: case ARR_AFTER_VAL: c = ']'; break;
            case OBJ_KEY: case IN_ESCAPE: c = '"'; break;
            case OBJ_COLON: c = ':'; break;
            case IN_STRING: c = g->utf8_left ? g->utf8_min : '"'; break;
            case IN_UNICODE: c = '0'; break;
            case IN_LITERAL: c = (unsigned char)literals[g->literal][g->literal_pos]; break;
            case IN_NUMBER:
                if (number_complete(g)) c = g->stack[g->depth - 1] == OBJECT ? '}' : ']';
                else c = '0';
                break;
            default: return -1;
        }
        if (len >= JSON_SUFFIX_CAP || !step(g, c)) return -1;
        suffix[len++] = (char)c;
    }
    return len;
}
