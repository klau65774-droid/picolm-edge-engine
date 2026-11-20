#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"
#include <stdint.h>

typedef struct {
    const char *data;
    size_t len;
    int leading_space;
} tokenizer_piece_t;

typedef struct {
    char  **vocab;       /* vocab[i] = string for token i */
    float  *scores;      /* BPE merge scores */
    int     vocab_size;
    size_t *vocab_len;
    tokenizer_piece_t *pieces;
    int    *hash;       /* Open addressing, zero means empty and IDs are stored plus one. */
    size_t  hash_mask;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t unk_id;
    int      add_bos;
    int      add_space_prefix;
} tokenizer_t;

/* Load tokenizer data from GGUF metadata pointers in model.
 * Returns 0 on success. */
int tokenizer_load(tokenizer_t *t, const model_t *m);

/* Encode a text string into token IDs.
 * tokens must have space for at least max_tokens entries.
 * Returns the number of tokens produced, or -1 on failure. */
int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos);

/* Span preserves raw byte tokens, including zero bytes. The text wrapper
 * returns the same borrowed, NUL-terminated storage for legacy callers. */
tokenizer_piece_t tokenizer_piece(const tokenizer_t *t, int prev_token, int token);
const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token);

/* Free tokenizer resources */
void tokenizer_free(tokenizer_t *t);

#endif /* TOKENIZER_H */
