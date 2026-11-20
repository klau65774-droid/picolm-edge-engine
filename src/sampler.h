#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct sampler_candidate sampler_candidate_t;

typedef struct {
    float temperature, top_p, min_p;
    int top_k, capacity;
    uint64_t rng_state;
    sampler_candidate_t *scratch;
} sampler_t;

/* top_k=0 and min_p=0 preserve the default nucleus distribution. */
int sampler_init(sampler_t *s, float temperature, float top_p, uint64_t seed,
                 int vocab_size, int top_k, float min_p);
void sampler_free(sampler_t *s);

/* Returns a finite-logit token ID, or -1 if no such token exists. */
int sampler_sample(sampler_t *s, const float *logits, int vocab_size);

#endif
