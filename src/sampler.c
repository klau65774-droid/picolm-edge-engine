#include "sampler.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

struct sampler_candidate {
    float weight;
    int token;
};

static double random_unit(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

static int higher(sampler_candidate_t a, sampler_candidate_t b) {
    return a.weight > b.weight || (a.weight == b.weight && a.token < b.token);
}

static void sift_down(sampler_candidate_t *heap, int n, int i) {
    sampler_candidate_t value = heap[i];
    while (i < n / 2) {
        int child = i * 2 + 1;
        if (child + 1 < n && higher(heap[child + 1], heap[child])) child++;
        if (!higher(heap[child], value)) break;
        heap[i] = heap[child];
        i = child;
    }
    heap[i] = value;
}

static float pop_weight(sampler_candidate_t *heap, int size) {
    sampler_candidate_t best = heap[0];
    size--;
    heap[0] = heap[size];
    if (size) sift_down(heap, size, 0);
    heap[size] = best;
    return best.weight;
}

int sampler_init(sampler_t *s, float temperature, float top_p, uint64_t seed,
                 int vocab_size, int top_k, float min_p) {
    *s = (sampler_t){
        .temperature = temperature, .top_p = top_p, .min_p = min_p,
        .top_k = top_k, .rng_state = seed ? seed : 42, .capacity = vocab_size
    };
    if (vocab_size <= 0 || (size_t)vocab_size > SIZE_MAX / sizeof(*s->scratch)) return -1;
    if (temperature > 0.0f) {
        s->scratch = malloc((size_t)vocab_size * sizeof(*s->scratch));
        if (!s->scratch) return -1;
    }
    return 0;
}

void sampler_free(sampler_t *s) {
    free(s->scratch);
    s->scratch = NULL;
    s->capacity = 0;
}

int sampler_sample(sampler_t *s, const float *logits, int vocab_size) {
    if (vocab_size > s->capacity) return -1;
    int best = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (isfinite(logits[i]) && (best < 0 || logits[i] > logits[best])) best = i;
    }
    if (best < 0 || s->temperature <= 0.0f || s->top_k == 1 || s->top_p <= 0.0f) return best;
    if (!s->scratch) return -1;

    sampler_candidate_t *values = s->scratch;
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (!isfinite(logits[i])) continue;
        float weight = expf((logits[i] - logits[best]) / s->temperature);
        if (weight > 0.0f && weight >= s->min_p) {
            values[count++] = (sampler_candidate_t){weight, i};
            sum += weight;
        }
    }
    if (!count) return best;
    if (s->top_p >= 1.0f && (s->top_k == 0 || s->top_k >= count)) {
        double threshold = sum * random_unit(&s->rng_state);
        for (int i = 0; i < count; i++) {
            threshold -= values[i].weight;
            if (threshold < 0.0) return values[i].token;
        }
        return values[count - 1].token;
    }

    /* Build once and pop just the filtered nucleus into the end of the array. */
    for (int i = count / 2; i > 0; i--) sift_down(values, count, i - 1);
    int next = count;
    double mass = 0.0;
    if (s->top_k > 0 && s->top_k < count) {
        int lower = count - s->top_k;
        for (; next > lower; next--) mass += pop_weight(values, next);
        sum = mass;
        next = count;
        mass = 0.0;
        do {
            mass += values[--next].weight;
        } while (next > lower && mass < sum * s->top_p);
    } else {
        do {
            mass += pop_weight(values, next--);
        } while (next > 0 && mass < sum * s->top_p);
    }
    double threshold = mass * random_unit(&s->rng_state);
    for (int i = count - 1; i >= next; i--) {
        threshold -= values[i].weight;
        if (threshold < 0.0) return values[i].token;
    }
    return values[next].token;
}
