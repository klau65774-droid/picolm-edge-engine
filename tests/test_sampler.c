#include "sampler.h"

#include "check.h"
#include <math.h>
#include <stdio.h>

static void only(const float *logits, int k, float min_p, float top_p, unsigned allowed) {
    sampler_t s;
    CHECK(sampler_init(&s, 1.0f, top_p, UINT64_C(128302132944576), 6, k, min_p) == 0);
    unsigned seen = 0;
    for (int i = 0; i < 512; i++) {
        int token = sampler_sample(&s, logits, 6);
        CHECK(token >= 0 && token < 6 && ((1u << token) & allowed));
        seen |= 1u << token;
    }
    CHECK(seen == allowed);
    sampler_free(&s);
}

int main(void) {
    float logits[6] = {2.0f, 1.0f, 0.0f, -INFINITY, NAN, -INFINITY};
    only(logits, 0, 0.0f, 1.0f, 7);
    only(logits, 2, 0.0f, 1.0f, 3);
    only(logits, 2, 0.0f, 0.7f, 1);
    only(logits, 0, 0.2f, 1.0f, 3);
    only(logits, 0, 0.0f, 0.9f, 3);
    sampler_t s;
    CHECK(sampler_init(&s, 0.0f, 0.9f, 42, 6, 0, 0.0f) == 0);
    logits[1] = logits[0];
    CHECK(sampler_sample(&s, logits, 6) == 0);
    CHECK(sampler_sample(&s, logits + 3, 3) == -1);
    CHECK(sampler_sample(&s, logits, 7) == -1);
    sampler_free(&s);
    puts("sampler: 2560 filtered draws and greedy/masked edge cases");
    return 0;
}
