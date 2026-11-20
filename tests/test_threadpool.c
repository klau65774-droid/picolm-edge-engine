#include "threadpool.h"

#include "check.h"
#include <stdatomic.h>
#include <stdio.h>

enum { ROWS = 97 };

typedef struct {
    atomic_int visits[ROWS];
    int size;
} context_t;

static void check_slice(void *data, int first, int end) {
    context_t *c = data;
    CHECK(first >= 0 && end <= c->size && first < end);
    for (int i = first; i < end; i++) atomic_fetch_add(&c->visits[i], 1);
}

static void run_generations(int rounds, int persistent, int threads) {
    threadpool_configure(threads, persistent);
    CHECK(threadpool_size() == threads);
    for (int i = 0; i < rounds; i++) {
        context_t c = {.size = (i * 41) % (ROWS + 1)};
        threadpool_run(c.size, check_slice, &c);
        for (int j = 0; j < ROWS; j++) CHECK(atomic_load(&c.visits[j]) == (j < c.size));
    }
    threadpool_shutdown();
    CHECK(threadpool_size() == 1);
}

int main(void) {
    run_generations(1024, 1, 4);
    run_generations(1024, 1, 6);
    run_generations(64, 1, 1);
    run_generations(32, 0, 4);
    puts("threadpool: 2144 generations across persistent, single and transient modes");
    return 0;
}
