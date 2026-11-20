#ifndef PICOLM_TEST_CHECK_H
#define PICOLM_TEST_CHECK_H

#include <stdio.h>
#include <stdlib.h>

/* Test expressions, including fixture setup, always run in Release builds. */
#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                       \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

#endif
