#ifndef SHA256_H
#define SHA256_H

#include <stddef.h>
#include <stdint.h>

#define SHA256_BYTES 32

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
} sha256_t;

void sha256_init(sha256_t *hash);
void sha256_update(sha256_t *hash, const void *data, size_t len);
void sha256_finish(sha256_t *hash, uint8_t digest[SHA256_BYTES]);

#endif
