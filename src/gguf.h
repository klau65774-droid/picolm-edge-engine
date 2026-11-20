#ifndef GGUF_H
#define GGUF_H

#include "model.h"

int gguf_parse(model_t *m, int max_seq_len);

#endif
