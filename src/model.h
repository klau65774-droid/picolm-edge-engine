#ifndef MODEL_H
#define MODEL_H

#include "quant.h"
#include "sha256.h"
#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747
#define MAX_LAYERS 64

/* Magic for KV cache files */
#define KVCACHE_MAGIC 0x4B564350  /* "KVCP" */
#define KVCACHE_VERSION 3

typedef enum { KV_FP16 = 0, KV_INT8 = 1 } kv_quant_t;

/* ---- Configuration ---- */

typedef struct {
    int n_embd;         /* embedding dimension (e.g. 2048) */
    int n_ffn;          /* feed-forward hidden size (e.g. 5632) */
    int n_heads;        /* number of attention heads (e.g. 32) */
    int n_kv_heads;     /* number of KV heads for GQA (e.g. 4) */
    int n_layers;       /* number of transformer layers (e.g. 22) */
    int vocab_size;     /* vocabulary size (e.g. 32000) */
    int max_seq_len;    /* maximum sequence length (e.g. 2048) */
    int head_dim;       /* = n_embd / n_heads */
    float rope_freq_base; /* RoPE theta base (e.g. 10000.0) */
    float rms_epsilon;
    int alignment;      /* GGUF data alignment */
    gguf_type_t weight_type; /* default weight quantization type */
} model_config_t;

/* ---- Per-layer weight pointers (into mmap) ---- */

typedef struct {
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_output;
    const void *ffn_norm;
    const void *ffn_gate;
    const void *ffn_down;
    const void *ffn_up;
    /* Per-tensor quantization types */
    gguf_type_t type_attn_norm;
    gguf_type_t type_attn_q;
    gguf_type_t type_attn_k;
    gguf_type_t type_attn_v;
    gguf_type_t type_attn_output;
    gguf_type_t type_ffn_norm;
    gguf_type_t type_ffn_gate;
    gguf_type_t type_ffn_down;
    gguf_type_t type_ffn_up;
} layer_weights_t;

typedef struct {
    const void *token_embd;
    gguf_type_t type_token_embd;
    const void *output_norm;
    gguf_type_t type_output_norm;
    const void *output;        /* final output projection (may alias token_embd) */
    gguf_type_t type_output;
    layer_weights_t layers[MAX_LAYERS];
} model_weights_t;

/* ---- Runtime state (pre-allocated buffers) ---- */

typedef struct {
    void *data;
    uint16_t *scales; /* One FP16 scale per head and position in INT8 mode. */
} kv_plane_t;

typedef struct {
    float *x;            /* current activation [n_embd] */
    float *xb;           /* buffer after norm / attention output [n_embd] */
    float *xb2;          /* second buffer [n_embd] */
    float *q;            /* query vector [n_embd] */
    float *hb;           /* FFN hidden buffer [n_ffn] */
    float *hb2;          /* FFN hidden buffer 2 [n_ffn] */
    float *logits;       /* output logits [vocab_size] */

    kv_plane_t key_cache;
    kv_plane_t val_cache;
    kv_quant_t kv_quant;

    /* Pre-computed RoPE cos/sin tables [max_seq_len * head_dim/2] */
    float *rope_cos;
    float *rope_sin;

    /* Pre-dequantized norm weights (small, keep in RAM) */
    float *norm_weights;
    float *attn_norm_w[MAX_LAYERS];
    float *ffn_norm_w[MAX_LAYERS];
    float *output_norm_w;

    /* Single allocation base */
    void *mem_block;
    size_t mem_size;

    /* Separate allocation for KV cache vectors and any scales. */
    void *kv_block;
    size_t kv_size;
} run_state_t;

/* ---- Model ---- */

typedef struct {
    model_config_t  config;
    model_weights_t weights;
    run_state_t     state;

    /* mmap bookkeeping */
    void  *mmap_addr;
    size_t mmap_size;
#ifdef _WIN32
    void  *file_handle;
    void  *map_handle;
#else
    int    fd;
#endif

    /* Tokenizer data offsets (filled by GGUF parser, used by tokenizer_load) */
    const void *tok_tokens_data;
    size_t      tok_tokens_size;
    uint64_t    tok_n_tokens;
    const void *tok_scores_data;
    uint64_t    tok_n_scores;
    uint32_t    tok_bos_id;
    uint32_t    tok_eos_id;
    uint32_t    tok_unk_id;
    int         tok_no_bos;
    int         tok_no_space_prefix;
    uint8_t     cache_digest[SHA256_BYTES];
    int         cache_digest_ready;
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len);
int model_load_ex(model_t *m, const char *path, int max_seq_len, kv_quant_t kv_quant);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
float *model_forward(model_t *m, int token, int pos);
/* Returns NULL after prefill with compute_logits=0, or for an invalid token/position. */
float *model_forward_ex(model_t *m, int token, int pos, int compute_logits);

/* Free all resources. */
void model_free(model_t *m);

/* ---- KV cache persistence ---- */

/* Save an integrity-checked cache for exactly this model and token prefix. */
int kvcache_save(model_t *m, const char *path, const int *tokens, int positions);

/* Returns positions, 0 on a cache miss, or -1 for invalid data or I/O failure.
 * A read or digest failure while copying discards the partially loaded state. */
int kvcache_load(model_t *m, const char *path, const int *tokens, int count, int trust_legacy);

#endif /* MODEL_H */
