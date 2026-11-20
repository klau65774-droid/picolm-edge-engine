#include "fileio.h"
#include "grammar.h"
#include "model.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

typedef struct {
    const char *model, *prompt, *cache, *ppl;
    int max_tokens, context, threads, json, bench, transient, top_k, trust_legacy_cache;
    float temperature, top_p, min_p;
    uint64_t seed;
    kv_quant_t kv_quant;
} options_t;

typedef struct {
    char *data;
    size_t len, capacity;
} text_t;

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return 1000.0 * (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1e6;
#endif
}

static double peak_rss_mb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS mem;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &mem, sizeof(mem))) return 0.0;
    return (double)mem.PeakWorkingSetSize / 1048576.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#ifdef __APPLE__
    return (double)usage.ru_maxrss / 1048576.0;
#else
    return (double)usage.ru_maxrss / 1024.0;
#endif
#endif
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s <model.gguf> [options]\n"
            "  -p <prompt>    Input prompt (or pipe UTF-8 text to stdin)\n"
            "  -n <int>       Generation budget (default: 256, upstream counting)\n"
            "  -t <float>     Temperature (default: 0.8, 0 means greedy)\n"
            "  -k <float>     Top-p nucleus probability (default: 0.9)\n"
            "  -s <int>       RNG seed (default: 42)\n"
            "  -c <int>       Limit the model context length\n"
            "  -j <int>       Number of threads (default: 4, up to 16)\n"
            "  --json         Constrain output to one complete JSON value\n"
            "  --cache <file> Save and reuse prompt KV state\n"
            "  --trust-legacy-cache  Allow identity-free KVCP v1/v2 input for this run\n"
            "  --kv-quant fp16|int8   KV representation (default: fp16)\n"
            "  --no-threadpool        Use temporary workers for comparison\n"
            "  --top-k <int>          Optional candidate limit (0 means disabled)\n"
            "  --min-p <float>        Optional probability relative to the best candidate\n"
            "  --bench               Emit benchmark JSON on stderr\n"
            "  --ppl <file>          Emit perplexity and NLL as JSON on stdout\n",
            program);
}

static int read_int(const char *s, int *out) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n < 0 || n > INT_MAX) return 0;
    *out = (int)n;
    return 1;
}

static int read_float(const char *s, float *out, float max) {
    char *end;
    errno = 0;
    float value = strtof(s, &end);
    if (errno || end == s || *end || !isfinite(value) || value < 0.0f || value > max) return 0;
    *out = value;
    return 1;
}

static int parse_options(options_t *o, int argc, char **argv) {
    *o = (options_t){.model = argv[1], .max_tokens = 256, .threads = 4, .temperature = 0.8f,
                     .top_p = 0.9f, .seed = 42, .kv_quant = KV_FP16};
    for (int i = 2; i < argc; i++) {
        const char *flag = argv[i];
        if (!strcmp(flag, "--json")) { o->json = 1; continue; }
        if (!strcmp(flag, "--bench")) { o->bench = 1; continue; }
        if (!strcmp(flag, "--no-threadpool")) { o->transient = 1; continue; }
        if (!strcmp(flag, "--trust-legacy-cache")) { o->trust_legacy_cache = 1; continue; }
        if (i + 1 == argc) goto invalid;
        const char *value = argv[++i];
        if (!strcmp(flag, "-p")) o->prompt = value;
        else if (!strcmp(flag, "--cache")) o->cache = value;
        else if (!strcmp(flag, "--ppl")) o->ppl = value;
        else if (!strcmp(flag, "-n") && read_int(value, &o->max_tokens)) {}
        else if (!strcmp(flag, "-c") && read_int(value, &o->context)) {}
        else if (!strcmp(flag, "-j") && read_int(value, &o->threads)) {}
        else if (!strcmp(flag, "--top-k") && read_int(value, &o->top_k)) {}
        else if (!strcmp(flag, "-t") && read_float(value, &o->temperature, INFINITY)) {}
        else if (!strcmp(flag, "-k") && read_float(value, &o->top_p, 1.0f)) {}
        else if (!strcmp(flag, "--min-p") && read_float(value, &o->min_p, 1.0f)) {}
        else if (!strcmp(flag, "--kv-quant") && !strcmp(value, "fp16")) o->kv_quant = KV_FP16;
        else if (!strcmp(flag, "--kv-quant") && !strcmp(value, "int8")) o->kv_quant = KV_INT8;
        else if (!strcmp(flag, "-s")) {
            char *end;
            errno = 0;
            o->seed = strtoull(value, &end, 10);
            if (errno || end == value || *end) goto invalid;
        } else {
invalid:
            fprintf(stderr, "Unknown option or invalid value: %s\n", flag);
            return 0;
        }
    }
    if (o->ppl && (o->prompt || o->json || o->cache)) {
        fprintf(stderr, "--ppl requires its own input file and fresh context\n");
        return 0;
    }
    if (o->trust_legacy_cache && !o->cache) {
        fprintf(stderr, "--trust-legacy-cache requires --cache\n");
        return 0;
    }
    return 1;
}

static int append_text(text_t *text, const char *bytes, size_t len) {
    if (len > SIZE_MAX - text->len - 1) return 0;
    size_t needed = text->len + len + 1;
    if (needed > text->capacity) {
        size_t next = text->capacity <= SIZE_MAX / 2 ? 2 * text->capacity : SIZE_MAX;
        if (next < 4096) next = 4096;
        if (next < needed) next = needed;
        char *p = realloc(text->data, next);
        if (!p) return 0;
        text->data = p;
        text->capacity = next;
    }
    memcpy(text->data + text->len, bytes, len);
    text->len += len;
    text->data[text->len] = '\0';
    return 1;
}

static int read_text(FILE *f, text_t *text) {
    char chunk[4096];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) != 0) {
        if (memchr(chunk, '\0', n) || !append_text(text, chunk, n)) return 0;
    }
    return !ferror(f) && text->len != 0;
}

static int *encode_text(const tokenizer_t *tok, const char *text, int *count) {
    size_t bytes = strlen(text);
    if (bytes > (size_t)(INT_MAX - 4) / 3 || bytes > (SIZE_MAX / sizeof(int) - 4) / 3) return NULL;
    int capacity = (int)(bytes * 3 + 4);
    int *ids = malloc((size_t)capacity * sizeof(*ids));
    if (!ids) return NULL;
    *count = tokenizer_encode(tok, text, ids, capacity, tok->add_bos);
    if (*count <= 0) {
        free(ids);
        return NULL;
    }
    return ids;
}

static const char *quant_name(kv_quant_t q) { return q == KV_INT8 ? "int8" : "fp16"; }

static void generation_error(const options_t *o, const char *reason, const char *message) {
    fprintf(stderr, "%s\n", message);
    if (o->bench) fprintf(stderr, "{\"finish_reason\":\"%s\"}\n", reason);
}

static int perplexity(const options_t *o, model_t *m, const tokenizer_t *tok) {
    text_t text = {0};
    FILE *f = open_utf8(o->ppl, "rb");
    int ok = f && read_text(f, &text);
    if (f) fclose(f);
    if (!ok) {
        fprintf(stderr, "Cannot read a nonempty UTF-8 evaluation file: %s\n", o->ppl);
        free(text.data);
        return 1;
    }
    int n;
    int *ids = encode_text(tok, text.data, &n);
    free(text.data);
    if (!ids || n < 2) {
        fprintf(stderr, "Perplexity requires at least one scoreable token\n");
        free(ids);
        return 1;
    }
    double total = 0.0;
    int context = m->config.max_seq_len;
    double start = now_ms();
    for (int i = 0; i < n - 1; i++) {
        const float *logits = model_forward_ex(m, ids[i], i % context, 1);
        double max = -INFINITY;
        for (int t = 0; t < tok->vocab_size; t++) {
            if (logits[t] > max) max = logits[t];
        }
        double sum = 0.0;
        for (int t = 0; t < tok->vocab_size; t++) sum += exp((double)logits[t] - max);
        total += (max - (double)logits[ids[i + 1]]) + log(sum);
    }
    free(ids);
    double nll = total / (n - 1), ppl = exp(nll);
    if (!isfinite(nll) || !isfinite(ppl)) {
        fprintf(stderr, "Perplexity was not finite\n");
        return 1;
    }
    printf("{\"perplexity\":%.10g,\"nll\":%.10g,\"total_nll\":%.10g,\"scored_tokens\":%d,"
           "\"context\":%d,\"elapsed_ms\":%.3f,\"peak_rss_mb\":%.3f,\"kv_cache_bytes\":%zu,"
           "\"threads\":%d,\"simd\":\"%s\",\"kv_quant\":\"%s\"}\n",
           ppl, nll, total, n - 1, context, now_ms() - start, peak_rss_mb(), m->state.kv_size,
           threadpool_size(), quant_simd(), quant_name(o->kv_quant));
    return 0;
}

static int generate(const options_t *o, model_t *m, const tokenizer_t *tok, const char *prompt) {
    int count;
    int *ids = encode_text(tok, prompt, &count);
    if (!ids) {
        fprintf(stderr, "Could not tokenize prompt\n");
        return 1;
    }
    int n_prompt = count < m->config.max_seq_len ? count : m->config.max_seq_len;
    int available = m->config.max_seq_len - n_prompt + 1;
    /* The original CLI samples once from the last prompt token, then takes -n
     * decode steps. Keep this behavior for existing -n callers. */
    int budget = o->max_tokens < available ? o->max_tokens + 1 : available;
    if (budget > available) budget = available;

    int status = 1, suffix_bytes = 0, generated = 0, printed_ids = 0;
    text_t json = {0};
    sampler_t sampler = {0};
    grammar_state_t grammar;
    grammar_init(&grammar);
    int *trace = NULL;
    if (o->bench) {
        trace = malloc((size_t)budget * sizeof(*trace));
        if (!trace) goto cleanup;
    }
    if (sampler_init(&sampler, o->temperature, o->top_p, o->seed, tok->vocab_size,
                     o->top_k, o->min_p) != 0) goto cleanup;

    int cached = o->cache ? kvcache_load(m, o->cache, ids, n_prompt, o->trust_legacy_cache) : 0;
    if (cached < 0) goto cleanup;
    int start_pos = cached > 0 ? cached - 1 : 0;
    fprintf(stderr, "Prompt: %d tokens, generating up to %d (threads=%d, simd=%s)\n",
            n_prompt, o->max_tokens, threadpool_size(), quant_simd());

    double start = now_ms();
    for (int i = start_pos; i < n_prompt - 1; i++) model_forward_ex(m, ids[i], i, 0);
    int pos = n_prompt - 1;
    int token = ids[pos];
    float *logits = model_forward_ex(m, token, pos, 1);
    double prefill_end = now_ms(), first = 0.0;
    enum { FINISH_BUDGET, FINISH_EOS, FINISH_JSON } finish = FINISH_BUDGET;
    const char *const finish_names[] = {"budget", "eos", "grammar_complete"};

    while (generated < budget) {
        if (o->json) {
            grammar_filter_t filter = grammar_apply(&grammar, tok, token, logits);
            if (filter == GRAMMAR_NO_FINITE_LOGITS) {
                generation_error(o, "nonfinite_logits", "Model supplied no finite logits");
                goto cleanup;
            }
            if (filter == GRAMMAR_NO_CANDIDATE) {
                generation_error(o, "grammar_no_candidate", "JSON grammar admits no token at this position");
                goto cleanup;
            }
        }
        int next = sampler_sample(&sampler, logits, tok->vocab_size);
        if (next < 0) {
            generation_error(o, "nonfinite_logits", "Model supplied no finite logits");
            goto cleanup;
        }
        if (trace) trace[printed_ids++] = next;
        generated++;
        if (o->json) {
            if (next != (int)tok->eos_id) {
                tokenizer_piece_t piece = tokenizer_piece(tok, token, next);
                if (!grammar_feed(&grammar, piece.data, piece.len) ||
                    !append_text(&json, piece.data, piece.len)) goto cleanup;
            }
        } else {
            tokenizer_piece_t piece = tokenizer_piece(tok, token, next);
            if (fwrite(piece.data, 1, piece.len, stdout) != piece.len || fflush(stdout)) goto cleanup;
        }
        if (generated == 1) first = now_ms();
        if (next == (int)tok->eos_id) {
            finish = FINISH_EOS;
            break;
        }
        if (o->json && grammar_is_complete(&grammar)) {
            finish = FINISH_JSON;
            break;
        }
        if (generated >= budget) break;
        token = next;
        logits = model_forward_ex(m, token, ++pos, 1);
    }
    if (o->json) {
        char suffix[JSON_SUFFIX_CAP];
        if (finish == FINISH_BUDGET) suffix_bytes = grammar_finish(&grammar, suffix);
        if (suffix_bytes < 0 || !grammar_can_end(&grammar) ||
            !append_text(&json, suffix, (size_t)suffix_bytes) ||
            fwrite(json.data, 1, json.len, stdout) != json.len) goto cleanup;
    }
    if (putchar('\n') == EOF || fflush(stdout)) goto cleanup;
    double end = now_ms();
    if (o->cache && kvcache_save(m, o->cache, ids, n_prompt) != 0) {
        fprintf(stderr, "Could not save KV cache: %s\n", o->cache);
        goto cleanup;
    }

    int prefill = n_prompt - start_pos;
    double prefill_ms = prefill_end - start, gen_ms = end - prefill_end;
    double prefill_rate = prefill_ms > 0.0 ? 1000.0 * prefill / prefill_ms : 0.0;
    double gen_rate = gen_ms > 0.0 ? 1000.0 * generated / gen_ms : 0.0;
    fprintf(stderr, "Prefill: %d tokens in %.2fs (%.1f tok/s)%s\n",
            prefill, prefill_ms / 1000.0, prefill_rate, start_pos ? " [partially cached]" : "");
    fprintf(stderr, "Generation: %d tokens in %.2fs (%.1f tok/s)\n",
            generated, gen_ms / 1000.0, gen_rate);
    fprintf(stderr, "Memory: %.2f MiB runtime state (%s KV cache)\n",
            (double)m->state.mem_size / 1048576.0, quant_name(o->kv_quant));
    if (o->bench) {
        double ttft = (first > 0.0 ? first : end) - start;
        fprintf(stderr,
                "{\"prefill_tokens\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%d,"
                "\"prefill_tok_s\":%.6f,\"decode_tok_s\":%.6f,\"ttft_ms\":%.3f,"
                "\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"peak_rss_mb\":%.3f,"
                "\"threads\":%d,\"simd\":\"%s\",\"kv_quant\":\"%s\",\"threadpool\":%s,"
                "\"kv_cache_bytes\":%zu,\"runtime_bytes\":%zu,\"json_suffix_bytes\":%d,"
                "\"finish_reason\":\"%s\","
                "\"token_ids\":[",
                prefill, n_prompt, generated, prefill_rate, gen_rate, ttft, prefill_ms, gen_ms,
                peak_rss_mb(), threadpool_size(), quant_simd(), quant_name(o->kv_quant),
                !o->transient && threadpool_size() > 1 ? "true" : "false",
                m->state.kv_size, m->state.mem_size, suffix_bytes, finish_names[finish]);
        for (int i = 0; i < printed_ids; i++) fprintf(stderr, "%s%d", i ? "," : "", trace[i]);
        fputs("]}\n", stderr);
    }
    status = 0;

cleanup:
    if (status) fprintf(stderr, "Generation failed\n");
    sampler_free(&sampler);
    free(trace);
    free(json.data);
    free(ids);
    return status;
}

static int picolm_main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        usage(argv[0]);
        return 0;
    }
    options_t options;
    if (argc < 2 || !parse_options(&options, argc, argv)) {
        usage(argv[0]);
        return 1;
    }

    text_t input = {0};
    const char *prompt = options.prompt;
    if (!options.ppl && !prompt) {
#ifdef _WIN32
        DWORD mode;
        int piped = !GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode);
#else
        int piped = !isatty(fileno(stdin));
#endif
        if (piped && read_text(stdin, &input)) prompt = input.data;
    }
    int status = 1;
    if (!options.ppl && (!prompt || !*prompt)) {
        fprintf(stderr, "Provide a prompt with -p or stdin\n");
        free(input.data);
        return 1;
    }

    model_t model;
    if (model_load_ex(&model, options.model, options.context, options.kv_quant) == 0) {
        tokenizer_t tokenizer;
        if (tokenizer_load(&tokenizer, &model) == 0) {
            threadpool_configure(options.threads, !options.transient);
            status = options.ppl ? perplexity(&options, &model, &tokenizer)
                                 : generate(&options, &model, &tokenizer, prompt);
            threadpool_shutdown();
            tokenizer_free(&tokenizer);
        }
        model_free(&model);
    }
    free(input.data);
    return status;
}

#ifdef _WIN32
int wmain(int argc, wchar_t **wide) {
    char **argv = calloc((size_t)argc + 1, sizeof(*argv));
    if (!argv) return 1;
    int status = 1, i = 0;
    for (; i < argc; i++) {
        int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, NULL, 0, NULL, NULL);
        if (!size) break;
        argv[i] = malloc((size_t)size);
        if (!argv[i] ||
            !WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, argv[i], size, NULL, NULL)) {
            break;
        }
    }
    if (i == argc) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
        status = picolm_main(argc, argv);
    }
    for (int j = 0; j < argc; j++) free(argv[j]);
    free(argv);
    return status;
}
#else
int main(int argc, char **argv) {
    return picolm_main(argc, argv);
}
#endif
