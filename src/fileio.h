#ifndef FILEIO_H
#define FILEIO_H

#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <errno.h>

static inline wchar_t *utf8_to_wide(const char *path) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (n <= 0 || (size_t)n > SIZE_MAX / sizeof(wchar_t)) return NULL;
    wchar_t *wide = malloc((size_t)n * sizeof(wchar_t));
    if (wide && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, n)) {
        free(wide);
        wide = NULL;
    }
    return wide;
}

static inline FILE *open_utf8(const char *path, const char *mode) {
    wchar_t *wide = utf8_to_wide(path);
    wchar_t *wide_mode = utf8_to_wide(mode);
    FILE *f = wide && wide_mode ? _wfopen(wide, wide_mode) : NULL;
    if (!wide || !wide_mode) errno = EINVAL;
    free(wide_mode);
    free(wide);
    return f;
}
#else
static inline FILE *open_utf8(const char *path, const char *mode) {
    return fopen(path, mode);
}
#endif

#endif
