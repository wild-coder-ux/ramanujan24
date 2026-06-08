/*
 * r24_pingala.c — Fibonacci-weighted hash + Hamming bit trick (Stage 0)
 *
 * v3 additions over v2:
 *   - hash24_pingala     : stopword-aware Fibonacci hash (unchanged from v2)
 *   - hash24_hamming     : Chandas-style binary presence vector
 *                          bit i = 1 if any content word hashes to bucket i
 *                          Hamming overlap = lexical match, 1 POPCNT instruction
 *   - score_hamming      : returns overlap count (0..24) between two bit vectors
 *
 * Compile:
 *   gcc -O3 -shared -fPIC r24_pingala.c -o libr24_pingala.so
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

static int squares[24] = {
    1,4,9,16,25,36,49,64,81,100,
    121,144,169,196,225,256,289,324,361,400,
    441,484,529,576
};

static float fib[20] = {
    1,1,2,3,5,8,13,21,34,55,
    89,144,233,377,610,987,1597,2584,4181,6765
};

static int   fib_ready = 0;
static float fib_norm[20];

static void init_fib() {
    if (fib_ready) return;
    float mx = fib[19];
    for (int i = 0; i < 20; i++)
        fib_norm[i] = fib[i] / mx;
    fib_ready = 1;
}

static const char* STOPWORDS[] = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","by","from","as","is","was","are","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","need",
    "it","its","this","that","these","those","he","she","they",
    "we","you","i","his","her","their","our","your","my",
    "who","which","what","when","where","how","why","not","no",
    "up","out","so","if","then","than","into","about","over",
    "after","before","between","through","during","while","also",
    NULL
};

static int is_stopword(const char* word) {
    char lower[64];
    int i;
    for (i = 0; word[i] && i < 63; i++)
        lower[i] = (char)tolower((unsigned char)word[i]);
    lower[i] = '\0';
    for (int j = 0; STOPWORDS[j]; j++)
        if (strcmp(lower, STOPWORDS[j]) == 0) return 1;
    return 0;
}

static unsigned int word_hash(const char* word) {
    unsigned int h = 5381;
    for (int i = 0; word[i]; i++)
        h = h * 31 + (unsigned char)word[i];
    return h;
}

/* ── hash24_pingala v2 (stopword-aware Fibonacci) ── */
void hash24_pingala(const char* text, float* out) {
    init_fib();
    char buf[8192];
    strncpy(buf, text, 8191);
    buf[8191] = '\0';

    float acc[24]      = {0};
    float total_weight = 0.0f;
    char* saveptr      = NULL;
    char* word         = strtok_r(buf, " \t\n\r.,;:!?()\"-", &saveptr);
    int   pos = 0, raw_pos = 0;

    while (word && raw_pos < 2000) {
        if (strlen(word) >= 3 && !is_stopword(word)) {
            unsigned int h = word_hash(word);
            float w = (pos < 20)
                ? fib_norm[pos]
                : 0.05f / (1.0f + (pos - 20) * 0.01f);
            for (int i = 0; i < 24; i++)
                acc[i] += w * ((h % squares[i]) / (float)squares[i]);
            total_weight += w;
            pos++;
        }
        word = strtok_r(NULL, " \t\n\r.,;:!?()\"-", &saveptr);
        raw_pos++;
    }

    if (total_weight > 0.0f)
        for (int i = 0; i < 24; i++)
            out[i] = acc[i] / total_weight;
    else
        for (int i = 0; i < 24; i++)
            out[i] = 0.0f;
}

/* ── hash24_hamming — Chandas binary presence vector ──
 * Each content word sets bit (word_hash % 24) = 1.
 * out[0..23] = 0.0 or 1.0
 * Overlap between query and passage = lexical match score.
 */
void hash24_hamming(const char* text, float* out) {
    for (int i = 0; i < 24; i++) out[i] = 0.0f;

    char buf[8192];
    strncpy(buf, text, 8191);
    buf[8191] = '\0';

    char* saveptr = NULL;
    char* word    = strtok_r(buf, " \t\n\r.,;:!?()\"-", &saveptr);
    int   raw_pos = 0;

    while (word && raw_pos < 2000) {
        if (strlen(word) >= 3 && !is_stopword(word)) {
            unsigned int h = word_hash(word);
            out[(int)(h % 24)] = 1.0f;
        }
        word = strtok_r(NULL, " \t\n\r.,;:!?()\"-", &saveptr);
        raw_pos++;
    }
}

/* ── score_hamming — bit overlap count 0..24 ── */
int score_hamming(const float* a, const float* b) {
    int overlap = 0;
    for (int i = 0; i < 24; i++)
        if (a[i] > 0.5f && b[i] > 0.5f) overlap++;
    return overlap;
}

/* ── hash24_plain (original) ── */
void hash24_plain(const char* text, float* out) {
    unsigned int h = 5381;
    for (int i = 0; text[i]; i++)
        h = h * 31 + (unsigned char)text[i];
    for (int i = 0; i < 24; i++)
        out[i] = (h % squares[i]) / (float)squares[i];
}
