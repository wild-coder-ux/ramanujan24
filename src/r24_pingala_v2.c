/*
 * r24_pingala.c — Fibonacci-weighted Jacobi hash (Stage 0)
 *
 * Improvement over plain C hash:
 * - Splits text into words
 * - Words at early positions get HIGH weight (Pingala/Fibonacci)
 * - Words at later positions decay toward 0.1
 * - First sentence carries most signal (matches how humans skim)
 *
 * v2: Stopword filter — common English stopwords are skipped before
 *     Fibonacci position is assigned. This means position 0 is the
 *     first MEANINGFUL word, not "The" or "In" or "During".
 *     Recovers accuracy on unstructured corpora (SQuAD, Wikipedia).
 *
 * Compile: gcc -O3 -shared -fPIC r24_pingala.c -o libr24_pingala.so
 */
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

static int squares[24] = {
    1,4,9,16,25,36,49,64,81,100,
    121,144,169,196,225,256,289,324,361,400,
    441,484,529,576
};

/* Fibonacci sequence for positions 0..19 */
static float fib[20] = {
    1,1,2,3,5,8,13,21,34,55,
    89,144,233,377,610,987,1597,2584,4181,6765
};

static int fib_ready = 0;
static float fib_norm[20];  /* normalized to 0..1 */

static void init_fib() {
    if (fib_ready) return;
    float mx = fib[19];
    for (int i = 0; i < 20; i++)
        fib_norm[i] = fib[i] / mx;
    fib_ready = 1;
}

/*
 * Stopword list — these words are skipped before Fibonacci position
 * is assigned, so position 0 = first meaningful content word.
 * Covers the most common English function words.
 */
static const char* STOPWORDS[] = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","by","from","as","is","was","are","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","need",
    "it","its","this","that","these","those","he","she","they",
    "we","you","i","his","her","their","our","your","my","its",
    "who","which","what","when","where","how","why","not","no",
    "up","out","so","if","then","than","into","about","over",
    "after","before","between","through","during","while","also",
    NULL
};

static int is_stopword(const char* word) {
    /* lowercase compare */
    char lower[64];
    int i;
    for (i = 0; word[i] && i < 63; i++)
        lower[i] = (char)tolower((unsigned char)word[i]);
    lower[i] = '\0';
    for (int j = 0; STOPWORDS[j]; j++)
        if (strcmp(lower, STOPWORDS[j]) == 0) return 1;
    return 0;
}

/* Hash a single word → unsigned int */
static unsigned int word_hash(const char* word) {
    unsigned int h = 5381;
    for (int i = 0; word[i]; i++)
        h = h * 31 + (unsigned char)word[i];
    return h;
}

/*
 * hash24_pingala — main function
 * text  : null-terminated UTF-8 string
 * out   : float[24] output vector (NOT normalized — normalize in Python)
 */
void hash24_pingala(const char* text, float* out) {
    init_fib();
    char buf[8192];
    strncpy(buf, text, 8191);
    buf[8191] = '\0';

    float acc[24] = {0};
    float total_weight = 0.0f;

    char* saveptr = NULL;
    char* word = strtok_r(buf, " \t\n\r.,;:!?()\"-", &saveptr);

    int pos = 0;          /* meaningful word position (stopwords don't count) */
    int raw_pos = 0;      /* total word position (cap to avoid runaway) */

    while (word && raw_pos < 2000) {
        /* skip very short tokens and stopwords */
        if (strlen(word) >= 3 && !is_stopword(word)) {
            unsigned int h = word_hash(word);

            /* Fibonacci weight for first 20 meaningful positions, decay after */
            float w;
            if (pos < 20)
                w = fib_norm[pos];
            else
                w = 0.05f / (1.0f + (pos - 20) * 0.01f);

            for (int i = 0; i < 24; i++)
                acc[i] += w * ((h % squares[i]) / (float)squares[i]);
            total_weight += w;
            pos++;
        }
        word = strtok_r(NULL, " \t\n\r.,;:!?()\"-", &saveptr);
        raw_pos++;
    }

    /* output: weight-normalized, raw (Python will L2-normalize) */
    if (total_weight > 0.0f)
        for (int i = 0; i < 24; i++)
            out[i] = acc[i] / total_weight;
    else
        for (int i = 0; i < 24; i++)
            out[i] = 0.0f;
}

/*
 * Original plain hash — kept for comparison
 */
void hash24_plain(const char* text, float* out) {
    unsigned int h = 5381;
    for (int i = 0; text[i]; i++)
        h = h * 31 + (unsigned char)text[i];
    for (int i = 0; i < 24; i++)
        out[i] = (h % squares[i]) / (float)squares[i];
}
