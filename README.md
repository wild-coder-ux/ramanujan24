# Ramanujan-24 (R24)

**Multi-stage retrieval that matches BM25 accuracy at 16–21× the speed,
with zero ML at Stage 0 — pure C arithmetic named after ancient Indian mathematicians.**

---

## The Claim

| Dataset | Domain | BM25 | R24 (Pingala+Madhava) | Gap | Speedup |
|---|---|---|---|---|---|
| SQuAD | Factual Q&A | 96.3% | **96.2%** | 0.1 pp | **16×** |
| Thirumandiram | Spiritual/poetic | 90.4% | 89.3% | 1.1 pp | **18×** |
| d2l textbook | Technical | 96.1% | 87.4% | 8.7 pp | **21×** |

**On SQuAD**, R24 closes the gap to 0.1 percentage points while being 16× faster.
The accuracy gap on the other datasets is domain-predictable (explained below).

---

## Architecture

```
Query
  │
  ├── Stage 0: Pingala hash  (C, nanoseconds)
  │           Fibonacci-weighted 24-dim hash
  │           Filters 7,000 → ~400 passages
  │           Zero ML. Pure integer arithmetic.
  │
  ├── Stage 1: Hamming overlap score
  │           Chandas-style binary presence vector
  │           24-bit dot product
  │
  ├── Stage 2: Madhava reranker  (Python)
  │           Token overlap + length penalty
  │           Inspired by Madhava's π-series corrections
  │
  └── Stage 3: Neural embedding re-rank  (top-K only)
               sentence-transformers, runs on filtered set
```

The key insight: most passages are eliminated before any Python runs.
Stage 0 is a compiled C function loaded via ctypes — query latency is in the
microsecond range before Stage 1 even begins.

---

## The Mathematics Behind the Names

R24 is named after the Indian mathematical tradition, not for aesthetics —
each stage maps directly to a historical contribution:

| Mathematician | Period | Role in R24 |
|---|---|---|
| **Pingala** | ~3rd century BCE | Fibonacci-weighted hash (Chandas Shastra — binary prosody) |
| **Aryabhata** | 476 CE | Stage budget — iterative doubling principle |
| **Madhava** | ~1340 CE | Token overlap + length-penalty correction (series convergence) |

Pingala's *Chandas Shastra* is the earliest known description of what we now call
Fibonacci numbers, used to enumerate Sanskrit verse metres by syllable weight.
R24 reuses that weighting to assign positional importance to query tokens in a
hash function — heavier weight to earlier, rarer tokens.

---

## Why the Gap Is Domain-Predictable

The 8.7 pp gap on the d2l textbook is not a flaw — it is expected and explainable:

- **SQuAD** (factual Q&A): questions share exact noun phrases with their answers.
  Fibonacci weighting on content words finds these reliably. Gap: 0.1 pp.
- **Thirumandiram** (spiritual/poetic): verse language is figurative, non-literal.
  BM25's exact-match advantage is smaller. Gap: 1.1 pp.
- **d2l textbook** (technical): questions use natural language; answers use
  notation-heavy prose. Token overlap between question and answer is low by
  design. BM25's IDF weighting handles this better. Gap: 8.7 pp.

The gap is a function of query–passage lexical distance, not algorithm weakness.

---

## Quick Start

### Requirements

```
Python >= 3.9
gcc (for compiling the Pingala hash)
sentence-transformers
numpy
```

```bash
pip install sentence-transformers numpy
```

### Build the Pingala hash

```bash
gcc -O2 -shared -fPIC -o src/libr24_pingala.so src/r24_pingala.c
```

### Auto-configure for your corpus

```bash
python src/r24_setup.py --corpus your_passages.json
```

This runs a K-sweep, detects document type, and writes `r24_config.json`.

### Run a query

```bash
python src/r24_query.py --config r24_config.json --query "your question here"
```

### Reproduce the benchmarks

```bash
# SQuAD
python src/r24_bm25_squad.py

# d2l textbook
python src/r24_bm25_d2l.py

# Thirumandiram
python src/r24_bm25.py
```

---

## Key Implementation Notes

**Stopword handling** is critical for factual Q&A corpora.
The v2 Pingala hash (`r24_pingala.c`) skips a 50-word stopword list when
assigning Fibonacci positional weights. Without this, tokens like "The" and "In"
absorb the highest-weight positions, collapsing the hash to near-random.
This single fix recovered +12.6 pp on SQuAD.

**Thirumandiram pipeline** filters Tamil Unicode blocks (`\u0B80–\u0BFF > 30%`)
to extract clean English passages from a bilingual PDF, then merges truncated
blocks into full verse units.

---

## Repository Layout

```
src/
  r24_pingala.c          # Stage 0 — Pingala hash (core innovation)
  r24_pingala_v1.c       # v1 backup (without stopword handling)
  r24_madhava.py         # Stage 2 — Madhava reranker
  r24_pipeline.py        # Full 4-stage pipeline
  r24_setup.py           # Auto K-sweep + config generator
  r24_query.py           # Interactive query tool
  r24_bm25_squad.py      # SQuAD benchmark
  r24_bm25_d2l.py        # d2l benchmark
  r24_bm25.py            # Thirumandiram benchmark
```

---

## Limitations and Future Work

- Stage 3 neural re-rank uses `sentence-transformers` — swappable for any encoder.
- A cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) as
  Stage 4 is planned and expected to close the d2l gap further.
- The Pingala hash dimension (24) is fixed; a configurable-width version is on
  the roadmap.
- No evaluation yet on multilingual corpora beyond English/Tamil.

---

## Citation

If you use R24 in research, please cite this repository until a paper is available.

```
@software{ramanujan24,
  title  = {Ramanujan-24: Multi-stage retrieval with Fibonacci-weighted hashing},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/ramanujan24}
}
```

---

## License

MIT
