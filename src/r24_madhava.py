"""
r24_madhava.py — Madhava Series Reranker for Ramanujan-24
==========================================================
Tests four scoring strategies:

  A) Pingala only          — baseline
  B) Pingala + Hamming     — Hamming blended INTO Stage 0 shortlist
  C) Pingala + Madhava     — Madhava series correction at Stage 0
  D) Combined              — Pingala + Hamming + Madhava at Stage 0

Key fix over v1: Hamming/Madhava now change WHICH passages survive
Stage 0, not just reorder the final 10. This is where recall is won.

Evaluated at both Recall@10 and Recall@1 (precision).

Run:
  python r24_madhava.py           # SQuAD (worst case)
  python r24_madhava.py --thiru   # Thirumandiram
  python r24_madhava.py --d2l     # d2l textbook
"""

import numpy as np
import ctypes, os, re, time, argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE  = 4
MAX_QUERIES = 500
STAGE0_K    = 1500   # overridden per dataset
STAGE1_DIMS = 24
STAGE2_DIMS = 96
STAGE1_K    = 200
STAGE2_K    = 50
STAGE3_K    = 10     # keep 10 for Recall@10; we also check top-1 separately
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Blend weights for Stage 0 combination
HAMMING_W   = 0.4    # weight of Hamming overlap in Stage 0 blend
MADHAVA_W   = 0.3    # weight of token overlap in Stage 0 blend
LENGTH_W    = 0.1    # penalty weight for over-long passages
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms

# ── Hash loaders ──────────────────────────────────────────────────────────────

def load_lib(so_name):
    path = os.path.abspath(so_name)
    if not os.path.exists(path): return None, None
    lib = ctypes.CDLL(path)

    pingala_fn = lib.hash24_pingala
    pingala_fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    pingala_fn.restype  = None

    hamming_fn = lib.hash24_hamming
    hamming_fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    hamming_fn.restype  = None

    return pingala_fn, hamming_fn

def hash_text(fn, text):
    arr = (ctypes.c_float * 24)()
    fn(text.encode("utf-8", errors="replace"), arr)
    v = np.array(list(arr), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def hamming_vec(fn, text):
    arr = (ctypes.c_float * 24)()
    fn(text.encode("utf-8", errors="replace"), arr)
    return np.array(list(arr), dtype=np.float32)

def build_matrix(fn, passages, use_norm=True):
    out = np.zeros((len(passages), 24), dtype=np.float32)
    for i, p in enumerate(passages):
        v = np.array([(ctypes.c_float * 24)()], dtype=np.float32)
        arr = (ctypes.c_float * 24)()
        fn(p.encode("utf-8", errors="replace"), arr)
        v = np.array(list(arr), dtype=np.float32)
        if use_norm:
            n = np.linalg.norm(v)
            v = v / (n + 1e-9)
        out[i] = v
    return out

# ── Madhava token scoring ─────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","it","its","this",
    "that","he","she","they","we","you","i","who","which","what","when",
    "where","how","why","not","no","so","if","then","than","also"
}

def content_words(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return set(w for w in words if w not in STOPWORDS)

def build_passage_word_sets(passages):
    """Pre-compute content word sets for all passages — O(N) once."""
    return [content_words(p) for p in passages]

def madhava_stage0_scores(query, passage_word_sets):
    """
    Madhava series token scores for all passages vs query.
    Term 2: token overlap ratio  (positive correction)
    Term 3: length mismatch      (negative correction)
    Returns float array shape (N,)
    """
    q_words = content_words(query)
    if not q_words:
        return np.zeros(len(passage_word_sets), dtype=np.float32)

    scores = np.zeros(len(passage_word_sets), dtype=np.float32)
    for i, p_words in enumerate(passage_word_sets):
        overlap  = len(q_words & p_words) / len(q_words)
        p_len    = max(len(p_words), 1)
        q_len    = max(len(q_words), 1)
        penalty  = min(1.0, max(0.0, (p_len / q_len - 5.0) / 20.0))
        scores[i] = MADHAVA_W * overlap - LENGTH_W * penalty
    return scores

# ── Stage 0 strategies ────────────────────────────────────────────────────────

def stage0_pingala(q_hash, hash_mat, k):
    return np.argsort(hash_mat @ q_hash)[::-1][:k]

def stage0_pingala_hamming(q_hash, q_hvec, hash_mat, hamming_mat, k):
    ping_scores = hash_mat    @ q_hash
    ham_scores  = hamming_mat @ q_hvec / 24.0   # normalise 0..1
    combined    = ping_scores + HAMMING_W * ham_scores
    return np.argsort(combined)[::-1][:k]

def stage0_pingala_madhava(q_hash, hash_mat, madhava_scores, k):
    ping_scores = hash_mat @ q_hash
    combined    = ping_scores + madhava_scores
    return np.argsort(combined)[::-1][:k]

def stage0_all(q_hash, q_hvec, hash_mat, hamming_mat, madhava_scores, k):
    ping_scores = hash_mat    @ q_hash
    ham_scores  = hamming_mat @ q_hvec / 24.0
    combined    = ping_scores + HAMMING_W * ham_scores + madhava_scores
    return np.argsort(combined)[::-1][:k]

# ── Neural cascade (unchanged) ────────────────────────────────────────────────

def neural_cascade(q_emb, stage0_ids, embs):
    survivors0 = stage0_ids

    s1k    = min(STAGE1_K, len(survivors0))
    cands1 = normalize(embs[survivors0, :STAGE1_DIMS])
    q24    = normalize(q_emb[None, :STAGE1_DIMS])
    surv1  = survivors0[np.argsort((cands1 @ q24.T).squeeze())[::-1][:s1k]]

    s2k    = min(STAGE2_K, len(surv1))
    cands2 = normalize(embs[surv1, :STAGE2_DIMS])
    q96    = normalize(q_emb[None, :STAGE2_DIMS])
    surv2  = surv1[np.argsort((cands2 @ q96.T).squeeze())[::-1][:s2k]]

    cands3 = normalize(embs[surv2])
    q768   = normalize(q_emb[None, :])
    scores = (cands3 @ q768.T).squeeze()
    if scores.ndim == 0: scores = scores[None]
    order  = np.argsort(scores)[::-1][:STAGE3_K]
    return surv2[order]

def brute_force(q_emb, embs):
    scores = (normalize(embs) @ normalize(q_emb[None, :]).T).squeeze()
    return np.argsort(scores)[::-1][:STAGE3_K]

# ── Data loaders ──────────────────────────────────────────────────────────────

def is_garbage(t):
    garbage = sum(1 for c in t if ord(c) > 127 or c in '@®°_$')
    return garbage / max(len(t), 1) > 0.05

def is_noise(t):
    if re.search(r'\b\d{3,}\b.*\b\d{3,}\b', t): return True
    if t.startswith('•') or t.startswith('·'):   return True
    if len(t.split()) < 10:                       return True
    tamil = sum(1 for c in t if '\u0B80' <= c <= '\u0BFF')
    if tamil / max(len(t), 1) > 0.3:             return True
    if is_garbage(t):                             return True
    return False

def load_pdf(pdf_path):
    import fitz
    doc, passages, buffer = fitz.open(pdf_path), [], ""
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] != 0: continue
            t = block[4].strip().replace("\n", " ")
            tamil = sum(1 for c in t if '\u0B80' <= c <= '\u0BFF')
            if tamil / max(len(t), 1) > 0.3 or is_garbage(t):
                if buffer and len(buffer.split()) >= 10 and not is_garbage(buffer):
                    passages.append(buffer)
                buffer = ""
                continue
            if is_noise(t): continue
            if buffer:
                if not buffer.rstrip().endswith(('.', '?', '!', '"', "'", '~')):
                    buffer = buffer.rstrip() + " " + t
                    continue
                else:
                    if len(buffer.split()) >= 10: passages.append(buffer)
                    buffer = t
            else:
                buffer = t
    if buffer and len(buffer.split()) >= 10 and not is_garbage(buffer):
        passages.append(buffer)
    return passages

def load_squad():
    from datasets import load_dataset
    ds = load_dataset("squad", split="validation")
    seen, passages, passage_idx = set(), [], {}
    for row in ds:
        ctx = row["context"]
        if ctx not in seen:
            seen.add(ctx)
            passage_idx[ctx] = len(passages)
            passages.append(ctx)
    queries, gold_ids = [], []
    for row in ds:
        if row["context"] not in passage_idx: continue
        queries.append(row["question"])
        gold_ids.append(passage_idx[row["context"]])
        if len(queries) >= MAX_QUERIES: break
    return passages, queries, gold_ids

def make_queries(passages, n=MAX_QUERIES):
    queries, gold_ids = [], []
    for i, p in enumerate(passages):
        first = p.split(".")[0].strip()
        if len(first) > 20:
            queries.append(first)
            gold_ids.append(i)
        if len(queries) >= n: break
    return queries, gold_ids

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thiru", action="store_true")
    parser.add_argument("--d2l",   action="store_true")
    args = parser.parse_args()

    global STAGE0_K

    if args.thiru:
        print("\n📄 Loading Thirumandiram…")
        passages = load_pdf("Thirumanthiram_text.pdf")
        queries, gold_ids = make_queries(passages)
        cache    = "thirumanthiram_vectors.npy"
        label    = "Thirumandiram"
        STAGE0_K = 4000
    elif args.d2l:
        print("\n📄 Loading d2l…")
        passages = load_pdf("d2l_deep_learning.pdf")
        queries, gold_ids = make_queries(passages)
        cache    = "d2l_deep_learning_vectors.npy"
        label    = "d2l"
        STAGE0_K = 4000
    else:
        print("\n📄 Loading SQuAD…")
        passages, queries, gold_ids = load_squad()
        cache    = "squad_vectors.npy"
        label    = "SQuAD"
        STAGE0_K = 1500

    N = len(passages)
    k = min(STAGE0_K, N)
    print(f"   {N:,} passages | {len(queries)} queries | Stage0_K={k}")

    # Load libs
    pingala_fn, hamming_fn = load_lib("libr24_pingala.so")
    if not pingala_fn:
        print("❌ libr24_pingala.so not found")
        print("   Compile: gcc -O3 -shared -fPIC r24_pingala.c -o libr24_pingala.so")
        return

    # Model + embeddings
    print(f"\n🤖 Loading model ({DEVICE.upper()})…")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)

    if os.path.exists(cache):
        embs = np.load(cache)
        if len(embs) == N:
            print(f"📦 Loaded cached embeddings")
        else:
            embs = None
    else:
        embs = None

    if embs is None:
        if DEVICE == "cuda": torch.cuda.empty_cache()
        embs = model.encode(passages, batch_size=BATCH_SIZE,
                            show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True)
        np.save(cache, embs)

    # Build matrices
    print("⚡ Building Pingala hash matrix…")
    hash_mat    = build_matrix(pingala_fn, passages, use_norm=True)
    print("⚡ Building Hamming bit matrix…")
    hamming_mat = build_matrix(hamming_fn, passages, use_norm=False)
    print("⚡ Pre-computing passage word sets for Madhava…")
    passage_word_sets = build_passage_word_sets(passages)

    # Embed queries
    print(f"🔍 Embedding {len(queries)} queries…")
    if DEVICE == "cuda": torch.cuda.empty_cache()
    q_embs = model.encode(queries, batch_size=BATCH_SIZE,
                          show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True)

    # Pre-compute query hash vectors
    q_hashes = np.array([hash_text(pingala_fn, q) for q in queries])
    q_hvecs  = np.array([hamming_vec(hamming_fn, q) for q in queries])

    # Brute-force ceiling
    print("\n⚡ Brute-force baseline…")
    bf_top1, bf_top10 = 0, 0
    for q_emb, gold in tqdm(zip(q_embs, gold_ids), total=len(queries)):
        result = brute_force(q_emb, embs)
        if gold == result[0]:  bf_top1  += 1
        if gold in result:     bf_top10 += 1
    bf_r1  = bf_top1  / len(queries) * 100
    bf_r10 = bf_top10 / len(queries) * 100

    # Four strategies
    strategies = [
        ("A: Pingala only",              False, False),
        ("B: Pingala + Hamming",         True,  False),
        ("C: Pingala + Madhava",         False, True),
        ("D: Pingala + Hamming + Madhava", True, True),
    ]

    results = {}
    for lbl, use_hamming, use_madhava in strategies:
        print(f"⚡ {lbl}…")
        hits1, hits10, times = 0, 0, []

        for i, (q_text, q_emb, gold) in enumerate(zip(queries, q_embs, gold_ids)):
            t0 = time.perf_counter()

            # Build Madhava scores if needed (per-query, O(N))
            m_scores = madhava_stage0_scores(q_text, passage_word_sets) \
                       if use_madhava else np.zeros(N, dtype=np.float32)

            # Stage 0 — blended shortlist
            if use_hamming and use_madhava:
                stage0 = stage0_all(q_hashes[i], q_hvecs[i],
                                    hash_mat, hamming_mat, m_scores, k)
            elif use_hamming:
                stage0 = stage0_pingala_hamming(q_hashes[i], q_hvecs[i],
                                                hash_mat, hamming_mat, k)
            elif use_madhava:
                stage0 = stage0_pingala_madhava(q_hashes[i], hash_mat,
                                                m_scores, k)
            else:
                stage0 = stage0_pingala(q_hashes[i], hash_mat, k)

            result = neural_cascade(q_emb, stage0, embs)
            times.append(time.perf_counter() - t0)

            if gold == result[0]: hits1  += 1
            if gold in result:    hits10 += 1

        results[lbl] = (hits1, hits10, np.mean(times) * 1000)

    # Print results
    n = len(queries)
    print(f"\n{'='*72}")
    print(f"  MADHAVA RESULTS — {label} | {N:,} passages | {n} queries")
    print(f"{'='*72}")
    print(f"  {'Method':<30} {'R@1':>7} {'R@10':>7} {'Ret@10':>8} {'ms/q':>7}")
    print(f"  {'-'*29:<30} {'-'*6:>7} {'-'*6:>7} {'-'*7:>8} {'-'*6:>7}")
    print(f"  {'Brute-force (768-dim)':<30} {bf_r1:>7.1f} {bf_r10:>7.1f} {'100.0':>8} {'—':>7}")

    for lbl, (h1, h10, ms) in results.items():
        r1  = h1  / n * 100
        r10 = h10 / n * 100
        ret = h10 / max(bf_top10, 1) * 100
        mark = " ✅" if ret >= 85 else " ⚠️"
        print(f"  {lbl:<30} {r1:>7.1f} {r10:>7.1f} {ret:>8.1f} {ms:>7.2f}{mark}")

    print(f"{'='*72}")
    print(f"\n  Weights: Hamming={HAMMING_W} Madhava={MADHAVA_W} LengthPenalty={LENGTH_W}")
    print(f"  R@1 = precision (top result correct)")
    print(f"  R@10 = recall (correct in top 10)\n")

if __name__ == "__main__":
    main()
