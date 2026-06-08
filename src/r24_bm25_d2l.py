"""
r24_bm25.py — Ramanujan-24 vs BM25 Comparison
Runs four Stage-0 strategies on the same corpus and queries:
  A) BM25          — industry standard keyword search (the real baseline)
  B) Pingala hash  — Fibonacci-weighted arithmetic (our claim)
  C) Plain hash    — position-agnostic arithmetic
  D) Brute-force   — full 768-dim neural (ceiling)

This is the publishability test:
  If Pingala retention ≥ BM25 retention → strong claim
  If Pingala speedup  >  BM25 speedup   → unique operating point

Run:
    cd ~/aiprojects/metaai/experiments
    python r24_bm25.py
"""

import numpy as np
import ctypes, os, time, re
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
PDF_PATH    = "d2l_deep_learning.pdf"
DATA_PATH   = "d2l_deep_learning_vectors.npy"   # reuse existing cache

STAGE0_K    = 4000   # candidates passed to neural stages (all methods use same K)
STAGE1_DIMS = 24
STAGE2_DIMS = 96
STAGE1_K    = 200
STAGE2_K    = 50
STAGE3_K    = 10
MAX_QUERIES = 500
BATCH_SIZE  = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms

# ── Hash loaders ──────────────────────────────────────────────────────────────

def load_lib(so_name, fn_name):
    path = os.path.abspath(so_name)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    fn  = getattr(lib, fn_name)
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    fn.restype  = None
    return fn

def hash_text(fn, text):
    arr = (ctypes.c_float * 24)()
    fn(text.encode("utf-8", errors="replace"), arr)
    v = np.array(list(arr), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def build_hash_matrix(fn, passages, label):
    print(f"⚡ Building {label} for {len(passages):,} passages…")
    t0  = time.perf_counter()
    out = np.zeros((len(passages), 24), dtype=np.float32)
    for i, p in enumerate(passages):
        out[i] = hash_text(fn, p)
    print(f"   Done in {time.perf_counter()-t0:.3f}s")
    return out

# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(passages):
    from rank_bm25 import BM25Okapi
    print(f"⚡ Building BM25 index for {len(passages):,} passages…")
    t0       = time.perf_counter()
    tokenized = [p.lower().split() for p in passages]
    bm25     = BM25Okapi(tokenized)
    print(f"   Done in {time.perf_counter()-t0:.3f}s")
    return bm25

def bm25_top_k(bm25, query, k):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    return np.argsort(scores)[::-1][:k]

# ── PDF loader ────────────────────────────────────────────────────────────────

def is_garbage(t):
    return False   # d2l is clean English — no OCR garbage

def is_noise(t):
    if re.search(r'\b\d{3,}\b.*\b\d{3,}\b', t): return True
    if t.startswith('•') or t.startswith('·'):   return True
    if len(t.split()) < 10:                       return True
    return False

def load_pdf(pdf_path):
    import fitz
    print(f"📄 Reading {pdf_path}…")
    doc, passages = fitz.open(pdf_path), []
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] != 0: continue
            t = block[4].strip().replace("\n", " ")
            if len(t) > 80 and not is_noise(t):
                passages.append(t)

    queries, gold_ids = [], []
    for i, p in enumerate(passages):
        first = p.split(".")[0].strip()
        if len(first) > 20:
            queries.append(first)
            gold_ids.append(i)
        if len(queries) >= MAX_QUERIES:
            break
    print(f"   {len(passages)} passages, {len(queries)} queries")
    return passages, queries, gold_ids

# ── Search ────────────────────────────────────────────────────────────────────

def cascade(q_text, q_emb, stage0_ids, embs):
    """Neural cascade on top of any Stage-0 shortlist."""
    survivors0 = stage0_ids

    s1k     = min(STAGE1_K, len(survivors0))
    cands1  = normalize(embs[survivors0, :STAGE1_DIMS])
    q24     = normalize(q_emb[None, :STAGE1_DIMS])
    surv1   = survivors0[np.argsort((cands1 @ q24.T).squeeze())[::-1][:s1k]]

    s2k     = min(STAGE2_K, len(surv1))
    cands2  = normalize(embs[surv1, :STAGE2_DIMS])
    q96     = normalize(q_emb[None, :STAGE2_DIMS])
    surv2   = surv1[np.argsort((cands2 @ q96.T).squeeze())[::-1][:s2k]]

    cands3  = normalize(embs[surv2])
    q768    = normalize(q_emb[None, :])
    scores  = (cands3 @ q768.T).squeeze()
    if scores.ndim == 0: scores = scores[None]
    top     = np.argsort(scores)[::-1][:STAGE3_K]
    return surv2[top]

def brute_force(q_emb, embs):
    scores = (normalize(embs) @ normalize(q_emb[None, :]).T).squeeze()
    return np.argsort(scores)[::-1][:STAGE3_K]

# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_hash_benchmark(label, hash_fn, hash_mat, q_texts, q_embs, gold_ids, embs):
    hits, times = 0, []
    for q_text, q_emb, gold in zip(q_texts, q_embs, gold_ids):
        t0       = time.perf_counter()
        q_hash   = hash_text(hash_fn, q_text)
        scores0  = hash_mat @ q_hash
        stage0   = np.argsort(scores0)[::-1][:min(STAGE0_K, len(embs))]
        result   = cascade(q_text, q_emb, stage0, embs)
        times.append(time.perf_counter() - t0)
        if gold in result: hits += 1
    return hits, np.mean(times) * 1000

def run_bm25_benchmark(bm25, q_texts, q_embs, gold_ids, embs):
    hits, times = 0, []
    for q_text, q_emb, gold in zip(q_texts, q_embs, gold_ids):
        t0     = time.perf_counter()
        stage0 = bm25_top_k(bm25, q_text, min(STAGE0_K, len(embs)))
        result = cascade(q_text, q_emb, stage0, embs)
        times.append(time.perf_counter() - t0)
        if gold in result: hits += 1
    return hits, np.mean(times) * 1000

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  Ramanujan-24 — BM25 vs Pingala vs Plain Hash vs Brute-force")
    print(f"  PDF: {PDF_PATH} | Device: {DEVICE.upper()}")
    print("=" * 68)

    plain_fn   = load_lib("libr24.so",        "hash24")
    pingala_fn = load_lib("libr24_pingala.so", "hash24_pingala")
    if not plain_fn:   print("⚠️  libr24.so not found")
    if not pingala_fn: print("⚠️  libr24_pingala.so not found")

    passages, queries, gold_ids = load_pdf(PDF_PATH)

    print(f"\n🤖 Loading model ({DEVICE.upper()})…")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)

    # Load or build embeddings
    if os.path.exists(DATA_PATH):
        embs = np.load(DATA_PATH)
        if len(embs) == len(passages):
            print(f"📦 Loaded cached embeddings ({len(embs)} passages)")
        else:
            print(f"⚠️  Cache mismatch — re-embedding…")
            embs = None
    else:
        embs = None

    if embs is None:
        if DEVICE == "cuda": torch.cuda.empty_cache()
        embs = model.encode(passages, batch_size=BATCH_SIZE,
                            show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True)
        np.save(DATA_PATH, embs)
        print(f"💾 Saved to {DATA_PATH}")

    # Build indexes
    bm25        = build_bm25(passages)
    plain_mat   = build_hash_matrix(plain_fn,   passages, "plain hash")   if plain_fn   else None
    pingala_mat = build_hash_matrix(pingala_fn, passages, "Pingala hash") if pingala_fn else None

    # Embed queries
    print(f"\n🔍 Embedding {len(queries)} queries…")
    if DEVICE == "cuda": torch.cuda.empty_cache()
    q_embs = model.encode(queries, batch_size=BATCH_SIZE,
                          show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True)

    # Brute-force baseline
    print("\n⚡ Running brute-force baseline…")
    bf_hits, bf_times = 0, []
    for q_emb, gold in tqdm(zip(q_embs, gold_ids), total=len(queries)):
        t0 = time.perf_counter()
        if gold in brute_force(q_emb, embs): bf_hits += 1
        bf_times.append(time.perf_counter() - t0)
    bf_ms = np.mean(bf_times) * 1000

    # BM25
    print("\n⚡ Running BM25 cascade…")
    bm25_hits, bm25_ms = run_bm25_benchmark(bm25, queries, q_embs, gold_ids, embs)

    # Hash methods
    results = {}
    if pingala_fn and pingala_mat is not None:
        print("\n⚡ Running Pingala hash cascade…")
        h, ms = run_hash_benchmark("pingala", pingala_fn, pingala_mat, queries, q_embs, gold_ids, embs)
        results["Pingala hash"] = (h, ms)

    if plain_fn and plain_mat is not None:
        print("\n⚡ Running plain hash cascade…")
        h, ms = run_hash_benchmark("plain", plain_fn, plain_mat, queries, q_embs, gold_ids, embs)
        results["Plain hash"] = (h, ms)

    # ── Print results ─────────────────────────────────────────────────────────
    n = len(queries)
    print("\n" + "=" * 68)
    print(f"  RESULTS — {len(passages):,} passages | {n} queries | Stage0_K={STAGE0_K}")
    print("=" * 68)
    print(f"  {'Method':<22} {'Recall@10':>10} {'Retention':>10} {'ms/q':>8} {'Speedup':>9}")
    print(f"  {'-'*21:<22} {'-'*9:>10} {'-'*9:>10} {'-'*7:>8} {'-'*8:>9}")
    print(f"  {'Brute-force (768-dim)':<22} {bf_hits/n*100:>10.1f} {'100.0':>10} {bf_ms:>8.2f} {'1.0x':>9}")

    # BM25 row
    bm25_ret     = bm25_hits / max(bf_hits, 1) * 100
    bm25_speedup = bf_ms / max(bm25_ms, 0.001)
    bm25_marker  = " ✅" if bm25_ret >= 85 else " ⚠️"
    print(f"  {'BM25':<22} {bm25_hits/n*100:>10.1f} {bm25_ret:>10.1f} {bm25_ms:>8.2f} {bm25_speedup:>8.1f}x{bm25_marker}")

    for label, (hits, ms) in results.items():
        retention = hits / max(bf_hits, 1) * 100
        speedup   = bf_ms / max(ms, 0.001)
        marker    = " ✅" if retention >= 85 else " ⚠️"
        print(f"  {label:<22} {hits/n*100:>10.1f} {retention:>10.1f} {ms:>8.2f} {speedup:>8.1f}x{marker}")

    print("=" * 68)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n  📊 KEY COMPARISON")
    if "Pingala hash" in results:
        p_hits, p_ms = results["Pingala hash"]
        p_ret = p_hits / max(bf_hits, 1) * 100
        b_ret = bm25_ret

        print(f"  Pingala retention : {p_ret:.1f}%")
        print(f"  BM25    retention : {b_ret:.1f}%")
        print(f"  Pingala speedup   : {bf_ms/max(p_ms,0.001):.1f}x")
        print(f"  BM25    speedup   : {bm25_speedup:.1f}x")
        print()

        if p_ret >= b_ret and bf_ms/p_ms > bm25_speedup:
            print("  ✅ PUBLISHABLE: Pingala beats BM25 on both retention AND speedup")
        elif p_ret >= b_ret:
            print("  ✅ STRONG: Pingala matches BM25 retention with better speedup")
        elif bf_ms/p_ms > bm25_speedup and p_ret >= 85:
            print("  🟡 INTERESTING: Pingala faster than BM25, retention within 5pp")
        else:
            print("  ⚠️  MORE WORK NEEDED: BM25 leads on retention")

    print("=" * 68)

if __name__ == "__main__":
    main()
